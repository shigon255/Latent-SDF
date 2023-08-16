import sys
from pathlib import Path
from typing import Any, Dict, Union, List

import imageio
import os
import numpy as np
import pyrallis
import torch
from PIL import Image
from loguru import logger
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from pyhocon import ConfigFactory

import utils
from confs.train_config import TrainConfig
from models.views_dataset import ViewsDataset
from models.dataset import Dataset # NeuS images dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import GeoNeuSRenderer, LatentPaintRenderer,  get_psnr
from stable_diffusion import StableDiffusion
from utils import make_path, tensor2numpy

'''
Latent-Paint Trainer
'''

class LatentPaintTrainer:
    def __init__(self, cfg: TrainConfig, args):
        self.cfg = cfg
        self.train_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load neus config
        self.neus_cfg_path = args.neus_cfg_path
        self.case = args.case

        f = open(self.neus_cfg_path)
        neus_cfg_text = f.read()
        neus_cfg_text = neus_cfg_text.replace('CASE_NAME', args.case)
        f.close()
        self.neus_cfg = ConfigFactory.parse_string(neus_cfg_text)
        self.neus_cfg['dataset.data_dir'] = self.neus_cfg['dataset.data_dir'].replace('CASE_NAME', args.case)

        utils.seed_everything(self.cfg.optim.seed)

        # Make dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        # self.mesh_model = self.init_mesh_model()
        
        # networks
        self.nerf_outside = NeRF(**self.conf['latent_model.nerf']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['latent_model.rendering_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)

        self.sdf_network.freeze() # freeze the sdf network
        params_to_train = []
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.renderer = LatentPaintRenderer(self.nerf_outside, 
                                                self.sdf_network,
                                                self.deviation_network,
                                                self.color_network,
                                                **self.conf['models.neus_renderer']
                                                )
        self.use_white_bkgd = args.use_white_bkgd

        self.diffusion = self.init_diffusion()
        self.text_z = self.calc_text_embeddings()
        self.optimizer = self.init_optimizer()
        self.dataloaders = self.init_dataloaders() # random view dataset
        self.img_dataset = Dataset(self.conf['dataset'])

        self.past_checkpoints = []
        if self.cfg.optim.resume:
            self.load_checkpoint(model_only=False)
        if args.load_from_neus:
            self.load_checkpoint_only_sdf(args.neus_ckpt_path)
        if self.cfg.optim.ckpt is not None:
            self.load_checkpoint(self.cfg.optim.ckpt, model_only=True)

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    # Do not use the mesh model
    '''
    def init_mesh_model(self) -> nn.Module:
        if self.cfg.render.backbone == 'texture-mesh':
            from src.latent_paint.models.textured_mesh import TexturedMeshModel
            model = TexturedMeshModel(self.cfg, device=self.device, render_grid_size=self.cfg.render.train_grid_size,
                                      latent_mode=True, texture_resolution=self.cfg.guide.texture_resolution).to(self.device)
        elif self.cfg.render.backbone == 'texture-rgb-mesh':
            from src.latent_paint.models.textured_mesh import TexturedMeshModel
            model = TexturedMeshModel(self.cfg, device=self.device, render_grid_size=self.cfg.render.train_grid_size,
                                      latent_mode=False, texture_resolution=self.cfg.guide.texture_resolution).to(self.device)
        else:
            raise NotImplementedError(f'--backbone {self.cfg.render.backbone} is not implemented!')

        model = model.to(self.device)
        logger.info(
            f'Loaded {self.cfg.render.backbone} Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model
    '''
    def init_diffusion(self) -> StableDiffusion:
        diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          latent_mode=self.mesh_model.latent_mode)
        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
        else:
            text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                text = f"{ref_text}, {d} view"
                text_z.append(self.diffusion.get_text_embeds([text]))
        return text_z

    def init_optimizer(self) -> Optimizer:
        # optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99), eps=1e-15)
        optimizer = torch.optim.Adam(self.params_to_train, lr=self.cfg.optim.lr, betas=(0.9, 0.99), eps=1e-15)
        return optimizer

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        train_dataloader = ViewsDataset(self.cfg.render, device=self.device, type='train', size=100).dataloader()
        val_loader = ViewsDataset(self.cfg.render, device=self.device, type='val',
                                  size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device, type='val',
                                        size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': train_dataloader, 'val': val_loader, 'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def train(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        # self.mesh_model.train()
        self.nerf_outside.train()
        self.deviation_network.train()
        self.color_network.train()

        pbar = tqdm(total=self.cfg.optim.iters, initial=self.train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        while self.train_step < self.cfg.optim.iters:
            # Keep going over dataloader until finished the required number of iterations
            for data in self.dataloaders['train']:
                self.train_step += 1
                pbar.update(1)

                self.optimizer.zero_grad()

                pred_rgbs, loss = self.train_render(data)

                self.optimizer.step()

                if self.train_step % self.cfg.log.save_interval == 0:
                    self.save_checkpoint(full=True)
                    self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                    # self.mesh_model.train()
                    self.nerf_outside.train()
                    self.deviation_network.train()
                    self.color_network.train()

                if np.random.uniform(0, 1) < 0.05:
                    # Randomly log rendered images throughout the training
                    self.log_train_renders(pred_rgbs)
        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
        # self.mesh_model.eval()
        self.nerf_outside.eval()
        self.deviation_network.eval()
        self.color_network.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
        for i, data in enumerate(dataloader):
            preds, textures = self.eval_render(data)

            # tensor 2 image, note that the color range will transform from [0, 1] to [0, 255]
            # and the tensor will be detach()
            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                Image.fromarray(pred).save(save_path / f"step_{self.train_step:05d}_{i:04d}_rgb.png")

        '''
        Project
        No texture can be shown/saved
        '''
        # Texture map is the same, so just take the last result
        # texture = tensor2numpy(textures[0])
        # Image.fromarray(texture).save(save_path / f"step_{self.train_step:05d}_texture.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"step_{self.train_step:05d}_{name}.mp4", video, fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'rgb')
        logger.info('Done!')

    def full_eval(self):
        try:
            self.evaluate(self.dataloaders['val_large'], self.final_renders_path, save_as_video=True)
        except:
            logger.error('failed to save result video')
        '''
        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path, guidance=self.diffusion)

            logger.info(f"\tDone!")
        '''

    def train_render(self, data: Dict[str, Any]):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']

        # outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius)
        # pred_rgb = outputs['image']
        # volume rendering, note that pred_rgb is latent, not the real rgb
        pred_rgb = self.render_single_image(theta, phi, radius, resolution_level=4)    

        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs]
        else:
            text_z = self.text_z

        # Guidance loss
        loss_guidance = self.diffusion.train_step(text_z, pred_rgb)
        loss = loss_guidance

        return pred_rgb, loss

    def render_single_image(self, theta, phi, radius, resolution_level):
        rays_o, rays_d = self.dataset.gen_random_ray_at_pose(theta, phi, radius, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_latent_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.img_dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            # background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            background_latent = torch.ones([1, 4]) if self.use_white_bkgd else None

            render_out_latent = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=0.5, # skip cosine annealing
                                              background_rgb=background_latent)

            out_latent_fine.append(render_out_latent['color_fine'].detach().cpu().numpy())

            del render_out_latent

        # img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        img_fine = (np.concatenate(out_latent_fine, axis=0).reshape([H, W, 4])) # latent image, do not multiply 256, since it'll be used to training
        return img_fine
    
    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        # dim = self.cfg.render.eval_grid_size
        # outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, decode_func=self.diffusion.decode_latents,
          #                                test=True ,dims=(dim,dim))
        # pred_rgb = outputs['image'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        pred_rgb = self.render_single_image(theta, phi, radius, resolution_level=4)
        # texture_rgb = outputs['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)

        # we don't have texture, just return the pred_rgb
        return pred_rgb, -1  #texture_rgb

    def log_train_renders(self, preds: torch.Tensor):
        if self.mesh_model.latent_mode:
            pred_rgb = self.diffusion.decode_latents(preds).permute(0, 2, 3, 1).contiguous()  # [1, 3, H, W]
        else:
            pred_rgb = preds.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
        save_path.parent.mkdir(exist_ok=True)

        pred_rgb = tensor2numpy(pred_rgb[0])

        Image.fromarray(pred_rgb).save(save_path)

    def load_checkpoint_only_sdf(self, ckpt_path):
        # For load SDF work from pretrained Geo-NeuS model
        # NeRF, variance, and radiance network should not be loaded
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine']) # assume that this checkpoint is saved from Geo-NeuS

        logger.info('End')

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(self.ckpt_path.glob('*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                logger.info(f"Latest checkpoint is {checkpoint}")
            else:
                logger.info("No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        def decode_texture_img(latent_texture_img):
            decoded_texture = self.diffusion.decode_latents(latent_texture_img)
            decoded_texture = F.interpolate(decoded_texture,
                                            (self.cfg.guide.texture_resolution, self.cfg.guide.texture_resolution),
                                            mode='bilinear', align_corners=False)
            return decoded_texture
        # load network
        self.nerf_outside.load_state_dict(checkpoint_dict['latent_nerf'])
        self.sdf_network.load_state_dict(checkpoint_dict['latent_sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint_dict['latent_variance_network_fine'])
        self.color_network.load_state_dict(checkpoint_dict['latent_color_network_fine'])
        '''
        if 'model' not in checkpoint_dict:
            if not self.mesh_model.latent_mode:
                # initialize the texture rgb image from the latent texture image
                checkpoint_dict['texture_img_rgb_finetune'] = decode_texture_img(checkpoint_dict['texture_img'])
            self.mesh_model.load_state_dict(checkpoint_dict)
            logger.info("loaded model.")
            return

        if not self.mesh_model.latent_mode:
            # initialize the texture rgb image from the latent texture image
            checkpoint_dict['model']['texture_img_rgb_finetune'] = \
            decode_texture_img(checkpoint_dict['model']['texture_img'])

        missing_keys, unexpected_keys = self.mesh_model.load_state_dict(checkpoint_dict['model'], strict=False)
        logger.info("loaded model.")
        if len(missing_keys) > 0:
            logger.warning(f"missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"unexpected keys: {unexpected_keys}")

        if model_only:
            return
        '''
        self.past_checkpoints = checkpoint_dict['checkpoints']
        self.train_step = checkpoint_dict['train_step'] + 1
        logger.info(f"load at step {self.train_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                logger.info("loaded optimizer.")
            except:
                logger.warning("Failed to load optimizer.")

    def save_checkpoint(self, full=False):

        name = f'step_{self.train_step:06d}'

        state = {
            'train_step': self.train_step,
            'checkpoints': self.past_checkpoints,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()

        # state['model'] = self.mesh_model.state_dict()
        state['latent_nerf'] = self.nerf_outside.state_dict()
        state['latent_sdf_network_fine'] = self.sdf_network.state_dict(),
        state['latent_variance_network_fine'] = self.deviation_network.state_dict(),
        state['latent_color_network_fine'] = self.color_network.state_dict(),

        file_path = f"{name}.pth"

        self.past_checkpoints.append(file_path)

        if len(self.past_checkpoints) > self.cfg.log.max_keep_ckpts:
            old_ckpt = self.ckpt_path / self.past_checkpoints.pop(0)
            old_ckpt.unlink(missing_ok=True)

        torch.save(state, self.ckpt_path / file_path)
