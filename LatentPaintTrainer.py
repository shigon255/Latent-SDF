import sys
from pathlib import Path
from typing import Any, Dict, Union, List

import imageio
import os
import cv2 as cv
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
from gpu_mem_track import MemTracker
import inspect
import trimesh

import utils
from confs.train_config import TrainConfig
from models.views_dataset import ViewsDataset
from models.dataset import Dataset # NeuS images dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import GeoNeuSRenderer, LatentPaintRenderer,  get_psnr
from stable_diffusion import StableDiffusion
from IF import IFDiffusion
from utils import make_path, tensor2numpy, numpy2image, near_far_from_sphere, read_intrinsic_inv, gen_random_ray_at_pose
import gc

'''
Latent-Paint Trainer
'''


class LatentPaintTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.train_step = 0
        self.device = torch.device(cfg.global_setting.gpu)
        self.half = cfg.global_setting.half

        # load neus config
        self.neus_cfg_path = cfg.neus.neus_cfg_path
        self.case = cfg.neus.case

        f = open(self.neus_cfg_path)
        neus_cfg_text = f.read()
        neus_cfg_text = neus_cfg_text.replace('CASE_NAME', cfg.neus.case)
        f.close()
        self.neus_cfg = ConfigFactory.parse_string(neus_cfg_text)
        self.neus_cfg['dataset.data_dir'] = self.neus_cfg['dataset.data_dir'].replace('CASE_NAME', cfg.neus.case)

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
        self.anneal_end = self.neus_cfg['train.anneal_end']

        # networks
        if self.half:
            self.nerf_outside = NeRF(**self.neus_cfg['latent_model.nerf']).half().to(self.device)
            self.color_network = RenderingNetwork(**self.neus_cfg['latent_model.rendering_network']).half().to(self.device)
            self.deviation_network = SingleVarianceNetwork(**self.neus_cfg['model.variance_network']).half().to(self.device)
            self.sdf_network = SDFNetwork(**self.neus_cfg['model.sdf_network']).half().to(self.device)
        else:
            self.nerf_outside = NeRF(**self.neus_cfg['latent_model.nerf']).to(self.device)
            self.color_network = RenderingNetwork(**self.neus_cfg['latent_model.rendering_network']).to(self.device)
            self.deviation_network = SingleVarianceNetwork(**self.neus_cfg['model.variance_network']).to(self.device)
            self.sdf_network = SDFNetwork(**self.neus_cfg['model.sdf_network']).to(self.device)

        self.sdf_network.eval()
        # self.sdf_network.freeze() # can not freeze, because gradient need to be calculated
        params_to_train = []
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.params_to_train = params_to_train
        self.renderer = LatentPaintRenderer(self.nerf_outside, 
                                                self.sdf_network,
                                                self.deviation_network,
                                                self.color_network,
                                                **self.neus_cfg['model.neus_renderer']
                                                )
        self.use_white_bkgd = cfg.neus.use_white_bkgd

        self.diffusion = self.init_diffusion()
        self.text_z = self.calc_text_embeddings()
        self.optimizer = self.init_optimizer(params_to_train)
        # self.dataloaders = self.init_dataloaders() # random view dataset
        # instead of load the whole Geo-NeuS dataset, only load the data we need(intrinsic)
        self.img_dataset = Dataset(self.neus_cfg['dataset'], device=self.device, half=self.half)
        self.intrinsic_inv = read_intrinsic_inv(self.neus_cfg['dataset']).to(torch.float16).to(self.device)
        self.train_H = self.cfg.render.train_grid_H
        self.train_W = self.cfg.render.train_grid_W
        self.eval_H = self.cfg.render.eval_grid_H
        self.eval_W = self.cfg.render.eval_grid_W

        self.eval_size = 1 # randomly evaluate 1 image
        self.full_eval_size = self.img_dataset.n_images # evaluate all images

        # frame = inspect.currentframe()     
        # self.gpu_tracker = MemTracker(frame) 
        # self.gpu_tracker.track()

        self.past_checkpoints = []
        if self.cfg.optim.resume:
            self.load_checkpoint(model_only=False)
        if cfg.neus.load_from_neus:
            self.load_checkpoint_only_sdf(cfg.neus.neus_ckpt_path)
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
    def init_diffusion(self):#  -> StableDiffusion:
        # diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
          #                                 concept_name=self.cfg.guide.concept_name,
            #                               latent_mode=True, half=self.half)
                                          # latent_mode=self.mesh_model.latent_mode)
        diffusion_model = IFDiffusion(self.device, half=self.half)
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

    def init_optimizer(self, params_to_train) -> Optimizer:
        # optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99), eps=1e-15)
        optimizer = torch.optim.Adam(params_to_train, lr=self.cfg.optim.lr, betas=(0.9, 0.99), eps=1e-15)
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
        # self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.evaluate(self.eval_renders_path)
        # self.mesh_model.train()
        self.nerf_outside.train()
        self.deviation_network.train()
        self.color_network.train()

        pbar = tqdm(total=self.cfg.optim.iters, initial=self.train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        while self.train_step < self.cfg.optim.iters:
            # Keep going over dataloader until finished the required number of iterations
            # for i, data in enumerate(self.dataloaders['train']):
            for i in range(self.img_dataset.n_images):
                self.train_step += 1
                pbar.update(1)

                self.optimizer.zero_grad()
                # pred_latents: (1, 4, 64, 64)
                # pred_rgb: (H/l, W/l, 3)
                pred_rgb, loss = self.train_render(i) # render ith image
                self.optimizer.step()
                # if np.random.uniform(0, 1) < 0.05:
                if np.random.uniform(0, 1) < 1.0:
                    # Randomly log rendered images throughout the training                
                    self.log_train_renders(pred_rgb, i)
                
                del pred_rgb

                if self.train_step % self.cfg.log.save_interval == 0:
                    self.save_checkpoint(full=True)
                    # self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                    self.evaluate(self.eval_renders_path)
                    # self.mesh_model.train()
                    self.nerf_outside.train()
                    self.deviation_network.train()
                    self.color_network.train()
        
        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        self.full_eval()
        logger.info('\tDone!')

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.train_step / self.anneal_end])

    def evaluate(self, save_path: Path, full_eval: bool = False):
        logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
        # self.mesh_model.eval()
        self.nerf_outside.eval()
        self.deviation_network.eval()
        self.color_network.eval()
        save_path.mkdir(exist_ok=True)

        eval_size = self.eval_size
        if full_eval:
            logger.info("Start full evaluation")
            all_preds = []
            eval_size = self.full_eval_size
        else:
            logger.info("Start normal evaluation")
        for i in range(eval_size):
            if not full_eval:
                img_idx = np.random.randint(self.img_dataset.n_images)
            else:
                img_idx = i
            pred_rgb, textures = self.eval_render(img_idx=img_idx) # note that textures contain dummy value

            # encode latent into rgb
            # pred_latents = pred_latents.permute(2, 0, 1).unsqueeze(0) # (train_grid_size, train_grid_size, 4) -> (1, 4, train_grid_size, train_grid_size)
            
            # pred_rgb = self.diffusion.decode_latents(pred_latents).permute(0, 2, 3, 1).contiguous() # -> (1, train_grid_size, train_grid_size, 3)
            # pred_rgb = tensor2numpy(pred_rgb[0])
            # pred_rgb = tensor2numpy(pred_rgb)
            pred_rgb = numpy2image(pred_rgb)

            if full_eval:
                all_preds.append(pred_rgb)
            else:
                cv.imwrite(os.path.join(save_path, f"step_{self.train_step:05d}_{i:04d}_rgb.png"), pred_rgb)

        # also store mesh
        # self.validate_mesh_vertex_color(world_space=True, resolution=512, threshold=self.cfg.log.mcube_threshold)
        '''
        Project
        No texture can be shown/saved
        '''
        # Texture map is the same, so just take the last result
        # texture = tensor2numpy(textures[0])
        # Image.fromarray(texture).save(save_path / f"step_{self.train_step:05d}_texture.png")

        if full_eval:
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"step_{self.train_step:05d}_{name}.mp4", video, fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'rgb')
        
        logger.info('Done!')

    def full_eval(self):
        try:
            # self.evaluate(self.dataloaders['val_large'], self.final_renders_path, save_as_video=True)
            self.evaluate(self.final_renders_path, full_eval=True)
        except:
            logger.error('failed to save result video')
        '''
        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path, guidance=self.diffusion)

            logger.info(f"\tDone!")
        '''

    def render_single_image(self, img_H, img_W, resolution_level, is_train, img_idx):
        # if self.cfg.optim.use_neus_view:
          #   rays_o, rays_d, intrinsic, intrinsic_inv, pose, image_gray = self.img_dataset.gen_rays_at(img_idx, H=img_H, W=img_W, resolution_level=resolution_level)
        # else:
          #   rays_o, rays_d = gen_random_ray_at_pose(theta, phi, radius, H=img_H, W=img_W, intrincis_inv=self.intrinsic_inv, resolution_level=resolution_level, half=self.half)
        # abandon using view dataset
        rays_o, rays_d, intrinsic, intrinsic_inv, pose, image_gray = self.img_dataset.gen_rays_at(img_idx, H=img_H, W=img_W, resolution_level=resolution_level)
        
        H, W, _ = rays_o.shape
        
        rays_o = rays_o.reshape(-1, 3).split(self.neus_cfg['train.batch_size'])
        rays_d = rays_d.reshape(-1, 3).split(self.neus_cfg['train.batch_size'])

        out_fine = []
        for idx, (rays_o_batch, rays_d_batch) in enumerate(zip(rays_o, rays_d)):
            rays_o_batch = rays_o_batch.to(self.device)
            rays_d_batch = rays_d_batch.to(self.device)
            near, far = near_far_from_sphere(rays_o_batch, rays_d_batch)
            
            # background_latent = torch.ones([1, 4]) if self.use_white_bkgd else None
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            # memory added from here
            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(), # skip cosine annealing
                                              background_rgb=background_rgb,
                                              intrinsics=intrinsic,
                                              intrinsics_inv=intrinsic_inv,
                                              poses=pose,
                                              images=image_gray)
          
            if is_train:
                if self.half:
                    out_fine.append(render_out['color_fine'].half())
                else:
                    out_fine.append(render_out['color_fine'])
            else:
                if self.half:
                    out_fine.append(render_out['color_fine'].half().detach().cpu().numpy())
                else:
                    out_fine.append(render_out['color_fine'].detach().cpu().numpy())
            
            del render_out
            '''
            if only del render_out here,
            we can even not use torch.cuda.empty_cache()
            '''
            # gc.collect()
            # torch.cuda.empty_cache()
            # print(f"after self.renderer.render", self.gpu_tracker.track())
        # if is_train:
          #   img_fine = (torch.cat(out_latent_fine, dim=0).reshape([H, W, 4])) # latent image, do not multiply 256, since it'll be used to training
        # else:
          #   img_fine = (np.concatenate(out_latent_fine, axis=0).reshape([H, W, 4]))
        # img_fine = (torch.cat(out_latent_fine, dim=0).reshape([H, W, 4]))
        if is_train:
            img_fine = (torch.cat(out_fine, dim=0).reshape([H, W, 3]))
        else:
            img_fine = (np.concatenate(out_fine, axis=0).reshape([H, W, 3]))
        
        return img_fine
    
    def train_render(self, img_idx):# , data: Dict[str, Any]):
        # theta = data['theta']
        # phi = data['phi']
        # radius = data['radius']
        # outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius)
        # pred_rgb = outputs['image']
        
        # (H/l, W/l, 3)
        pred_rgb = self.render_single_image(img_idx=img_idx, img_H=self.train_H, img_W=self.train_W, resolution_level=self.neus_cfg['train.train_resolution_level'], is_train=True)    
        # (1. 3. H/l, W/l)
        pred_rgb = pred_rgb.permute(2, 0, 1).unsqueeze(0)
        
        # pred_latents = pred_latents.permute((2, 0, 1)).unsqueeze(0)
        # text embeddings
        text_z = self.text_z
        # if self.cfg.guide.append_direction:
        #     dirs = data['dir']  # [B,]
        #     text_z = self.text_z[dirs]
        # else:
        #     text_z = self.text_z
        
        # loss_guidance = self.diffusion.train_step(text_z, pred_latents)
        loss_guidance = self.diffusion.train_step(text_z, pred_rgb)
        loss = loss_guidance # Note: this loss value will be 0. The real loss value can't be calculated
        loss = -1

        pred_rgb = pred_rgb.squeeze(0).permute(1, 2, 0)
        return pred_rgb, loss
    
    def eval_render(self, img_idx):
        '''
        create the latent image
        '''
        # theta = data['theta']
        # phi = data['phi']
        # radius = data['radius']
        # dim = self.cfg.render.eval_grid_size
        # outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, decode_func=self.diffusion.decode_latents,
          #                                test=True ,dims=(dim,dim))
        # pred_rgb = outputs['image'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        # randomly select image
        pred_rgb = self.render_single_image(img_H=self.eval_H, img_W=self.eval_W, resolution_level=self.neus_cfg['train.validate_resolution_level'], is_train=False, img_idx=img_idx)        
        # texture_rgb = outputs['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        return pred_rgb, -1  

    def log_train_renders(self, preds: torch.Tensor, img_idx):
        # if self.mesh_model.latent_mode:
          #   pred_rgb = self.diffusion.decode_latents(preds).permute(0, 2, 3, 1).contiguous()  # [1, 3, H, W]
        # else:
          #   pred_rgb = preds.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        # (1, 4, train_grid_size, train_grid_size) -> (train_grid_size, train_grid_size, 3)
        # pred_rgb = self.diffusion.decode_latents(preds).permute(0, 2, 3, 1).contiguous()
        logger.info(f"log image {img_idx} at step {self.train_step}")
        pred_rgb = preds
        save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
        save_path.parent.mkdir(exist_ok=True)

        pred_rgb = tensor2numpy(pred_rgb)
        # pred_rgb = numpy2image(pred_rgb)

        # Image.fromarray(pred_rgb).save(save_path)
        cv.imwrite(os.path.join(save_path), pred_rgb)

    # TODO: figure out how to find the color of vertices
    def validate_mesh_vertex_color(self, world_space=False, resolution=64, threshold=0.0, name=None):
        print('Start exporting textured mesh')

        bound_min = torch.tensor(self.img_dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.img_dataset.object_bbox_max, dtype=torch.float32)
        vertices, triangles = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution,
                                                               threshold=threshold)
        print(f'Vertices count: {vertices.shape[0]}')

        vertices = torch.tensor(vertices, dtype=torch.float32)
        vertices_batch = vertices.split(self.neus_cfg['train.batch_size'])
        render_iter = len(vertices_batch)

        vertex_colors = []
        for iter in tqdm(range(render_iter)):
            feature_vector = self.sdf_network.sdf_hidden_appearance(vertices_batch[iter])[:, 1:]
            gradients = self.sdf_network.gradient(vertices_batch[iter]).squeeze()
            dirs = -gradients
            # vertex color: (self.neus_cfg['train.batch_size'], 4)
            vertex_color = self.color_network(vertices_batch[iter], gradients, dirs,
                                                feature_vector).reshape(self.neus_cfg['train.batch_size'], 4)  # BGR to RGB
            print(vertex_color.shape)
            vertex_color = vertex_color.view(self.neus_cfg['train.batch_size'] // 2, 2, 4).permute(2, 0, 1).unsqueeze(0)
            print(vertex_color.shape)
            vertex_color = self.diffusion.decode_latents(vertex_color).permute(0, 2, 3, 1).contiguous() # -> (1, self.neus_cfg['train.batch_size'] // 2, 2, 3)
            print(vertex_color.shape)
            vertex_color = vertex_color.squeeze(0).view(1, self.neus_cfg['train.batch_size'], 3) # -> (self.neus_cfg['train.batch_size'], 3)
            print(vertex_color.shape)
            vertex_color = vertex_color.detach().cpu().numpy()
            vertex_colors.append(vertex_color)
        vertex_colors = np.concatenate(vertex_colors)
        print(f'validate point count: {vertex_colors.shape[0]}')
        vertices = vertices.detach().cpu().numpy()

        if world_space:
            vertices = vertices * self.img_dataset.scale_mats_np[0][0, 0] + self.img_dataset.scale_mats_np[0][:3, 3][None]

        os.makedirs(os.path.join(self.exp_path, 'meshes'), exist_ok=True)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
        if name is not None:
            mesh.export(os.path.join(self.exp_path, 'meshes', f'{name}.ply'))
        else:
            mesh.export(os.path.join(self.exp_path, 'meshes', '{:0>8d}_vertex_color.ply'.format(self.train_step)))

        logger.info('End')

    def load_checkpoint_only_sdf(self, ckpt_path):
        # For load SDF work from pretrained Geo-NeuS model
        # NeRF, variance, and radiance network should not be loaded
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine']) # assume that this checkpoint is saved from Geo-NeuS
        self.nerf_outside.load_state_dict(checkpoint['nerf']) # assume that this checkpoint is saved from Geo-NeuS
        self.color_network.load_state_dict(checkpoint['color_network_fine']) # assume that this checkpoint is saved from Geo-NeuS
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine']) # assume that this checkpoint is saved from Geo-NeuS

        logger.info('End')

    def load_checkpoint_from_neus(self, ckpt_path):
        # For load SDF, radiance, NeRF, variance network from pretrained Geo-NeuS model
        # NeRF, variance, and radiance network should not be loaded
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine']) # assume that this checkpoint is saved from Geo-NeuS
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])

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
