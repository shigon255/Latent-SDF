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
        self.latent = cfg.global_setting.latent
        self.color_ch = 4 if self.latent else 3

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

        model_name = "latent_model" if self.latent else "model"

        # networks
        if self.half:
            self.nerf_outside = NeRF(**self.neus_cfg[model_name+'.nerf']).half().to(self.device)
            self.color_network = RenderingNetwork(**self.neus_cfg[model_name+'.rendering_network']).half().to(self.device)
            self.deviation_network = SingleVarianceNetwork(**self.neus_cfg[model_name+'.variance_network']).half().to(self.device)
            self.sdf_network = SDFNetwork(**self.neus_cfg[model_name+'.sdf_network']).half().to(self.device)
        else:
            self.nerf_outside = NeRF(**self.neus_cfg[model_name+'.nerf']).to(self.device)
            self.color_network = RenderingNetwork(**self.neus_cfg[model_name+'.rendering_network']).to(self.device)
            self.deviation_network = SingleVarianceNetwork(**self.neus_cfg[model_name+'.variance_network']).to(self.device)
            self.sdf_network = SDFNetwork(**self.neus_cfg[model_name+'.sdf_network']).to(self.device)

        self.sdf_network.eval()
        self.deviation_network.eval()
        self.color_network.train()
        if self.latent:
            self.nerf_outside.train()
        else:
            self.nerf_outside.eval()
        self.sdf_network.freeze() # can not freeze, because gradient need to be calculated
        self.deviation_network.freeze()
        params_to_train = []
        # params_to_train += list(self.nerf_outside.parameters())
        # params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        if self.latent:
            params_to_train += list(self.nerf_outside.parameters())


        self.params_to_train = params_to_train
        self.renderer = LatentPaintRenderer(self.nerf_outside, 
                                                self.sdf_network,
                                                self.deviation_network,
                                                self.color_network,
                                                color_ch=self.color_ch,
                                                **self.neus_cfg[model_name+'.neus_renderer']
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

        self.linear_rgb_estimator = torch.tensor([
            #   R       G       B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ], dtype=torch.float32).to(self.device)

        # inverse linear approx to find latent
        A = self.linear_rgb_estimator.T
        regularizer = 1e-2
        self.linear_rgb_estimator_inv = torch.pinverse(A.T @ A + regularizer * torch.eye(4, dtype=torch.float32).to(self.device)) @ A.T
        if self.half:
            self.linear_rgb_estimator = self.linear_rgb_estimator.to(torch.float16)
            self.linear_rgb_estimator_inv = self.linear_rgb_estimator_inv.to(torch.float16)
            
        # frame = inspect.currentframe()     
        # self.gpu_tracker = MemTracker(frame) 
        # self.gpu_tracker.track()

        self.past_checkpoints = []
        if self.cfg.optim.resume:
            self.load_checkpoint(model_only=False)
        if cfg.neus.load_from_neus:
            if self.latent:
                self.load_checkpoint_only_sdf_dev(cfg.neus.neus_ckpt_path)
            else:
                self.load_checkpoint_from_neus(cfg.neus.neus_ckpt_path)
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
        if self.latent:
            diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                            concept_name=self.cfg.guide.concept_name,
                                            latent_mode=True, half=self.half)
        else:
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
        # self.mesh_model.train()

        self.evaluate(self.eval_renders_path)
        self.color_network.train()
        if self.latent:
            self.nerf_outside.train()

        pbar = tqdm(total=self.cfg.optim.iters, initial=self.train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        while self.train_step < self.cfg.optim.iters:
            # Keep going over dataloader until finished the required number of iterations
            # for i, data in enumer ate(self.dataloaders['train']):
            for i in range(self.img_dataset.n_images):
                self.train_step += 1
                pbar.update(1)
                torch.autograd.set_detect_anomaly(True)
                self.optimizer.zero_grad()
                # pred: (H, W, color_ch)
                pred, loss = self.train_render(i) # render ith image
                nn.utils.clip_grad_norm_(self.color_network.parameters(), 1.0)
                self.optimizer.step()
                if np.random.uniform(0, 1) < 0.05:
                    # Randomly log rendered images throughout the training                
                    self.log_train_renders(pred, i)
                
                del pred

                if self.train_step % self.cfg.log.save_interval == 0:
                    self.save_checkpoint(full=True)
                    # self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                    # self.mesh_model.train()

                    self.evaluate(self.eval_renders_path)
                    self.color_network.train()
                    if self.latent:
                        self.nerf_outside.train()
        
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

        self.color_network.eval()
        if self.latent:
            self.nerf_outside.eval()
        save_path.mkdir(exist_ok=True)

        eval_size = self.eval_size
        if full_eval:
            logger.info("Start full evaluation")
            all_preds = []
            eval_size = self.full_eval_size
        else:
            logger.info("Start normal evaluation")
        
        # render eval_size images at random views from NeuS dataset
        for i in range(eval_size):
            if not full_eval:
                img_idx = np.random.randint(self.img_dataset.n_images)
            else:
                img_idx = i
            
            # pred: (H, W, color_Ch) tensor
            pred, textures = self.eval_render(img_idx=img_idx) # note that textures contain dummy value

            if self.latent:
                # encode latent into rgb
                # (H, W, 4) -> (1, 4, H, W)
                pred = pred.permute(2, 0, 1).unsqueeze(0)
                # (1, 4, H, W)-> (train_grid_size, train_grid_size, 3)
                pred = self.diffusion.decode_latents(pred).permute(0, 2, 3, 1).contiguous().squeeze(0)

            pred_cpu = pred.detach().cpu()
            del pred
            pred_cpu = tensor2numpy(pred_cpu)

            if full_eval:
                all_preds.append(pred_cpu)
            else:
                cv.imwrite(os.path.join(save_path, f"step_{self.train_step:05d}_{i:04d}_rgb.png"), pred_cpu)

        # also store mesh
        self.validate_mesh_vertex_color(world_space=True, resolution=512, threshold=self.cfg.log.mcube_threshold, half=self.cfg.global_setting.half)
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
        # abandon using view dataset
        # if self.cfg.optim.use_neus_view:
          #   rays_o, rays_d, intrinsic, intrinsic_inv, pose, image_gray = self.img_dataset.gen_rays_at(img_idx, H=img_H, W=img_W, resolution_level=resolution_level)
        # else:
          #   rays_o, rays_d = gen_random_ray_at_pose(theta, phi, radius, H=img_H, W=img_W, intrincis_inv=self.intrinsic_inv, resolution_level=resolution_level, half=self.half)

        # generate ray from thie view
        rays_o, rays_d, intrinsic, intrinsic_inv, pose, image_gray = self.img_dataset.gen_rays_at(img_idx, H=img_H, W=img_W, resolution_level=resolution_level)
        
        H, W, _ = rays_o.shape
        
        rays_o = rays_o.reshape(-1, 3).split(self.neus_cfg['train.batch_size'])
        rays_d = rays_d.reshape(-1, 3).split(self.neus_cfg['train.batch_size'])

        out_fine = []
        for idx, (rays_o_batch, rays_d_batch) in enumerate(zip(rays_o, rays_d)):
            rays_o_batch = rays_o_batch.to(self.device)
            rays_d_batch = rays_d_batch.to(self.device)
            near, far = near_far_from_sphere(rays_o_batch, rays_d_batch)
            
            background_rgb = torch.ones([1, self.color_ch]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              # cos_anneal_ratio=self.get_cos_anneal_ratio(), # skip cosine annealing
                                              cos_anneal_ratio=0.5, # skip cosine annealing
                                              background_rgb=background_rgb,
                                              intrinsics=intrinsic,
                                              intrinsics_inv=intrinsic_inv,
                                              poses=pose,
                                              images=image_gray)
          
            # training: return torch tensor
            # testing: return detached tensor
            # if is_train:
            #     if self.half:
            #         out_fine.append(render_out['color_fine'].half())
            #     else:
            #         out_fine.append(render_out['color_fine'])
            # else:
            #     if self.half:
            #         out_fine.append(render_out['color_fine'].half().detach().cpu())
            #     else:
            #         out_fine.append(render_out['color_fine'].detach().cpu())
            if self.half:
                out_fine.append(render_out['color_fine'].half())
            else:
                out_fine.append(render_out['color_fine'])
            sampled_color = render_out['sampled_color']
            del render_out
            '''
            if only del render_out here,
            we can even not use torch.cuda.empty_cache()
            '''
            # gc.collect()
            # torch.cuda.empty_cache()
            # print(f"after self.renderer.render", self.gpu_tracker.track())

        img_fine = torch.cat(out_fine, dim=0).reshape([H, W, self.color_ch])
        # print(img_fine.shape)
        return img_fine, sampled_color
    
    def check_all_grad(self):    
        for name, param in self.color_network.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"Layer {name} has NaN gradients")


    def train_render(self, img_idx):# , data: Dict[str, Any]):

        # abandon view dataset
        # theta = data['theta'] 
        # phi = data['phi']
        # radius = data['radius']
        
        # pred: (H, W, color_ch)
        # Note: sampled_color for debug
        pred, sampled_color = self.render_single_image(img_idx=img_idx, img_H=self.train_H, img_W=self.train_W, resolution_level=self.neus_cfg['train.train_resolution_level'], is_train=True)    
        if not self.latent:
            # turn BGR to RGB to fit the diffusion model
            pred = pred[..., [2, 1, 0]]
        # (H, W, color_ch) -> (1, color_ch, H, W)
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        
        # text embeddings
        text_z = self.text_z

        # Currently do not use guide
        # if self.cfg.guide.append_direction:
        #     dirs = data['dir']  # [B,]
        #     text_z = self.text_z[dirs]
        # else:
        #     text_z = self.text_z
        
        # loss_guidance = self.diffusion.train_step(text_z, pred_latents)

        # for debug
        pred.retain_grad()
        sampled_color.retain_grad()
        # print(pred.grad)
        # print(sampled_color.grad)

        loss_guidance = self.diffusion.train_step(text_z, pred, params_to_train=self.params_to_train)
        loss = loss_guidance # Note: this loss value will be 0. The real loss value can't be calculated
        loss = -1

        # for debug
        # print("==============pred_rgb.grad=========")
        # print(pred.grad)
        # print("=========sampled_color.grad")
        # print(sampled_color.grad)
        # print(torch.any(torch.isnan(pred.grad)))
        # print(torch.any(torch.isnan(sampled_color)))
        self.check_all_grad()
        # raise NotImplementedError

        pred = pred
        if not self.latent:
            # (1, 3, H, W) -> (H, W, 3)
            # RGB -> BGR to save image through cv2
            pred = pred[..., [2, 1, 0]].squeeze(0).permute(1, 2, 0)

        # Note:
        #   For pixel-based. pred is (H, W, 3)
        #   For latent-based, pred is (1, 4, H, W) to be further fed into decoder
        return pred, loss
    
    def eval_render(self, img_idx):
        '''
        create the latent image
        '''
        # abandon view from viewdataset
        # theta = data['theta']
        # phi = data['phi']
        # radius = data['radius']

        # dim = self.cfg.render.eval_grid_size
        # outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, decode_func=self.diffusion.decode_latents,
          #                                test=True ,dims=(dim,dim))
        # pred_rgb = outputs['image'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)

        # pred: (H, W, color_ch)
        pred, _ = self.render_single_image(img_H=self.eval_H, img_W=self.eval_W, resolution_level=self.neus_cfg['train.validate_resolution_level'], is_train=False, img_idx=img_idx)        

        # texture_rgb = outputs['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)

        return pred, -1  

    def log_train_renders(self, pred: torch.Tensor, img_idx):
        # if self.mesh_model.latent_mode:
          #   pred_rgb = self.diffusion.decode_latents(preds).permute(0, 2, 3, 1).contiguous()  # [1, 3, H, W]
        # else:
          #   pred_rgb = preds.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        # (1, 4, train_grid_size, train_grid_size) -> (train_grid_size, train_grid_size, 3)
        # pred_rgb = self.diffusion.decode_latents(preds).permute(0, 2, 3, 1).contiguous()

        logger.info(f"log image {img_idx} at step {self.train_step}")
        # pred: 
        #   For pixel based: (H, W, color_ch)
        #   For latent based: (1, 4, H, W)
        if self.latent:
            # decode the latent image to RGB image
            # (1, 4, train_grid_size, train_grid_size) -> (train_grid_size, train_grid_size, 3)
            pred = self.diffusion.decode_latents(pred).permute(0, 2, 3, 1).contiguous().squeeze(0)
        
        save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
        save_path.parent.mkdir(exist_ok=True)

        # detach, to numpy, then * 255, clip, turn into uint8
        pred = tensor2numpy(pred)

        # Note: PIL use RGB, while cv2 use BGR
        # Image.fromarray(pred_rgb).save(save_path)
        cv.imwrite(os.path.join(save_path), pred)

    # TODO: figure out how to find the color of vertices
    def validate_mesh_vertex_color(self, world_space=False, resolution=64, threshold=0.0, name=None, half=True):
        print('Start exporting textured mesh')
        dtype = torch.float16 if half else torch.float32

        bound_min = torch.tensor(self.img_dataset.object_bbox_min, dtype=dtype)
        bound_max = torch.tensor(self.img_dataset.object_bbox_max, dtype=dtype)
        vertices, triangles = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution,
                                                               threshold=threshold)
        print(f'Vertices count: {vertices.shape[0]}')

        vertices = torch.tensor(vertices, dtype=dtype)
        vertices_batch = vertices.split(self.neus_cfg['train.batch_size'])
        render_iter = len(vertices_batch)

        vertex_colors = []
        for iter in tqdm(range(render_iter)):
            feature_vector = self.sdf_network.sdf_hidden_appearance(vertices_batch[iter])[:, 1:]
            gradients = self.sdf_network.gradient(vertices_batch[iter]).squeeze()
            dirs = -gradients
            # vertex color: (self.neus_cfg['train.batch_size'], color_ch) (BGR for pixel based)
            vertex_color = self.color_network(vertices_batch[iter], gradients, dirs,
                                                feature_vector)
            if self.latent:
                # Note: please aware that if the color decode from latent is BGR or RGB
                # latent2RGB: 3 * 4 matrix that can transform latent -> RGB
                latent2RGB = self.linear_rgb_estimator 
                # (b, 4) -> (b, 3)
                vertex_color = (vertex_color @ latent2RGB).detach().cpu().numpy()
            else:
                # BGR -> RGB
                vertex_color = vertex_color.detach().cpu().numpy()[..., ::-1]
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

    def load_checkpoint_only_sdf_dev(self, ckpt_path):
        # For load SDF work from pretrained Geo-NeuS model
        # NeRF, variance, and radiance network should not be loaded
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine']) # assume that this checkpoint is saved from Geo-NeuS
        # self.nerf_outside.load_state_dict(checkpoint['nerf']) # assume that this checkpoint is saved from Geo-NeuS
        # self.color_network.load_state_dict(checkpoint['color_network_fine']) # assume that this checkpoint is saved from Geo-NeuS
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
