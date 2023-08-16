import torch
import logging
import argparse
import pyrallis

from GeoNeusTrainer import GeoNeusTrainer
from LatentPaintTrainer import LatentPaintTrainer
from confs.train_config import TrainConfig

@pyrallis.wrap()
def main(cfg: TrainConfig):
    print('Hello Wooden')

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(cfg.global_setting.gpu)
    if cfg.global_setting.mode == "latent_paint":
        trainer = LatentPaintTrainer(cfg)
        if cfg.log.eval_only:
            trainer.full_eval()
        else:
            trainer.train()
    else:
        trainer = GeoNeusTrainer(cfg.neus.neus_cfg_path, cfg.global_setting.mode, cfg.neus.case, cfg.neus.is_continue, cfg.neus.checkpoint, cfg.neus.suffix)

        if cfg.global_setting.mode == 'train':
            trainer.train()
        elif cfg.global_setting.mode == 'validate_mesh':
            trainer.validate_mesh(world_space=True, resolution=512, threshold=cfg.neus.mcube_threshold, dilation=cfg.neus.dilation) # 512
        # elif args.mode == 'validate_mesh_womask':
        #     runner.validate_mesh_womask(world_space=True, resolution=512, threshold=args.mcube_threshold, dilation=args.dilation) # 512
        # elif args.mode == 'validate_mesh_ori':
        #     runner.validate_mesh_ori(world_space=True, resolution=512, threshold=args.mcube_threshold) # 512
        elif cfg.global_setting.mode == 'validate_image':
            trainer.validate_image()
        elif cfg.global_setting.mode == 'eval_image':
            trainer.eval_image()
        elif cfg.global_setting.mode.startswith('interpolate'):  # Interpolate views given two image indices
            _, img_idx_0, img_idx_1 = cfg.global_setting.mode.split('_')
            img_idx_0 = int(img_idx_0)
            img_idx_1 = int(img_idx_1)
            trainer.interpolate_view(img_idx_0, img_idx_1)

if __name__ == "__main__":
    main()