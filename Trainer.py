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

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = argparse.ArgumentParser()
    parser.add_argument('--neus_cfg_path', type=str, default='./confs/womask.conf')
    parser.add_argument('--load_from_neus', default=True, action="store_true")
    parser.add_argument('--neus_ckpt_path', type=str, default='')
    parser.add_argument('--mode', type=str, default='latent_paint')
    parser.add_argument('--use_white_bkgd', default=False, action="store_true")
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument("--suffix", default="")
    parser.add_argument("--dilation", type=int, default=15)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    if args.mode == "latent_paint":
        trainer = LatentPaintTrainer(cfg, args)
        if cfg.log.eval_only:
            trainer.full_eval()
        else:
            trainer.train()
    else:
        runner = GeoNeusTrainer(args.neus_cfg_path, args.mode, args.case, args.is_continue, args.checkpoint, args.suffix)

        if args.mode == 'train':
            runner.train()
        elif args.mode == 'validate_mesh':
            runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold, dilation=args.dilation) # 512
        # elif args.mode == 'validate_mesh_womask':
        #     runner.validate_mesh_womask(world_space=True, resolution=512, threshold=args.mcube_threshold, dilation=args.dilation) # 512
        # elif args.mode == 'validate_mesh_ori':
        #     runner.validate_mesh_ori(world_space=True, resolution=512, threshold=args.mcube_threshold) # 512
        elif args.mode == 'validate_image':
            runner.validate_image()
        elif args.mode == 'eval_image':
            runner.eval_image()
        elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
            _, img_idx_0, img_idx_1 = args.mode.split('_')
            img_idx_0 = int(img_idx_0)
            img_idx_1 = int(img_idx_1)
            runner.interpolate_view(img_idx_0, img_idx_1)

if __name__ == "__main__":
    main()