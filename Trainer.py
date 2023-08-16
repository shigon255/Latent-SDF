import torch
import logging
import argparse

from GeoNeusTrainer import GeoNeusTrainer
from LatentPaintTrainer import LatentPaintTrainer

if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--train_from_geo_neus', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument("--suffix", default="")
    parser.add_argument("--dilation", type=int, default=15)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.checkpoint, args.suffix)

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
