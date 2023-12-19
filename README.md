# Latent-SDF
## Intro
+ This is the repository of our final project of Deep Learning, 2023 summer, NYCU.
+ Our goal is to combine two method: [Geo-NeuS](https://github.com/GhiXu/Geo-Neus) and [Latent-NeRF](https://github.com/eladrich/latent-nerf). We aim to use latent paint method in Latent NeRF, but instead of using mesh and latent texture, we replace the mesh with the SDF network obtained from Geo-NeuS, and replace the latent texture with trainable latent radiance network.
+ This project is based on [Geo-NeuS](https://github.com/GhiXu/Geo-Neus) and [Latent-NeRF](https://github.com/eladrich/latent-nerf). Part of the code is from [Textured-NeuS](https://github.com/xrr-233/Textured-NeuS).

## Environment
+ Environment need to satisfy the requirement in [Geo-NeuS](https://github.com/GhiXu/Geo-Neus) and [Latent-NeRF](https://github.com/eladrich/latent-nerf), please refer to these 2 repo to build your environment.

## Usage
1. train the model from [Geo-NeuS](https://github.com/GhiXu/Geo-Neus), and put the model in neus_ckpt
    + change the model path in confs/train_config.py, neus_ckpt_path
2. make sure the dataset of [Geo-NeuS](https://github.com/GhiXu/Geo-Neus) exist
    + change the dataset path in confs/womask.conf
    + change the case name in train_config.py, case
3. Execute: python Trainer.py --log.exp_name "exp_name" --guide.text "guide text", you will see the result in experiments/exp_name/

## Result
+ Although we do not get better result as Latent-NeRF, but our network's storage requirement is relatively small(3.8 MB), which is far smaller than regular mesh.

## Future work
+ Complete the README
+ GeoNeusTrainer compability
+ Complete pixel-based training(using Deepfloyd IF model)
+ Use different differentiable SDF rendering method
+ Filter spurious region in SDF networks
+ Finding optimal network architectures and hyperparmeters
+ Also train background NeRF(idea from NeRF++)
