# Latent-SDF
+ This is the repository of our final project in DLP, 2023 summer, NYCU.
+ Our goal is to combine two method: [Geo-NeuS](https://github.com/GhiXu/Geo-Neus) and [Latent-NeRF](https://github.com/eladrich/latent-nerf). We want to use latent paint in Latent NeRF, but instead of using mesh and latent texture, we replace the mesh with the SDF network obtained from Geo-NeuS, and replace the latent textuer with trainable latent radiance network.
+ This project is based on [Geo-NeuS](https://github.com/GhiXu/Geo-Neus) and [Latent-NeRF](https://github.com/eladrich/latent-nerf)
+ TODO
    + Training size is too small(can only use 32), is there any better idea to calculate SDS loss?
    + Inference part. We need to use stable diffusion decoder to decode the rendered latent image into RGB image
