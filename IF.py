from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import T5Tokenizer, T5EncoderModel, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DiffusionPipeline
from diffusers.utils import pt_to_pil
# suppress partial model loading warning
logging.set_verbosity_error()

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from loguru import logger
import cv2 as cv


class IFDiffusion(nn.Module):
    def __init__(self, device, model_name="DeepFloyd/IF-I-M-v1.0", half=True):
        super().__init__()

        print("DNS error! please do not use this part for now")
        raise NotImplementedError
        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                logger.info(f'loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            logger.warning(f'try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')
        
        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        logger.info(f'loading diffusion with {model_name}...')

        if half:
            self.pipeline = DiffusionPipeline.from_pretrained(model_name, variant="fp16", torch_dtype=torch.float16, use_auth_token=self.token, safety_checker=None, use_safetensors=False).to(device)
        else:
            self.pipeline = DiffusionPipeline.from_pretrained(model_name, variant="fp32", torch_dtype=torch.float32, use_auth_token=self.token, safety_checker=None, use_safetensors=False).to(device)
        # self.pipeline.enable_model_cpu_offload()
        if half:
            self.alphas = self.pipeline.scheduler.alphas_cumprod.half().to(device)    
        else:
            self.alphas = self.pipeline.scheduler.alphas_cumprod.to(device)    

        self.generator = torch.manual_seed(1)

    def get_text_embeds(self, prompt):
        # # Tokenize text and get embeddings
        # text_input = self.pipeline.tokenizer(prompt, padding='max_length', max_length=self.pipeline.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        # with torch.no_grad():
        #     text_embeddings = self.pipeline.text_encoder(text_input.input_ids.to(self.device))[0]

        # # Do the same for unconditional embeddings
        # uncond_input = self.pipeline.tokenizer([''] * len(prompt), padding='max_length', max_length=self.pipeline.tokenizer.model_max_length, return_tensors='pt')

        # with torch.no_grad():
        #     uncond_embeddings = self.pipeline.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # # Cat for final embeddings
        # text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embeddings, _ = self.pipeline.encode_prompt(prompt)
        return text_embeddings
    
    def train_step(self, text_embeds, inputs, guidance_scale=100, params_to_train=None):
        pred_rgb_64 = F.interpolate(inputs, (64, 64), mode='bilinear', align_corners=False)
        # print("pred_rgb_64", pred_rgb_64)
        # print("requires_grad: ", pred_rgb_64.requires_grad)
        # pred_rgb_64 = inputs
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(pred_rgb_64)
            # print(noise.shape)
            latents_noisy = self.pipeline.scheduler.add_noise(pred_rgb_64, noise, t)
            # pred noise
            # print(latents_noisy.shape)
            # latent_model_input = torch.cat([latents_noisy] * 2)
            # print(latent_model_input.shape)
            # print(text_embeds.shape)
            noise_pred = self.pipeline.unet(latents_noisy, t, encoder_hidden_states=text_embeds)[0]
            # print(noise_pred)
            # print(noise_pred.shape)
            
        # no guidance
        # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # noise_pred_uncond, _ = noise_pred_uncond.split(latent_model_input.shape[1], dim=1)
        # noise_pred_text, predicted_variance = noise_pred_text.split(latent_model_input.shape[1], dim=1)
        # print(noise_pred_uncond.shape)
        # print(noise_pred_text.shape)
        # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
        noise_pred, _ = noise_pred.split(latents_noisy.shape[1], dim=1)
        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        # print(w.shape)
        # print(noise_pred.shape)
        # print(noise.shape)
        # print(w)        
        grad = w * (noise_pred - noise)
        # print("before clip grad: ", grad)
        # grad = grad.clamp(-0.01, 0.01)
        # print("after clip grad: ", grad)
    

        # print("any: ", grad.any())
        if math.isnan(grad[0, 0, 0, 0]):
            raise NotImplementedError
        # with torch.autograd.detect_anomaly():
        # pred_rgb_64.backward(gradient=grad, retain_graph=True)
        torch.autograd.grad(outputs=pred_rgb_64, inputs=params_to_train, grad_outputs=grad)


        return 0 # dummy loss value

    def generate_image(self, prompt):
        # embeds = self.get_text_embeds(prompt)
        embeds, _ = self.pipeline.encode_prompt(prompt)
        
        image = self.pipeline(
            prompt_embeds=embeds, generator=self.generator, output_type="pt"
        ).images
        pt_to_pil(image)[0].save("./if_stage_I.png")

        return image
        # if half:
        #     self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=self.token).half().to(self.device)
        # else:
        #     self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=self.token).to(self.device)

if __name__ == "__main__" :
    model = IFDiffusion(device='cuda:1')
    image = model.generate_image('A blue cat')
