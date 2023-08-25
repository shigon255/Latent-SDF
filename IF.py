from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel,logging,CLIPProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DiffusionPipeline
from diffusers.utils import pt_to_pil
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class IFDiffusion(nn.Module):
    def __init__(self, device, model_name="DeepFloyd/IF-I-M-v1.0", half=True):
        super().__init__()

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
                
        self.pipeline = DiffusionPipeline.from_pretrained(model_name, variant="fp16", torch_dtype=torch.float16)
        self.pipeline.enable_model_cpu_offload()

        self.generator = torch.manual_seed(1)

    def get_text_embeds(self, prompt):
        # text embeds
        prompt_embeds, negative_embeds = self.pipeline.encode_prompt(prompt)

        return prompt_embeds, negative_embeds
    
    def train_step(self, prompt_embeds, negative_embeds, inputs, guidance_scale=100):
        pass

    def generate_image(self, prompt):
        prompt_embeds, negative_embeds = self.get_text_embeds(prompt)
        
        image = self.pipeline(
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=self.generator, output_type="pt"
        ).images
        pt_to_pil(image)[0].save("./if_stage_I.png")

        # if half:
        #     self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=self.token).half().to(self.device)
        # else:
        #     self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=self.token).to(self.device)

model = IFDiffusion(device='cuda:1')
model.generate_image('A dog')