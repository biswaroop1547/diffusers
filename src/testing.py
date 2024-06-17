from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline
from diffusers.utils import load_image
import torch

# prompt = "a red car fully visible, vintage style photograph"

prompt = {"single": "A photo of a bunny", "rest": "A photo of a tiger"}

pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16).to("cuda")

generator = torch.Generator(device="cuda").manual_seed(1)

images = pipeline(
            prompt,
            num_inference_steps=30,
            guidance_scale=9.5,
            negative_prompt="lowres, child, getty, bad anatomy, bad hands, pixelated",
            generator=generator,
            single_layer_idxs=[3, 4, 21, 22],
        ).images[0]

images.save("test_sd3.png")

