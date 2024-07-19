import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from pathlib import Path
import os

# Load the model
model_path = "flowers-102-categories/model.pth"
model_state_dict = torch.load(model_path)

# Assuming the model is a UNet2DModel, initialize it with the loaded state dict
model = UNet2DModel(
    sample_size=128,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
      ),
)
model.load_state_dict(model_state_dict)

# Initialize the DDPMPipeline with the loaded model
# Note: You might need to adjust the scheduler according to your specific setup
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

# Prepare the model for evaluation

# Generate a batch of images
# Note: Adjust `num_inference_steps` and `guidance_scale` as needed
generated_images = pipeline(num_inference_steps=1500).images

# Save the generated images
output_dir = Path("generated_images")
output_dir.mkdir(exist_ok=True)

for i, image in enumerate(generated_images):
    image_path = output_dir / f"image_{i}.png"
    image.save(image_path)

print(f"Saved generated images to {output_dir}")