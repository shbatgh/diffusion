from accelerate import PartialState
import torch
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision=None,
    variant=None,
    torch_dtype=torch.float16,
)
            

# load attention processors
pipeline.load_lora_weights("output/pytorch_lora_weights.safetensors")

distributed_state = PartialState()
pipeline.to(distributed_state.device)

#pipeline.enable_xformers_memory_efficient_attention()
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

# Assume two processes
with (torch.inference_mode()):
    for i in range(1, 1000):
        with distributed_state.split_between_processes(["medical image of nerves taken at Langevin Institute", "medical image of nerves taken at Langevin Institute"]) as prompt:
            result = pipeline(prompt, num_inference_steps=30).images[0]
            result.save(f"inference2/result_{distributed_state.process_index}_{i}.png")