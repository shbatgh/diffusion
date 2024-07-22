from dataclasses import dataclass
import time
import torch
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
import functools

# torch disable grad
torch.set_grad_enabled(False)

# set variables
n_experiments = 2
unet_runs_per_experiment = 50

@dataclass
class UNet2DConditionOutput:
    sample: torch.Tensor

# load inputs
def generate_inputs():
    sample = torch.randn((2, 4, 64, 64), device="cuda", dtype=torch.float16)
    timestep = torch.rand(1, device="cuda", dtype=torch.float16) * 999
    encoder_hidden_states = torch.randn((2, 77, 768), device="cuda", dtype=torch.float16)
    return sample, timestep, encoder_hidden_states

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision=None,
    variant=None,
    torch_dtype=torch.float16,
)
pipeline.to("cuda")
            
# load attention processors
pipeline.load_lora_weights("output/pytorch_lora_weights.safetensors")

unet = pipeline.unet
unet.eval()
unet.to(memory_format=torch.channels_last)  # use channels_last memory format
unet.forward = functools.partial(unet.forward, return_dict=False)  # set return_dict=False as default

# warmup
for _ in range(3):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet(*inputs)

# trace
print("tracing..")
unet_traced = torch.jit.trace(unet, inputs)
unet_traced.eval()
print("done tracing")


# warmup and optimize graph
for _ in range(5):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet_traced(*inputs)


# benchmarking
with torch.inference_mode():
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet_traced(*inputs)
        torch.cuda.synchronize()
        print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet(*inputs)
        torch.cuda.synchronize()
        print(f"unet inference took {time.time() - start_time:.2f} seconds")

# save the model
unet_traced.save("unet_traced.pt")

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

pipeline.enable_xformers_memory_efficient_attention()

unet_traced = torch.jit.load("unet_traced.pt")

class TracedUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        setattr(self, "config", pipeline.unet.config)
        self.in_channels = pipeline.unet.config.in_channels
        self.device = pipeline.unet.device

    def forward(self, latent_model_input, t, encoder_hidden_states):
        sample = unet_traced(latent_model_input, t, encoder_hidden_states)
        return UNet2DConditionOutput(sample=sample)
    
pipeline.unet = TracedUNet()