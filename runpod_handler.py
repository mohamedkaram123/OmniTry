"""
RunPod Serverless Handler for OmniTry Virtual Try-On
Handles requests from RunPod serverless infrastructure
"""

import runpod
import torch
import random
import numpy as np
import torchvision.transforms as T
import math
import peft
from peft import LoraConfig
from safetensors import safe_open
from omegaconf import OmegaConf
import os
import io
import base64
from PIL import Image

from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline


# Global variables for model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weight_dtype = torch.bfloat16
args = OmegaConf.load('configs/omnitry_v1_unified.yaml')
pipeline = None
transformer = None


def seed_everything(seed=0):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model():
    """Load the OmniTry model and pipeline"""
    global pipeline, transformer
    
    print("🔄 Loading OmniTry model...")
    
    # Initialize model & pipeline
    transformer = FluxTransformer2DModel.from_pretrained(
        f'{args.model_root}/transformer'
    ).requires_grad_(False).to(dtype=weight_dtype)
    
    pipeline = FluxFillPipeline.from_pretrained(
        args.model_root, 
        transformer=transformer.eval(), 
        torch_dtype=weight_dtype
    )

    # VRAM saving
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_tiling()

    # Insert LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            'x_embedder',
            'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0', 
            'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out', 
            'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2', 
            'norm1_context.linear', 'norm1.linear', 'norm.linear', 'proj_mlp', 'proj_out'
        ]
    )
    transformer.add_adapter(lora_config, adapter_name='vtryon_lora')
    transformer.add_adapter(lora_config, adapter_name='garment_lora')

    with safe_open(args.lora_path, framework="pt") as f:
        lora_weights = {k: f.get_tensor(k) for k in f.keys()}
        transformer.load_state_dict(lora_weights, strict=False)

    # Hack lora forward
    def create_hacked_forward(module):
        def lora_forward(self, active_adapter, x, *args, **kwargs):
            result = self.base_layer(x, *args, **kwargs)
            if active_adapter is not None:
                torch_result_dtype = result.dtype
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result = result + lora_B(lora_A(dropout(x))) * scaling
            return result
        
        def hacked_lora_forward(self, x, *args, **kwargs):
            return torch.cat((
                lora_forward(self, 'vtryon_lora', x[:1], *args, **kwargs),
                lora_forward(self, 'garment_lora', x[1:], *args, **kwargs),
            ), dim=0)
        
        return hacked_lora_forward.__get__(module, type(module))

    for n, m in transformer.named_modules():
        if isinstance(m, peft.tuners.lora.layer.Linear):
            m.forward = create_hacked_forward(m)

    print("✅ Model loaded successfully!")


def handler(event):
    """
    RunPod handler function
    
    Expected input format:
    {
        "input": {
            "person_image": "base64_encoded_image",
            "clothing_image": "base64_encoded_image",
            "category": "top clothes",
            "steps": 20,
            "guidance_scale": 30.0,
            "seed": -1
        }
    }
    """
    try:
        job_input = event.get("input", {})
        
        # Validate required fields
        if "person_image" not in job_input or "clothing_image" not in job_input:
            return {
                "error": "Missing required fields: person_image and clothing_image"
            }
        
        if "category" not in job_input:
            return {
                "error": "Missing required field: category"
            }
        
        # Get parameters
        person_image_b64 = job_input["person_image"]
        clothing_image_b64 = job_input["clothing_image"]
        category = job_input["category"]
        steps = job_input.get("steps", 20)
        guidance_scale = job_input.get("guidance_scale", 30.0)
        seed = job_input.get("seed", -1)
        
        # Validate category
        if category not in args.object_map:
            return {
                "error": f"Invalid category. Must be one of: {list(args.object_map.keys())}"
            }
        
        # Decode base64 images
        person_img = Image.open(io.BytesIO(base64.b64decode(person_image_b64))).convert('RGB')
        object_img = Image.open(io.BytesIO(base64.b64decode(clothing_image_b64))).convert('RGB')
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        seed_everything(seed)

        # Resize model
        max_area = 1024 * 1024
        oW = person_img.width
        oH = person_img.height

        ratio = math.sqrt(max_area / (oW * oH))
        ratio = min(1, ratio)
        tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16
        transform = T.Compose([
            T.Resize((tH, tW)),
            T.ToTensor(),
        ])
        person_tensor = transform(person_img)

        # Resize and pad garment
        ratio = min(tW / object_img.width, tH / object_img.height)
        transform = T.Compose([
            T.Resize((int(object_img.height * ratio), int(object_img.width * ratio))),
            T.ToTensor(),
        ])
        object_tensor_padded = torch.ones_like(person_tensor)
        object_tensor = transform(object_img)
        new_h, new_w = object_tensor.shape[1], object_tensor.shape[2]
        min_x = (tW - new_w) // 2
        min_y = (tH - new_h) // 2
        object_tensor_padded[:, min_y: min_y + new_h, min_x: min_x + new_w] = object_tensor

        # Prepare prompts & conditions
        prompts = [args.object_map[category]] * 2
        img_cond = torch.stack([person_tensor, object_tensor_padded]).to(dtype=weight_dtype, device=device)
        mask = torch.zeros_like(img_cond).to(img_cond)

        # Generate
        with torch.no_grad():
            result_img = pipeline(
                prompt=prompts,
                height=tH,
                width=tW,
                img_cond=img_cond,
                mask=mask,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator(device).manual_seed(seed),
            ).images[0]

        # Convert result to base64
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        result_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
        
        return {
            "image": result_base64,
            "seed": seed,
            "status": "success"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }


if __name__ == "__main__":
    # Load model on startup
    load_model()
    
    # Start RunPod handler
    runpod.serverless.start({
        "handler": handler
    })
