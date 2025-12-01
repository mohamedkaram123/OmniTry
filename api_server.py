"""
FastAPI server for OmniTry Virtual Try-On
Provides REST API endpoints for virtual try-on functionality
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import diffusers
import transformers
import copy
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
from typing import Optional

from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline


# Initialize FastAPI app
app = FastAPI(
    title="OmniTry Virtual Try-On API",
    description="API for virtual try-on of clothes and accessories",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
device = None
weight_dtype = None
pipeline = None
transformer = None
args = None


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
    global device, weight_dtype, pipeline, transformer, args
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.bfloat16
    args = OmegaConf.load('configs/omnitry_v1_unified.yaml')

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


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "OmniTry Virtual Try-On API is running",
        "available_categories": list(args.object_map.keys()) if args else []
    }


@app.get("/health")
async def health():
    """Health check endpoint for RunPod"""
    return {"status": "healthy"}


@app.post("/try-on")
async def virtual_try_on(
    person_image: UploadFile = File(..., description="Image of the person"),
    clothing_image: UploadFile = File(..., description="Image of the clothing/accessory"),
    category: str = Form(..., description="Category of the item (e.g., 'top clothes', 'dress', 'shoe')"),
    steps: int = Form(20, description="Number of inference steps (1-50)"),
    guidance_scale: float = Form(30.0, description="Guidance scale (1-50)"),
    seed: int = Form(-1, description="Random seed for reproducibility (-1 for random)"),
    return_base64: bool = Form(False, description="Return image as base64 string")
):
    """
    Virtual try-on endpoint
    
    Args:
        person_image: Image of the person
        clothing_image: Image of the clothing or accessory
        category: Category of the item to try on
        steps: Number of inference steps (default: 20)
        guidance_scale: Guidance scale (default: 30)
        seed: Random seed (-1 for random)
        return_base64: Return image as base64 instead of binary
    
    Returns:
        Generated image with the person wearing the item
    """
    try:
        # Validate category
        if category not in args.object_map:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid category. Must be one of: {list(args.object_map.keys())}"
            )
        
        # Validate parameters
        if not (1 <= steps <= 50):
            raise HTTPException(status_code=400, detail="Steps must be between 1 and 50")
        if not (1 <= guidance_scale <= 50):
            raise HTTPException(status_code=400, detail="Guidance scale must be between 1 and 50")
        
        # Load images
        person_img = Image.open(io.BytesIO(await person_image.read())).convert('RGB')
        object_img = Image.open(io.BytesIO(await clothing_image.read())).convert('RGB')
        
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

        # Return image
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        if return_base64:
            img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
            return {
                "image": img_base64,
                "seed": seed,
                "format": "base64"
            }
        else:
            return Response(content=img_byte_arr.read(), media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during try-on: {str(e)}")


@app.get("/categories")
async def get_categories():
    """Get available clothing/accessory categories"""
    return {
        "categories": list(args.object_map.keys()),
        "descriptions": args.object_map
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
