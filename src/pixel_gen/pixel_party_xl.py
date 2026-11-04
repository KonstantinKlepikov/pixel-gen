# https://huggingface.co/pixelparty/pixel-party-xl
from functools import lru_cache

import torch
from diffusers import DiffusionPipeline, LCMScheduler, UNet2DConditionModel

from .constants import LORA_ID, MODEL_ID


@lru_cache
def get_pipe():
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        unet=UNet2DConditionModel.from_pretrained(
            'pixelparty/pixel-party-xl',
            torch_dtype=torch.float16,
        ),
        use_safetensors=True,
        variant='fp16',
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights(LORA_ID, adapter_name='lora')
    pipe.set_adapters(['lora'], adapter_weights=[1.0])
    return pipe
