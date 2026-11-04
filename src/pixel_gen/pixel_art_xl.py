# https://huggingface.co/nerijs/pixel-art-xl
from functools import lru_cache

from diffusers import DiffusionPipeline, LCMScheduler

from .constants import LORA_ID, MODEL_ID


@lru_cache
def get_pipe():
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, variant='fp16')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights(LORA_ID, adapter_name='lora')
    pipe.load_lora_weights(
        'nerijs/pixel-art-xl',
        weight_name='pixel-art-xl.safetensors',
        adapter_name='pixel',
    )

    pipe.set_adapters(['lora', 'pixel'], adapter_weights=[1.0, 1.2])
    return pipe
