import gc

import torch


def clear_device(device: torch.device) -> None:
    # Run garbage collection
    gc.collect()

    # Clear CUDA cache and reset memory stats
    with device:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# clear_device(DEVICE)
