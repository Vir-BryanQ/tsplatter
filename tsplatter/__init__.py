import torch
from datetime import datetime

def preallocate_vmem(vram=38):
    required_elements = int(vram * 1024 * 1024 * 1024 / 4)
    while True:
        try:
            occupied = torch.empty(required_elements , dtype=torch.float32, device='cuda')
            del occupied
            break
        except RuntimeError as e:
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            if 'CUDA out of memory' in str(e):
                total_memory, free_memory = torch.cuda.mem_get_info(torch.cuda.current_device())
                print(f"*** CUDA OOM: {free_memory / (1024 * 1024)}MiB is free [{t}]")
            else:
                print(f"### {e} [{t}]")

def release_vmem():
    torch.cuda.empty_cache()

preallocate_vmem()

from .data.thermalmap_dataparser import ThermalMapDataParserSpecification

__all__ = [
    "__version__",
    ThermalMapDataParserSpecification,
]
