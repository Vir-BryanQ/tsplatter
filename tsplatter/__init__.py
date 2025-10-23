import torch

TARGET_PERCENTAGE = 0.90
device = torch.cuda.current_device()
total_memory, free_memory = torch.cuda.mem_get_info(device)
required_elements = free_memory * TARGET_PERCENTAGE // 4
occupied = torch.empty(required_elements , dtype=torch.float32, device='cuda')
del occupied

from .data.thermalmap_dataparser import ThermalMapDataParserSpecification

__all__ = [
    "__version__",
    ThermalMapDataParserSpecification,
]
