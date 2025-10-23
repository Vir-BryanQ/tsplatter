import torch

def calculate_required_elements(gb):
    bytes_per_element = 4
    gb_to_byte = 1024**3
    required_elements = (gb * gb_to_byte) // bytes_per_element
    return required_elements

device = torch.cuda.current_device()
total_memory, free_memory = torch.cuda.mem_get_info(device_id)
TARGET_PERCENTAGE = 0.90
target_allocation_bytes = int(free_memory * TARGET_PERCENTAGE)
required_elements = calculate_required_elements(target_allocation_bytes)
occupied = torch.empty(required_elements, dtype=torch.float32, device='cuda')
del occupied

from .data.thermalmap_dataparser import ThermalMapDataParserSpecification

__all__ = [
    "__version__",
    ThermalMapDataParserSpecification,
]
