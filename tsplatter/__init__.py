import torch
from shuffle import VRAM

print(VRAM)


from .data.thermalmap_dataparser import ThermalMapDataParserSpecification

__all__ = [
    "__version__",
    ThermalMapDataParserSpecification,
]
