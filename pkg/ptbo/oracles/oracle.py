from torch import Tensor, Size
from abc import ABC, abstractmethod

class Oracle(ABC):
    def __init__(self, inputs: int | Size, outputs: int | Size) -> None:
        super().__init__()
        if isinstance(inputs, int):
            inputs = Size([inputs])
        if isinstance(outputs, int):
            outputs = Size([outputs])
        self.inputs = inputs
        self.outputs = outputs
    
    @abstractmethod
    def solve(self, eta: Tensor) -> Tensor:
        pass
