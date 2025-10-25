from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.nn.module_v3 import Module, module_dataclass
from max.nn.module_v3 import Linear

import numpy as np
from max.experimental import tensor


@module_dataclass
class MLP(Module):
    fc1: Linear
    fc2: Linear

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x))


if __name__ == "__main__":
    # Create a NumPy array of 10 float32 ones
    np_array = np.ones(10).astype(np.float32)

    # Convert to MAX tensor via DLPack
    x = tensor.Tensor.from_dlpack(np_array).to(DeviceRef.GPU())

    model = MLP(fc1=Linear(10, 20), fc2=Linear(20, 5))
    out = model(x)

    print("Model:", model)

    print("Input:", x)
    print("Output:", out)

    # Count parameters
    total_params = sum(param.num_elements() for name, param in model.parameters)
    print(f"Total parameters: {total_params}")
