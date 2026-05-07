import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x).relu()

model = Model().eval()
x = torch.randn(2, 4)

torch.onnx.export(
    model,
    x,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
)

print("exported model.onnx")