import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

import onnx
import onnxruntime as ort
print("onnx", onnx.__version__)
print("onnxruntime", ort.__version__)
print("providers", ort.get_available_providers())