import torch

x = torch.randn((4096, 4096), device="cuda")
y = x @ x
torch.cuda.synchronize()

print(y.shape)
print(y.device)
print("ok")