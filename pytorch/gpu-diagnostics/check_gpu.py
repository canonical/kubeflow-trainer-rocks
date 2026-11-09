import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")

assert torch.cuda.is_available(), "CUDA not available"

device = torch.device("cuda:0")
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)

torch.cuda.synchronize()
c = a @ b
torch.cuda.synchronize()

print(f"Result shape: {c.shape}, sum: {c.sum().item():.2f}")
print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
print(f"Memory reserved:  {torch.cuda.memory_reserved(device) / 1e6:.1f} MB")
