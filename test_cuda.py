import torch

print(torch.cuda.is_available())  # True if CUDA is available
print(torch.cuda.device_count())  # Number of GPUs
print(torch.cuda.get_device_name(0))  # Name of GPU (if one exists)