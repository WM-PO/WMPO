import torch

n_gpu = torch.cuda.device_count()
# total_mem = torch.cuda.get_device_properties().total_memory
      
import torch
import time
def consume_multi_gpu_memory_and_compute(num_gpus=1, memory_gb=10, iterations=10):
    """
    Consume memory and perform computations on multiple GPUs.

    Parameters:
    num_gpus (int): Number of GPUs to use.
    memory_gb (int): Amount of memory to consume per GPU, in gigabytes.
    iterations (int): Number of computation iterations per GPU.
    """
    devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]

    # Check available GPUs
    if torch.cuda.device_count() < num_gpus:
        print(f"Requested {num_gpus} GPUs, but only {torch.cuda.device_count()} are available.")
        return
    
    large_tensors = []
    for device in devices:
        # Allocate memory on each GPU
        num_elements = int(memory_gb * (256 * 10**6) / 4) # Assuming float32 (4 bytes)
        tensor = torch.randn(num_elements, device=device, dtype=torch.float32)
        large_tensors.append(tensor)
        print(f"Allocated {memory_gb} GB on {device}")
    index=0
    while True:
        index+=1
        for tensor in large_tensors:
            # Perform some computations
            tensor=(tensor+tensor)/2
            # tensor = torch.sin(tensor) * torch.cos(tensor)
        print(f"Iteration {index+1}/{iterations} completed across {num_gpus} GPUs.")
    
    print("Finished intensive multi-GPU usage.")

# Adjust the parameters below based on the number of GPUs and their capabilities
consume_multi_gpu_memory_and_compute(num_gpus=n_gpu, memory_gb=3, iterations=1000000000000000000)

    