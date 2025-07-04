import torch
import time
import psutil
import GPUtil
import numpy as np
from datetime import datetime

def check_gpu_availability():
    """Check if GPU is available and get basic info"""
    print("=== GPU Availability Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("No CUDA-compatible GPU found")
    print()

def gpu_performance_test():
    """Run GPU performance benchmark"""
    print("=== GPU Performance Test ===")
    
    if not torch.cuda.is_available():
        print("GPU not available, skipping performance test")
        return
    
    device = torch.device("cuda")
    
    # Matrix multiplication test
    sizes = [1000, 2000, 4000, 8000]
    
    for size in sizes:
        print(f"Testing {size}x{size} matrix multiplication...")
        
        # Generate random matrices
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warm up
        torch.cuda.synchronize()
        for _ in range(3):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            result = torch.mm(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        gflops = (2 * size**3) / (avg_time * 1e9)
        
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Performance: {gflops:.2f} GFLOPS")
        print()

def monitor_gpu_usage():
    """Monitor GPU memory and utilization"""
    print("=== GPU Monitoring ===")
    
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("No GPUs found for monitoring")
            return
        
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name}")
            print(f"  Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
            print(f"  GPU Load: {gpu.load*100:.1f}%")
            print(f"  Temperature: {gpu.temperature}°C")
            print()
    except Exception as e:
        print(f"Error monitoring GPU: {e}")

def memory_stress_test():
    """Test GPU memory allocation"""
    print("=== GPU Memory Stress Test ===")
    
    if not torch.cuda.is_available():
        print("GPU not available, skipping memory test")
        return
    
    device = torch.device("cuda")
    tensors = []
    
    try:
        print("Allocating GPU memory...")
        for i in range(100):
            tensor = torch.randn(1000, 1000, device=device)
            tensors.append(tensor)
            
            if i % 10 == 0:
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_cached = torch.cuda.memory_reserved(device) / 1024**3
                print(f"  Step {i}: Allocated {memory_allocated:.2f}GB, Cached {memory_cached:.2f}GB")
        
        print("Memory allocation successful")
        
    except RuntimeError as e:
        print(f"Memory allocation failed: {e}")
    
    finally:
        # Clean up
        del tensors
        torch.cuda.empty_cache()
        print("Memory cleaned up")
        print()

def cpu_gpu_comparison():
    """Compare CPU vs GPU performance"""
    print("=== CPU vs GPU Performance Comparison ===")
    
    size = 2000
    iterations = 5
    
    # CPU test
    print("Testing CPU performance...")
    cpu_device = torch.device("cpu")
    a_cpu = torch.randn(size, size, device=cpu_device)
    b_cpu = torch.randn(size, size, device=cpu_device)
    
    start_time = time.time()
    for _ in range(iterations):
        result_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    
    print(f"CPU time: {cpu_time:.4f}s")
    
    # GPU test
    if torch.cuda.is_available():
        print("Testing GPU performance...")
        gpu_device = torch.device("cuda")
        a_gpu = torch.randn(size, size, device=gpu_device)
        b_gpu = torch.randn(size, size, device=gpu_device)
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            result_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("GPU not available for comparison")
    print()

def main():
    """Main function to run all GPU tests"""
    print(f"GPU Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    check_gpu_availability()
    monitor_gpu_usage()
    gpu_performance_test()
    memory_stress_test()
    cpu_gpu_comparison()
    
    print("GPU test suite completed!")

if __name__ == "__main__":
    main()