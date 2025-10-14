import argparse
import time
import threading
from typing import Tuple

import torch


def exponential_backoff_matrix_multiplication(initial_size: Tuple[int, int], device: torch.device, gpu_id: int):
    """
    Perform matrix multiplication on the GPU with exponential backoff.

    Args:
        initial_size (Tuple[int, int]): Initial size of the matrices (rows, cols).
        device (torch.device): The GPU device to use.
        gpu_id (int): The GPU ID for logging purposes.
    """
    size = initial_size
    while True:
        try:
            # Allocate matrices on the GPU
            print(f"GPU {gpu_id}: Attempting to allocate matrices of size {size}...")
            A = torch.randn(size, device=device)
            B = torch.randn(size, device=device)

            # Perform matrix multiplication
            C = torch.matmul(A, B)
            torch.cuda.synchronize()  # Ensure the computation is done
            time.sleep(0.01)  # Small delay to avoid overwhelming the GPU
            size = (int(size[0] * 1.5), int(size[1] * 1.5))

        except RuntimeError as e:
            if "out of memory" in str(e):
                # Reduce matrix size exponentially
                size = (size[0] // 2, size[1] // 2)
                print(f"GPU {gpu_id}: Out of memory. Reducing matrix size to {size}...")
                if size[0] < 1 or size[1] < 1:
                    raise RuntimeError(f"GPU {gpu_id}: Matrix size too small. GPU memory is insufficient.")
                torch.cuda.empty_cache()  # Clear GPU memory
            else:
                raise e


def touch_gpu_worker(gpu_id: int, initial_size: Tuple[int, int]):
    """
    Worker function to touch a specific GPU.
    
    Args:
        gpu_id (int): The GPU ID to use.
        initial_size (Tuple[int, int]): Initial matrix size.
    """
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Starting GPU worker for GPU {gpu_id}: {torch.cuda.get_device_name(device)}")
    
    try:
        exponential_backoff_matrix_multiplication(initial_size, device, gpu_id)
    except KeyboardInterrupt:
        print(f"GPU {gpu_id}: Exiting...")
    except Exception as e:
        print(f"GPU {gpu_id}: An error occurred: {e}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Maximize GPU utilization using matrix multiplications.")
    parser.add_argument("--gpu", type=int, default=0, 
                       help="GPU to use: non-negative integer (0,1,2,...) for specific GPU ID, or -1 for all GPUs (default: 0).")
    args = parser.parse_args()

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    # Validate GPU argument
    if args.gpu < -1:
        print("Invalid GPU ID. Use non-negative integer for specific GPU or -1 for all GPUs.")
        return

    # Get available GPU count
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")

    # Initial matrix size (adjust based on your GPU memory)
    initial_size = (10000, 10000)  # Start with a large matrix

    if args.gpu == -1:
        # Use all GPUs with threading
        print("Using all available GPUs...")
        threads = []
        
        for gpu_id in range(gpu_count):
            thread = threading.Thread(target=touch_gpu_worker, args=(gpu_id, initial_size))
            thread.daemon = True
            threads.append(thread)
            thread.start()
        
        try:
            # Wait for all threads
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("Exiting all GPU workers...")
    else:
        # Use specific GPU
        if args.gpu >= gpu_count:
            print(f"GPU {args.gpu} is not available. Only {gpu_count} GPUs detected.")
            return
        
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)} (ID: {args.gpu})")
        
        try:
            exponential_backoff_matrix_multiplication(initial_size, device, args.gpu)
        except KeyboardInterrupt:
            print("Exiting...")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
