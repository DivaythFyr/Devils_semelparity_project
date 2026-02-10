import time
import random
import torch as th

def test_builtin_vs_fastrand_int01_single_number_generators(n_iterations=1_000_000):
    """
    Compare the speed of Python's built-in random number generator
    vs various fastrand functions for generating single random integers (0 or 1).
    """
    print(f"Testing single-value random integer (0 or 1) generation with {n_iterations:,} iterations\n")
    
    # Test Python's built-in random
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        random.randint(0, 1)
    builtin_time = time.perf_counter() - start_time
    print(f"Python built-in random.randint(0, 1):         {builtin_time:.4f} seconds")
    
    # Test fastrand.xorshift128plusrandint
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        fastrand.xorshift128plusrandint(0, 1)
    xorshift_time = time.perf_counter() - start_time
    print(f"fastrand.xorshift128plusrandint(0, 1):        {xorshift_time:.4f} seconds")
    
    # Test fastrand.pcg32randint
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        fastrand.pcg32randint(0, 1)
    pcg32_time = time.perf_counter() - start_time
    print(f"fastrand.pcg32randint(0, 1):                  {pcg32_time:.4f} seconds")
    
    # Summary
    times = [
        ("Python built-in", builtin_time),
        ("fastrand.xorshift128plusrandint", xorshift_time),
        ("fastrand.pcg32randint", pcg32_time)
    ]
    fastest = min(times, key=lambda x: x[1])
    print(f"\nFastest: {fastest[0]} ({fastest[1]:.4f} seconds)")
    
def test_torch_randint_vs_fastrand_pcg32randint(n_iterations=1_000_000):
    """
    Compare the speed of torch.randint vs fastrand.pcg32randint for generating random 0 or 1.
    """
    print(f"Testing torch.randint vs fastrand.pcg32randint with {n_iterations:,} iterations\n")

    # Test torch.randint (scalar generation in a loop)
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        th.randint(0, 2, (1,))
    torch_time = time.perf_counter() - start_time
    print(f"torch.randint(0, 2, (1,)):        {torch_time:.4f} seconds")

    # Test fastrand.pcg32randint
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        fastrand.pcg32randint(0, 1)
    fastrand_time = time.perf_counter() - start_time
    print(f"fastrand.pcg32randint(0, 1):      {fastrand_time:.4f} seconds")

    # Summary
    if fastrand_time < torch_time:
        speedup = torch_time / fastrand_time
        print(f"\nfastrand.pcg32randint is {speedup:.2f}x faster")
    else:
        speedup = fastrand_time / torch_time
        print(f"\ntorch.randint is {speedup:.2f}x faster")
        
# Global tensors
global_tensors_x = [th.rand(1000) for _ in range(10)]
global_tensors_y = [th.rand(1000) for _ in range(10)]

class TensorHolder:
    def __init__(self):
        self.x = [th.rand(1000) for _ in range(10)]
        self.y = [th.rand(1000) for _ in range(10)]

def calculate_pairwise_distances(x, y):
    # x, y: [N]
    dx = x.unsqueeze(1) - x.unsqueeze(0)
    dy = y.unsqueeze(1) - y.unsqueeze(0)
    return dx ** 2 + dy ** 2

def test_global_vs_object_tensors(n_iterations=2000):
    """
    Compare speed of pairwise distance calculation using global tensors vs tensors inside an object.
    """
    print(f"Testing global tensors vs object tensors for pairwise distance calculation ({n_iterations} iterations)\n")

    # Test global tensors
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        for i in range(10):
            calculate_pairwise_distances(global_tensors_x[i], global_tensors_y[i])
    global_time = time.perf_counter() - start_time
    print(f"Global tensors: {global_time:.4f} seconds")

    # Test object tensors
    holder = TensorHolder()
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        for i in range(10):
            calculate_pairwise_distances(holder.x[i], holder.y[i])
    object_time = time.perf_counter() - start_time
    print(f"Object tensors: {object_time:.4f} seconds")

    # Summary
    if object_time < global_time:
        speedup = global_time / object_time
        print(f"\nObject tensors are {speedup:.2f}x faster")
    else:
        speedup = object_time / global_time
        print(f"\nGlobal tensors are {speedup:.2f}x faster")
    


if __name__ == "__main__":
    test_global_vs_object_tensors()
    
    # test_builtin_vs_fastrand_int01_single_number_generators()
    # test_torch_randint_vs_fastrand_pcg32randint()
    
    # for _ in range(10):
    #     print(fastrand.pcg32randint(0, 1))