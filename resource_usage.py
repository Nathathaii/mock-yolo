import time
import psutil
import subprocess
import json

def memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss  # in bytes

def gpu_usage():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True)
        result.check_returncode()
        lines = result.stdout.strip().split("\n")
        gpu_usages = {}
        for i, line in enumerate(lines):
            used, total = map(int, line.split(", "))
            usage_percentage = (used / total) * 100
            gpu_usages[i] = usage_percentage
        return gpu_usages
    except subprocess.CalledProcessError as e:
        print(f"Error calling nvidia-smi: {e}")
        return {}

def print_resource_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = memory_usage() / (1024 ** 2)
    print(f"CPU usage: {cpu_usage}%")
    print(f"RAM usage: {ram_usage:.2f} MB")
    gpu_usages = gpu_usage()
    for gpu_id, usage in gpu_usages.items():
        print(f"GPU {gpu_id} usage: {usage:.2f}%")

    return cpu_usage, ram_usage, gpu_usages

def print_total_usage(cpu_usage_init, ram_usage_init, gpu_usage_init, cpu_usage_end, ram_usage_end, gpu_usage_end):
    print("\nTotal used---------------")
    cpu_used = cpu_usage_end - cpu_usage_init
    ram_used = ram_usage_end - ram_usage_init

    print(f"CPU used: {cpu_used} %")
    print(f"RAM used: {ram_used:.2f} MB")

    for gpu_id in gpu_usage_init:
        init_used = gpu_usage_init[gpu_id]
        end_used = gpu_usage_end[gpu_id]
        used = end_used - init_used
        print(f"GPU {gpu_id} used: {used:.2f}%")

# ## at beginning of the function
# print("Initial resource usage:")
# cpu_usage_init, ram_usage_init, gpu_usage_init = print_resource_usage()
# start_time = time.time()

# ## at the end of the function
# end_time = time.time()
# print("Resource usage after evaluation:")
# cpu_usage_end, ram_usage_end, gpu_usage_end = print_resource_usage()
# print(f"Total time taken: {end_time - start_time:.2f} seconds")
# print_total_usage(cpu_usage_init, ram_usage_init, gpu_usage_init, cpu_usage_end, ram_usage_end, gpu_usage_end)