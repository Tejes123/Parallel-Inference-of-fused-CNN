import threading
import time
import psutil 
import os 
from pynvml import *
import GPUtil 

# cpu_readings = [] 
# ram_readings = [] 

# stop_cpu_monitoring = False 
# interval = 0.5

# def monitor_cpu():
#     pid = os.getpid()
#     process = psutil.Process(pid = pid)

#     """ CPU Utilization  """

#     cpu_one = process.cpu_percent(interval = None)

#     while not stop_cpu_monitoring:
#         # time.sleep(interval)
#         usage = process.cpu_percent(interval = 1)
#         cpu_readings.append(usage)

#     """ RAM Utilization """

#     memory_info = process.memory_info()
#     rss_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB

stop_gpu_monitoring = False
gpu_util = []
gpu_mem = []  

def monitor_gpu():
    # Initialize NVML
    nvmlInit()

    handle = nvmlDeviceGetHandleByIndex(0)

    while not stop_gpu_monitoring:

        # Get memory information  
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        gpu_mem.append(memory_info.used // 1024**2)
        # print(f"Used Memory: {memory_info.used // 1024**2} MB")

        # Get GPU utilization
        # utilization = nvmlDeviceGetUtilizationRates(handle)
        # gpu_util.append(utilization.gpu)
        # print(f"GPU Utilization: {utilization.gpu}%")
        gpu = GPUtil.getGPUs()[0]
        utilization = (gpu.load * 100)
        gpu_util.append(utilization)

        time.sleep(1)

thread = threading.Thread(target = monitor_gpu)
thread.start()

time.sleep(10)

stop_gpu_monitoring = True 
# # thread.join()

print(gpu_util)
print(gpu_mem)

