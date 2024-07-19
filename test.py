import subprocess
import re

def get_gpu_processes():
    # Run nvidia-smi to get GPU process information
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

    # Parse the output
    processes = []
    for line in output.strip().split('\n'):
        pid, memory_usage = line.split(', ')
        processes.append({'pid': int(pid), 'memory_usage': int(memory_usage)})

    return processes

# Example usage
processes = get_gpu_processes()
for p in processes:
    print(f"Process ID: {p['pid']}, Memory Used: {p['memory_usage']} MB")