import matplotlib.pyplot as plt

def read_memory_profile(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    timestamps = []
    memory_usage = []

    for line in lines:
        if line.startswith('MEM'):
            parts = line.split()
            timestamps.append(float(parts[1]))
            memory_usage.append(float(parts[2]))

    return timestamps, memory_usage

def plot_memory_usage(timestamps, memory_usage):
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, memory_usage, label='Memory Usage')
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (MiB)')
    plt.title('Memory Usage Over Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    timestamps, memory_usage = read_memory_profile('mprofile.dat')
    plot_memory_usage(timestamps, memory_usage)