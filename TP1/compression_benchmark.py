import os
import time
import threading
import psutil
from arithmetic import compress_text_file, compress_image_file

def monitor_resources(stop_event, max_cpu, max_memory):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu_usage = process.cpu_percent(interval=0.01)
        memory_usage = process.memory_info().rss
        max_cpu[0] = max(max_cpu[0], cpu_usage)
        max_memory[0] = max(max_memory[0], memory_usage)

def run_compression(input_file, output_file, is_image=False):
    max_cpu = [0]
    max_memory = [0]
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, max_cpu, max_memory))
    monitor_thread.start()

    start_time = time.time()
    if is_image:
        compress_image_file(input_file, output_file)
    else:
        compress_text_file(input_file, output_file)
    end_time = time.time()

    stop_event.set()
    monitor_thread.join()

    original_size = os.path.getsize(input_file)
    compressed_size = os.path.getsize(output_file)
    compression_ratio = 1 - (compressed_size / original_size)
    compression_time = end_time - start_time

    return compression_time, compression_ratio, max_cpu[0], max_memory[0]

def main():
    text_input_dir = 'Test-Files/Text-Files/'
    image_input_dir = 'Test-Files/Image-Files/'
    output_dir = 'Compressed-Files/'
    os.makedirs(output_dir, exist_ok=True)

    text_files = [f for f in os.listdir(text_input_dir) if f.endswith('.txt')]
    image_files = [f for f in os.listdir(image_input_dir) if f.endswith('.tiff')]
    text_results = {'time': [], 'ratio': [], 'cpu': [], 'memory': []}
    image_results = {'time': [], 'ratio': [], 'cpu': [], 'memory': []}

    # Process text files
    for file in text_files:
        input_file = os.path.join(text_input_dir, file)
        output_file = os.path.join(output_dir, file + '.compressed')

        file_results = {'time': [], 'ratio': [], 'cpu': [], 'memory': []}

        for _ in range(10):
            compression_time, compression_ratio, max_cpu, max_memory = run_compression(input_file, output_file)
            file_results['time'].append(compression_time)
            file_results['ratio'].append(compression_ratio)
            file_results['cpu'].append(max_cpu)
            file_results['memory'].append(max_memory)

        avg_time = sum(file_results['time']) / 10
        avg_ratio = sum(file_results['ratio']) / 10
        avg_cpu = sum(file_results['cpu']) / 10
        avg_memory = sum(file_results['memory']) / 10

        print(f"Results for {file}:")
        print(f"Average Compression Time: {avg_time:.2f} seconds")
        print(f"Average Compression Ratio: {avg_ratio:.2f}")
        print(f"Average Max CPU Usage: {avg_cpu:.2f}%")
        print(f"Average Max Memory Usage: {avg_memory} bytes\n")

        text_results['time'].extend(file_results['time'])
        text_results['ratio'].extend(file_results['ratio'])
        text_results['cpu'].extend(file_results['cpu'])
        text_results['memory'].extend(file_results['memory'])

    # Process image files
    for file in image_files:
        input_file = os.path.join(image_input_dir, file)
        output_file = os.path.join(output_dir, file + '.compressed')

        file_results = {'time': [], 'ratio': [], 'cpu': [], 'memory': []}

        for _ in range(3):
            compression_time, compression_ratio, max_cpu, max_memory = run_compression(input_file, output_file, is_image=True)
            file_results['time'].append(compression_time)
            file_results['ratio'].append(compression_ratio)
            file_results['cpu'].append(max_cpu)
            file_results['memory'].append(max_memory)

        avg_time = sum(file_results['time']) / 3
        avg_ratio = sum(file_results['ratio']) / 3
        avg_cpu = sum(file_results['cpu']) / 3
        avg_memory = sum(file_results['memory']) / 3

        print(f"Results for {file}:")
        print(f"Average Compression Time: {avg_time:.2f} seconds")
        print(f"Average Compression Ratio: {avg_ratio:.2f}")
        print(f"Average Max CPU Usage: {avg_cpu:.2f}%")
        print(f"Average Max Memory Usage: {avg_memory} bytes\n")

        image_results['time'].extend(file_results['time'])
        image_results['ratio'].extend(file_results['ratio'])
        image_results['cpu'].extend(file_results['cpu'])
        image_results['memory'].extend(file_results['memory'])

    text_avg_time = sum(text_results['time']) / len(text_results['time'])
    text_avg_ratio = sum(text_results['ratio']) / len(text_results['ratio'])
    text_avg_cpu = sum(text_results['cpu']) / len(text_results['cpu'])
    text_avg_memory = sum(text_results['memory']) / len(text_results['memory'])

    image_avg_time = sum(image_results['time']) / len(image_results['time'])
    image_avg_ratio = sum(image_results['ratio']) / len(image_results['ratio'])
    image_avg_cpu = sum(image_results['cpu']) / len(image_results['cpu'])
    image_avg_memory = sum(image_results['memory']) / len(image_results['memory'])

    print("Overall Text Files Results:")
    print(f"Average Compression Time: {text_avg_time:.2f} seconds")
    print(f"Average Compression Ratio: {text_avg_ratio:.2f}")
    print(f"Average Max CPU Usage: {text_avg_cpu:.2f}%")
    print(f"Average Max Memory Usage: {text_avg_memory} bytes\n")

    print("Overall Image Files Results:")
    print(f"Average Compression Time: {image_avg_time:.2f} seconds")
    print(f"Average Compression Ratio: {image_avg_ratio:.2f}")
    print(f"Average Max CPU Usage: {image_avg_cpu:.2f}%")
    print(f"Average Max Memory Usage: {image_avg_memory} bytes")

if __name__ == "__main__":
    main()