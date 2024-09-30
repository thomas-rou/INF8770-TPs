import os
import time
import threading
import psutil
import math
from Compression_Arithmetic.Arithmetic import ArithmeticCompressor
from Compression_LZ77.LZ77_tests import LZ77Compressor

def monitor_resources(stop_event, max_cpu, max_memory):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu_usage = process.cpu_percent(interval=0.01)
        memory_usage = process.memory_info().rss
        max_cpu[0] = max(max_cpu[0], cpu_usage)
        max_memory[0] = max(max_memory[0], memory_usage)

def measure_resources(compression_method, input_file_path, compressed_file_path):
    # Get the process object for the current process
    process = psutil.Process(os.getpid())

    # CPU times before starting the compression
    start_cpu_times = process.cpu_times()

    # Variables for maximum CPU and memory usage
    max_cpu = [0]
    max_memory = [0]
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, max_cpu, max_memory))
    monitor_thread.start()

    # Perform compression
    compression_method(input_file_path, compressed_file_path)

    # Stop monitoring thread
    stop_event.set()
    monitor_thread.join()

    # CPU times after compression finishes
    end_cpu_times = process.cpu_times()

    # Calculate CPU time (user + system)
    user_time = end_cpu_times.user - start_cpu_times.user
    system_time = end_cpu_times.system - start_cpu_times.system
    active_cpu_time = user_time + system_time

    # Maximum memory usage during compression
    mem_usage = max_memory[0] / (1024 * 1024)  # Convert to MiB

    return max_cpu[0], mem_usage, active_cpu_time

def perform_compression(compressor, input_file_path, compressed_file_path, is_image=False):
    print("\n=== Compression ===")
    if compressor.__class__.__name__ == 'ArithmeticCompressor':
        if is_image:
            cpu_usage_percent, mem_usage, compression_time = measure_resources(compressor.compress_image_file, input_file_path, compressed_file_path)
        else:
            cpu_usage_percent, mem_usage, compression_time = measure_resources(compressor.compress_text_file, input_file_path, compressed_file_path)
    else:
        cpu_usage_percent, mem_usage, compression_time = measure_resources(compressor.compress, input_file_path, compressed_file_path)

    compressed_file_size = os.path.getsize(compressed_file_path)
    print(f"Taille du fichier compressé : {math.ceil(compressed_file_size / 1024)} KB")
    print(f"Temps de compression : {compression_time} secondes")
    print(f"Mémoire maximale utilisée pendant la compression : {mem_usage} MiB")
    print(f"Pourcentage moyen d'utilisation du CPU pendant la compression : {cpu_usage_percent}%")

    return compression_time, compressed_file_size, cpu_usage_percent, mem_usage

def run_compression(input_file, output_file, is_image=False):
    compressor = ArithmeticCompressor()
    compression_time, compressed_file_size, cpu_usage_percent, mem_usage = perform_compression(compressor, input_file, output_file, is_image)

    original_size = os.path.getsize(input_file)
    compression_ratio = 1 - (compressed_file_size / original_size)

    return compression_time, compression_ratio, cpu_usage_percent, mem_usage

def run_lz77_compression(input_file, output_file):
    compressor = LZ77Compressor()
    compression_time, compressed_file_size, cpu_usage_percent, mem_usage = perform_compression(compressor, input_file, output_file)

    original_size = os.path.getsize(input_file)
    compression_ratio = 1 - (compressed_file_size / original_size)

    return compression_time, compression_ratio, cpu_usage_percent, mem_usage

def main():
    text_input_dir = 'Test-Files/Text-Files/'
    image_input_dir = 'Test-Files/Image-Files/'
    output_dir = 'Test-Files/Compressed-Files/'
    os.makedirs(output_dir, exist_ok=True)

    text_files = [f for f in os.listdir(text_input_dir) if f.endswith('.txt')]
    image_files = [f for f in os.listdir(image_input_dir) if f.endswith('.tiff')]
    text_results = {'arithmetic': {'time': [], 'ratio': [], 'cpu': [], 'memory': []},
                    'lz77': {'time': [], 'ratio': [], 'cpu': [], 'memory': []}}
    image_results = {'arithmetic': {'time': [], 'ratio': [], 'cpu': [], 'memory': []},
                     'lz77': {'time': [], 'ratio': [], 'cpu': [], 'memory': []}}

    with open('compression_results.txt', 'w') as f:
        # Process text files
        for file in text_files:
            input_file = os.path.join(text_input_dir, file)
            output_file_arithmetic = os.path.join(output_dir, file + '.arithmetic.compressed')
            output_file_lz77 = os.path.join(output_dir, file + '.lz77.compressed')

            file_results_arithmetic = {'time': [], 'ratio': [], 'cpu': [], 'memory': []}
            file_results_lz77 = {'time': [], 'ratio': [], 'cpu': [], 'memory': []}

            for _ in range(10):
                compression_time, compression_ratio, max_cpu, max_memory = run_compression(input_file, output_file_arithmetic)
                file_results_arithmetic['time'].append(compression_time)
                file_results_arithmetic['ratio'].append(compression_ratio)
                file_results_arithmetic['cpu'].append(max_cpu)
                file_results_arithmetic['memory'].append(max_memory)

                compression_time, compression_ratio, max_cpu, max_memory = run_lz77_compression(input_file, output_file_lz77)
                file_results_lz77['time'].append(compression_time)
                file_results_lz77['ratio'].append(compression_ratio)
                file_results_lz77['cpu'].append(max_cpu)
                file_results_lz77['memory'].append(max_memory)

            for method, file_results in zip(['arithmetic', 'lz77'], [file_results_arithmetic, file_results_lz77]):
                avg_time = sum(file_results['time']) / 10
                avg_ratio = sum(file_results['ratio']) / 10
                avg_cpu = sum(file_results['cpu']) / 10
                avg_memory = sum(file_results['memory']) / 10

                text_results[method]['time'].extend(file_results['time'])
                text_results[method]['ratio'].extend(file_results['ratio'])
                text_results[method]['cpu'].extend(file_results['cpu'])
                text_results[method]['memory'].extend(file_results['memory'])

                f.write(f"Text File: {file} ({method}):\n")
                f.write(f"Average Compression Time: {avg_time:.2f} seconds\n")
                f.write(f"Average Compression Ratio: {avg_ratio:.2f}\n")
                f.write(f"Average Max CPU Usage: {avg_cpu:.2f}%\n")
                f.write(f"Average Max Memory Usage: {avg_memory} bytes\n\n")

        # Process image files
        for file in image_files:
            input_file = os.path.join(image_input_dir, file)
            output_file_arithmetic = os.path.join(output_dir, file + '.arithmetic.compressed')
            output_file_lz77 = os.path.join(output_dir, file + '.lz77.compressed')

            file_results_arithmetic = {'time': [], 'ratio': [], 'cpu': [], 'memory': []}
            file_results_lz77 = {'time': [], 'ratio': [], 'cpu': [], 'memory': []}

            for _ in range(3):
                compression_time, compression_ratio, max_cpu, max_memory = run_compression(input_file, output_file_arithmetic, is_image=True)
                file_results_arithmetic['time'].append(compression_time)
                file_results_arithmetic['ratio'].append(compression_ratio)
                file_results_arithmetic['cpu'].append(max_cpu)
                file_results_arithmetic['memory'].append(max_memory)

                compression_time, compression_ratio, max_cpu, max_memory = run_lz77_compression(input_file, output_file_lz77)
                file_results_lz77['time'].append(compression_time)
                file_results_lz77['ratio'].append(compression_ratio)
                file_results_lz77['cpu'].append(max_cpu)
                file_results_lz77['memory'].append(max_memory)

            for method, file_results in zip(['arithmetic', 'lz77'], [file_results_arithmetic, file_results_lz77]):
                avg_time = sum(file_results['time']) / 3
                avg_ratio = sum(file_results['ratio']) / 3
                avg_cpu = sum(file_results['cpu']) / 3
                avg_memory = sum(file_results['memory']) / 3

                image_results[method]['time'].extend(file_results['time'])
                image_results[method]['ratio'].extend(file_results['ratio'])
                image_results[method]['cpu'].extend(file_results['cpu'])
                image_results[method]['memory'].extend(file_results['memory'])

                f.write(f"Image File: {file} ({method}):\n")
                f.write(f"Average Compression Time: {avg_time:.2f} seconds\n")
                f.write(f"Average Compression Ratio: {avg_ratio:.2f}\n")
                f.write(f"Average Max CPU Usage: {avg_cpu:.2f}%\n")
                f.write(f"Average Max Memory Usage: {avg_memory} bytes\n\n")

        for method in ['arithmetic', 'lz77']:
            text_avg_time = sum(text_results[method]['time']) / len(text_results[method]['time'])
            text_avg_ratio = sum(text_results[method]['ratio']) / len(text_results[method]['ratio'])
            text_avg_cpu = sum(text_results[method]['cpu']) / len(text_results[method]['cpu'])
            text_avg_memory = sum(text_results[method]['memory']) / len(text_results[method]['memory'])

            image_avg_time = sum(image_results[method]['time']) / len(image_results[method]['time'])
            image_avg_ratio = sum(image_results[method]['ratio']) / len(image_results[method]['ratio'])
            image_avg_cpu = sum(image_results[method]['cpu']) / len(image_results[method]['cpu'])
            image_avg_memory = sum(image_results[method]['memory']) / len(image_results[method]['memory'])

            f.write(f"Overall Text Files Results ({method}):\n")
            f.write(f"Average Compression Time: {text_avg_time:.2f} seconds\n")
            f.write(f"Average Compression Ratio: {text_avg_ratio:.2f}\n")
            f.write(f"Average Max CPU Usage: {text_avg_cpu:.2f}%\n")
            f.write(f"Average Max Memory Usage: {text_avg_memory} bytes\n\n")

            f.write(f"Overall Image Files Results ({method}):\n")
            f.write(f"Average Compression Time: {image_avg_time:.2f} seconds\n")
            f.write(f"Average Compression Ratio: {image_avg_ratio:.2f}\n")
            f.write(f"Average Max CPU Usage: {image_avg_cpu:.2f}%\n")
            f.write(f"Average Max Memory Usage: {image_avg_memory} bytes\n\n")

if __name__ == "__main__":
    main()