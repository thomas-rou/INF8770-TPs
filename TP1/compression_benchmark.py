import os
import time
import threading
import psutil
from Compression_Arithmetic.Arithmetic import ArithmeticCompressor
from Compression_LZ77.LZ77_tests import LZ77Compressor

def monitor_resources(stop_event, max_cpu, max_memory):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu_usage = process.cpu_percent(interval=0.01)
        memory_usage = process.memory_info().rss
        max_cpu[0] = max(max_cpu[0], cpu_usage)
        max_memory[0] = max(max_memory[0], memory_usage)

def run_compression(input_file, output_file, is_image=False):
    compressor = ArithmeticCompressor()
    max_cpu = [0]
    max_memory = [0]
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, max_cpu, max_memory))
    monitor_thread.start()

    start_time = time.time()
    if is_image:
        compressor.compress_image_file(input_file, output_file)
    else:
        compressor.compress_text_file(input_file, output_file)
    end_time = time.time()

    stop_event.set()
    monitor_thread.join()

    original_size = os.path.getsize(input_file)
    compressed_size = os.path.getsize(output_file)
    compression_ratio = 1 - (compressed_size / original_size)
    compression_time = end_time - start_time

    return compression_time, compression_ratio, max_cpu[0], max_memory[0]

def run_lz77_compression(input_file, output_file):
    compressor = LZ77Compressor()
    max_cpu = [0]
    max_memory = [0]
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, max_cpu, max_memory))
    monitor_thread.start()

    start_time = time.time()
    compressor.compress(input_file, output_file)
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
    output_dir = 'Test-Files/Compressed-Files/'
    os.makedirs(output_dir, exist_ok=True)

    text_files = [f for f in os.listdir(text_input_dir) if f.endswith('.txt')]
    image_files = [f for f in os.listdir(image_input_dir) if f.endswith('.tiff')]
    text_results = {'arithmetic': {'time': [], 'ratio': [], 'cpu': [], 'memory': []},
                    'lz77': {'time': [], 'ratio': [], 'cpu': [], 'memory': []}}
    image_results = {'arithmetic': {'time': [], 'ratio': [], 'cpu': [], 'memory': []},
                     'lz77': {'time': [], 'ratio': [], 'cpu': [], 'memory': []}}

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

    with open('compression_results.txt', 'w') as f:
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