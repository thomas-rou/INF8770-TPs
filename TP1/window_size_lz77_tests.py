import os
import math
import time
import psutil
import threading
from Compression_LZ77.LZ77_tests import LZ77Compressor

def monitor_resources(stop_event, max_cpu, max_memory):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu_usage = process.cpu_percent(interval=0.01)
        memory_usage = process.memory_info().rss
        max_cpu[0] = max(max_cpu[0], cpu_usage)
        max_memory[0] = max(max_memory[0], memory_usage)

def measure_resources(compression_method, input_file_path, compressed_file_path):
    max_cpu = [0]
    max_memory = [0]
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, max_cpu, max_memory))
    monitor_thread.start()

    start_time = time.time()
    compression_method(input_file_path, compressed_file_path)
    end_time = time.time()

    stop_event.set()
    monitor_thread.join()

    compression_time = end_time - start_time
    cpu_usage_percent = max_cpu[0]
    mem_usage = max_memory[0] / (1024 * 1024)  # Convert to MiB

    return cpu_usage_percent, mem_usage, compression_time

def perform_compression(compressor, input_file_path, compressed_file_path, is_image=False):
    print("\n=== Compression ===")
    cpu_usage_percent, mem_usage, compression_time = measure_resources(compressor.compress, input_file_path, compressed_file_path)

    compressed_file_size = os.path.getsize(compressed_file_path)
    print(f"Taille du fichier compressé : {math.ceil(compressed_file_size / 1024)} KB")
    print(f"Temps de compression : {compression_time} secondes")
    print(f"Mémoire maximale utilisée pendant la compression : {mem_usage} MiB")
    print(f"Pourcentage moyen d'utilisation du CPU pendant la compression : {cpu_usage_percent}%")

    return compression_time, compressed_file_size, cpu_usage_percent, mem_usage

def run_lz77_compression(input_file, output_file, window_size):
    compressor = LZ77Compressor(window_size=window_size)
    compression_time, compressed_file_size, cpu_usage_percent, mem_usage = perform_compression(compressor, input_file, output_file)

    original_size = os.path.getsize(input_file)
    compression_ratio = 1 - (compressed_file_size / original_size)

    return compression_time, compression_ratio, cpu_usage_percent, mem_usage

def main():
    text_input_dir = 'Test-Files/Text-Files/'
    output_dir = 'Test-Files/Compressed-Files/'
    os.makedirs(output_dir, exist_ok=True)

    text_files = [f for f in os.listdir(text_input_dir) if f.endswith('.txt')]
    window_sizes = [20, 100, 200, 300, 400]  # Different window sizes to test

    for window_size in window_sizes:
        print(f"Testing LZ77 with window size: {window_size}")
        for file in text_files:
            input_file = os.path.join(text_input_dir, file)
            output_file_lz77 = os.path.join(output_dir, f"{file}.lz77.{window_size}.compressed")

            compression_time, compression_ratio, max_cpu, max_memory = run_lz77_compression(input_file, output_file_lz77, window_size)
            print(f"File: {file}, Window Size: {window_size}, Time: {compression_time:.2f}s, Ratio: {compression_ratio:.2f}, CPU: {max_cpu:.2f}%, Memory: {max_memory} bytes")

if __name__ == "__main__":
    main()