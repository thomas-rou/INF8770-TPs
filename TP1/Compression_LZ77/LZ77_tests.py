import logging
import os
import math
import time
import psutil
import threading
from memory_profiler import memory_usage

# L'algorithme LZ77 utilisé ci-dessous est tiré de celui-ci : https://github.com/TimGuite/lz77/tree/master
class LZ77Compressor:
    def __init__(self):
        pass

    def compress(self, input_data: bytes, max_offset: int = 2047, max_length: int = 31) -> [(int, int, bytes)]:
        """Compress the input data into a list of length, offset, char values"""
        input_array = bytearray(input_data)
        window = bytearray()
        output = []

        while len(input_array) > 0:
            length, offset = self.best_length_offset(window, input_array, max_length, max_offset)
            char = input_array[0:1]
            output.append((offset, length, char))
            window.extend(input_array[:length])
            input_array = input_array[length:]

        return output

    def to_bytes(self, compressed_representation: [(int, int, bytes)], offset_bits: int = 11, length_bits: int = 5) -> bytearray:
        """Turn the compression representation into a byte array"""
        output = bytearray()
        assert (offset_bits + length_bits) % 8 == 0, "Offset and length bits must add up to a multiple of 8."
        offset_length_bytes = int((offset_bits + length_bits) / 8)

        for value in compressed_representation:
            offset, length, char = value
            assert offset < 2 ** offset_bits, f"Offset {offset} too large."
            assert length < 2 ** length_bits, f"Length {length} too large."

            offset_length_value = (offset << length_bits) + length
            for count in range(offset_length_bytes):
                output.append((offset_length_value >> (8 * (offset_length_bytes - count - 1))) & 0b11111111)

            if char is not None:
                if offset == 0:
                    output.extend(char)
            else:
                output.append(0)

        return output

    def best_length_offset(self, window: bytearray, input_array: bytearray, max_length: int = 15, max_offset: int = 4095) -> (int, int):
        """Return the offset and length with the biggest length of the input array as a substring"""
        cut_window = window[-max_offset:] if max_offset < len(window) else window
        if len(input_array) == 0:
            return (0, 0)

        length, offset = (1, 0)
        if input_array[0:1] not in cut_window:
            best_length = self.repeating_length_from_start(input_array[0:1], input_array[1:])
            return (min((length + best_length), max_length), offset)

        length = 0
        for index in range(1, (len(cut_window) + 1)):
            char = cut_window[-index:-index + 1]
            if char == input_array[0:1]:
                found_offset = index
                found_length = self.repeating_length_from_start(cut_window[-index:], input_array)
                if found_length > length:
                    length = found_length
                    offset = found_offset

        return (min(length, max_length), offset)

    def repeating_length_from_start(self, window: bytearray, input_array: bytearray) -> int:
        """Get the maximum repeating length of the input from the start of the window"""
        length = 0
        while length < len(window) and length < len(input_array) and window[length] == input_array[length]:
            length += 1
        return length

    def compress_file(self, input_file: str, output_file: str, max_offset: int = 2047, max_length: int = 31, offset_bits: int = 11, length_bits: int = 5):
        """Open and read an input file, compress it, and write the compressed values to the output file"""
        try:
            with open(input_file, 'rb') as f:
                input_array = f.read()
        except FileNotFoundError:
            print(f"Could not find input file at: {input_file}")
            raise

        compressed_input = self.to_bytes(self.compress(input_array, max_offset, max_length), offset_bits, length_bits)
        with open(output_file, "wb") as f:
            f.write(compressed_input)

    def monitor_resources(self, stop_event, max_cpu, max_memory):
        process = psutil.Process(os.getpid())
        while not stop_event.is_set():
            cpu_usage = process.cpu_percent(interval=0.01)
            memory_usage = process.memory_info().rss
            max_cpu[0] = max(max_cpu[0], cpu_usage)
            max_memory[0] = max(max_memory[0], memory_usage)

    def show_compression_results(self, input_file, output_file, compression_ratio, compression_time, max_cpu, max_memory):
        print(f"Compression complete. Output of '{input_file}' written to '{output_file}'.")
        print(f"Compression Ratio: {compression_ratio:.2f}")
        print(f"Compression Time: {compression_time:.2f} seconds")
        print(f"Max CPU Usage: {max_cpu:.2f}%")
        print(f"Max Memory Usage: {max_memory} bytes")

def measure_resources(func, *args):
    max_cpu = [0]
    max_memory = [0]
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=lambda: monitor_resources(stop_event, max_cpu, max_memory))
    monitor_thread.start()

    start_time = time.time()
    func(*args)
    end_time = time.time()

    stop_event.set()
    monitor_thread.join()

    return max_cpu[0], max_memory[0] / (1024 * 1024), end_time - start_time

def monitor_resources(stop_event, max_cpu, max_memory):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu_usage = process.cpu_percent(interval=0.01)
        memory_usage = process.memory_info().rss
        max_cpu[0] = max(max_cpu[0], cpu_usage)
        max_memory[0] = max(max_memory[0], memory_usage)

def get_file_paths():
    file_type = input("Voulez-vous compresser une image ou un fichier texte ? (image/texte) : ").strip().lower()
    if file_type not in ['image', 'texte']:
        raise ValueError("Type de fichier non reconnu. Veuillez choisir 'image' ou 'texte'.")

    input_file_path = input("Veuillez entrer le chemin du fichier à compresser : ").strip()

    compressed_file_path = 'compressed.bin'
    decompressed_file_path = 'decompressed'

    if file_type == 'image':
        decompressed_file_path += '.tiff'
    elif file_type == 'texte':
        decompressed_file_path += '.txt'

    return input_file_path, compressed_file_path, decompressed_file_path

def print_file_size(file_path, description):
    try:
        file_size = os.path.getsize(file_path)
        print(f"{description} : {math.ceil(file_size / 1024)} KB")
    except FileNotFoundError:
        print(f"Le fichier {file_path} est introuvable. Veuillez vérifier le chemin.")
        raise

def perform_compression(compressor, input_file_path, compressed_file_path, max_offset, max_length, offset_bits, length_bits):
    print("\n=== Compression ===")
    cpu_usage_percent, mem_usage, compression_time = measure_resources(
        compressor.compress_file, input_file_path, compressed_file_path, max_offset, max_length, offset_bits, length_bits
    )

    compressed_file_size = os.path.getsize(compressed_file_path)
    print(f"Taille du fichier compressé : {math.ceil(compressed_file_size / 1024)} KB")
    print(f"Temps de compression : {compression_time} secondes")
    print(f"Mémoire maximale utilisée pendant la compression : {mem_usage} MiB")
    print(f"Pourcentage moyen d'utilisation du CPU pendant la compression : {cpu_usage_percent}%")

    return compression_time, compressed_file_size

def calculate_compression_ratio(initial_size, compressed_size):
    compression_ratio = 1 - ((initial_size - compressed_size) / initial_size)
    print(f"Taux de compression : {round(compression_ratio, 2)} %")

def main():
    try:
        input_file_path, compressed_file_path, decompressed_file_path = get_file_paths()

        compressor = LZ77Compressor()

        # Affiche la taille initiale du fichier
        print_file_size(input_file_path, "Taille initiale du fichier")

        # Set parameters based on file type
        file_type = input_file_path.split('.')[-1]
        if file_type in ['tiff', 'jpg', 'png']:
            max_offset = 4095
            max_length = 15  # Ensure max_length is less than 2^4 (16)
            offset_bits = 12
            length_bits = 4
        else:
            max_offset = 2047
            max_length = 31  # Ensure max_length is less than 2^5 (32)
            offset_bits = 11
            length_bits = 5

        # Effectue la compression
        initial_file_size = os.path.getsize(input_file_path)
        compression_time, compressed_file_size = perform_compression(
            compressor, input_file_path, compressed_file_path, max_offset, max_length, offset_bits, length_bits
        )

        # Calcule le taux de compression
        calculate_compression_ratio(initial_file_size, compressed_file_size)

        # Effectue la décompression
        # compressor.decompress_file(compressed_file_path, decompressed_file_path)

    except ValueError as e:
        print(e)
    except FileNotFoundError:
        pass

if __name__ == '__main__':
    main()