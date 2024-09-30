import sys
import os
import math
import time
import psutil
import threading
import struct
from memory_profiler import memory_usage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Algorithme basé sur : https://github.com/ahmedfgad/ArithmeticEncodingPython/blob/main/pyae.py
# et : https://colab.research.google.com/github/gabilodeau/INF8770/blob/master/Codage%20arithmetique.ipynb#scrollTo=txaWtZTeiov0

MAX_RANGE = 2**32 - 1
HALF_RANGE = 2**31
QUARTER_RANGE = 2**30

class ArithmeticCompressor:
    def __init__(self):
        pass

    def compute_symbol_probabilities(self, data):
        frequency_table = {}
        for symbol in data:
            frequency_table[symbol] = frequency_table.get(symbol, 0) + 1

        total = len(data)
        ranges = {}
        cumulative_lower_boundaries = {}
        cumulative_bound = 0

        for symbol, freq in sorted(frequency_table.items()):
            symbol_freq_ratio = freq / total
            ranges[symbol] = symbol_freq_ratio
            cumulative_lower_boundaries[symbol] = cumulative_bound
            cumulative_bound += symbol_freq_ratio

        return ranges, cumulative_lower_boundaries

    def arithmetic_compression(self, data, ranges, cumulative_lower_boundaries):
        low = 0
        high = MAX_RANGE
        bitstream = []
        pending_bits = 0

        for symbol in data:
            symbol_low = cumulative_lower_boundaries[symbol]
            symbol_high = symbol_low + ranges[symbol]
            range_size = high - low + 1
            high = low + int(range_size * symbol_high) - 1
            low = low + int(range_size * symbol_low)

            while True:
                if high < HALF_RANGE:
                    bitstream.append(0)
                    bitstream.extend([1] * pending_bits)
                    pending_bits = 0
                    low <<= 1
                    high = (high << 1) + 1
                elif low >= HALF_RANGE:
                    bitstream.append(1)
                    bitstream.extend([0] * pending_bits)
                    pending_bits = 0
                    low = (low - HALF_RANGE) << 1
                    high = ((high - HALF_RANGE) << 1) + 1
                elif low >= QUARTER_RANGE and high < HALF_RANGE + QUARTER_RANGE:
                    pending_bits += 1
                    low = (low - QUARTER_RANGE) << 1
                    high = ((high - QUARTER_RANGE) << 1) + 1
                else:
                    break

        pending_bits += 1
        if low < QUARTER_RANGE:
            bitstream.append(0)
            bitstream.extend([1] * pending_bits)
        else:
            bitstream.append(1)
            bitstream.extend([0] * pending_bits)

        return bitstream

    def write_bitstream_to_file(self, bitstream, filename, ranges, cumulative_lower_boundaries, total_symbols, is_image=False, width=None, height=None, mode=None):
        with open(filename, 'wb') as file:
            if is_image and width is not None and height is not None:
                file.write(struct.pack('I', width))
                file.write(struct.pack('I', height))
                file.write(mode.encode('utf-8'))
            file.write(struct.pack('I', total_symbols))
            file.write(struct.pack('I', len(ranges)))

            for symbol in ranges:
                if is_image:
                    file.write(struct.pack('B', symbol))
                else:
                    symbol_encoded = symbol.encode('utf-8')
                    file.write(struct.pack('B', len(symbol_encoded)))
                    file.write(symbol_encoded)
                file.write(struct.pack('d', ranges[symbol]))
                file.write(struct.pack('d', cumulative_lower_boundaries[symbol]))

            file.write(struct.pack('I', len(bitstream)))

            byte = 0
            bit_count = 0
            for bit in bitstream:
                if bit == 1:
                    byte |= (1 << (7 - bit_count))
                bit_count += 1
                if bit_count == 8:
                    file.write(struct.pack('B', byte))
                    byte = 0
                    bit_count = 0
            if bit_count > 0:
                file.write(struct.pack('B', byte))

    def read_bitstream_from_file(self, filename, is_image=False):
        with open(filename, 'rb') as file:
            width = height = None
            mode = None
            if is_image:
                width = struct.unpack('I', file.read(4))[0]
                height = struct.unpack('I', file.read(4))[0]
                mode = file.read(3).decode('utf-8')
            total_symbols = struct.unpack('I', file.read(4))[0]
            num_symbols = struct.unpack('I', file.read(4))[0]

            probabilities = {}
            cumulative_probabilities = {}
            for _ in range(num_symbols):
                if is_image:
                    symbol = struct.unpack('B', file.read(1))[0]
                else:
                    symbol_length = struct.unpack('B', file.read(1))[0]
                    symbol = file.read(symbol_length).decode('utf-8')
                probability = struct.unpack('d', file.read(8))[0]
                cumulative_probability = struct.unpack('d', file.read(8))[0]
                probabilities[symbol] = probability
                cumulative_probabilities[symbol] = cumulative_probability

            bitstream_length = struct.unpack('I', file.read(4))[0]
            bitstream = []
            byte = file.read(1)
            while byte:
                byte = ord(byte)
                for i in range(8):
                    bitstream.append((byte >> (7 - i)) & 1)
                byte = file.read(1)
            if is_image:
                return bitstream[:bitstream_length], probabilities, cumulative_probabilities, total_symbols, width, height, mode
            else:
                return bitstream[:bitstream_length], probabilities, cumulative_probabilities, total_symbols

    def arithmetic_decompression(self, bitstream, ranges, cumulative_lower_boundaries, total_symbols):
        low = 0
        high = MAX_RANGE
        value = 0

        bitstream = iter(bitstream)
        for _ in range(32):
            value = (value << 1) | self.next_bit(bitstream)

        decoded_data = []

        for _ in range(total_symbols):
            range_size = high - low + 1
            cum_value = ((value - low + 1) * 1.0 / range_size)

            for symbol, cum_prob in cumulative_lower_boundaries.items():
                if cum_prob <= cum_value < cum_prob + ranges[symbol]:
                    decoded_data.append(symbol)
                    symbol_low = cumulative_lower_boundaries[symbol]
                    symbol_high = symbol_low + ranges[symbol]
                    high = low + int(range_size * symbol_high) - 1
                    low = low + int(range_size * symbol_low)

                    while True:
                        if high < HALF_RANGE:
                            low <<= 1
                            high = (high << 1) + 1
                            value = (value << 1) | self.next_bit(bitstream)
                        elif low >= HALF_RANGE:
                            low = (low - HALF_RANGE) << 1
                            high = ((high - HALF_RANGE) << 1) + 1
                            value = ((value - HALF_RANGE) << 1) | self.next_bit(bitstream)
                        elif low >= QUARTER_RANGE and high < HALF_RANGE + QUARTER_RANGE:
                            low = (low - QUARTER_RANGE) << 1
                            high = ((high - QUARTER_RANGE) << 1) + 1
                            value = ((value - QUARTER_RANGE) << 1) | self.next_bit(bitstream)
                        else:
                            break
                    break

        return decoded_data

    def next_bit(self, bitstream):
        try:
            return next(bitstream)
        except StopIteration:
            return 0

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

    def compress_text_file(self, input_file, output_file):
        with open(input_file, 'r') as file:
            data = file.read()

            max_cpu = [0]
            max_memory = [0]
            stop_event = threading.Event()
            monitor_thread = threading.Thread(target=self.monitor_resources, args=(stop_event, max_cpu, max_memory))
            monitor_thread.start()

            start_time = time.time()

            ranges, cumulative_lower_boundaries = self.compute_symbol_probabilities(data)
            bit_output = self.arithmetic_compression(data, ranges, cumulative_lower_boundaries)

            end_time = time.time()

            stop_event.set()
            monitor_thread.join()

            self.write_bitstream_to_file(bit_output, output_file, ranges, cumulative_lower_boundaries, len(data))

            original_size = os.path.getsize(input_file)
            compressed_size = os.path.getsize(output_file)
            compression_ratio = 1 - (compressed_size / original_size)
            compression_time = end_time - start_time

            self.show_compression_results(input_file, output_file, compression_ratio, compression_time, max_cpu[0], max_memory[0])

    def decompress_text_file(self, input_file, output_file):
        bitstream, ranges, cumulative_lower_boundaries, total_symbols = self.read_bitstream_from_file(input_file)
        decoded_data = self.arithmetic_decompression(bitstream, ranges, cumulative_lower_boundaries, total_symbols)

        with open(output_file, 'w') as file:
            file.write(''.join(decoded_data))

        print(f"Decompression complete. Decompressed data written to '{output_file}'.")

    def compress_image_file(self, input_file, output_file):
        image = Image.open(input_file)
        mode = image.mode
        data = np.array(image).flatten()
        width, height = image.size
        print(f"Compressing image '{input_file}' with dimensions: {width}x{height} and mode: {mode}...")

        max_cpu = [0]
        max_memory = [0]
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=self.monitor_resources, args=(stop_event, max_cpu, max_memory))
        monitor_thread.start()

        start_time = time.time()

        ranges, cumulative_lower_boundaries = self.compute_symbol_probabilities(data)
        bit_output = self.arithmetic_compression(data, ranges, cumulative_lower_boundaries)

        end_time = time.time()

        stop_event.set()
        monitor_thread.join()

        self.write_bitstream_to_file(bit_output, output_file, ranges, cumulative_lower_boundaries, len(data), is_image=True, width=width, height=height, mode=mode)

        original_size = os.path.getsize(input_file)
        compressed_size = os.path.getsize(output_file)
        compression_ratio = 1 - (compressed_size / original_size)
        compression_time = end_time - start_time

        self.show_compression_results(input_file, output_file, compression_ratio, compression_time, max_cpu[0], max_memory[0])

    def decompress_image_file(self, input_file, output_file):
        bitstream, ranges, cumulative_lower_boundaries, total_symbols, width, height, mode = self.read_bitstream_from_file(input_file, is_image=True)
        decoded_data = self.arithmetic_decompression(bitstream, ranges, cumulative_lower_boundaries, total_symbols)
        decoded_data = np.array(decoded_data, dtype=np.uint8)

        if width is None or height is None:
            image_size = int(np.sqrt(total_symbols))
            width = height = image_size

        if mode.strip().upper() == 'L':
            decoded_image = decoded_data.reshape((height, width))
        else:
            decoded_image = decoded_data.reshape((height, width, len(mode)))

        decoded_image = Image.fromarray(decoded_image, mode)
        decoded_image.save(output_file)

        print(f"Decompression complete. Decompressed image written to '{output_file}'.")

        # plt.imshow(decoded_image, cmap="gray" if mode == 'L' else None)
        # plt.title("Reconstructed Image")
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()

def measure_cpu_usage(func, *args):
    process = psutil.Process()

    process.cpu_percent(interval=None)
    time.sleep(0.1)

    start_time = time.time()
    func(*args)
    end_time = time.time()

    cpu_percent = process.cpu_percent(interval=None)
    elapsed_time = end_time - start_time

    return cpu_percent, elapsed_time

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

def perform_compression(compressor, input_file_path, compressed_file_path, is_image=False):
    print("\n=== Compression ===")
    compression_start_time = time.time()
    if is_image:
        mem_usage = memory_usage((compressor.compress_image_file, (input_file_path, compressed_file_path)), interval=0.1)
        cpu_usage_percent, _ = measure_cpu_usage(compressor.compress_image_file, input_file_path, compressed_file_path)
    else:
        mem_usage = memory_usage((compressor.compress_text_file, (input_file_path, compressed_file_path)), interval=0.1)
        cpu_usage_percent, _ = measure_cpu_usage(compressor.compress_text_file, input_file_path, compressed_file_path)
    compression_end_time = time.time()

    compressed_file_size = os.path.getsize(compressed_file_path)
    print(f"Taille du fichier compressé : {math.ceil(compressed_file_size / 1024)} KB")
    print(f"Temps de compression : {compression_end_time - compression_start_time} secondes")
    print(f"Mémoire maximale utilisée pendant la compression : {max(mem_usage)} MiB")
    print(f"Pourcentage moyen d'utilisation du CPU pendant la compression : {cpu_usage_percent}%")

    return compression_end_time - compression_start_time, compressed_file_size

def perform_decompression(compressor, compressed_file_path, decompressed_file_path, is_image=False):
    print("\n=== Décompression ===")
    decompression_start_time = time.time()
    if is_image:
        mem_usage = memory_usage((compressor.decompress_image_file, (compressed_file_path, decompressed_file_path)), interval=0.1)
        cpu_usage_percent, _ = measure_cpu_usage(compressor.decompress_image_file, compressed_file_path, decompressed_file_path)
    else:
        mem_usage = memory_usage((compressor.decompress_text_file, (compressed_file_path, decompressed_file_path)), interval=0.1)
        cpu_usage_percent, _ = measure_cpu_usage(compressor.decompress_text_file, compressed_file_path, decompressed_file_path)
    decompression_end_time = time.time()

    print(f"Temps de décompression : {decompression_end_time - decompression_start_time} secondes")
    print(f"Mémoire maximale utilisée pendant la décompression : {max(mem_usage)} MiB")
    print(f"Pourcentage moyen d'utilisation du CPU pendant la décompression : {cpu_usage_percent}%")

    return decompression_end_time - decompression_start_time

def calculate_compression_ratio(initial_size, compressed_size):
    compression_ratio = ((initial_size - compressed_size) / initial_size) * 100
    print(f"Taux de compression : {round(compression_ratio, 2)} %")

def main():
    try:
        input_file_path, compressed_file_path, decompressed_file_path = get_file_paths()
        is_image = input_file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))

        compressor = ArithmeticCompressor()

        # Affiche la taille initiale du fichier
        print_file_size(input_file_path, "Taille initiale du fichier")

        # Effectue la compression
        initial_file_size = os.path.getsize(input_file_path)
        compression_time, compressed_file_size = perform_compression(compressor, input_file_path, compressed_file_path, is_image)

        # Calcule le taux de compression
        calculate_compression_ratio(initial_file_size, compressed_file_size)

        # Effectue la décompression
        perform_decompression(compressor, compressed_file_path, decompressed_file_path, is_image)

    except ValueError as e:
        print(e)
    except FileNotFoundError:
        pass

if __name__ == '__main__':
    main()