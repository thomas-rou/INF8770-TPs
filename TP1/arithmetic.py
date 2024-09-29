import sys
import os
import struct
import time
import psutil
import threading




# Maximum range for encoding
# Based on : https://github.com/ahmedfgad/ArithmeticEncodingPython/blob/main/pyae.py
# and : https://colab.research.google.com/github/gabilodeau/INF8770/blob/master/Codage%20arithmetique.ipynb#scrollTo=txaWtZTeiov0

MAX_RANGE = 2**32 - 1

## Calculate frequency table
def compute_symbol_probabilities(data):
    frequency_table = {}
    for symbol in data:
        frequency_table[symbol] = frequency_table.get(symbol, 0) + 1

    total = len(data)
    probabilities = {}
    cumulative_probabilities = {}
    cum_prob = 0

    for symbol, freq in sorted(frequency_table.items()):
        prob = freq / total
        probabilities[symbol] = prob
        cumulative_probabilities[symbol] = cum_prob
        cum_prob += prob

    return probabilities, cumulative_probabilities

def arithmetic_compression(data, probabilities, cumulative_probabilities):
    low = 0
    high = MAX_RANGE
    bitstream = []
    pending_bits = 0

    for symbol in data:
        symbol_low = cumulative_probabilities[symbol]
        symbol_high = symbol_low + probabilities[symbol]
        range_size = high - low + 1
        high = low + int(range_size * symbol_high) - 1
        low = low + int(range_size * symbol_low)

        while True:
            if high < 2**31:
                bitstream.append(0)
                bitstream.extend([1] * pending_bits)
                pending_bits = 0
                low <<= 1
                high = (high << 1) + 1
            elif low >= 2**31:
                bitstream.append(1)
                bitstream.extend([0] * pending_bits)
                pending_bits = 0
                low = (low - 2**31) << 1
                high = ((high - 2**31) << 1) + 1
            elif low >= 2**30 and high < 2**31 + 2**30:
                pending_bits += 1
                low = (low - 2**30) << 1
                high = ((high - 2**30) << 1) + 1
            else:
                break

    pending_bits += 1
    if low < 2**30:
        bitstream.append(0)
        bitstream.extend([1] * pending_bits)
    else:
        bitstream.append(1)
        bitstream.extend([0] * pending_bits)

    return bitstream

def write_bitstream_to_file(bitstream, filename, probabilities, cumulative_probabilities, total_symbols):
    with open(filename, 'wb') as file:
        file.write(struct.pack('I', total_symbols))
        file.write(struct.pack('I', len(probabilities)))

        for symbol in probabilities:
            symbol_encoded = symbol.encode('utf-8')
            file.write(struct.pack('B', len(symbol_encoded)))
            file.write(symbol_encoded)
            file.write(struct.pack('d', probabilities[symbol]))
            file.write(struct.pack('d', cumulative_probabilities[symbol]))

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

def read_bitstream_from_file(filename):
    with open(filename, 'rb') as file:
        total_symbols = struct.unpack('I', file.read(4))[0]
        num_symbols = struct.unpack('I', file.read(4))[0]

        probabilities = {}
        cumulative_probabilities = {}
        for _ in range(num_symbols):
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
        return bitstream[:bitstream_length], probabilities, cumulative_probabilities, total_symbols

def arithmetic_decompression(bitstream, probabilities, cumulative_probabilities, total_symbols):
    MAX_RANGE = 2**32 - 1
    HALF_RANGE = 2**31
    QUARTER_RANGE = 2**30

    low = 0
    high = MAX_RANGE
    value = 0

    bitstream = iter(bitstream)
    for _ in range(32):
        value = (value << 1) | next_bit(bitstream)

    decoded_data = []

    for _ in range(total_symbols):
        range_size = high - low + 1
        cum_value = ((value - low + 1) * 1.0 / range_size)

        for symbol, cum_prob in cumulative_probabilities.items():
            if cum_prob <= cum_value < cum_prob + probabilities[symbol]:
                decoded_data.append(symbol)
                symbol_low = cumulative_probabilities[symbol]
                symbol_high = symbol_low + probabilities[symbol]
                high = low + int(range_size * symbol_high) - 1
                low = low + int(range_size * symbol_low)

                while True:
                    if high < HALF_RANGE:
                        low <<= 1
                        high = (high << 1) + 1
                        value = (value << 1) | next_bit(bitstream)
                    elif low >= HALF_RANGE:
                        low = (low - HALF_RANGE) << 1
                        high = ((high - HALF_RANGE) << 1) + 1
                        value = ((value - HALF_RANGE) << 1) | next_bit(bitstream)
                    elif low >= QUARTER_RANGE and high < HALF_RANGE + QUARTER_RANGE:
                        low = (low - QUARTER_RANGE) << 1
                        high = ((high - QUARTER_RANGE) << 1) + 1
                        value = ((value - QUARTER_RANGE) << 1) | next_bit(bitstream)
                    else:
                        break
                break

    return decoded_data

def next_bit(bitstream):
    try:
        return next(bitstream)
    except StopIteration:
        return 0

def monitor_resources(stop_event, max_cpu, max_memory):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu_usage = process.cpu_percent(interval=0.1)
        memory_usage = process.memory_info().rss
        max_cpu[0] = max(max_cpu[0], cpu_usage)
        max_memory[0] = max(max_memory[0], memory_usage)

def main():
    if len(sys.argv) != 4:
        print("Usage: python arithmetic.py <compress|decompress> <input_file> <output_file>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == 'compress':
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        if not os.path.isfile(input_file):
            print(f"Error: File '{input_file}' not found.")
            sys.exit(1)

        with open(input_file, 'r') as file:
            data = file.read()

         # Initialize variables for monitoring
        max_cpu = [0]
        max_memory = [0]
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, max_cpu, max_memory))
        monitor_thread.start()

        start_time = time.time()

        probabilities, cumulative_probabilities = compute_symbol_probabilities(data)
        bit_output = arithmetic_compression(data, probabilities, cumulative_probabilities)

        end_time = time.time()

        # Stop monitoring
        stop_event.set()
        monitor_thread.join()

        write_bitstream_to_file(bit_output, output_file, probabilities, cumulative_probabilities, len(data))

        # Calculate compression ratio
        original_size = os.path.getsize(input_file)
        compressed_size = os.path.getsize(output_file)
        compression_ratio = 1 - (compressed_size / original_size)

        # Calculate compression time
        compression_time = end_time - start_time

        print(f"Compression complete. Output written to '{output_file}'.")
        print(f"Compression Ratio: {compression_ratio:.2f}")
        print(f"Compression Time: {compression_time:.2f} seconds")
        print(f"Max CPU Usage: {max_cpu[0]:.2f}%")
        print(f"Max Memory Usage: {max_memory[0]} bytes")

    elif mode == 'decompress':
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        if not os.path.isfile(input_file):
            print(f"Error: File '{input_file}' not found.")
            sys.exit(1)

        bitstream, probabilities, cumulative_probabilities, total_symbols = read_bitstream_from_file(input_file)
        decoded_data = arithmetic_decompression(bitstream, probabilities, cumulative_probabilities, total_symbols)

        with open(output_file, 'w') as file:
            file.write(''.join(decoded_data))

        print(f"Decompression complete. Decompressed data written to '{output_file}'.")

    else:
        print("Invalid mode. Use 'compress' or 'decompress'.")
        sys.exit(1)

if __name__ == "__main__":
    main()