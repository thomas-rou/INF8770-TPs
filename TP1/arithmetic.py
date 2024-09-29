import time
#import psutil
import sys
import os
import numpy as np
import struct
import base64

# Maximum range for encoding
# Based on : https://github.com/ahmedfgad/ArithmeticEncodingPython/blob/main/pyae.py
# and : https://colab.research.google.com/github/gabilodeau/INF8770/blob/master/Codage%20arithmetique.ipynb#scrollTo=txaWtZTeiov0

MAX_RANGE = 2**32 - 1

## Calculate frequency table
def computeSymbolProbabilities(data):
    # Calculate frequency table
    frequency_table = {}
    for symbol in data:
        if symbol in frequency_table:
            frequency_table[symbol] += 1
        else:
            frequency_table[symbol] = 1

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

def arithmeticCompression(data, probabilities, cumulative_probabilities):
    low = 0
    high = MAX_RANGE

    # Output bitstream
    bitstream = []
    pending_bits = 0
    compressedCode = []

    encoded_symbols_num = 0

    for symbol in data:
        encoded_symbols_num += 1
        symbol_low = cumulative_probabilities[symbol]
        symbol_high = symbol_low + probabilities[symbol]

        # Update range based on symbol's cumulative probability
        range_size = high - low + 1
        high = low + int(range_size * symbol_high) - 1
        low = low + int(range_size * symbol_low)

        # Store symbol with current low and high values for debugging/compressed code
        compressedCode.append((symbol, low, high))

        # Normalize the range to avoid precision overflow
        while True:
            if high < 2**31:
                bitstream.append(0)
                bitstream += [1] * pending_bits
                pending_bits = 0
                low <<= 1
                high = (high << 1) + 1
            elif low >= 2**31:
                bitstream.append(1)
                bitstream += [0] * pending_bits
                pending_bits = 0
                low = (low - 2**31) << 1
                high = ((high - 2**31) << 1) + 1
            elif low >= 2**30 and high < 2**31 + 2**30:
                pending_bits += 1
                low = (low - 2**30) << 1
                high = ((high - 2**30) << 1) + 1
            else:
                break

    # Final bits to finalize the range
    pending_bits += 1
    if low < 2**30:
        bitstream.append(0)
        bitstream += [1] * pending_bits
    else:
        bitstream.append(1)
        bitstream += [0] * pending_bits

    return compressedCode, bitstream

def writeBitstreamToFile(bitstream, filename, probabilities, cumulative_probabilities, total_symbols):
    with open(filename, 'wb') as file:
        # Write the length of the original data
        file.write(struct.pack('I', total_symbols))

        # Write the number of unique symbols
        file.write(struct.pack('I', len(probabilities)))

        # Write the probabilities and cumulative probabilities
        for symbol in probabilities:
            symbol_encoded = symbol.encode('utf-8')
            file.write(struct.pack('B', len(symbol_encoded)))  # Write the length of the symbol
            file.write(symbol_encoded)  # Write the symbol itself
            file.write(struct.pack('d', probabilities[symbol]))  # Write the probability
            file.write(struct.pack('d', cumulative_probabilities[symbol]))  # Write the cumulative probability

        # Write the bitstream length
        file.write(struct.pack('I', len(bitstream)))

        # Write the bitstream as bytes
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

def readBitstreamFromFile(filename):
    with open(filename, 'rb') as file:
        # Read the length of the original data
        total_symbols = struct.unpack('I', file.read(4))[0]

        # Read the number of unique symbols
        num_symbols = struct.unpack('I', file.read(4))[0]

        # Read the probabilities and cumulative probabilities
        probabilities = {}
        cumulative_probabilities = {}
        for _ in range(num_symbols):
            symbol_length = struct.unpack('B', file.read(1))[0]
            symbol = file.read(symbol_length).decode('utf-8')
            probability = struct.unpack('d', file.read(8))[0]
            cumulative_probability = struct.unpack('d', file.read(8))[0]
            probabilities[symbol] = probability
            cumulative_probabilities[symbol] = cumulative_probability

        # Read the bitstream length
        bitstream_length = struct.unpack('I', file.read(4))[0]

        # Read the bitstream as bytes
        bitstream = []
        byte = file.read(1)
        while byte:
            byte = ord(byte)
            for i in range(8):
                bitstream.append((byte >> (7 - i)) & 1)
            byte = file.read(1)
        return bitstream[:bitstream_length], probabilities, cumulative_probabilities, total_symbols

def arithmeticDecompression(bitstream, probabilities, cumulative_probabilities, total_symbols):
    MAX_RANGE = 2**32 - 1
    HALF_RANGE = 2**31
    QUARTER_RANGE = 2**30

    low = 0
    high = MAX_RANGE
    value = 0

    # Initialize the value from the bitstream
    bitstream = iter(bitstream)  # Ensure bitstream is an iterator
    for _ in range(32):
        value = (value << 1) | next_bit(bitstream)

    decoded_data = []
    nbr_decoded_symbols = 0

    for _ in range(total_symbols):
        range_size = high - low + 1
        cum_value = ((value - low + 1) * 1.0 / range_size)  # Subtract a tiny value to avoid precision issues

        # Find the symbol for the current value
        for symbol, cum_prob in cumulative_probabilities.items():
            if cum_prob <= cum_value < cum_prob + probabilities[symbol]:
                decoded_data.append(symbol)
                nbr_decoded_symbols += 1

                symbol_low = cumulative_probabilities[symbol]
                symbol_high = symbol_low + probabilities[symbol]

                # Update the range
                high = low + int(range_size * symbol_high) - 1
                low = low + int(range_size * symbol_low)

                # Normalize range
                while True:
                    if high < HALF_RANGE:
                        # high and low are in the lower half
                        low <<= 1
                        high = (high << 1) + 1
                        value = (value << 1) | next_bit(bitstream)
                    elif low >= HALF_RANGE:
                        # high and low are in the upper half
                        low = (low - HALF_RANGE) << 1
                        high = ((high - HALF_RANGE) << 1) + 1
                        value = ((value - HALF_RANGE) << 1) | next_bit(bitstream)
                    elif low >= QUARTER_RANGE and high < HALF_RANGE + QUARTER_RANGE:
                        # low is in the upper quarter and high is in the lower quarter
                        low = (low - QUARTER_RANGE) << 1
                        high = ((high - QUARTER_RANGE) << 1) + 1
                        value = ((value - QUARTER_RANGE) << 1) | next_bit(bitstream)
                    else:
                        break
                break

    print(f"Decoded {nbr_decoded_symbols} symbols")
    return decoded_data

def next_bit(bitstream):
    try:
        return next(bitstream)
    except StopIteration:
        return 0  # Return 0 if the bitstream is exhausted

def readProbabilitiesFromFile(filename):
    probabilities = {}
    cumulative_probabilities = {}
    total_prob = 0.0

    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the first line
            parts = line.split()
            if len(parts) == 3:
                symbol, prob, cum_prob = parts
                probabilities[symbol] = float(prob)
                cumulative_probabilities[symbol] = float(cum_prob)
            elif len(parts) == 2:
                symbol, prob = parts
                probabilities[symbol] = float(prob)
                cumulative_probabilities[symbol] = total_prob
                total_prob += float(prob)
            else:
                print(f"Skipping malformed line: {line.strip()}")

    return probabilities, cumulative_probabilities

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python script.py <compress|decompress> <input_file> <output_file>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == 'compress':
        print("Compressing data...")
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        if not os.path.isfile(input_file):
            print(f"Error: File '{input_file}' not found.")
            sys.exit(1)

        with open(input_file, 'r') as file:
            data = file.read()

        probabilities, cumulative_probabilities = computeSymbolProbabilities(data)
        compressedCode, bit_output = arithmeticCompression(data, probabilities, cumulative_probabilities)

        writeBitstreamToFile(bit_output, output_file, probabilities, cumulative_probabilities, len(data))

        print(f"Compression complete. Output written to '{output_file}'.")

    elif mode == 'decompress':
        print("Decompressing data...")
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        if not os.path.isfile(input_file):
            print(f"Error: File '{input_file}' not found.")
            sys.exit(1)

        bitstream, probabilities, cumulative_probabilities, total_symbols = readBitstreamFromFile(input_file)

        print(f"total_symbols: {total_symbols}")

        decoded_data = arithmeticDecompression(bitstream, probabilities, cumulative_probabilities, total_symbols)

        with open(output_file, 'w') as file:
            file.write(''.join(decoded_data))

        print(f"Decompression complete. Decompressed data written to '{output_file}'.")

    else:
        print("Invalid mode. Use 'compress' or 'decompress'.")
        sys.exit(1)

if __name__ == "__main__":
    main()