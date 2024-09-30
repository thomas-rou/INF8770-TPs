import logging
import os
import math
import time
import psutil
from memory_profiler import memory_usage
import bitarray


# L'algorithme LZ77 utilisé ci-dessous est tiré de celui-ci : https://github.com/TimGuite/lz77/tree/master
def compress(
    input_data: bytes, max_offset: int = 2047, max_length: int = 31
) -> [(int, int, bytes)]:
    """Compress the input data into a list of length, offset, char values"""
    
    # Convert input data to a bytearray
    input_array = bytearray(input_data)

    # Create a bytearray for the sliding window
    window = bytearray()

    ## Store output in this list
    output = []

    while len(input_array) > 0:
        length, offset = best_length_offset(window, input_array, max_length, max_offset)
        char = input_array[0:1]  # Take the first byte
        output.append((offset, length, char))
        window.extend(input_array[:length])
        input_array = input_array[length:]

    return output

def to_bytes(
    compressed_representation: [(int, int, bytes)],
    offset_bits: int = 11,
    length_bits: int = 5,
) -> bytearray:
    """Turn the compression representation into a byte array"""
    output = bytearray()

    assert (
        offset_bits + length_bits
    ) % 8 == 0, f"Please provide offset_bits and length_bits which add up to a multiple of 8, so they can be efficiently packed. Received {offset_bits} and {length_bits}."
    offset_length_bytes = int((offset_bits + length_bits) / 8)

    for value in compressed_representation:
        offset, length, char = value
        assert (
            offset < 2 ** offset_bits
        ), f"Offset of {offset} is too large, only have {offset_bits} to store this value"
        assert (
            length < 2 ** length_bits
        ), f"Length of {length} is too large, only have {length_bits} to store this value"

        offset_length_value = (offset << length_bits) + length
        logging.debug(f"Offset: {offset}")
        logging.debug(f"Length: {length}")
        logging.debug(f"Offset and length: 0b{offset_length_value:b}")

        for count in range(offset_length_bytes):
            output.append(
                (offset_length_value >> (8 * (offset_length_bytes - count - 1)))
                & (0b11111111)
            )

        if char is not None:
            if offset == 0:
                output.extend(char)
        else:
            output.append(0)

    return output

def best_length_offset(
    window: bytearray, input_array: bytearray, max_length: int = 15, max_offset: int = 4095
) -> (int, int):
    """Take the window and an input array and return the offset and length
    with the biggest length of the input array as a substring"""

    if max_offset < len(window):
        cut_window = window[-max_offset:]
    else:
        cut_window = window

    # Return (0, 0) if the input array is empty
    if len(input_array) == 0:
        return (0, 0)

    # Initialise result parameters - best case so far
    length, offset = (1, 0)

    # This should also catch the empty window case
    if input_array[0:1] not in cut_window:
        best_length = repeating_length_from_start(input_array[0:1], input_array[1:])
        return (min((length + best_length), max_length), offset)

    # Best length now zero to allow occurrences to take priority
    length = 0

    # Test for every string in the window, in reverse order to keep the offset as low as possible
    for index in range(1, (len(cut_window) + 1)):
        # Get the character at this offset
        char = cut_window[-index:-index + 1]
        if char == input_array[0:1]:
            found_offset = index
            # Collect any further strings which can be found
            found_length = repeating_length_from_start(
                cut_window[-index:], input_array
            )
            if found_length > length:
                length = found_length
                offset = found_offset

    return (min(length, max_length), offset)

def repeating_length_from_start(window: bytearray, input_array: bytearray) -> int:
    """Get the maximum repeating length of the input from the start of the window"""
    if len(window) == 0 or len(input_array) == 0:
        return 0

    length = 0
    while length < len(window) and length < len(input_array) and window[length] == input_array[length]:
        length += 1

    return length


def compress_file(input_file: str, output_file: str):
    """Open and read an input file, compress it, and write the compressed
    values to the output file"""
    try:
        with open(input_file, 'rb') as f:
            input_array = f.read()
    except FileNotFoundError:
        print(f"Could not find input file at: {input_file}")
        raise
    except Exception:
        raise

    compressed_input = to_bytes(compress(input_array))

    with open(output_file, "wb") as f:
        f.write(compressed_input)
        
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
    
def perform_compression(input_file_path, compressed_file_path):
    print("\n=== Mesure du temps de compression ===")
    
    # Mesurer le temps de compression
    compression_start_time = time.time()
    compress_file(input_file_path, compressed_file_path)
    compression_end_time = time.time()
    
    compressed_file_size = os.path.getsize(compressed_file_path)
    print(f"Taille du fichier compressé : {math.ceil(compressed_file_size / 1024)} KB")
    print(f"Temps de compression : {compression_end_time - compression_start_time} secondes")
    
    # Mesurer la mémoire et le CPU utilisés pendant la compression
    print("\n=== Mesure de la mémoire et du CPU ===")
    mem_usage = memory_usage((compress_file, (input_file_path, compressed_file_path)), interval=0.1)
    cpu_usage_percent, _ = measure_cpu_usage(compress_file, input_file_path, compressed_file_path)

    print(f"Mémoire maximale utilisée pendant la compression : {max(mem_usage)} MiB")
    print(f"Pourcentage moyen d'utilisation du CPU pendant la compression : {cpu_usage_percent}%")
    
    return compression_end_time - compression_start_time, compressed_file_size
    
def calculate_compression_ratio(initial_size, compressed_size):
    compression_ratio = ((initial_size - compressed_size) / initial_size) * 100
    print(f"Taux de compression : {round(compression_ratio, 2)} %")
        
def main():
    input_file_path, compressed_file_path, decompressed_file_path = get_file_paths()
    
    # Afficher la taille du fichier original
    print_file_size(input_file_path, "Taille originale du fichier")
    
    initial_file_size = os.path.getsize(input_file_path)
    compression_time, compressed_file_size = perform_compression(input_file_path, compressed_file_path)
    
    # Afficher le taux de compression
    calculate_compression_ratio(initial_file_size, compressed_file_size)
    
if __name__ == "__main__":
    main()