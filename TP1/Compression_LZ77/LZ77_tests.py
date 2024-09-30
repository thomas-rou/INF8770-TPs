import os
import math
import time
import psutil
import threading
from bitarray import bitarray

# La classe LZ77Compressor contenant l'algorithme de compression et décompression LZ77 en python
# provient du repo GitHub suivant : https://github.com/manassra/LZ77-Compressor/tree/master
class LZ77Compressor:
    """
    A simplified implementation of the LZ77 Compression Algorithm
    """
    MAX_WINDOW_SIZE = 400

    def __init__(self, window_size=20):
        self.window_size = min(window_size, self.MAX_WINDOW_SIZE)
        self.lookahead_buffer_size = 15  # length of match is at most 4 bits

    def compress(self, input_file_path, output_file_path=None, verbose=False):
        data = None
        i = 0
        output_buffer = bitarray(endian='big')

        try:
            with open(input_file_path, 'rb') as input_file:
                data = input_file.read()
        except IOError:
            print('Could not open input file ...')
            raise

        while i < len(data):
            match = self.findLongestMatch(data, i)

            if match:
                (bestMatchDistance, bestMatchLength) = match

                output_buffer.append(True)
                output_buffer.frombytes(bytes([bestMatchDistance >> 4]))
                output_buffer.frombytes(bytes([((bestMatchDistance & 0xf) << 4) | bestMatchLength]))

                if verbose:
                    print("<1, %i, %i>" % (bestMatchDistance, bestMatchLength), end='')

                i += bestMatchLength
            else:
                output_buffer.append(False)
                output_buffer.frombytes(bytes([data[i]]))

                if verbose:
                    print("<0, %s>" % data[i], end='')

                i += 1

        output_buffer.fill()

        if output_file_path:
            try:
                with open(output_file_path, 'wb') as output_file:
                    output_file.write(output_buffer.tobytes())
                    print("File was compressed successfully and saved to output path ...")
                    return None
            except IOError:
                print('Could not write to output file path. Please check if the path is correct ...')
                raise

        return output_buffer

    def decompress(self, input_file_path, output_file_path=None):
        data = bitarray(endian='big')
        output_buffer = []

        try:
            with open(input_file_path, 'rb') as input_file:
                data.fromfile(input_file)
        except IOError:
            print('Could not open input file ...')
            raise

        while len(data) >= 9:
            flag = data.pop(0)

            if not flag:
                byte = data[0:8].tobytes()
                output_buffer.append(byte)
                del data[0:8]
            else:
                byte1 = ord(data[0:8].tobytes())
                byte2 = ord(data[8:16].tobytes())

                del data[0:16]
                distance = (byte1 << 4) | (byte2 >> 4)
                length = (byte2 & 0xf)

                for i in range(length):
                    output_buffer.append(output_buffer[-distance])

        out_data = b''.join(output_buffer)

        if output_file_path:
            try:
                with open(output_file_path, 'wb') as output_file:
                    output_file.write(out_data)
                    print('File was decompressed successfully and saved to output path ...')
                    return None
            except IOError:
                print('Could not write to output file path. Please check if the path is correct ...')
                raise
        return out_data

    def findLongestMatch(self, data, current_position):
        end_of_buffer = min(current_position + self.lookahead_buffer_size, len(data) + 1)
        best_match_distance = -1
        best_match_length = -1

        for j in range(current_position + 2, end_of_buffer):
            start_index = max(0, current_position - self.window_size)
            substring = data[current_position:j]

            for i in range(start_index, current_position):
                repetitions = len(substring) // (current_position - i)
                last = len(substring) % (current_position - i)
                matched_string = data[i:current_position] * repetitions + data[i:i + last]

                if matched_string == substring and len(substring) > best_match_length:
                    best_match_distance = current_position - i
                    best_match_length = len(substring)

        if best_match_distance > 0 and best_match_length > 0:
            return (best_match_distance, best_match_length)
        return None

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

def perform_compression(compressor, input_file_path, compressed_file_path):
    print("\n=== Compression ===")
    cpu_usage_percent, mem_usage, compression_time = measure_resources(compressor.compress, input_file_path, compressed_file_path)

    compressed_file_size = os.path.getsize(compressed_file_path)
    print(f"Taille du fichier compressé : {math.ceil(compressed_file_size / 1024)} KB")
    print(f"Temps de compression : {compression_time} secondes")
    print(f"Mémoire maximale utilisée pendant la compression : {mem_usage} MiB")
    print(f"Pourcentage moyen d'utilisation du CPU pendant la compression : {cpu_usage_percent}%")

    return compression_time, compressed_file_size

def perform_decompression(compressor, compressed_file_path, decompressed_file_path):
    print("\n=== Décompression ===")
    cpu_usage_percent, mem_usage, decompression_time = measure_resources(compressor.decompress, compressed_file_path, decompressed_file_path)

    print(f"Temps de décompression : {decompression_time} secondes")
    print(f"Mémoire maximale utilisée pendant la décompression : {mem_usage} MiB")
    print(f"Pourcentage moyen d'utilisation du CPU pendant la décompression : {cpu_usage_percent}%")

    return decompression_time

def calculate_compression_ratio(initial_size, compressed_size):
    compression_ratio = ((initial_size - compressed_size) / initial_size) * 100
    print(f"Taux de compression : {round(compression_ratio, 2)} %")

def main():
    try:
        input_file_path, compressed_file_path, decompressed_file_path = get_file_paths()
        window_size = int(input("Veuillez entrer la taille de la fenêtre (20 par défaut) : ") or 20)

        compressor = LZ77Compressor(window_size)

        # Affiche la taille initiale du fichier
        print_file_size(input_file_path, "Taille initiale du fichier")

        # Effectue la compression
        initial_file_size = os.path.getsize(input_file_path)
        compression_time, compressed_file_size = perform_compression(compressor, input_file_path, compressed_file_path)

        # Calcule le taux de compression
        calculate_compression_ratio(initial_file_size, compressed_file_size)

        # Effectue la décompression
        perform_decompression(compressor, compressed_file_path, decompressed_file_path)

    except ValueError as e:
        print(e)
    except FileNotFoundError:
        pass

if __name__ == '__main__':
    main()
