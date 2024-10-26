import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt


QUANT_PARAM = 1.25
DEAD_ZONE = 25

# Note : We used the standard RGB to YUV and YUV to RGB using weighted contribution instead of the given formula
# since the given formula was causing color distortion

# 1) RGB/YUV

# 1.1) RGB to YUV with 4:2:0 subsampling
def rgb_to_yuv(image, subsampling='4:2:0'):
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = (B - Y) * 0.492
    V = (R - Y) * 0.877

    if subsampling == '4:4:4' :
        return Y, U, V
    elif subsampling == '4:2:2':
        U[:, 1::2] = U[:, ::2]  # Subsample U horizontally by 2
        U[:, ::2] = 0  # Set the rest to 0
        V[:, 1::2] = V[:, ::2]  # Subsample V horizontally by 2
        V[:, ::2] = 0  # Set the rest to 0
    elif subsampling == '4:1:1':
        U[:, 1::4] = U[:, ::4]  # Subsample U horizontally by 4
        U[:, ::4] = 0  # Set the rest to 0
        V[:, 1::4] = V[:, ::4]  # Subsample V horizontally by 4
        V[:, ::4] = 0  # Set the rest to 0
    elif subsampling == '4:0:0':
        U[:, :] = 0  # Discard U
        V[:, :] = 0  # Discard V
    elif subsampling == '4:2:0':
        U[1::2, :] = U[::2, :]  # Subsample U both horizontally and vertically by 2
        U[:, 1::2] = U[:, ::2]
        U[::2, :] = 0  # Set the rest to 0
        V[1::2, :] = V[::2, :]  # Subsample V both horizontally and vertically by 2
        V[:, 1::2] = V[:, ::2]
        V[::2, :] = 0  # Set the rest to 0
    else:
        raise ValueError("Unsupported subsampling format. Use '4:4:4', '4:2:2', '4:1:1', '4:0:0', or '4:2:0'.")
    return Y, U, V

# 1.2) YUV to RGB with 4:2:0 upsampling
def yuv_to_rgb(Y, U, V):
    # U = cv2.resize(U, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)  # Upsample U
    # V = cv2.resize(V, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)  # Upsample V
    R = Y + 1.140 * V
    G = Y - 0.394 * U - 0.581 * V
    B = Y + 2.032 * U
    rgb_image = np.stack((R, G, B), axis=-1)
    return rgb_image.clip(0, 255).astype(np.uint8)

# 2) Wavelet transform (DWT)

# 2.1) DWT
def dwt2d(channel):
    coeffs = pywt.wavedec2(channel, 'haar', level=3)
    return coeffs

# 2.2) Reverse DWT
def idwt2d(coeffs):
    return pywt.waverec2(coeffs, 'haar')

# 3) Quantification

# 3.1) Quantification with dead-zone
def quantize(coeffs, quant_param=2):
    quantized_coeffs = []
    for c in coeffs:
        quantized_c = np.where(np.abs(c) < DEAD_ZONE, 0, np.round(c / quant_param) * quant_param)
        quantized_coeffs.append(quantized_c)
    return quantized_coeffs

# 3.2) Reverse Quantification
def dequantize(coeffs, quant_param=2):
    dequantized_coeffs = []
    for c in coeffs:
        dequantized_c = c * quant_param
        dequantized_coeffs.append(dequantized_c)
    return dequantized_coeffs

# Étape finale : Compression et décompression de l'image
def compress_decompress_image(image, quant_param=2):
    # 1) Conversion RGB vers YUV
    Y, U, V = rgb_to_yuv(image)

    # 2) DWT sur chaque canal
    Y_dwt = dwt2d(Y)
    U_dwt = dwt2d(U)
    V_dwt = dwt2d(V)

    # 3) Quantification
    Y_dwt_quant = [quantize(coeff, quant_param) for coeff in Y_dwt]
    U_dwt_quant = [quantize(coeff, quant_param) for coeff in U_dwt]
    V_dwt_quant = [quantize(coeff, quant_param) for coeff in V_dwt]

    # Reverse quantification
    Y_dwt_dequant = [dequantize(coeff, quant_param) for coeff in Y_dwt_quant]
    U_dwt_dequant = [dequantize(coeff, quant_param) for coeff in U_dwt_quant]
    V_dwt_dequant = [dequantize(coeff, quant_param) for coeff in V_dwt_quant]

    # Reverse DWT pour la reconstruction
    Y_reconstructed = idwt2d(Y_dwt_dequant)
    U_reconstructed = idwt2d(U_dwt_dequant)
    V_reconstructed = idwt2d(V_dwt_dequant)

    # Reconstruction RGB
    reconstructed_image = yuv_to_rgb(Y_reconstructed, U_reconstructed, V_reconstructed)

    return reconstructed_image

# Chargement et affichage de l'image originale et reconstruite
def display_images(original_image, compressed_image):
    plt.figure(figsize=(12,6))

    # Image originale
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')

    # Image reconstruite
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB))
    plt.title('Image Reconstruite')

    plt.show()

if __name__ == '__main__':
    # Charger l'image en tant que tableau NumPy
    image = cv2.imread('./lenna.tiff')
    if image is None:
        print("Error: Image not found or unable to load.")
    else:
        # Compression et décompression de l'image
        compressed_image = compress_decompress_image(image, quant_param=QUANT_PARAM)

        # Affichage des images
        display_images(image, compressed_image)