import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

# 1) RGB/YUV

# 1.1) RGB to YUV
def rgb_to_yuv(image):
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    Y = (R + 2*G + B) / 4
    U = B - G
    V = R - G
    return Y, U, V

# 1.2) YUV to RGB
def yuv_to_rgb(Y, U, V):
    G = (Y - (U + V) / 4)
    R = V + G
    B = U + G
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

# 3.1) Quantification
def quantize(coeffs, quant_param=50):
    return [(np.round(c / quant_param) * quant_param) for c in coeffs]

# 3.2) Reverse Quantification
def dequantize(coeffs, quant_param=50):
    return [(c * quant_param) for c in coeffs]

# Étape finale : Compression et décompression de l'image
def compress_decompress_image(image, quant_param=50):
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
        compressed_image = compress_decompress_image(image, quant_param=50)

        # Affichage des images
        display_images(image, compressed_image)