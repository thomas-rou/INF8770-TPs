import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Fonction pour la conversion RGB -> YUV avec sous-échantillonnage 4:2:0
def convert_rgb_to_yuv(image: np.ndarray, subsampling: str) -> tuple:
    # Extraction des canaux RGB de l'image
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Conversion en YUV
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

# Fonction pour la conversion inverse YUV -> RGB
def convert_yuv_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    R = Y + 1.140 * V
    G = Y - 0.394 * U - 0.581 * V
    B = Y + 2.032 * U
    
    rgb_image = np.stack((R, G, B), axis=-1)
    
    return rgb_image.clip(0, 255).astype(np.uint8)

# Fonction pour appliquer la Transformée en Ondelettes Discrète (DWT)
def apply_dwt(image: np.ndarray, levels: int = 3) -> list:
    
    subbands = []
    current_image = image
    
    for _ in range(levels):
        # Filtrage passe-bas en X
        f1 = (current_image[:, ::2] + current_image[:, 1::2]) / 2
        f1h = (f1[::2, :] - f1[1::2, :]) / 2
        f11 = (f1[::2, :] + f1[1::2, :]) / 2
        
        # Filtrage passe-haut en X (fh)
        fh = (current_image[:, ::2] - current_image[:, 1::2]) / 2
        fh1 = (fh[::2, :] + fh[1::2, :]) / 2
        fhh = (fh[::2, :] - fh[1::2, :]) / 2
        
        subbands.append((f11, f1h, fh1, fhh))
        
        # On garde la sous-bande LL pour la prochaine itération
        current_image = f11
        
    return subbands

# Fonction pour appliquer la Transformée en Ondelettes Discrète Inverse (IDWT)
def apply_idwt(subbands: list, levels: int) -> np.ndarray:
    # On part du niveau le plus bas
    current_image = subbands[-1][0]  # LL au niveau le plus bas

    for i in range(levels - 1, -1, -1):
        # Récupération des sous-bandes
        f11, f1h, fh1, fhh = subbands[i]

        # Reconstruction en X
        f1 = np.zeros((f11.shape[0] * 2, f11.shape[1]))  # Placeholder pour la reconstruction en X
        f1[::2, :] = f11 + f1h
        f1[1::2, :] = f11 - f1h

        fh = np.zeros((fh1.shape[0] * 2, fh1.shape[1]))  # Placeholder pour la reconstruction en X
        fh[::2, :] = fh1 + fhh
        fh[1::2, :] = fh1 - fhh

        # Reconstruction en Y
        current_image = np.zeros((f1.shape[0], f1.shape[1] * 2))  # Placeholder pour la reconstruction finale
        current_image[:, ::2] = f1 + fh
        current_image[:, 1::2] = f1 - fh

    return current_image

# Fonction pour la quantification avec un quantificateur à zone morte
def dead_zone_quantize(signal: list, step_size, dead_zone_width) -> list:
    quantized_subbands = []

    # Appliquer la quantification sur chaque sous-bande séparément
    for subband in signal:
        quantized_subband = np.zeros_like(subband)

        # Applique la quantification à zone morte
        quantized_subband = np.where(
            np.abs(subband) > dead_zone_width / 2,
            np.sign(subband) * np.floor((np.abs(subband) - dead_zone_width / 2) / step_size),
            0
        )
        quantized_subbands.append(quantized_subband)

    return quantized_subbands

# Fonction inverse de la quantification
def inverse_quantize(quantized_coefficients: list, step_size: float) -> list:
    dequantized_subbands = []
    for subband in quantized_coefficients:
        dequantized_subband = subband * step_size
        dequantized_subbands.append(dequantized_subband)
    return dequantized_subbands


# Fonction pour afficher une image à partir d'un tableau NumPy
def display_image(image_array: np.ndarray, title: str = ""):
    plt.imshow(image_array, cmap='gray' if len(image_array.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Pipeline JPEG2000
def jpeg2000_pipeline(image_path: str, levels: int = 3, quantization_step_size: float = 1, dead_zone_width: float = 1):
    # Charger l'image
    image = Image.open(image_path)
    image_np = np.array(image)
    
    # Afficher l'image originale
    display_image(image_np, "Image Originale")

    # Conversion RGB -> YUV (sans sous-échantillonnage pour simplifier)
    Y, U, V = convert_rgb_to_yuv(image_np, '4:2:0')
    
    # Afficher les composantes Y, U, V
    display_image(Y, "Composante Y")
    display_image(U, "Composante U")
    display_image(V, "Composante V")
    
    # Appliquer la DWT sur chaque composante
    Y_subbands = apply_dwt(Y, levels)
    U_subbands = apply_dwt(U, levels)
    V_subbands = apply_dwt(V, levels)
    
    # Afficher la sous-bande LL après DWT
    display_image(Y_subbands[-1][0], "Sous-bande LL après DWT - Y")
    display_image(U_subbands[-1][0], "Sous-bande LL après DWT - U")
    display_image(V_subbands[-1][0], "Sous-bande LL après DWT - V")
    
    # Quantification des coefficients de la DWT
    quantized_Y_subbands = dead_zone_quantize(Y_subbands, quantization_step_size, dead_zone_width)
    quantized_U_subbands = dead_zone_quantize(U_subbands, quantization_step_size, dead_zone_width)
    quantized_V_subbands = dead_zone_quantize(V_subbands, quantization_step_size, dead_zone_width)
    
    # Afficher la sous-bande quantifiée
    display_image(quantized_Y_subbands[-1][0], "Sous-bande LL quantifiée - Y")
    display_image(quantized_U_subbands[-1][0], "Sous-bande LL quantifiée - U")
    display_image(quantized_V_subbands[-1][0], "Sous-bande LL quantifiée - V")

    # --- Processus inverse ---
    # Inverse de la quantification
    Y_subbands = inverse_quantize(quantized_Y_subbands, quantization_step_size)
    U_subbands = inverse_quantize(quantized_U_subbands, quantization_step_size)
    V_subbands = inverse_quantize(quantized_V_subbands, quantization_step_size)
    
    # Afficher la sous-bande après inverse quantification
    display_image(Y_subbands[-1][0], "Sous-bande LL après inverse quantification - Y")
    display_image(U_subbands[-1][0], "Sous-bande LL après inverse quantification - U")
    display_image(V_subbands[-1][0], "Sous-bande LL après inverse quantification - V")

    # Inverse de la DWT
    Y_reconstructed = apply_idwt(Y_subbands, levels)
    U_reconstructed = apply_idwt(U_subbands, levels)
    V_reconstructed = apply_idwt(V_subbands, levels)
    
    # Afficher les composantes Y, U, V reconstruites
    display_image(Y_reconstructed, "Composante Y reconstruite")
    display_image(U_reconstructed, "Composante U reconstruite")
    display_image(V_reconstructed, "Composante V reconstruite")
    
    # Conversion YUV -> RGB
    reconstructed_image = convert_yuv_to_rgb(Y_reconstructed, U_reconstructed, V_reconstructed)
    
    # Afficher l'image reconstruite
    display_image(reconstructed_image, "Image Reconstruite")

if __name__ == "__main__":
    # Exemple d'utilisation du pipeline avec une image test
    jpeg2000_pipeline("RGB.jpg")
