import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from SSIM_PIL import compare_ssim
import cv2

# Fonction pour la conversion RGB -> YUV
def convert_rgb_to_yuv(image: np.ndarray, subsampling: str = '4:2:0') -> tuple:
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Conversion en YUV
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = (B - Y) * 0.492
    V = (R - Y) * 0.877
    
    if subsampling == '4:4:4':
        return Y, U, V
    elif subsampling == '4:2:2':
        U = U[:, ::2]
        V = V[:, ::2]
    elif subsampling == '4:1:1':
        U = U[:, ::4]
        V = V[:, ::4]
    elif subsampling == '4:0:0':
        U = np.zeros_like(Y)
        V = np.zeros_like(Y)
    elif subsampling == '4:2:0':
        U = U[::2, ::2]
        V = V[::2, ::2]
    else:
        raise ValueError("Unsupported subsampling format. Use '4:4:4', '4:2:2', '4:1:1', '4:0:0', or '4:2:0'.")

    return Y, U, V

def convert_yuv_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    # Redimensionner U et V pour correspondre à Y en doublant les lignes et/ou les colonnes si nécessaire
    if U.shape != Y.shape:
        U = np.repeat(U, Y.shape[0] // U.shape[0], axis=0)
        U = np.repeat(U, Y.shape[1] // U.shape[1], axis=1)
    if V.shape != Y.shape:
        V = np.repeat(V, Y.shape[0] // V.shape[0], axis=0)
        V = np.repeat(V, Y.shape[1] // V.shape[1], axis=1)
    
    # Calculer les canaux RGB
    R = Y + 1.140 * V
    G = Y - 0.394 * U - 0.581 * V
    B = Y + 2.032 * U
    
    # Combiner les canaux en une seule image RGB
    rgb_image = np.stack((R, G, B), axis=-1)
    
    return rgb_image.clip(0, 255).astype(np.uint8)


# Fonction pour appliquer la Transformée en Ondelettes Discrète (DWT)
def apply_dwt(image: np.ndarray, levels: int = 3) -> list:
    subbands = []
    current_image = image

    for _ in range(levels):
        # Vérifie si les dimensions sont impaires et duplique le dernier pixel si nécessaire
        if current_image.shape[1] % 2 != 0:
            current_image = np.concatenate((current_image, current_image[:, -1:]), axis=1)
        if current_image.shape[0] % 2 != 0:
            current_image = np.concatenate((current_image, current_image[-1:, :]), axis=0)

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
def display_image(image_array: np.ndarray, title: str = "", image_array2: np.ndarray = None, title2: str = None):
    if image_array2 is None:
        # Affichage d'une seule image
        plt.imshow(image_array, cmap='gray' if len(image_array.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        # Affichage de deux images côte à côte
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 ligne, 2 colonnes
        axes[0].imshow(image_array, cmap='gray' if len(image_array.shape) == 2 else None)
        axes[0].set_title(title)
        axes[0].axis('off')
        
        # Convert images to PIL format
        original_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        reconstructed_pil = Image.fromarray(cv2.cvtColor(image_array2, cv2.COLOR_BGR2RGB))

        # Calculate SSIM
        ssim_value = compare_ssim(original_pil, reconstructed_pil)
        
        axes[1].imshow(image_array2, cmap='gray' if len(image_array2.shape) == 2 else None)
        axes[1].set_title(f'{title2}\nSSIM: {ssim_value:.4f}' if title2 else "")
        axes[1].axis('off')
        
        plt.show()

# Pipeline JPEG2000
def jpeg2000_pipeline(image_path: str, show_steps: bool = False, levels: int = 3, quantization_step_size: float = 1, dead_zone_width: float = 1):
    # Charger l'image
    image = Image.open(image_path)
    image_np = np.array(image)
    
    # Afficher l'image originale
    # display_image(image_np, "Image Originale")

    # Conversion RGB -> YUV (sans sous-échantillonnage pour simplifier)
    Y, U, V = convert_rgb_to_yuv(image_np, '4:2:0')
    
    # Afficher les composantes Y, U, V
    if show_steps:
        display_image(Y, "Composante Y")
        display_image(U, "Composante U")
        display_image(V, "Composante V")
    
    # Appliquer la DWT sur chaque composante
    Y_subbands = apply_dwt(Y, levels)
    U_subbands = apply_dwt(U, levels)
    V_subbands = apply_dwt(V, levels)
    
    # Afficher la sous-bande LL après DWT
    if show_steps:
        display_image(Y_subbands[-1][0], "Sous-bande LL après DWT - Y")
        display_image(U_subbands[-1][0], "Sous-bande LL après DWT - U")
        display_image(V_subbands[-1][0], "Sous-bande LL après DWT - V")
    
    # Quantification des coefficients de la DWT
    quantized_Y_subbands = dead_zone_quantize(Y_subbands, quantization_step_size, dead_zone_width)
    quantized_U_subbands = dead_zone_quantize(U_subbands, quantization_step_size, dead_zone_width)
    quantized_V_subbands = dead_zone_quantize(V_subbands, quantization_step_size, dead_zone_width)
    
    # Afficher la sous-bande quantifiée
    if show_steps:
        display_image(quantized_Y_subbands[-1][0], "Sous-bande LL quantifiée - Y")
        display_image(quantized_U_subbands[-1][0], "Sous-bande LL quantifiée - U")
        display_image(quantized_V_subbands[-1][0], "Sous-bande LL quantifiée - V")

    # --- Processus inverse ---
    # Inverse de la quantification
    Y_subbands = inverse_quantize(quantized_Y_subbands, quantization_step_size)
    U_subbands = inverse_quantize(quantized_U_subbands, quantization_step_size)
    V_subbands = inverse_quantize(quantized_V_subbands, quantization_step_size)
    
    # Afficher la sous-bande après inverse quantification
    if show_steps:
        display_image(Y_subbands[-1][0], "Sous-bande LL après inverse quantification - Y")
        display_image(U_subbands[-1][0], "Sous-bande LL après inverse quantification - U")
        display_image(V_subbands[-1][0], "Sous-bande LL après inverse quantification - V")

    # Inverse de la DWT
    Y_reconstructed = apply_idwt(Y_subbands, levels)
    U_reconstructed = apply_idwt(U_subbands, levels)
    V_reconstructed = apply_idwt(V_subbands, levels)
    
    # Afficher les composantes Y, U, V reconstruites
    if show_steps:
        display_image(Y_reconstructed, "Composante Y reconstruite")
        display_image(U_reconstructed, "Composante U reconstruite")
        display_image(V_reconstructed, "Composante V reconstruite")
    
    # Conversion YUV -> RGB
    reconstructed_image = convert_yuv_to_rgb(Y_reconstructed, U_reconstructed, V_reconstructed)
    
    # Afficher l'image reconstruite
    display_image(image_np, "Image Original", reconstructed_image, "Image Reconstruite")

if __name__ == "__main__":
    # Exemple d'utilisation du pipeline avec une image test
    jpeg2000_pipeline("RGB.jpg", True)
