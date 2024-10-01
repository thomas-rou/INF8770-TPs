import os
import math
from collections import Counter
from PIL import Image

def calculer_entropie_texte(fichier):
    with open(fichier, "r", encoding="utf-8") as file:
        texte = file.read()

    frequences = Counter(texte)

    total_characters = sum(frequences.values())
    probabilites = {char: count / total_characters for char, count in frequences.items()}

    entropie = -sum(p * math.log2(p) for p in probabilites.values() if p > 0)

    taille_originale = total_characters * 8
    taille_compressee = total_characters * entropie
    taux_compression = 1 - (taille_compressee / taille_originale)

    print(f"Fichier texte : {fichier}")
    print(f"L'entropie du texte est d'environ {entropie:.4f} bits par caractère.")
    print(f"Taille originale : {taille_originale} bits")
    print(f"Taille compressée théorique : {taille_compressee:.2f} bits")
    print(f"Taux de compression théorique : {taux_compression:.2%}\n")

def calculer_entropie_image(fichier):
    with Image.open(fichier) as img:
        img = img.convert("L")
        pixels = list(img.getdata())

    frequences = Counter(pixels)

    total_pixels = sum(frequences.values())
    probabilites = {pixel: count / total_pixels for pixel, count in frequences.items()}

    entropie = -sum(p * math.log2(p) for p in probabilites.values() if p > 0)

    taille_originale = total_pixels * 8
    taille_compressee = total_pixels * entropie
    taux_compression = 1 - (taille_compressee / taille_originale)

    print(f"Fichier image : {fichier}")
    print(f"L'entropie de l'image est d'environ {entropie:.4f} bits par pixel.")
    print(f"Taille originale : {taille_originale} bits")
    print(f"Taille compressée théorique : {taille_compressee:.2f} bits")
    print(f"Taux de compression théorique : {taux_compression:.2%}\n")


text_folder = "../Test-Files/Text-Files/"
image_folder = "../Test-Files/Image-Files/"


for filename in os.listdir(text_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(text_folder, filename)
        calculer_entropie_texte(filepath)


for filename in os.listdir(image_folder):
    if filename.endswith(".tiff"):
        filepath = os.path.join(image_folder, filename)
        calculer_entropie_image(filepath)
