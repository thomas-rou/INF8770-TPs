import matplotlib.pyplot as plt
import numpy as np

# Vérités terrain
verites_terrain = [
    {"temps": 2.30, "type": "C"},
    {"temps": 4.50, "type": "C"},
    {"temps": 8.80, "type": "C"},
    {"temps": 10.70, "type": "C"},
    {"temps": 12.03, "type": "C"},
    {"temps": 13.10, "type": "C"},
    {"temps": 14.66, "type": "C"},
    {"temps": 19.16, "type": "F"},
    {"temps": 19.53, "type": "C"},
    {"temps": 21.43, "type": "C"},
    {"temps": 27.33, "type": "C"},
    {"temps": 32.46, "type": "C"},
    {"temps": 33.96, "type": "C"},
    {"temps": 37.70, "type": "C"},
    {"temps": 38.70, "type": "C"},
    {"temps": 45.26, "type": "C"},
    {"temps": 46.73, "type": "C"},
    {"temps": 49.23, "type": "F"},
    {"temps": 52.60, "type": "C"},
    {"temps": 55.40, "type": "C"},
    {"temps": 56.66, "type": "C"},
    {"temps": 61.20, "type": "F"},
    {"temps": 61.40, "type": "C"},
    {"temps": 66.00, "type": "C"},
    {"temps": 71.00, "type": "C"},
    {"temps": 72.63, "type": "C"},
    {"temps": 77.70, "type": "C"},
    {"temps": 79.70, "type": "C"},
    {"temps": 85.20, "type": "C"},
    {"temps": 87.03, "type": "C"},
    {"temps": 89.40, "type": "C"},
    {"temps": 91.33, "type": "C"},
    {"temps": 93.43, "type": "C"},
    {"temps": 93.76, "type": "C"},
    {"temps": 99.33, "type": "C"},
]

# EdgeDetector
edge_detector = [
    {"frame": 57, "temps": 2.28, "type": "C"},
    {"frame": 220, "temps": 8.80, "type": "C"},
    {"frame": 260, "temps": 10.40, "type": "C"},
    {"frame": 367, "temps": 14.68, "type": "C"},
    {"frame": 484, "temps": 19.36, "type": "F"},
    {"frame": 536, "temps": 21.44, "type": "C"},
    {"frame": 568, "temps": 22.72, "type": "F"},
    {"frame": 684, "temps": 27.36, "type": "C"},
    {"frame": 812, "temps": 32.48, "type": "C"},
    {"frame": 850, "temps": 34.00, "type": "C"},
    {"frame": 943, "temps": 37.72, "type": "C"},
    {"frame": 968, "temps": 38.72, "type": "C"},
    {"frame": 1132, "temps": 45.28, "type": "C"},
    {"frame": 1147, "temps": 45.88, "type": "F"},
    {"frame": 1169, "temps": 46.76, "type": "C"},
    {"frame": 1233, "temps": 49.32, "type": "C"},
    {"frame": 1235, "temps": 49.40, "type": "F"},
    {"frame": 1386, "temps": 55.44, "type": "C"},
    {"frame": 1418, "temps": 56.72, "type": "C"},
    {"frame": 1536, "temps": 61.44, "type": "C"},
    {"frame": 1539, "temps": 61.56, "type": "F"},
    {"frame": 1651, "temps": 66.04, "type": "C"},
    {"frame": 1776, "temps": 71.04, "type": "C"},
    {"frame": 1817, "temps": 72.68, "type": "C"},
    {"frame": 1944, "temps": 77.76, "type": "C"},
    {"frame": 1957, "temps": 78.28, "type": "F"},
    {"frame": 1970, "temps": 78.80, "type": "F"},
    {"frame": 1993, "temps": 79.72, "type": "C"},
    {"frame": 2132, "temps": 85.28, "type": "C"},
    {"frame": 2178, "temps": 87.12, "type": "C"},
    {"frame": 2237, "temps": 89.48, "type": "C"},
    {"frame": 2285, "temps": 91.40, "type": "C"},
    {"frame": 2338, "temps": 93.52, "type": "C"},
    {"frame": 2423, "temps": 96.92, "type": "F"},
    {"frame": 2484, "temps": 99.36, "type": "C"},
]

# HistogramDetector
histogram_detector = [
    {"frame": 57, "temps": 2.28, "type": "C"},
    {"frame": 112, "temps": 4.48, "type": "C"},
    {"frame": 220, "temps": 8.80, "type": "C"},
    {"frame": 260, "temps": 10.40, "type": "C"},
    {"frame": 301, "temps": 12.04, "type": "F"},
    {"frame": 327, "temps": 13.08, "type": "C"},
    {"frame": 367, "temps": 14.68, "type": "C"},
    {"frame": 488, "temps": 19.52, "type": "C"},
    {"frame": 514, "temps": 20.56, "type": "F"},
    {"frame": 684, "temps": 27.36, "type": "C"},
    {"frame": 812, "temps": 32.48, "type": "C"},
    {"frame": 850, "temps": 34.00, "type": "C"},
    {"frame": 943, "temps": 37.72, "type": "C"},
    {"frame": 968, "temps": 38.72, "type": "C"},
    {"frame": 1132, "temps": 45.28, "type": "C"},
    {"frame": 1160, "temps": 46.40, "type": "F"},
    {"frame": 1231, "temps": 49.24, "type": "F"},
    {"frame": 1316, "temps": 52.64, "type": "C"},
    {"frame": 1386, "temps": 55.44, "type": "C"},
    {"frame": 1418, "temps": 56.72, "type": "C"},
    {"frame": 1651, "temps": 66.04, "type": "C"},
    {"frame": 1776, "temps": 71.04, "type": "C"},
    {"frame": 1817, "temps": 72.68, "type": "C"},
    {"frame": 1944, "temps": 77.76, "type": "C"},
    {"frame": 1993, "temps": 79.72, "type": "C"},
    {"frame": 2132, "temps": 85.28, "type": "C"},
    {"frame": 2178, "temps": 87.12, "type": "C"},
    {"frame": 2237, "temps": 89.48, "type": "C"},
    {"frame": 2285, "temps": 91.40, "type": "F"},
    {"frame": 2338, "temps": 93.52, "type": "C"},
    {"frame": 2482, "temps": 99.28, "type": "C"},
]

def separer_par_type(donnees):
    coupures = [d for d in donnees if d["type"] == "C"]
    fondues = [d for d in donnees if d["type"] == "F"]
    return coupures, fondues

def correspondances(detections, verites, tolerance=0.5):
    nb_correct = 0
    verites_non_match = verites.copy()

    for det in detections:
        # Chercher une vérité terrain non encore matchée qui correspond
        correspondance = None
        for ver in verites_non_match:
            if det["type"] == ver["type"] and abs(det["temps"] - ver["temps"]) <= tolerance:
                correspondance = ver
                break
        if correspondance:
            nb_correct += 1
            verites_non_match.remove(correspondance)

    return nb_correct

verites_coupure = [v for v in verites_terrain if v["type"] == "C"]
verites_fondu = [v for v in verites_terrain if v["type"] == "F"]

edge_coupures, edge_fondues = separer_par_type(edge_detector)

hist_coupures, hist_fondues = separer_par_type(histogram_detector)

nb_correct_edge_coupure = correspondances(edge_coupures, verites_coupure)
precision_edge_coupure = (nb_correct_edge_coupure / len(edge_coupures)) * 100 if len(edge_coupures) > 0 else 0
rappel_edge_coupure = (nb_correct_edge_coupure / len(verites_coupure)) * 100 if len(verites_coupure) > 0 else 0

nb_correct_edge_fondu = correspondances(edge_fondues, verites_fondu)
precision_edge_fondu = (nb_correct_edge_fondu / len(edge_fondues)) * 100 if len(edge_fondues) > 0 else 0
rappel_edge_fondu = (nb_correct_edge_fondu / len(verites_fondu)) * 100 if len(verites_fondu) > 0 else 0

nb_correct_hist_coupure = correspondances(hist_coupures, verites_coupure)
precision_hist_coupure = (nb_correct_hist_coupure / len(hist_coupures)) * 100 if len(hist_coupures) > 0 else 0
rappel_hist_coupure = (nb_correct_hist_coupure / len(verites_coupure)) * 100 if len(verites_coupure) > 0 else 0

nb_correct_hist_fondu = correspondances(hist_fondues, verites_fondu)
precision_hist_fondu = (nb_correct_hist_fondu / len(hist_fondues)) * 100 if len(hist_fondues) > 0 else 0
rappel_hist_fondu = (nb_correct_hist_fondu / len(verites_fondu)) * 100 if len(verites_fondu) > 0 else 0

print("=== Résultats ===")
print(f"EdgeDetector - Coupure :")
print(f"  Détections correctes : {nb_correct_edge_coupure}")
print(f"  Précision : {precision_edge_coupure:.2f}%")
print(f"  Rappel : {rappel_edge_coupure:.2f}%\n")

print(f"EdgeDetector - Fondu :")
print(f"  Détections correctes : {nb_correct_edge_fondu}")
print(f"  Précision : {precision_edge_fondu:.2f}%")
print(f"  Rappel : {rappel_edge_fondu:.2f}%\n")

print(f"HistogramDetector - Coupure :")
print(f"  Détections correctes : {nb_correct_hist_coupure}")
print(f"  Précision : {precision_hist_coupure:.2f}%")
print(f"  Rappel : {rappel_hist_coupure:.2f}%\n")

print(f"HistogramDetector - Fondu :")
print(f"  Détections correctes : {nb_correct_hist_fondu}")
print(f"  Précision : {precision_hist_fondu:.2f}%")
print(f"  Rappel : {rappel_hist_fondu:.2f}%\n")

types_transition = ['Coupure', 'Fondu']
algorithmes = ['EdgeDetector', 'HistogramDetector']

precision = [
    [precision_edge_coupure, precision_hist_coupure],
    [precision_edge_fondu, precision_hist_fondu],
]

rappel = [
    [rappel_edge_coupure, rappel_hist_coupure],
    [rappel_edge_fondu, rappel_hist_fondu],
]

x = np.arange(len(types_transition))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 8))

offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

bar1 = ax.bar(x + offsets[0], precision[0], width, label='Précision EdgeDetector', color='skyblue')
bar2 = ax.bar(x + offsets[1], rappel[0], width, label='Rappel EdgeDetector', color='lightgreen')

bar3 = ax.bar(x + offsets[2], precision[1], width, label='Précision HistogramDetector', color='salmon')
bar4 = ax.bar(x + offsets[3], rappel[1], width, label='Rappel HistogramDetector', color='orange')

ax.set_ylabel('Pourcentage (%)')
ax.set_title('Comparaison de la Précision et du Rappel pour Soccer')
ax.set_xticks(x)
ax.set_xticklabels(types_transition)
ax.set_ylim(0, 110)
ax.legend()

def ajouter_valeurs(bars):
    for bar in bars:
        hauteur = bar.get_height()
        ax.annotate(f'{hauteur:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, hauteur),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

ajouter_valeurs(bar1)
ajouter_valeurs(bar2)
ajouter_valeurs(bar3)
ajouter_valeurs(bar4)

plt.tight_layout()
plt.show()