import matplotlib.pyplot as plt
import numpy as np

# Vérités terrain
verites_terrain = [
    {"temps": 0.5, "type": "F"},
    {"temps": 3.63, "type": "C"},
    {"temps": 6.88, "type": "C"},
    {"temps": 9.58, "type": "F"},
    {"temps": 11.38, "type": "F"},
    {"temps": 17.08, "type": "C"},
    {"temps": 17.62, "type": "C"},
    {"temps": 24.63, "type": "C"},
    {"temps": 32.63, "type": "C"},
    {"temps": 41.67, "type": "F"},
    {"temps": 46.83, "type": "C"},
    {"temps": 49.46, "type": "C"},
    {"temps": 52.33, "type": "C"},
    {"temps": 55.54, "type": "C"},
]

# EdgeDetector
edge_detector = [
    {"frame": 12, "temps": 0.40, "type": "F"},
    {"frame": 109, "temps": 3.64, "type": "C"},
    {"frame": 207, "temps": 6.91, "type": "C"},
    {"frame": 294, "temps": 9.81, "type": "F"},
    {"frame": 358, "temps": 11.95, "type": "F"},
    {"frame": 513, "temps": 17.12, "type": "C"},
    {"frame": 529, "temps": 17.65, "type": "C"},
    {"frame": 739, "temps": 24.66, "type": "C"},
    {"frame": 978, "temps": 32.63, "type": "C"},
    {"frame": 1270, "temps": 42.38, "type": "F"},
    {"frame": 1404, "temps": 46.85, "type": "C"},
    {"frame": 1483, "temps": 49.48, "type": "C"},
    {"frame": 1569, "temps": 52.35, "type": "C"},
    {"frame": 1665, "temps": 55.56, "type": "C"},
]

# HistogramDetector
histogram_detector = [
    {"frame": 2, "temps": 0.07, "type": "F"},
    {"frame": 109, "temps": 3.64, "type": "C"},
    {"frame": 207, "temps": 6.91, "type": "C"},
    {"frame": 288, "temps": 9.61, "type": "F"},
    {"frame": 513, "temps": 17.12, "type": "C"},
    {"frame": 739, "temps": 24.66, "type": "C"},
    {"frame": 978, "temps": 32.63, "type": "C"},
    {"frame": 1248, "temps": 41.64, "type": "F"},
    {"frame": 1404, "temps": 46.85, "type": "C"},
    {"frame": 1483, "temps": 49.48, "type": "C"},
    {"frame": 1569, "temps": 52.35, "type": "C"},
    {"frame": 1665, "temps": 55.56, "type": "C"},
    {"frame": 1880, "temps": 62.73, "type": "F"},
]

def separer_par_type(donnees):
    coupures = [d for d in donnees if d["type"] == "C"]
    fondues = [d for d in donnees if d["type"] == "F"]
    return coupures, fondues

def correspondances(detections, verites, tolerance=0.5):
    nb_correct = 0
    verites_non_match = verites.copy()

    for det in detections:
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
ax.set_title('Comparaison de la Précision et du Rappel pour Athletisme')
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
