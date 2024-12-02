import matplotlib.pyplot as plt
from edgeDetection import EdgeDetector
from histogramDetection import HistogramBasedDetector

class Combiner:
    def __init__(self, edge_detector, histogram_detector, tolerance=0.1):
        self.edge_detector = edge_detector
        self.histogram_detector = histogram_detector
        self.tolerance = tolerance

    def prioritize_fades(self, edge_fades, histogram_fades):
        prioritized_fades = edge_fades[:]
        for fade in histogram_fades:
            if not any(abs(fade[0] - e[0]) < self.tolerance and abs(fade[1] - e[1]) < self.tolerance for e in edge_fades):
                prioritized_fades.append(fade)
        return prioritized_fades

    def merge_cuts(self, edge_cuts, histogram_cuts):
        all_cuts = edge_cuts + histogram_cuts
        all_cuts.sort(key=lambda x: x[0])
        merged_cuts = []
        for cut in all_cuts:
            if not merged_cuts or abs(merged_cuts[-1][0] - cut[0]) >= self.tolerance:
                merged_cuts.append(cut)
        return merged_cuts

    def compare_results(self, edge_transitions, histogram_transitions):
        both_detected = []
        edge_only = []
        histogram_only = []
        for edge in edge_transitions:
            found = False
            for histogram in histogram_transitions:
                if abs(edge[0] - histogram[0]) < self.tolerance:
                    both_detected.append(edge)
                    found = True
                    break
            if not found:
                edge_only.append(edge)
        for histogram in histogram_transitions:
            if not any(abs(histogram[0] - edge[0]) < self.tolerance for edge in edge_transitions):
                histogram_only.append(histogram)
        return both_detected, edge_only, histogram_only

    def visualize_results(self, edge_fades, edge_cuts, histogram_fades, histogram_cuts):
        plt.figure(figsize=(12, 6))

        offset = 0.1

        # Athlétisme
        # ground_truth_fades = [12, 41]  # En secondes
        # ground_truth_cuts = [17, 24, 32]  # En secondes

        # Soccer
        ground_truth_fades = [19.167, 49.234, 61.2]  # En secondes
        ground_truth_cuts = [
            2.3, 4.5, 8.8, 10.7, 12.033, 13.1, 14.667,
            19.534, 21.434, 27.333, 32.467, 33.967,
            37.7, 38.7, 45.267, 46.734, 61.4, 66.0,
            71.0, 72.633, 77.734, 79.734, 85.2, 87.034,
            89.4, 91.367, 93.433, 93.767, 99.367
        ]  # En secondes

        for gt_fade in ground_truth_fades:
            plt.axvline(gt_fade, color='green', linestyle='--', label="Ground Truth - Fondu" if 'Ground Truth - Fondu' not in plt.gca().get_legend_handles_labels()[1] else "")

        for gt_cut in ground_truth_cuts:
            plt.axvline(gt_cut, color='purple', linestyle='--', label="Ground Truth - Coupure" if 'Ground Truth - Coupure' not in plt.gca().get_legend_handles_labels()[1] else "")


        for fade in edge_fades:
            middle_time = (fade[0] + fade[1]) / 2
            plt.scatter(middle_time, 1, color='red', s=100, label="EdgeDetector - Fondu" if 'EdgeDetector - Fondu' not in plt.gca().get_legend_handles_labels()[1] else "")
        for cut in edge_cuts:
            plt.scatter(cut[0] - offset, 1, color='black', s=100, label="EdgeDetector - Coupure" if 'EdgeDetector - Coupure' not in plt.gca().get_legend_handles_labels()[1] else "")

        for fade in histogram_fades:
            middle_time = (fade[0] + fade[1]) / 2
            plt.scatter(middle_time, 1.5, color='green', marker='x', s=100, label="Histogram - Fondu" if 'Histogram - Fondu' not in plt.gca().get_legend_handles_labels()[1] else "")
        for cut in histogram_cuts:
            plt.scatter(cut[0] + offset, 1.5, color='blue', marker='x', s=100, label="Histogram - Coupure" if 'Histogram - Coupure' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.yticks([1, 1.5], ["EdgeDetector", "Histogram"])
        plt.xlabel("Temps (secondes)")
        plt.title("Visualisation des transitions détectées")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def process_video(self, video_path):
        self.edge_detector.reset()
        self.histogram_detector.reset()

        print("Exécution de EdgeDetection...")
        self.edge_detector.detect_transitions(video_path)
        edge_fades, edge_cuts = self.edge_detector.get_transitions()

        print("Exécution de HistogramBasedDetector...")
        self.histogram_detector.process_video(video_path)
        histogram_fades, histogram_cuts = self.histogram_detector.get_transitions()

        print("Priorisation des fondues...")
        prioritized_fades = self.prioritize_fades(edge_fades, histogram_fades)

        print("Fusion des coupures...")
        merged_cuts = self.merge_cuts(edge_cuts, histogram_cuts)

        print("Visualisation des résultats...")
        self.visualize_results(edge_fades, edge_cuts, histogram_fades, histogram_cuts)

        print("Comparaison des résultats...")
        both_detected, edge_only, histogram_only = self.compare_results(
            edge_fades + edge_cuts, histogram_fades + histogram_cuts
        )

        print("\nCompte-rendu des transitions détectées :")
        print(f"Transitions détectées par les deux algorithmes : {len(both_detected)}")
        for t in both_detected:
            print(f"- Transition détectée à {t[0]:.2f} secondes.")

        print(f"\nTransitions détectées uniquement par EdgeDetection : {len(edge_only)}")
        for t in edge_only:
            print(f"- Transition détectée à {t[0]:.2f} secondes.")

        print(f"\nTransitions détectées uniquement par HistogramBasedDetector : {len(histogram_only)}")
        for t in histogram_only:
            print(f"- Transition détectée à {t[0]:.2f} secondes.")

        print("\nRésultats combinés :")
        for t in prioritized_fades:
            print(f"- Fondu priorisé détecté entre {t[0]:.2f} secondes et {t[1]:.2f} secondes.")
        for t in merged_cuts:
            print(f"- Coupure fusionnée détectée à {t[0]:.2f} secondes.")


        return {
            "both_detected": both_detected,
            "edge_only": edge_only,
            "histogram_only": histogram_only,
            "prioritized_fades": prioritized_fades,
            "merged_cuts": merged_cuts,
        }

def main():
    video_path = "../VideodataTP3/Soccer.mp4"

    edge_detector = EdgeDetector(Verbose=True)
    histogram_detector = HistogramBasedDetector(threshold_cut=0.6, threshold_effect=0.3, bins=16, Verbose=True)

    combiner = Combiner(edge_detector, histogram_detector, tolerance=0.3)

    results = combiner.process_video(video_path)

    print("\nRésumé des résultats prioritaires :")
    print(f"Fondues priorisées par EdgeDetection : {len([fade for fade in results['prioritized_fades'] if fade in edge_detector.get_transitions()[0]])}")
    print(f"Coupures fusionnées : {len(results['merged_cuts'])}")

if __name__ == "__main__":
    main()