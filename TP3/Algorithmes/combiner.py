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
    histogram_detector = HistogramBasedDetector(threshold_cut=0.6, threshold_effect=0.3, bins=16, Verbose=False)

    combiner = Combiner(edge_detector, histogram_detector, tolerance=0.3)

    results = combiner.process_video(video_path)

    print("\nRésumé des résultats prioritaires :")
    print(f"Fondues priorisées par EdgeDetection : {len([fade for fade in results['prioritized_fades'] if fade in edge_detector.get_transitions()[0]])}")
    print(f"Coupures fusionnées : {len(results['merged_cuts'])}")

if __name__ == "__main__":
    main()