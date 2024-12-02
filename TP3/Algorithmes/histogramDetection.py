import cv2
import numpy as np
import matplotlib.pyplot as plt

class HistogramBasedDetector:
    def __init__(self, threshold_cut=0.5, threshold_effect=0.2, bins=16, Verbose=False):
        self.threshold_cut = threshold_cut
        self.threshold_effect = threshold_effect
        self.bins = bins
        self.min_frame_gap = 0
        self.previous_histogram = None
        self.last_transition_frame = -self.min_frame_gap
        self.Verbose = Verbose
        self.transitions = []
        self.distances = []
        self.timestamps = []

    def reset(self):
        """
        Réinitialise les données internes pour une nouvelle exécution.
        """
        self.previous_histogram = None
        self.last_transition_frame = -self.min_frame_gap
        self.transitions = []
        self.distances = []
        self.timestamps = []

    def calculate_histogram(self, frame):
        histogram = []
        for channel in cv2.split(frame):
            hist = cv2.calcHist([channel], [0], None, [self.bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histogram.append(hist)
        return np.concatenate(histogram)

    def calculate_distance(self, hist_a, hist_b):
        return np.sqrt(np.sum((hist_a - hist_b) ** 2))

    def detect_transition(self, current_histogram, frame_count):
        if self.previous_histogram is None:
            self.previous_histogram = current_histogram
            return None
        timestamp = frame_count / 30
        distance = self.calculate_distance(self.previous_histogram, current_histogram)
        self.distances.append(distance)

        self.previous_histogram = current_histogram

        if frame_count - self.last_transition_frame < self.min_frame_gap:
            return None

        if distance > self.threshold_cut:
            self.last_transition_frame = frame_count
            return "Coupure"
        elif distance > self.threshold_effect:
            self.last_transition_frame = frame_count
            return "Effet"
        else:
            return None

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if self.Verbose:
                print(f"Erreur : Impossible de lire la vidéo '{video_path}'")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.min_frame_gap = int(fps)
        self.last_transition_frame = -self.min_frame_gap
        if self.Verbose:
            print(f"FPS de la vidéo : {fps:.2f}")
        frame_count = 0
        self.transitions = []  # Réinitialiser les transitions

        if self.Verbose:
            print("Début du traitement de la vidéo...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 360))
            current_histogram = self.calculate_histogram(frame)
            transition = self.detect_transition(current_histogram, frame_count)
            timestamp = frame_count / fps

            if transition:
                self.transitions.append((frame_count, transition, timestamp))
                if self.Verbose:
                    print(f"Transition détectée ({transition}) à {timestamp:.2f} secondes.")
            if frame_count > 0:
                self.timestamps.append(timestamp)

            frame_count += 1

        cap.release()
        if self.Verbose:
            print("Nombre de transitions détectées :", len(self.transitions))
            print("Fin du traitement.")
        return self.transitions

    def get_transitions(self):
        """
        Retourne séparément toutes les fondues et coupures détectées.
        """
        fades = [(t[2], t[2]) for t in self.transitions if t[1] == "Effet"]
        cuts = [(t[2], t[2]) for t in self.transitions if t[1] == "Coupure"]
        return fades, cuts


    def plot_distances(self):
        """
        Plot the distances between consecutive frames.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.timestamps, self.distances, label='Distance euclidienne entre les histogrammes')
         # Athlétisme
        ground_truth_fades = [12, 41]  # En secondes
        ground_truth_cuts = [17, 24, 32]  # En secondes

        # Soccer
        # ground_truth_fades = [19.167, 49.234, 61.2]  # En secondes
        # ground_truth_cuts = [
        #     2.3, 4.5, 8.8, 10.7, 12.033, 13.1, 14.667,
        #     19.534, 21.434, 27.333, 32.467, 33.967,
        #     37.7, 38.7, 45.267, 46.734, 61.4, 66.0,
        #     71.0, 72.633, 77.734, 79.734, 85.2, 87.034,
        #     89.4, 91.367, 93.433, 93.767, 99.367
        # ]  # En secondes

        for gt_fade in ground_truth_fades:
            plt.axvline(gt_fade, color='green', linestyle='--', label="Ground Truth - Fondu" if 'Ground Truth - Fondu' not in plt.gca().get_legend_handles_labels()[1] else "")

        for gt_cut in ground_truth_cuts:
            plt.axvline(gt_cut, color='purple', linestyle='--', label="Ground Truth - Coupure" if 'Ground Truth - Coupure' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.xlabel('Temps (secondes)')
        plt.ylabel('Distance')
        plt.title('Distance entre les histogrammes de trames consécutives')
        plt.legend()
        plt.show()

def main():
    detector = HistogramBasedDetector(threshold_cut=0.53, threshold_effect=0.25, bins=16, Verbose=True)

    detector.process_video("../VideodataTP3/Athletisme.mp4")

    fades, cuts = detector.get_transitions()

    print("\nFondues détectées :")
    for fade in fades:
        print(f"- Fondu détecté entre {fade[0]:.2f} secondes et {fade[1]:.2f} secondes.")

    print("\nCoupures détectées :")
    for cut in cuts:
        print(f"- Coupure détectée à {cut[0]:.2f} secondes.")

    detector.plot_distances()

if __name__ == "__main__":
    main()