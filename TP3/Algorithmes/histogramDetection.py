import cv2
import numpy as np

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

    def reset(self):
        """
        Réinitialise les données internes pour une nouvelle exécution.
        """
        self.previous_histogram = None
        self.last_transition_frame = -self.min_frame_gap
        self.transitions = []

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

        distance = self.calculate_distance(self.previous_histogram, current_histogram)

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

            if transition:
                timestamp = frame_count / fps
                self.transitions.append((frame_count, transition, timestamp))
                if self.Verbose:
                    print(f"Transition détectée ({transition}) à {timestamp:.2f} secondes.")

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

if __name__ == "__main__":
    main()