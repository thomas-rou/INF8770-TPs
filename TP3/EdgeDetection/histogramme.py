import cv2
import numpy as np

class HistogramBasedDetector:
    def __init__(self, threshold_cut=0.5, threshold_effect=0.2, bins=16):
        self.threshold_cut = threshold_cut
        self.threshold_effect = threshold_effect
        self.bins = bins
        self.min_frame_gap = 0
        self.previous_histogram = None
        self.last_transition_frame = -self.min_frame_gap

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
            print(f"Erreur : Impossible de lire la vidéo '{video_path}'")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.min_frame_gap = int(fps)
        self.last_transition_frame = -self.min_frame_gap
        print(f"FPS de la vidéo : {fps:.2f}")
        frame_count = 0
        transitions = []

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
                transitions.append((timestamp, transition))
                print(f"Transition détectée ({transition}) à {timestamp:.2f} secondes.")

            frame_count += 1

        cap.release()
        print("Nombre de transitions détectées :", len(transitions))
        print("Fin du traitement.")
        return transitions
    
def main():
    detector = HistogramBasedDetector(threshold_cut=0.6, threshold_effect=0.3, bins=16)

    detector.process_video("../VideodataTP3/Soccer.mp4")
        
if __name__ == "__main__":
    main()