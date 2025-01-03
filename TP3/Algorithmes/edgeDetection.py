import numpy as np
import cv2
import matplotlib.pyplot as plt

# Code inspiré de : https://github.com/abhilas0/edge_detection/blob/master/edge_detection.ipynb
# et https://github.com/gabilodeau/INF8770/blob/master/Gradients%20et%20extraction%20d'aretes%20sur%20une%20image.ipynb
# Utilisation de GitHub Copilot pour contribuer à l'écriture du code par l'auto-complétion et la suggestion de code de lecture de vidéo


class EdgeDetector:
    def __init__(self, Verbose=False):
        self.transitions = []
        self.rho_in = []
        self.rho_out = []
        self.rho_max = []
        self.threshold = []
        self.Verbose = Verbose

    def reset(self):
        self.transitions = []
        self.rho_in = []
        self.rho_out = []
        self.rho_max = []
        self.threshold = []

    def compute_gradients(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
        gradient_orientation = np.arctan2(Gy, Gx)
        return Gx, Gy, gradient_magnitude, gradient_orientation

    def dilate_edges(self, edges, radius=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        dilated_edges = cv2.dilate(edges.astype(np.uint8), kernel)
        return dilated_edges

    def detect_transitions(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erreur : Impossible de lire la vidéo '{video_path}'")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        prev_edges = None
        prev_dilated_edges = None
        frame_count = 0
        cut_count = 0
        fade_count = 0
        last_transition_frame = -10
        window_size = 5
        recent_rho = []
        fade_start_frame = None
        possible_fade_frame_count = 0
        r = 2
        BASE_THRESHOLD = 0.75

        if self.Verbose:
            print("Début du traitement de la vidéo...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if self.Verbose:
                    print("Fin de la vidéo.")
                break

            frame = cv2.resize(frame, (640, 360))
            _, _, gradient_magnitude, _ = self.compute_gradients(frame)
            edges = gradient_magnitude > 220
            dilated_edges = self.dilate_edges(edges)

            if prev_edges is not None and prev_dilated_edges is not None:
                rho_in = 1 - np.sum(prev_dilated_edges * edges) / np.sum(edges)
                rho_out = 1 - np.sum(prev_edges * dilated_edges) / np.sum(prev_edges)
                self.rho_in.append(rho_in)
                self.rho_out.append(rho_out)
                rho = max(rho_in, rho_out)
                self.rho_max.append(rho)

                if len(self.rho_max) > 1:
                    mu = np.mean(self.rho_max)
                    sigma = np.std(self.rho_max)
                    threshold = mu + r * sigma
                else:
                    threshold = BASE_THRESHOLD

                self.threshold.append(threshold)

                if (frame_count - last_transition_frame) > 5:
                    if rho > threshold and abs(rho - np.average(recent_rho)) > 0.3:
                        transition_type = "Coupure"
                        self.transitions.append((frame_count, frame_count, transition_type, frame_count / fps, frame_count / fps))
                        if self.Verbose:
                            print(f"{transition_type} à la frame {frame_count} ({frame_count / fps:.2f} s)")
                        last_transition_frame = frame_count
                        cut_count += 1
                        fade_start_frame = None

                if fade_start_frame is None and (rho_in > rho_out) and (rho_in - rho_out > 0.025) and (rho < threshold):
                    possible_fade_frame_count += 1
                    if possible_fade_frame_count >= 4:
                        fade_start_frame = frame_count - 2
                        possible_fade_frame_count = 0
                elif fade_start_frame is not None and (rho_out > rho_in or (frame_count - fade_start_frame) > 30):
                    transition_type = "Fondu"
                    self.transitions.append((fade_start_frame, frame_count, transition_type, fade_start_frame / fps, frame_count / fps))
                    if self.Verbose:
                        print(f"{transition_type} de la frame {fade_start_frame} à la frame {frame_count} ({fade_start_frame / fps:.2f} s - {frame_count / fps:.2f} s)")
                    fade_count += 1
                    fade_start_frame = None
                    last_transition_frame = frame_count
                else:
                    possible_fade_frame_count = 0

                recent_rho.append(rho)
                if len(recent_rho) > window_size:
                    recent_rho.pop(0)

            prev_edges = edges
            prev_dilated_edges = dilated_edges
            frame_count += 1

        if self.Verbose:
            print(f"Nombre total de coupures détectées : {cut_count}")
            print(f"Nombre total de fondus détectés : {fade_count}")
            self.plot_edge_change_fraction(fps)
        cap.release()

    def get_transitions(self):
        fades = [(t[3], t[4]) for t in self.transitions if t[2] == "Fondu"]
        cuts = [(t[3], t[4]) for t in self.transitions if t[2] == "Coupure"]
        return fades, cuts

    def plot_edge_change_fraction(self, fps):
        timestamps = [frame / fps for frame in range(len(self.rho_in))]
        plt.figure(figsize=(12, 6))

        plt.plot(timestamps, self.rho_in, 'x', label="Rho In", color="blue")
        plt.plot(timestamps, self.rho_out, 'o', label="Rho Out", color="red")
        #plt.plot(timestamps, self.rho_max, 's', label="Rho Max", color="green")

        plt.title("Évolution des fractions de changement d'arêtes (Rho) entre trames successives")
        plt.xlabel("Temps (secondes)")
        plt.ylabel("Fraction de changement d'arêtes (Rho)")

        plt.plot(timestamps, self.threshold, 'k--', label="Seuil de détection")

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

        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    video_path = "../VideodataTP3/Athletisme.mp4"
    edge_detector = EdgeDetector(Verbose=True)
    edge_detector.detect_transitions(video_path)

    # Récupération des fondues et coupures
    fades, cuts = edge_detector.get_transitions()

    print("\nFondues détectées :")
    for fade in fades:
        print(f"- Fondu détecté entre {fade[0]:.2f} secondes et {fade[1]:.2f} secondes.")

    print("\nCoupures détectées :")
    for cut in cuts:
        print(f"- Coupure détectée à {cut[0]:.2f} secondes.")

if __name__ == "__main__":
    main()