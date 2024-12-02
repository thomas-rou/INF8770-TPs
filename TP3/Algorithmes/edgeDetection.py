import numpy as np
import cv2
import matplotlib.pyplot as plt

class EdgeDetector:
    def __init__(self, Verbose=False):
        self.transitions = []
        self.rho_in = []
        self.rho_out = []
        self.rho_max = []
        self.threshold = []
        self.Verbose = Verbose
        
    def reset(self):
        """
        Réinitialise les données internes pour une nouvelle exécution.
        """
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
        cap.release()

    def get_transitions(self):
        """
        Retourne deux listes distinctes :
        - Les fondues détectées (avec début et fin).
        - Les coupures détectées.
        """
        fades = [(t[3], t[4]) for t in self.transitions if t[2] == "Fondu"]
        cuts = [(t[3], t[4]) for t in self.transitions if t[2] == "Coupure"]
        return fades, cuts
    
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