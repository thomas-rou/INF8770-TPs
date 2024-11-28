import numpy as np
import cv2
import matplotlib.pyplot as plt

class EdgeDetector:
    def __init__(self):
        self.transitions = []
        self.rho_in = []
        self.rho_out = []
        self.rho_max = []

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
        last_transition_frame = -10
        cut_count = 0
        window_size = 5
        recent_rho = []

        print("Début du traitement de la vidéo...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fin de la vidéo.")
                break

            frame = cv2.resize(frame, (640, 360))
            _, _, gradient_magnitude, gradient_orientation = self.compute_gradients(frame)
            edges = gradient_magnitude > 220
            dilated_edges = self.dilate_edges(edges)

            if prev_edges is not None and prev_dilated_edges is not None:
                rho_in = 1 - np.sum(prev_dilated_edges * edges) / np.sum(edges)
                rho_out = 1 - np.sum(prev_edges * dilated_edges) / np.sum(prev_edges)
                self.rho_in.append(rho_in)
                self.rho_out.append(rho_out)
                rho = max(rho_in, rho_out)
                self.rho_max.append(rho)

                if rho > 0.5 and (frame_count - last_transition_frame) > 5:
                    if rho > 0.75 and abs(rho - np.average(recent_rho)) > 0.3:
                        transition_type = "Coupure"
                        self.transitions.append((frame_count, transition_type, frame_count / fps))
                        print(f"Transition détectée : {transition_type} à la frame {frame_count} ({frame_count / fps:.2f} s)")
                        last_transition_frame = frame_count
                        cut_count += 1

                recent_rho.append(rho)
                if len(recent_rho) > window_size:
                    recent_rho.pop(0)

            if frame_count % 50 == 0:
                #print(f"Frame {frame_count} traitée...")
                pass

            prev_edges = edges
            prev_dilated_edges = dilated_edges
            frame_count += 1

        print(f"Nombre total de coupures détectées : {cut_count}")
        cap.release()

    def report_transitions(self):
        # print("\nRésumé des transitions détectées :")
        # for frame, transition_type, timestamp in self.transitions:
        #     print(f"- Frame {frame}: {transition_type} à {timestamp:.2f} secondes")
        # print(f"\nTotal des transitions détectées : {len(self.transitions)}")
        return

    def visualize_edges(self, gradient_magnitude, threshold=80):
        edges = gradient_magnitude > threshold
        plt.figure(figsize=(10, 10))
        plt.imshow(edges, cmap="binary")
        plt.axis("off")
        plt.show()

    def plot_edge_change_fraction(self, fps):
        timestamps = [frame / fps for frame in range(len(self.rho_in))]
        plt.figure(figsize=(12, 6))

        plt.plot(timestamps, self.rho_in, 'x', label="Rho In", color="blue")
        plt.plot(timestamps, self.rho_out, 'o', label="Rho Out", color="red")
        plt.plot(timestamps, self.rho_max, 's', label="Rho Max", color="green")

        plt.title("Évolution des fractions de changement d'arêtes (Rho) entre frames successives")
        plt.xlabel("Temps (secondes)")
        plt.ylabel("Fraction de changement d'arêtes (Rho)")

        plt.axhline(0.5, color="black", linestyle="--", label="Seuil de détection")
        plt.axhline(0.75, color="black", linestyle="--", label="Seuil de coupure")

        ground_truth = {
            "Fondu": [12, 41],
            "Coupure": [17, 24, 32]
        }

        for gt_type, times in ground_truth.items():
            for t in times:
                plt.axvline(t, color="green" if gt_type == "Fondu" else "purple", linestyle="--", label=f"{gt_type} Ground Truth" if t == times[0] else "")

        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    video_path = "../VideodataTP3/Soccer.mp4"
    edge_detector = EdgeDetector()
    edge_detector.detect_transitions(video_path)
    edge_detector.report_transitions()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    edge_detector.plot_edge_change_fraction(fps)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        _, _, gradient_magnitude, _ = edge_detector.compute_gradients(frame)
        edge_detector.visualize_edges(gradient_magnitude, threshold=200)
    cap.release()

if __name__ == "__main__":
    main()