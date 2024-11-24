import numpy as np
import cv2
import matplotlib.pyplot as plt

class EdgeDetector:
    def __init__(self):
        self.transitions = []
        self.mean_diffs = []

    def compute_gradients(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calcul des gradients avec les filtres Sobel d'OpenCV
        Gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
        return Gx, Gy, gradient_magnitude

    def detect_transitions(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Erreur : Impossible de lire la vidéo '{video_path}'")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        prev_gradient_magnitude = None
        frame_count = 0
        last_transition_frame = -10  # Pour éviter des détections consécutives trop proches

        print("Début du traitement de la vidéo...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fin de la vidéo.")
                break

            # Réduction de la taille de la frame (optionnel, pour accélérer)
            frame = cv2.resize(frame, (640, 360))

            _, _, gradient_magnitude = self.compute_gradients(frame)

            if prev_gradient_magnitude is not None:
                diff = np.abs(gradient_magnitude - prev_gradient_magnitude)
                mean_diff = np.mean(diff)
                self.mean_diffs.append(mean_diff)

                if mean_diff > 65 and (frame_count - last_transition_frame) > 10:
                    self.transitions.append((frame_count, "Coupure", frame_count / fps))
                    print(f"Transition détectée : Coupure à la frame {frame_count} ({frame_count / fps:.2f} s)")
                    last_transition_frame = frame_count
                elif mean_diff > 45 and (frame_count - last_transition_frame) > 10:
                    self.transitions.append((frame_count, "Fondu", frame_count / fps))
                    print(f"Transition détectée : Fondu à la frame {frame_count} ({frame_count / fps:.2f} s)")
                    last_transition_frame = frame_count

            if frame_count % 50 == 0:
                print(f"Frame {frame_count} traitée...")

            prev_gradient_magnitude = gradient_magnitude
            frame_count += 1

        cap.release()

    def report_transitions(self):
        print("\nRésumé des transitions détectées :")
        for frame, transition_type, timestamp in self.transitions:
            print(f"- Frame {frame}: {transition_type} à {timestamp:.2f} secondes")

        print(f"\nTotal des transitions détectées : {len(self.transitions)}")

    def visualize_edges(self, gradient_magnitude, threshold=80):
        edges = gradient_magnitude > threshold
        plt.figure(figsize=(10, 10))
        plt.imshow(edges, cmap="binary")
        plt.axis("off")
        plt.show()

    def plot_mean_diffs(self, fps):
        timestamps = [frame / fps for frame in range(len(self.mean_diffs))]

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, self.mean_diffs, label="Mean Diff par frame", color="blue")
        plt.title("Évolution des Mean Diff entre frames successives")
        plt.xlabel("Temps (secondes)")
        plt.ylabel("Mean Difference")
        plt.axhline(50, color="orange", linestyle="--", label="Seuil pour Fondus")
        plt.axhline(70, color="red", linestyle="--", label="Seuil pour Coupures")
        
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
    video_path = "../VideodataTP3/Athletisme.mp4"
    edge_detector = EdgeDetector()
    
    edge_detector.detect_transitions(video_path)

    edge_detector.report_transitions()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    edge_detector.plot_mean_diffs(fps)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        _, _, gradient_magnitude = edge_detector.compute_gradients(frame)
        edge_detector.visualize_edges(gradient_magnitude, threshold=80)
    cap.release()


if __name__ == "__main__":
    main()