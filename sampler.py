import os
import cv2
import numpy as np

class FrameSampler:
    def __init__(self, config):
        self.config = config
        self.temp_frames_dir = "tmp"
        os.makedirs(self.temp_frames_dir, exist_ok=True)

    def _cleanup_temp_frames(self):
        """Remove all contents of the temporary frames directory."""
        if os.path.exists(self.temp_frames_dir):
            for filename in os.listdir(self.temp_frames_dir):
                file_path = os.path.join(self.temp_frames_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    def sample(self, video_path, duration):
        self._cleanup_temp_frames()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.config.sampling_strategy == "uniform":
            indices = np.linspace(0, total_frames - 1, self.config.num_frames).astype(int)
        elif self.config.sampling_strategy == "dense":
            step = max(1, int(fps / self.config.sampling_rate))
            indices = np.arange(0, total_frames, step)
        else:
            raise ValueError("Unsupported sampling strategy")

        frames_with_info = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            max_edge_len = self.config.max_edge_len
            height, width = frame.shape[:2]
            max_dim = max(height, width)
            if max_dim > max_edge_len:
                scale = max_edge_len / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            timestamp = idx / fps
            frame_path = os.path.join(self.temp_frames_dir, f"frame_{idx}.jpg")
            cv2.imwrite(frame_path, frame)
            frames_with_info.append({
                "frame": frame,
                "frame_index": idx,
                "timestamp_sec": timestamp,
                "label": f"{int(timestamp)}s",
                "frame_path": frame_path
            })
            

        cap.release()
        return frames_with_info