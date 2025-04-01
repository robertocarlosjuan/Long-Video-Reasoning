import os
import cv2
import torch
import numpy as np

class FrameSampler:
    def __init__(self, config):
        self.config = config
        self.temp_frames_dir = self.config.temp_frames_dir
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

    def get_free_memory(self, device=0):
        torch.cuda.empty_cache()  # Optional: frees unused cached memory
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        free_inside_reserved = reserved - allocated
        total_memory = torch.cuda.get_device_properties(device).total_memory
        free_memory = total_memory - reserved + free_inside_reserved
        return free_memory
    
    def estimate_image_memory(self, height, width, num_frames=1, channels=3, dtype_size=4):
        return num_frames * channels * height * width * dtype_size  # bytes
    
    def bytes_to_MB(self, x): return x / (1024 ** 2)

    def find_max_image_dim(self, device=0, safety_factor=0.8, max_dim=2048, num_frames=1):
        free_bytes = self.get_free_memory(device)
        target_bytes = free_bytes * safety_factor
        dtype_size = 4  # float32

        for dim in range(64, max_dim, 16):
            est_mem = self.estimate_image_memory(dim, dim, num_frames=num_frames, dtype_size=dtype_size)
            if est_mem > target_bytes:
                return dim - 16  # Previous dimension was the last safe one
        return max_dim

    def sample(self, video_path):
        self._cleanup_temp_frames()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sampling_strategy = self.config.sampling_strategy

        if sampling_strategy == "uniform":
            indices = np.linspace(0, total_frames - 1, self.config.num_frames).astype(int)
        elif sampling_strategy == "dense":
            step = max(1, int(fps / self.config.sampling_rate))
            indices = np.arange(0, total_frames, step)
        else:
            raise ValueError("Unsupported sampling strategy")
        
        actual_max = self.find_max_image_dim(num_frames=len(indices))
        max_edge_len = self.config.max_edge_len 
        if max_edge_len == -1:
            max_edge_len = actual_max
        if max_edge_len > actual_max:
            return None
        
        frames_with_info = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            height, width = frame.shape[:2]
            max_dim = max(height, width)
            if max_dim > max_edge_len:
                scale = max_edge_len / max_dim
                width = int(width * scale)
                height = int(height * scale)
                frame = cv2.resize(frame, (width, height))
            timestamp = int(idx / fps)
            frame_path = os.path.join(self.temp_frames_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(frame_path, frame)
            frames_with_info.append({
                "frame": frame,
                "frame_index": idx,
                "timestamp_sec": timestamp,
                "label": f"{int(timestamp)}s",
                "frame_path": frame_path,
                "frame_width": width,
                "frame_height": height,
                "sampling_strategy": sampling_strategy
            })
            

        cap.release()
        return frames_with_info