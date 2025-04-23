import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
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
    
    def generate_frames(self, video_uids, video_dir):
        if self.config.sampling_strategy == "shot_detection":
            video_frames_dir = os.path.join(self.temp_frames_dir, video_uid)
            if os.path.exists(video_frames_dir):
                return video_frames_dir
            raise ValueError(f"Video {video_uid} not found in {video_frames_dir}: Please run shot_detection first")
        print(f"Generating frames for {len(video_uids)} videos")
        for video_uid in tqdm(video_uids):
            video_path = os.path.join(video_dir, video_uid + ".mp4")
            video_frames_dir = os.path.join(self.temp_frames_dir, video_uid)
            if os.path.exists(video_frames_dir):
                continue
            os.makedirs(video_frames_dir)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.arange(0, total_frames, fps)
            for idx in indices:
                frame_path = os.path.join(video_frames_dir, f"frame_{int(idx/fps)}.jpg")
                if os.path.exists(frame_path):
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                try:
                    cv2.imwrite(frame_path, frame)
                except Exception as e:
                    print(f"Error writing frame {idx} to {frame_path}: {e}")
        cap.release()
        return video_frames_dir

    def sample(self, video_uid):
        video_frames_dir = os.path.join(self.temp_frames_dir, video_uid)
        sampling_strategy = self.config.sampling_strategy
        if sampling_strategy in ["uniform", "dense"]:
            video_frames_paths = {int(x.split("frame_")[1].split(".")[0]): os.path.join(video_frames_dir, x) for x in os.listdir(video_frames_dir)}
        else:
            video_frames_paths = {tuple(map(int, x.split(".")[0].split("_")[1:])): os.path.join(video_frames_dir, x) for x in os.listdir(video_frames_dir)}
        total_frames = max(video_frames_paths.keys())


        if sampling_strategy == "uniform":
            indices = np.linspace(0, total_frames - 1, self.config.num_frames).astype(int)
        elif sampling_strategy == "dense":
            step = max(1, int(1 / self.config.sampling_rate))
            indices = np.arange(0, total_frames, step)
        elif sampling_strategy == "shot_detection":
            indices = sorted(video_frames_paths.keys(), key=lambda x: x[0])
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
            frame_path = video_frames_paths[idx]
            frame = cv2.imread(frame_path)
            height, width = frame.shape[:2]
            max_dim = max(height, width)
            if max_dim > max_edge_len:
                scale = max_edge_len / max_dim
                width = int(width * scale)
                height = int(height * scale)
                frame = cv2.resize(frame, (width, height))
            if sampling_strategy in ["uniform", "dense"]:
                label = f"{int(idx)}s"
            elif sampling_strategy == "shot_detection":
                label = f"{int(idx[0])}s - {int(idx[1])}s"
            else:
                raise ValueError("Unsupported sampling strategy")

            frames_with_info.append({
                "frame": frame,
                "frame_index": idx,
                "timestamp_sec": idx,
                "label": label,
                "frame_path": frame_path,
                "frame_width": width,
                "frame_height": height,
                "sampling_strategy": sampling_strategy
            })
        return frames_with_info