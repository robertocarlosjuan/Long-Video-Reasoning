import sys
sys.path.append('/nethome/che321/flash/LVTR/shot_detection')
import os
import tempfile
import json
import imageio
import re
import csv
import time
import cv2
import shutil
from pathlib import Path
from collections import defaultdict
from prompt_builder import PromptBuilder
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import select_best_by_edit_distance, check_segmentation_validity
from shot_detection.shot_detector import img_main

class RecursiveBatchRunner:
    def __init__(self, config, dataset_loader, sampler, inference_engine):
        self.config = config
        Path(self.config.temp_frames_dir).mkdir(exist_ok=True, parents=True)
        self.dataset_loader = dataset_loader
        self.sampler = sampler
        self.inference_engine = inference_engine
        self.prompt_builder = PromptBuilder(config)
        self.cache = self._load_cache() if not config.ignore_cache else {}
        self.results = []

    def _load_cache(self):
        if os.path.exists(self.config.cache_path):
            with open(self.config.cache_path, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.config.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _save_results(self):
        print("Saving results to ", self.config.output_path)
        print(self.results)
        with open(self.config.output_path, "w") as f:
            json.dump(self.results, f, indent=2)

    def _extract_segments(self, text):
        if self.config.prompt_style in ["coarse_cot", "coarse_cot_shot_detection"]:
            lines = text.splitlines()
            # Pattern to find lines like "1. 0s - 128s:"
            segment_pattern = re.compile(r'^\s*\d+\.\s*([\d]+)s\s*-\s*([\d]+)s:')
            
            relevant_segments = []
            confidence_scores = []
            num_segments_listed = 0
            current_range = None
            
            for line in lines:
                # Check if the line indicates a new segment range, e.g., "1. 0s - 128s:"
                match = segment_pattern.match(line)
                if match:
                    start = int(match.group(1))
                    end = int(match.group(2))
                    current_range = (start, end)

                if "Confidence Score:" in line:
                    confidence_score = line.split("Confidence Score:")[-1].strip().replace('[', '').replace(']', '').lower()
                    confidence_score = float(confidence_score) if confidence_score.isdigit() else 0.0
                
                # Check if the line says "Verdict: Relevant"
                if "Verdict:" in line:
                    # Trim and compare
                    verdict_str = line.split("Verdict:")[-1].strip().replace('[', '').replace(']', '').lower()
                    if verdict_str == "relevant" and current_range:
                        relevant_segments.append(current_range)
                        confidence_scores.append(confidence_score)
                        current_range = None
                    else:
                        current_range = None
                    num_segments_listed += 1
            
            return relevant_segments, confidence_scores, num_segments_listed
        else:
            segments_str = re.findall(r"\[(\d+(?:\.\d+)?)(?:s)?\]\s*-\s*\[(\d+(?:\.\d+)?)(?:s)?\]", text)
            segments = []
            for start_str, end_str in segments_str:
                try:
                    start = int(float(start_str))
                    end = int(float(end_str))
                    segments.append((start, end))
                except ValueError:
                    pass
            return segments, None
    
    def _evaluate(self, segments, clip_start, clip_end):
        correct = False
        for start, end in segments:
            if clip_start >= start and clip_end <= end:
                correct = True
        return correct
    
    def _select_frames(self, segments, frames_dir):
        selected_frames_list = []
        for start, end in segments:
            segment = []
            for i in range(start, end):
                segment.append(os.path.join(frames_dir, f'frame_{i}.jpg'))
            selected_frames_list.append(segment)
        if len(selected_frames_list) == 0:
            return [[os.path.join(frames_dir, x) for x in os.listdir(frames_dir)]]
        return selected_frames_list

    def _report_metrics(self):
        total = len(self.results)
        correct = sum(1 for r in self.results if r["correct_segment"])
        accuracy = correct / total if total > 0 else 0.0
        print("\n===== EVALUATION =====")
        print(f"Total Queries: {total}")
        print(f"Correct Segment Identified: {correct}")
        print(f"Accuracy: {accuracy:.3f}")

        video_map = defaultdict(list)
        query_map = defaultdict(list)
        for r in self.results:
            video_map[r["video_uid"]].append(r["correct_segment"])
            query_key = r["query"].strip().lower().split(" ")[0]  # group by first word as a simple proxy
            query_map[query_key].append(r["correct_segment"])

        print("\n===== PER-VIDEO BREAKDOWN =====")
        for video, flags in video_map.items():
            acc = sum(flags) / len(flags)
            print(f"{video}: {sum(flags)}/{len(flags)} correct ({acc:.2f})")

        print("\n===== PER-QUERY CATEGORY BREAKDOWN =====")
        for qword, flags in sorted(query_map.items()):
            acc = sum(flags) / len(flags)
            print(f"{qword}: {sum(flags)}/{len(flags)} correct ({acc:.2f})")

        self._plot_metrics(query_map)

    def _plot_metrics(self, query_map):
        categories = sorted(query_map.keys())
        accuracies = [sum(flags) / len(flags) for flags in [query_map[c] for c in categories]]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(categories, accuracies)
        plt.ylabel("Accuracy")
        plt.xlabel("Query Category (First Word)")
        plt.title("Accuracy by Query Category")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{acc:.2f}", ha="center", va="bottom")
        plt.savefig(self.config.output_plot_path)
        print(f"Saved accuracy plot to {self.config.output_plot_path}")

    def get_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return num_frames, fps

    def generate_shots(self, video_path, temp_dir, max_segments=20, sample_rate=30, segment = []):
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        Path(temp_dir).mkdir(exist_ok=True, parents=True)
        if len(segment) == 0:
            img_main(video_path, temp_dir, max_segments=max_segments, sample_rate=float(sample_rate), start_sec = 0, max_edge_len = self.config.max_edge_len, target_seconds=self.config.target_seconds)
        else:
            cropped_video_path = os.path.join(temp_dir, f"{video_path}_cropped.mp4")
            self.crop_and_save_video(video_path, cropped_video_path, segment)
            img_main(cropped_video_path, temp_dir, max_segments=max_segments, sample_rate=float(sample_rate), start_sec = segment[0], max_edge_len = self.config.max_edge_len, target_seconds=self.config.target_seconds)

    def crop_and_save_video(self, video_path, output_path, segment):
        """
        Crops a video from start_sec to end_sec and saves it to output_path.

        Parameters:
            input_path (str): Path to the input video file.
            output_path (str): Path to save the cropped video.
            start_sec (float): Start time in seconds.
            end_sec (float): End time in seconds.
        """
        start_sec, end_sec = segment
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        # Clamp frames to video length
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()


    def run(self):
        print("Running recursive batch runner")
        max_segments = 20
        instances = self.dataset_loader.load_instances()
        video_dir = os.path.join(self.config.dataset_path, "v2", "nlq_videos", "full_scale")
        processed_ids = []
        if os.path.exists(self.config.output_path):
            with open(self.config.output_path, "r") as f:
                self.results = json.load(f)
                processed_ids = [x["id"] for x in self.results]
            print("Processed ids: ", processed_ids)

        for ctr, instance in enumerate(tqdm(instances)):
            if instance["id"] in processed_ids:
                continue
            video_uid = instance["video_uid"]
            video_path = os.path.join(video_dir, video_uid+'.mp4')
            num_frames, fps = self.get_video_info(video_path)
            full_segment = [0, num_frames/fps]
            conversation = [{'stage': 'segment_selection', 'selected_segments': [full_segment]}]
            queue = [[0, full_segment, conversation, 'segment_selection']]
            query = instance["query"]
            video_temp_dir = os.path.join(self.config.temp_frames_dir, video_uid)
            os.makedirs(video_temp_dir, exist_ok=True)
            all_conversations = []
            clip_start = instance.get("clip_start_sec", 0)
            clip_end = instance.get("clip_end_sec", 0)
            # Ground truth answer
            cropped_video_path = os.path.join(video_temp_dir, f"{video_uid}_ground_truth.mp4")
            print("Cropping video for ground truth", cropped_video_path, [int(clip_start), int(clip_end)+1])
            self.crop_and_save_video(video_path, cropped_video_path, [clip_start, clip_end])
            print("saved ground truth video")
            stage, num_inference_attempts, message = self.prompt_builder.build(
                query=query,
                video_path=cropped_video_path,
                turn_index=1
            )
            ground_truth_responses = self.inference_engine.run(message, num_inference_attempts)
            while len(queue) > 0:
                queue.sort(key=lambda x: x[0])
                priority, segment, conversation, next_stage = queue.pop(0)
                print("QUEUE", priority, segment, next_stage)
                if next_stage == 'segment_selection':
                    print("Generating shots for ", video_uid)
                    if segment != full_segment:
                        self.generate_shots(video_path, video_temp_dir, max_segments=max_segments, sample_rate=fps, segment = segment)
                    else:
                        self.generate_shots(video_path, video_temp_dir, max_segments=max_segments, sample_rate=fps)
                    frames_info = self.sampler.sample(video_uid)
                    if frames_info is None:
                        print(f"Skipping {video_uid} due to possible memory issue")
                        continue
                    frame_labels = [f["label"] for f in frames_info]
                    frames_paths = [f["frame_path"] for f in frames_info]
                    if not frames_info:
                        print(f"Skipping {video_uid} — no frames sampled")
                        continue
                    stage, num_inference_attempts, message = self.prompt_builder.build(
                        query=query,
                        frames_paths=frames_paths,
                        timestamps=frame_labels,
                        prior_response=None,
                        turn_index=0,
                        selected_frames_list=[]
                    )
                    print("Running inference for ", video_uid)
                    responses = self.inference_engine.run(message, num_inference_attempts)
                    segments = []
                    is_correct = False
                    min_relevant_segments = float('inf')
                    selected_segments = []
                    selected_confidence_scores = []
                    selected_response = None
                    for r in responses:
                        segments, confidence_scores, num_segments_listed = self._extract_segments(r)
                        is_correct = self._evaluate(
                            segments,
                            clip_start=clip_start,
                            clip_end=clip_end,
                        )
                        if is_correct and (len(segments) < min_relevant_segments):
                            min_relevant_segments = len(segments)
                            selected_segments = segments
                            selected_response = r
                            selected_confidence_scores = confidence_scores

                    conversation.append({
                        "stage": "segment_selection",
                        "prompt": message[0]["content"][-1]["text"],
                        "responses": responses,
                        "selected_response": selected_response,
                        "selected_segments": selected_segments,
                        "selected_confidence_scores": selected_confidence_scores,
                    })
                    # recurse only on correct segments
                    for s in selected_segments:
                        next_stage = "segment_selection" if abs(s[1]-s[0]) > self.config.target_seconds else "answer"
                        print("CONSIDERING SEGMENT", s)
                        if clip_start >= s[0] and clip_end <= s[1]:
                            print("GROUND TRUTH SEGMENT")
                            queue.append((priority+1, s, conversation, next_stage))
                            break
                        elif (clip_start > s[0] and clip_start < s[1]) and ((s[1]-clip_start) >= (clip_end-s[1])):
                            queue.append((priority+1, [s[0], clip_start], conversation, next_stage))
                            break
                        elif clip_end > s[0] and clip_end < s[1]:
                            queue.append((priority+1, [clip_end, s[1]], conversation, next_stage))
                            break
                    if len(queue) == 0:
                        result = {
                            "id": instance["id"],
                            "video_uid": video_uid,
                            "clip_uid": instance["clip_uid"],
                            "query": query,
                            "duration_sec": instance.get("duration_sec"),
                            "clip_start_sec": clip_start,
                            "clip_end_sec": clip_end,
                            "video_start_frame": instance.get("video_start_frame"),
                            "video_end_frame": instance.get("video_end_frame"),
                            "frame_width": frames_info[0]["frame_width"],
                            "frame_height": frames_info[0]["frame_height"],
                            "sampling_strategy": frames_info[0]["sampling_strategy"],
                            "template": instance.get("template"),
                            "conversation": conversation,
                            "answer": "Segment selection failed",
                            "ground_truth_responses": ground_truth_responses
                            }
                        self.results.append(result)

                        print("###################### GONNA SAVE \n")
                        self._save_results()
                        if not self.config.ignore_cache:
                            self._save_cache()
                    # recurse on all segments with highest confidence scores first
                    # for s, c in zip(selected_segments, selected_confidence_scores):
                    #     # Prioritize segments with higher confidence scores
                    #     new_priority = ((priority//10)*10)+10+(10-c)
                    #     queue.append((new_priority, s, conversation))

                    ####################
                    

                    ####################
                elif next_stage == 'answer':
                    print("ANSWERING ", video_uid)
                    print("Cropping video for ", video_uid)
                    # Answering inference
                    cropped_video_path = os.path.join(video_temp_dir, f"{video_uid}_cropped.mp4")
                    self.crop_and_save_video(video_path, cropped_video_path, segment)
                    stage, num_inference_attempts, message = self.prompt_builder.build(
                        query=query,
                        video_path=cropped_video_path,
                        turn_index=1
                    )
                    responses = self.inference_engine.run(message, num_inference_attempts)
                    conversation.append({
                        "stage": "answer",
                        "prompt": message[0]["content"][-1]["text"],
                        "responses": responses,
                    })
                    all_conversations.append(conversation)

                    

                    result = {
                        "id": instance["id"],
                        "video_uid": video_uid,
                        "clip_uid": instance["clip_uid"],
                        "query": query,
                        "duration_sec": instance.get("duration_sec"),
                        "clip_start_sec": clip_start,
                        "clip_end_sec": clip_end,
                        "video_start_frame": instance.get("video_start_frame"),
                        "video_end_frame": instance.get("video_end_frame"),
                        "frame_width": frames_info[0]["frame_width"],
                        "frame_height": frames_info[0]["frame_height"],
                        "sampling_strategy": frames_info[0]["sampling_strategy"],
                        "template": instance.get("template"),
                        "conversation": all_conversations,
                        "answer": responses[0],
                        "ground_truth_responses": ground_truth_responses
                    }
                    self.results.append(result)

                    print("###################### GONNA SAVE \n")
                    self._save_results()
                    if not self.config.ignore_cache:
                        self._save_cache()
                    break

        if not self.config.ignore_cache:
            self._save_cache()

        self._save_results()
        self._report_metrics()