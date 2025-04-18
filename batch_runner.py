import os
import tempfile
import json
import imageio
import numpy as np
import cv2
import re
import csv
import time
from collections import defaultdict
from prompt_builder import PromptBuilder
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import select_best_by_edit_distance, check_segmentation_validity

class BatchRunner:
    def __init__(self, config, dataset_loader, inference_engine):
        self.config = config
        self.dataset_loader = dataset_loader
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
        with open(self.config.output_path, "w") as f:
            json.dump(self.results, f, indent=2)

    def _extract_segments(self, text):
        if self.config.prompt_style == "coarse_cot":
            lines = text.splitlines()
            # Pattern to find lines like "1. 0s - 128s:"
            segment_pattern = re.compile(r'^\s*\d+\.\s*([\d]+)s\s*-\s*([\d]+)s:')
            
            relevant_segments = []
            num_segments_listed = 0
            current_range = None
            
            for line in lines:
                # Check if the line indicates a new segment range, e.g., "1. 0s - 128s:"
                match = segment_pattern.match(line)
                if match:
                    start = int(match.group(1))
                    end = int(match.group(2))
                    current_range = (start, end)
                
                # Check if the line says "Verdict: Relevant"
                if "Verdict:" in line:
                    # Trim and compare
                    verdict_str = line.split("Verdict:")[-1].strip().replace('[', '').replace(']', '').lower()
                    if verdict_str == "relevant" and current_range:
                        relevant_segments.append(current_range)
                        current_range = None
                    else:
                        current_range = None
                    num_segments_listed += 1
            
            return relevant_segments, num_segments_listed
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

    def get_frame_labels(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampling_strategy = self.config.sampling_strategy
        indices = []
        if sampling_strategy == "uniform":
            indices = np.linspace(0, total_frames - 1, self.config.num_frames).astype(int)
        elif sampling_strategy == "dense":
            step = max(1, int(1 / self.config.sampling_rate))
            indices = np.arange(0, total_frames, step)
        else:
            raise ValueError("Unsupported sampling strategy")
        return indices

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

    def run(self):
        instances = self.dataset_loader.load_instances()
        video_dir = os.path.join(self.config.dataset_path, "v2", "nlq_videos", "full_scale")
        video_uids = list(set([instance["video_uid"] for instance in instances]))
        video_uids.reverse()
        if os.path.exists(self.config.output_path):
            with open(self.config.output_path, "r") as f:
                self.results = json.load(f)
                processed_ids = [x["id"] for x in self.results]
        else:
            processed_ids = []

        if not os.path.exists(self.config.temporal_selection_json):
            for ctr, instance in enumerate(tqdm(instances)):
                if instance["id"] in processed_ids:
                    continue
                loading_time = time.time()
                video_uid = instance["video_uid"]
                video_path = os.path.join(video_dir, video_uid + ".mp4")
                frame_labels = self.get_frame_labels(video_path)
                sampling_strategy = self.config.sampling_strategy
                query = instance["query"]
                turns = self.prompt_builder.num_turns()
                prior_response = None
                conversation = []
                selected_frames_list = []
                segments = []
                is_correct = False
                for turn_index in range(turns-1):
                    cache_key = f"{video_uid}:{turn_index}"
                    should_use_cache = self.prompt_builder.uses_query(turn_index) is False

                    stage, num_inference_attempts, message = self.prompt_builder.build(
                        query=query,
                        video_path=video_path,
                        timestamps=frame_labels,
                        segments=segments,
                        prior_response=prior_response,
                        turn_index=turn_index
                    )

                    if ctr == 0:
                        print(stage)
                        print(message)

                    if should_use_cache and cache_key in self.cache:
                        responses = self.cache[cache_key]
                    else:
                        responses = self.inference_engine.run(message, num_inference_attempts)
                        if should_use_cache:
                            self.cache[cache_key] = responses

                        conversation.append({
                                "prompt": message[0]["content"][-1]["text"],
                                "responses": responses
                            })
                        if stage == 'segmentation':
                            required_segments = [f"{frame_labels[i]} - {frame_labels[i+1]}" for i in range(len(frame_labels)-1)]
                            candidate_prior = []
                            # Ensure all segments are present
                            for r in responses:
                                missing_segments = [x for x in required_segments if x not in r]
                                if len(missing_segments)>0:
                                    continue
                                elif not check_segmentation_validity(r, prior_response):
                                    continue
                                else:
                                    candidate_prior.append(r)
                            # Select the best response by consensus on edit distance
                            if len(candidate_prior) > 0:
                                prior_response = select_best_by_edit_distance(candidate_prior)
                            else:
                                prior_response = responses[0]
                                for response in responses[1:]:
                                    if len(response) > len(prior_response):
                                        prior_response = response
                            conversation[-1]['chosen_response'] = prior_response
                        elif stage == 'segment_selection':
                            correct_list = []
                            num_pred_relevant = []
                            num_segments_listed_list = []
                            for r in responses:
                                segments, num_segments_listed = self._extract_segments(r)
                                is_correct = self._evaluate(
                                    segments,
                                    clip_start=instance.get("clip_start_sec", 0),
                                    clip_end=instance.get("clip_end_sec", 0),
                                )
                                correct_list.append(is_correct)
                                num_pred_relevant.append(len(segments))
                                num_segments_listed_list.append(num_segments_listed)
                            conversation[-1]['correct_relevant_segments'] = list(zip(correct_list, num_pred_relevant, num_segments_listed_list))
                            best = 1
                            for i in range(len(correct_list)):
                                if correct_list[i]:
                                    if num_segments_listed_list[i] == self.config.num_frames-1 and num_pred_relevant[i]/num_segments_listed_list[i] < best:
                                        best = num_pred_relevant[i]/num_segments_listed_list[i]
                                        prior_response = responses[i]
                            if best == 1:
                                prior_response = None
                        else:
                            prior_response = responses[0]

                result = {
                    "id": instance["id"],
                    "video_uid": video_uid,
                    "clip_uid": instance["clip_uid"],
                    "query": query,
                    "duration_sec": instance.get("duration_sec"),
                    "clip_start_sec": instance.get("clip_start_sec"),
                    "clip_end_sec": instance.get("clip_end_sec"),
                    "video_start_frame": instance.get("video_start_frame"),
                    "video_end_frame": instance.get("video_end_frame"),
                    "sampling_strategy": sampling_strategy,
                    "template": instance.get("template"),
                    "conversation": conversation
                }
                self.results.append(result)

                if ctr < 10 or ctr % 10 == 0:
                    self._save_results()
                    if not self.config.ignore_cache:
                        self._save_cache()
        else:
            with open(self.config.temporal_selection_json, "r") as f:
                temporal_selection = json.load(f)
            turns = self.prompt_builder.num_turns()
            for ctr, instance in enumerate(tqdm(temporal_selection)):
                video_uid = instance["video_uid"]
                query = instance["query"]
                video_path = os.path.join(self.config.dataset_path, "v2", "nlq_videos", "full_scale", video_uid + ".mp4")
                frame_labels = self.get_frame_labels(video_path)
                segments = self._extract_segments(instance['response'])
                responses = []
                for selected_frames in selected_frames_list:
                    stage, num_inference_attempts, message = self.prompt_builder.build(
                        query=query,
                        video_path=video_path,
                        timestamps=frame_labels,
                        segments=segments,
                        prior_response=instance["response"],
                        turn_index=turns-1,
                    )
                    response = self.inference_engine.run(message, num_inference_attempts)[0]
                    responses.append(f"####### Segment {selected_frames[0]} - {selected_frames[-1]} #######\n{response}")
                instance["answer"] = '\n\n'.join(responses)

                self.results.append(instance)
                if ctr % 10 == 0:
                    self._save_results()
                    if not self.config.ignore_cache:
                        self._save_cache()

        if not self.config.ignore_cache:
            self._save_cache()

        self._save_results()
        self._report_metrics()