import os
import tempfile
import json
import imageio
import re
import csv
import time
from collections import defaultdict
from prompt_builder import PromptBuilder
import matplotlib.pyplot as plt
from tqdm import tqdm
class BatchRunner:
    def __init__(self, config, dataset_loader, sampler, inference_engine):
        self.config = config
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
        with open(self.config.output_path, "w") as f:
            json.dump(self.results, f, indent=2)

    def _extract_segments(self, text):
        segments_str = re.findall(r"\[(\d+(?:\.\d+)?)(?:s)?\]\s*-\s*\[(\d+(?:\.\d+)?)(?:s)?\]", text)
        segments = []
        for start_str, end_str in segments_str:
            try:
                start = int(float(start_str))
                end = int(float(end_str))
                segments.append((start, end))
            except ValueError:
                pass
        return segments
    
    def _evaluate(self, segments, clip_start, clip_end):
        for start, end in segments:
            if clip_start >= start and clip_end <= end:
                return True
        return False
    
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

    def run(self):
        instances = self.dataset_loader.load_instances()

        if not os.path.exists(self.config.temporal_selection_json):
            for ctr, instance in enumerate(tqdm(instances)):
                if ctr < 20:
                    continue
                loading_time = time.time()
                video_uid = instance["video_uid"]
                query = instance["query"]

                video_path = os.path.join(self.config.dataset_path, "v2", "nlq_videos", "full_scale", video_uid + ".mp4")
                frames_info = self.sampler.sample(video_path)
                if frames_info is None:
                    print(f"Skipping {video_uid} due to possible memory issue")
                    continue
                frame_labels = [f["label"] for f in frames_info]
                frames_paths = [f["frame_path"] for f in frames_info]
                if not frames_info:
                    print(f"Skipping {video_uid} — no frames sampled")
                    continue
                frames_dir = os.path.dirname(frames_paths[0])
                frame_width = frames_info[0]["frame_width"]
                frame_height = frames_info[0]["frame_height"]
                sampling_strategy = frames_info[0]["sampling_strategy"]
                turns = self.prompt_builder.num_turns()
                prior_response = None
                conversation = []
                selected_frames_list = []
                segments = []
                is_correct = False
                for turn_index in range(turns-1):
                    cache_key = f"{video_uid}:{turn_index}"
                    should_use_cache = self.prompt_builder.uses_query(turn_index) is False

                    stage, message = self.prompt_builder.build(
                        query=query,
                        frames_paths=frames_paths,
                        timestamps=frame_labels,
                        prior_response=prior_response,
                        turn_index=turn_index,
                        selected_frames_list=selected_frames_list
                    )

                    if ctr == 0:
                        print(stage)
                        print(message)

                    if should_use_cache and cache_key in self.cache:
                        prior_response = self.cache[cache_key]
                    else:
                        start_time = time.time()
                        print("loading duration: ", start_time-loading_time)
                        response = self.inference_engine.run(message)[0]
                        print("inference duration: ", time.time()-start_time)
                        if should_use_cache:
                            self.cache[cache_key] = response
                        prior_response = response
                    
                    conversation.append({
                        "prompt": message[0]["content"][-1]["text"],
                        "response": prior_response
                    })

                    if stage == 'segment_selection':
                        segments = self._extract_segments(prior_response)
                        is_correct = self._evaluate(
                            segments,
                            clip_start=instance.get("clip_start_sec", 0),
                            clip_end=instance.get("clip_end_sec", 0)
                        )
                    

                result = {
                    "id": instance["id"],
                    "video_uid": video_uid,
                    "clip_uid": instance["clip_uid"],
                    "query": query,
                    "response": prior_response,
                    "chosen_segments": segments,
                    "correct_segment": is_correct,
                    "duration_sec": instance.get("duration_sec"),
                    "clip_start_sec": instance.get("clip_start_sec"),
                    "clip_end_sec": instance.get("clip_end_sec"),
                    "video_start_frame": instance.get("video_start_frame"),
                    "video_end_frame": instance.get("video_end_frame"),
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "sampling_strategy": sampling_strategy,
                    "template": instance.get("template"),
                    "conversation": conversation
                }
                self.results.append(result)

                if ctr % 10 == 0:
                    self._save_results()
        else:
            with open(self.config.temporal_selection_json, "r") as f:
                temporal_selection = json.load(f)
            turns = self.prompt_builder.num_turns()
            for ctr, instance in enumerate(tqdm(temporal_selection)):
                if ctr < 20:
                    continue
                video_uid = instance["video_uid"]
                query = instance["query"]
                video_path = os.path.join(self.config.dataset_path, "v2", "nlq_videos", "full_scale", video_uid + ".mp4")
                frames_info = self.sampler.sample(video_path)
                if frames_info is None:
                    print(f"Skipping {video_uid} due to possible memory issue")
                    continue
                frames_paths = [f["frame_path"] for f in frames_info]
                frame_labels = [f["label"] for f in frames_info]
                frames_dir = os.path.dirname(frames_paths[0])
                segments = self._extract_segments(instance['response'])
                selected_frames_list = self._select_frames(segments, frames_dir)
                responses = []
                for selected_frames in selected_frames_list:
                    stage, message = self.prompt_builder.build(
                        query=query,
                        frames_paths=frames_paths,
                        timestamps=frame_labels,
                        prior_response=instance["response"],
                        turn_index=turns-1,
                        selected_frames_list=selected_frames
                    )
                    response = self.inference_engine.run(message)[0]
                    responses.append(f"####### Segment {selected_frames[0]} - {selected_frames[-1]} #######\n{response}")
                instance["answer"] = '\n\n'.join(responses)

                self.results.append(instance)
                if ctr % 10 == 0:
                    self._save_results()

        if not self.config.ignore_cache:
            self._save_cache()

        self._save_results()
        self._report_metrics()