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

        with open(self.config.output_csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "id", "video_uid", "clip_uid", "query", "correct_segment",
                "duration_sec", "clip_start_sec", "clip_end_sec",
                "video_start_frame", "video_end_frame", "template"
            ])
            writer.writeheader()
            for row in self.results:
                writer.writerow({k: row[k] for k in writer.fieldnames})

    def _extract_segments(self, text):
        return re.findall(r"\[(\d+)(?:s)?\]\s*-\s*\[(\d+)(?:s)?\]", text)

    def _evaluate(self, response, clip_start, clip_end):
        if not response:
            return False
        segments = self._extract_segments(response)
        for start_str, end_str in segments:
            try:
                start = int(start_str)
                end = int(end_str)
                if clip_start >= start and clip_end <= end:
                    return True
            except ValueError:
                continue
        return False

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
        for ctr, instance in enumerate(tqdm(instances)):
            loading_time = time.time()
            video_uid = instance["video_uid"]
            query = instance["query"]

            video_path = os.path.join(self.config.dataset_path, "v2", "nlq_videos", "full_scale", video_uid + ".mp4")
            frames_info = self.sampler.sample(video_path, instance["duration_sec"])
            frame_labels = [f["label"] for f in frames_info]
            frames_paths = [f["frame_path"] for f in frames_info]
            
            turns = self.prompt_builder.num_turns()
            prior_response = None
            conversation = []

            for turn_index in range(turns):
                cache_key = f"{video_uid}:{turn_index}"
                should_use_cache = self.prompt_builder.uses_query(turn_index) is False

                message = self.prompt_builder.build(
                    query=query,
                    frames_paths=frames_paths,
                    timestamps=frame_labels,
                    prior_response=prior_response,
                    turn_index=turn_index
                )

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

            is_correct = self._evaluate(
                prior_response,
                clip_start=instance.get("clip_start_sec", 0),
                clip_end=instance.get("clip_end_sec", 0)
            )

            result = {
                "id": instance["id"],
                "video_uid": video_uid,
                "clip_uid": instance["clip_uid"],
                "query": query,
                "response": prior_response,
                "correct_segment": is_correct,
                "duration_sec": instance.get("duration_sec"),
                "clip_start_sec": instance.get("clip_start_sec"),
                "clip_end_sec": instance.get("clip_end_sec"),
                "video_start_frame": instance.get("video_start_frame"),
                "video_end_frame": instance.get("video_end_frame"),
                "template": instance.get("template"),
                "conversation": conversation
            }
            self.results.append(result)

            if ctr % 10 == 0:
                self._save_results()

        if not self.config.ignore_cache:
            self._save_cache()

        self._save_results()
        self._report_metrics()