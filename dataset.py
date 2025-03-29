import os
import json
import cv2

class Ego4DLoader:
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.dataset_path
        self.annotation_file = os.path.join(self.dataset_path, config.annotation_file)
        self.videos_metadata = self._load_metadata()

    def _load_metadata(self):
        with open(os.path.join(self.dataset_path, "ego4d.json")) as f:
            data = json.load(f)
        return {x['video_uid']: x for x in data['videos']}

    def load_instances(self):
        with open(self.annotation_file) as f:
            val = json.load(f)

        instances = []
        for d in val['videos']:
            video_id = d['video_uid']
            video_metadata = self.videos_metadata.get(video_id, {})
            duration = video_metadata.get("duration_sec", 0)
            for clip in d['clips']:
                clip_id = clip['clip_uid']
                for annotation in clip['annotations']:
                    annotation_id = annotation['annotation_uid']
                    for i, sample in enumerate(annotation['language_queries']):
                        instances.append({
                            "id": f"{annotation_id}_{i}",
                            "video_uid": video_id,
                            "clip_uid": clip_id,
                            "query": sample.get('query'),
                            "duration_sec": duration,
                            'clip_start_sec': sample.get('clip_start_sec'),
                            'clip_end_sec': sample.get('clip_end_sec'),
                            'video_start_frame': sample.get('video_start_frame'),
                            'video_end_frame': sample.get('video_end_frame'),
                            'template': sample.get('template')
                        })
        return instances