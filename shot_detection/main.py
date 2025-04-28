from shot_detector import img_main
import os
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    sample_rate = 30
    max_segments = 20
    outPath = '/nethome/che321/flash/LVTR/shot_detection/frames'
    Path(outPath).mkdir(exist_ok=True, parents=True)
    video_folder = '/nethome/che321/flash/datasets/Ego4D/v2/nlq_videos/full_scale'
    for video in tqdm(os.listdir(video_folder)):
        if video.endswith('.mp4'):
            out_prefix = outPath + "/"+video.split('.')[0]  
            if not os.path.exists(out_prefix):
                os.makedirs(out_prefix)
            else:
                continue
            img_main(video_folder+'/'+video, out_prefix, max_segments)