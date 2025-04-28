import cv2
import os
import numpy as np
from tqdm import tqdm
import gooleNet_KTS as google_kts
from pathlib import Path

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip



def overlay_segment_label(img, start_time, end_time):
    label = f"{start_time}s - {end_time}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, label, (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img

def sample_img_from_shot(cps,videoFile,out_prefix, max_edge_len=224, start_sec = 0):
    cap = cv2.VideoCapture(videoFile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width > height:
        height = int(max_edge_len * height / width)
        width = max_edge_len
    else:
        width = int(max_edge_len * width / height)
        height = max_edge_len
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cps[0])

    img_idx=int((cps[0]+cps[1])/2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, img_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Error reading frame {img_idx} for {videoFile}")
        return
    start_time = int(cps[0]/fps) + start_sec
    end_time = int(cps[1]/fps) + start_sec
    rlt_img_file=out_prefix+f'/frame_{start_time}_{end_time}'+".jpg"
    try:
        frame = cv2.resize(frame, (width, height))
        frame = overlay_segment_label(frame, start_time, end_time)
        print(f"Writing image {rlt_img_file}")
        cv2.imwrite(rlt_img_file, frame)
    except Exception as e:
        print(f"Error writing image {rlt_img_file}: {e}")

def merge_to_target_count(cps, target_segments=20, min_length=200, max_iters=30):
    """
    Merges adjacent segments in a (n, 2) array until the count is ≤ target_segments.

    Parameters:
        cps (np.ndarray): (n, 2) array of (start, end) segments.
        target_segments (int): Target number of segments to reduce to.
        min_length (int): Initial minimum length for a segment.
        max_iters (int): Safety cap on iterations to prevent infinite loop.

    Returns:
        merged (np.ndarray): (m, 2) array with m ≤ target_segments.
    """
    merged = cps.copy()
    iters = 0

    def merge_pass(segments, min_length):
        segments = segments.tolist()  # work with mutable list of lists
        i = 0
        while i < len(segments):
            start, end = segments[i]
            length = end - start

            if length < min_length:
                # Determine merge direction: left or right
                if i == 0:
                    # No left neighbor, must merge right
                    segments[i+1] = [start, segments[i+1][1]]
                    del segments[i]
                    continue
                elif i == len(segments) - 1:
                    # No right neighbor, must merge left
                    segments[i-1][1] = end
                    del segments[i]
                    i -= 1
                    continue
                else:
                    # Compare left and right neighbor lengths
                    left_len = segments[i-1][1] - segments[i-1][0]
                    right_len = segments[i+1][1] - segments[i+1][0]

                    if left_len <= right_len:
                        # Merge left
                        segments[i-1][1] = end
                        del segments[i]
                        i -= 1
                        continue
                    else:
                        # Merge right
                        segments[i+1][0] = start
                        del segments[i]
                        continue
            else:
                i += 1

        return np.array(segments)

    while len(merged) > target_segments and iters < max_iters:
        merged = merge_pass(merged, min_length)
        min_length += 100
        iters += 1
    
    if len(merged) == 1:
        return cps

    return merged
    
    
def shot2Video(cps,videoFile,shot_file):
    cap = cv2.VideoCapture(videoFile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cps[0])

     # create summary video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(shot_file, fourcc, fps, (width, height))
    for i in range(cps[1]-cps[0]+1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    cap.release()
    

def shot_to_video_audio(cps, video_file, shot_file):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_time = cps[0] / fps
    end_time = cps[1] / fps
    print('CPS', cps)
    print('TIME', start_time, end_time)
    ffmpeg_extract_subclip(video_file, start_time, end_time, targetname=shot_file)

    
    
def shot2Frames(cps,videoFile,out_dir):
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    cap = cv2.VideoCapture(videoFile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cps[0])
    stem=Path(videoFile).stem
    for i in range(cps[1]-cps[0]+1):
        ret, frame = cap.read()
        if not ret:
            break
        idx=i+cps[0]
        filename=os.path.join(out_dir,f'{stem}_{idx}.jpg')
        cv2.imwrite(filename,frame)
    cap.release()


def shot_key_frame_detect(input_path, outPath, sample_rate=1):
    if os.path.isdir(input_path):
        video_names=os.listdir(input_path)
        video_names=[vname for vname in video_names if vname.endswith('.mp4')]
    else:
        video_names=[os.path.basename(input_path)]
        input_path= os.path.dirname(input_path)

    Path(outPath).mkdir(exist_ok=True, parents=True)
    video_proc = google_kts.VideoPreprocessor(sample_rate)
    for video in video_names:
        videoname=f'{input_path}/{video}'
        out_prefix=outPath+'/'+Path(video).stem
        rlt_seg_file = out_prefix +".txt"
        if os.path.exists(rlt_seg_file):       #already exist
            continue
        print(f'short detection from {video}...')
        n_frames, cps, nfps,picks = video_proc.seg_shot_detect(videoname, batch_size=100, max_len=2000, max_overlap=100, vmax=1.5*sample_rate)
        seg_num = len(cps)
  #save shots' locations
        with open(rlt_seg_file,'wt') as f:
            for n in range(seg_num): 
                start=cps[n][0] 
                end=cps[n][1]
                line=f'{n} {start} {end }\n'
                f.write(line)
        #save key frames of each shot
        for i in range(seg_num):
            sample_img_from_shot(cps[i], videoname, out_prefix, i)

def video_image_main(video_file, sample_rate=1):
     _, videoname=os.path.split(video_file)
#  videoname,_=os.path.splittext(videoname)     #alternative appraoch
#   videoname = videoname.stem
     outPath = './shotRlt'
     videoname=videoname.split('.')[0]
     out_prefix=outPath+'/'+videoname
     rlt_seg_file = out_prefix +".txt"
     if os.path.exists(rlt_seg_file):       #already exist
        return
     Path(outPath).mkdir(exist_ok=True, parents=True)
     video_proc = google_kts.VideoPreprocessor(sample_rate)
 
# shot detection 
     n_frames, cps, nfps,picks = video_proc.seg_shot_detect(video_file, batch_size=100, max_len=2000, max_overlap=100, vmax=1.5*sample_rate)
     seg_num = len(cps)
#save shots' locations
     with open(rlt_seg_file,'wt') as f:
         for n in range(seg_num): 
             start=cps[n][0] 
             end=cps[n][1]
             line=f'{n} {start} {end }\n'
             f.write(line)

     if not os.path.exists(outPath):
         os.makedirs(outPath)
     os.makedirs('{}/video'.format(outPath), exist_ok=True)
     os.makedirs('{}/image'.format(outPath), exist_ok=True)
     for i in range(seg_num):
         #save video shots
         video_outFile='{}/video/{}_{}.mp4'.format(outPath,videoname,i)
         shot_to_video_audio(cps[i],video_file, video_outFile)
        #  shot2Video(cps[i],video_file,outFile)
         #save key frames of each shot
         image_out_prefix = '{}/image/{}'.format(outPath, videoname)
         sample_img_from_shot(cps[i], video_file, image_out_prefix, i)


def video_main(video_file, outPath, output_mode='video_seg', sample_rate=30):
    video_proc = google_kts.VideoPreprocessor(sample_rate)
    _, videoname=os.path.split(video_file)
#  videoname,_=os.path.splittext(videoname)     #alternative appraoch
#   videoname = videoname.stem
    videoname=videoname.split('.')[0]
    out_prefix=outPath+'/'+videoname
    rlt_seg_file = out_prefix +".txt"

# shot detection 
    n_frames, cps, nfps,picks = video_proc.seg_shot_detect(video_file, batch_size=100, max_len=2000, max_overlap=100, vmax=1.5*sample_rate)
    seg_num = len(cps)
#save shots' locations
    with open(rlt_seg_file,'wt') as f:
        for n in range(seg_num): 
            start=cps[n][0]
            end=cps[n][1]
            line=f'{n} {start} {end }\n'
            f.write(line)
# #save video shots
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    if output_mode == 'video_seg':      #save segment into video chips
        for i in range(seg_num):
            outFile=out_prefix+f'_{i}'+".mp4"
            shot2Video(cps[i],video_file,outFile)
    elif output_mode == 'video_frame':    #save frames of segments into respective folders
        for i in range(seg_num):
            subfolder = os.path.join(outPath, f'{videoname}-{i}')
            Path(subfolder).mkdir(parents=True, exist_ok=True)
            shot2Frames(cps[i],video_file,subfolder)
            
def split_frames_into_segments(x_frames, n_segments):
    """
    Split x_frames into n_segments. 
    Each segment is [start_frame, end_frame] (both inclusive).
    Returns a numpy array of shape (n_segments, 2).
    """
    frames_per_segment = x_frames / n_segments
    segments = []
    
    for i in range(n_segments):
        start = int(round(i * frames_per_segment))
        end = int(round((i + 1) * frames_per_segment)) - 1
        end = min(end, x_frames - 1)  # Prevent going beyond last frame
        segments.append([start, end])
    
    return np.array(segments)

def split_frames_by_seconds(x_frames, fps, target_seconds=20):
    """
    Split frames into even segments where each segment is approximately target_seconds long.
    Returns a numpy array of [start_frame, end_frame] per segment.
    
    Args:
    - x_frames (int): total number of frames
    - fps (float): frames per second
    - target_seconds (float): approximate length per segment in seconds
    
    Returns:
    - numpy.ndarray: shape (n_segments, 2)
    """
    frames_per_segment = target_seconds * fps
    n_segments = int(np.ceil(x_frames / frames_per_segment))
    
    segments = []
    for i in range(n_segments):
        start = int(round(i * frames_per_segment))
        end = int(round((i + 1) * frames_per_segment)) - 1
        end = min(end, x_frames - 1)  # Make sure not to go out of bounds
        segments.append([start, end])
    
    return np.array(segments)

def img_main(video_file, out_prefix, max_segments=20, sample_rate=30, start_sec = 0, max_edge_len = 224, target_seconds=20):
    video_proc = google_kts.VideoPreprocessor(sample_rate)
#check video length, temporarily ignore videos longer than 23,000 frames
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    #     _, videoname=os.path.split(video_file)
    #     videoname=videoname.split('.')[0]
    videoname = Path(video_file).stem 
# shot detection
    n_frames, cps, nfps,picks = video_proc.seg_shot_detect(video_file, batch_size=100, max_len=2000,max_overlap=100, vmax=1.5*sample_rate)
    cps = cps[cps[:, 0] < cps[:, 1]]
    cps = merge_to_target_count(cps, target_segments=max_segments)
    seg_num = len(cps)

#save shots' locations
    # rlt_seg_file = out_prefix+".txt"
    # with open(rlt_seg_file,'wt') as f:
    #     for n in range(seg_num): 
    #         start=cps[n][0] 
    #         end=cps[n][1]
    #         line=f'{n} {start} {end }\n'
    #         f.write(line)
#save key frames of each shot
    if seg_num <= 1:
        # cps = split_frames_into_segments(n_frames, max_segments)
        cps = split_frames_by_seconds(n_frames, sample_rate, target_seconds=target_seconds)
        seg_num = len(cps)
    for i in range(seg_num):
        sample_img_from_shot(cps[i], video_file, out_prefix, max_edge_len=max_edge_len, start_sec=start_sec)
         
# if __name__ == '__main__':
#     sample_rate = 30
#     max_segments = 20
#     outPath = '/nethome/che321/flash/LVTR/shot_detection/frames'
#     Path(outPath).mkdir(exist_ok=True, parents=True)
#     video_proc = google_kts.VideoPreprocessor(sample_rate)
#     video_folder = '/nethome/che321/flash/datasets/Ego4D/v2/nlq_videos/full_scale'
#     for video in tqdm(os.listdir(video_folder)):
#         if video.endswith('.mp4'):
#             out_prefix = outPath + "/"+video.split('.')[0]  
#             if not os.path.exists(out_prefix):
#                 os.makedirs(out_prefix)
#             else:
#                 continue
#             img_main(video_folder+'/'+video, out_prefix, max_segments)

    # mode = sys.argv[1]
    # video_path = sys.argv[2]
    # outPath = sys.argv[3]
    # sample_rate = int(sys.argv[4])
    # Path(outPath).mkdir(exist_ok=True, parents=True)
    # video_proc = google_kts.VideoPreprocessor(sample_rate)
    
    # video_names=os.listdir(video_path)
    # video_names=[vname for vname in video_names if vname.endswith('.mp4')]
    # for video in video_names:
    #     print(f'Preprocessing video {video}...\n')
    #     if mode == 'video':
    #        video_main(video_path+'/'+video,'video_seg')
           
    #     if mode == "image":
    #         img_main(video_path+'/'+video)
