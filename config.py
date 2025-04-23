# config.py
import torch
import os

class TestConfig:
    dataset_path = "/nethome/che321/flash/datasets/Ego4D"
    annotation_file = "v2/annotations/nlq_val.json"
    max_edge_len = 224
    sampling_rate = 1  # frames per second for dense sampling
    num_frames = 8  # for uniform sampling
    sampling_strategy = "dense"  # or "dense"
    prompt_style = "frame_labeling" #"stage_analysis"  # single style used per run
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    cache_path = "cached_stage_outputs.json"
    ignore_cache = True  # set to True to force fresh run
    output_path = "test_outputs.json"
    output_csv_path = "test_outputs.csv"
    output_plot_path = "test_outputs.png"
    temp_frames_dir = "frames"

class BaselineConfig:
    dataset_path = "/nethome/che321/flash/datasets/Ego4D"
    annotation_file = "v2/annotations/nlq_val.json"
    max_edge_len = 224
    sampling_rate = 1  # frames per second for dense sampling
    num_frames = 8  # for uniform sampling
    sampling_strategy = "dense"  # or "dense"
    prompt_style = "baseline" #"stage_analysis"  # single style used per run
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    cache_path = "baseline_cached_stage_outputs.json"
    ignore_cache = True  # set to True to force fresh run
    output_path = "baseline_outputs.json"
    output_csv_path = "baseline_outputs.csv"
    output_plot_path = "baseline_outputs.png"
    temp_frames_dir = "frames"
    temporal_selection_json = "baseline_temporal_selection.json"

class Qwen72BDenseSamplingConfig:
    dataset_path = "/nethome/che321/flash/datasets/Ego4D"
    annotation_file = "v2/annotations/nlq_val.json"
    max_edge_len = 112
    sampling_rate = 1  # frames per second for dense sampling
    num_frames = 8  # for uniform sampling
    sampling_strategy = "dense"  # or "dense"
    prompt_style = "frame_labeling" #"stage_analysis"  # single style used per run
    model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
    cache_path = "dense_cached_stage_outputs.json"
    ignore_cache = True  # set to True to force fresh run
    output_path = "Qwen72B_Ego4D_dense_sampling.json"
    output_csv_path = "Qwen72B_Ego4D_dense_sampling.csv"
    output_plot_path = "Qwen72B_Ego4D_dense_sampling.png"
    temp_frames_dir = "frames"
    temporal_selection_json = "Qwen72B_Ego4D_dense_sampling_temporal_selection.json"
    if os.path.exists(temporal_selection_json): # final answer
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        sampling_strategy = "dense"
        sampling_rate = 1
        max_edge_len = 224

class Qwen72BCoarseSamplingConfig:
    dataset_path = "/nethome/che321/flash/datasets/Ego4D"
    annotation_file = "v2/annotations/nlq_val.json"
    max_edge_len = 1024
    sampling_rate = 1  # frames per second for dense sampling
    num_frames = 8  # for uniform sampling
    sampling_strategy = "uniform"  # or "dense"
    prompt_style = "stage_analysis" #"stage_analysis"  # single style used per run
    model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
    cache_path = "coarse_cached_stage_outputs.json"
    ignore_cache = True  # set to True to force fresh run
    output_path = "Qwen72B_Ego4D_coarse_sampling.json"
    output_csv_path = "Qwen72B_Ego4D_coarse_sampling.csv"
    output_plot_path = "Qwen72B_Ego4D_coarse_sampling.png"
    temp_frames_dir = "frames"
    temporal_selection_json = "Qwen72B_Ego4D_coarse_sampling_temporal_selection.json"
    if os.path.exists(temporal_selection_json): # final answer
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        sampling_strategy = "dense"
        sampling_rate = 1
        max_edge_len = 224

class Qwen72BCoarseCoTSamplingConfig:
    dataset_path = "/nethome/che321/flash/datasets/Ego4D"
    annotation_file = "v2/annotations/nlq_val.json"
    max_edge_len = 224
    sampling_rate = 1  # frames per second for dense sampling
    num_frames = 8  # for uniform sampling
    sampling_strategy = "uniform"  # or "dense"
    prompt_style = "coarse_cot" #"stage_analysis"  # single style used per run
    model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
    cache_path = "coarse_cot_cached_stage_outputs.json"
    ignore_cache = False  # set to True to force fresh run
    output_path = "Qwen72B_Ego4D_coarse_cot_sampling.json"
    output_csv_path = "Qwen72B_Ego4D_coarse_cot_sampling.csv"
    output_plot_path = "Qwen72B_Ego4D_coarse_cot_sampling.png"
    temp_frames_dir = "frames"
    temporal_selection_json = "Qwen72B_Ego4D_coarse_cot_sampling_temporal_selection.json"
    if os.path.exists(temporal_selection_json): # final answer
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        sampling_strategy = "dense"
        sampling_rate = 1
        max_edge_len = 224

class Qwen72BCoarseCoTShotDetectionConfig:
    dataset_path = "/nethome/che321/flash/datasets/Ego4D"
    annotation_file = "v2/annotations/nlq_val.json"
    max_edge_len = 224
    sampling_rate = 1  # frames per second for dense sampling
    num_frames = 8  # for uniform sampling
    sampling_strategy = "shot_detection"  # or "dense"
    prompt_style = "coarse_cot_shot_detection" #"stage_analysis"  # single style used per run
    model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
    cache_path = "coarse_cot_shot_detection_cached_stage_outputs.json"
    ignore_cache = False  # set to True to force fresh run
    output_path = "Qwen72B_Ego4D_coarse_cot_shot_detection.json"
    output_csv_path = "Qwen72B_Ego4D_coarse_cot_shot_detection.csv"
    output_plot_path = "Qwen72B_Ego4D_coarse_cot_shot_detection.png"
    temp_frames_dir = "shot_detection/overlay_frames"
    temporal_selection_json = "Qwen72B_Ego4D_coarse_cot_shot_detection_temporal_selection.json"
    if os.path.exists(temporal_selection_json): # final answer
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        sampling_strategy = "uniform"
        num_frames = 8
        max_edge_len = 224
        temp_frames_dir = "frames"

class CLIPSBERTDenseSamplingConfig:
    dataset_path = "/nethome/che321/flash/datasets/Ego4D"
    annotation_file = "v2/annotations/nlq_val.json"
    max_edge_len = 112
    sampling_rate = 1  # frames per second for dense sampling
    num_frames = 8  # for uniform sampling
    sampling_strategy = "dense"  # or "dense"
    prompt_style = "frame_labeling" #"stage_analysis"  # single style used per run
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    cache_path = "cached_stage_outputs.json"
    ignore_cache = True  # set to True to force fresh run
    output_path = "Qwen72B_Ego4D_dense_sampling.json"
    output_csv_path = "Qwen72B_Ego4D_dense_sampling.csv"
    output_plot_path = "Qwen72B_Ego4D_dense_sampling.png"
    temp_frames_dir = "frames"
    temporal_selection_json = "Qwen72B_Ego4D_dense_sampling_temporal_selection.json"