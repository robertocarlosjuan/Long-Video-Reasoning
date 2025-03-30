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
    temp_frames_dir = "tmp/test"

class Qwen72BDenseSamplingConfig:
    dataset_path = "/nethome/che321/flash/datasets/Ego4D"
    annotation_file = "v2/annotations/nlq_val.json"
    max_edge_len = 112
    sampling_rate = 1  # frames per second for dense sampling
    num_frames = 8  # for uniform sampling
    sampling_strategy = "dense"  # or "dense"
    prompt_style = "frame_labeling" #"stage_analysis"  # single style used per run
    model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
    cache_path = "cached_stage_outputs.json"
    ignore_cache = True  # set to True to force fresh run
    output_path = "Qwen72B_Ego4D_dense_sampling.json"
    output_csv_path = "Qwen72B_Ego4D_dense_sampling.csv"
    output_plot_path = "Qwen72B_Ego4D_dense_sampling.png"
    temp_frames_dir = "tmp/dense"

class Qwen72BCoarseSamplingConfig:
    dataset_path = "/nethome/che321/flash/datasets/Ego4D"
    annotation_file = "v2/annotations/nlq_val.json"
    max_edge_len = 1024
    sampling_rate = 1  # frames per second for dense sampling
    num_frames = 8  # for uniform sampling
    sampling_strategy = "uniform"  # or "dense"
    prompt_style = "stage_analysis" #"stage_analysis"  # single style used per run
    model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
    cache_path = "cached_stage_outputs.json"
    ignore_cache = True  # set to True to force fresh run
    output_path = "Qwen72B_Ego4D_coarse_sampling.json"
    output_csv_path = "Qwen72B_Ego4D_coarse_sampling.csv"
    output_plot_path = "Qwen72B_Ego4D_coarse_sampling.png"
    temp_frames_dir = "tmp/coarse"
