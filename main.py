import argparse
from config import (
    Qwen72BDenseSamplingConfig,
    Qwen72BCoarseSamplingConfig,
    TestConfig,
    BaselineConfig,
)
from dataset import Ego4DLoader
from sampler import FrameSampler
from inference_engine import InferenceEngine
from batch_runner import BatchRunner

# Map string names to config classes
config_presets = {
    "dense": Qwen72BDenseSamplingConfig,
    "coarse": Qwen72BCoarseSamplingConfig,
    "test": TestConfig,
    "baseline": BaselineConfig,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="test",
        choices=config_presets.keys(),
        help="Choose which config to use (dense, coarse, test)"
    )
    args = parser.parse_args()

    # Instantiate the selected config
    Config = config_presets[args.config]
    config = Config()

    print(f"Using config: {args.config}")
    print(f"Model: {config.model_path}")

    # Initialize components and run
    dataset_loader = Ego4DLoader(config)
    sampler = FrameSampler(config)
    inference_engine = InferenceEngine(config)
    runner = BatchRunner(config, dataset_loader, sampler, inference_engine)
    runner.run()
