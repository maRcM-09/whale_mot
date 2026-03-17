# in this script we run the detector experiment

from pathlib import Path
import argparse
import yaml

from whale_mot.pipelines.detection_only import run_detection_only

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_detection_only(cfg)

if __name__ == "__main__":
    main()