# This file is to run the tracker on its own

import argparse
import yaml

from whale_mot.pipelines.tracking_only import run_tracking_only

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_tracking_only(cfg)

if __name__ == "__main__":
    main()