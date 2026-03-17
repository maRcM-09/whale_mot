from whale_mot.trackers.norfair_tracker import NorfairTracker
def build_tracker(cfg: dict):
    name = cfg["name"].lower()

    if name == "norfair":
        return NorfairTracker(
            distance_threshold=cfg.get("distance_threshold", 30.0)
        )
    raise ValueError(f"Unknown tracker: {name}")