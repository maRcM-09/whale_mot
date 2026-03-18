from whale_mot.trackers.norfair_tracker import NorfairTracker
from whale_mot.trackers.bytetrack import ByteTrackTracker
from whale_mot.trackers.botsort import BoTSORTTracker

def build_tracker(cfg: dict):
    name = cfg["name"].lower()

    if name == "norfair":
        return NorfairTracker(
            distance_threshold=cfg.get("distance_threshold", 30.0)
        )
    
    elif name == "bytetrack":
        return ByteTrackTracker(
            min_confidence=cfg.get("min_confidence", 0.25),
            track_threshold=cfg.get("track_threshold", 0.5),
            match_threshold=cfg.get("match_threshold", 0.5),
            track_buffer=cfg.get("track_buffer", 30),
            frame_rate=cfg.get("frame_rate", 30)
        )
    elif name == "botsort":
        return BoTSORTTracker(
            reid_weights=cfg.get("reid_weights", None),
            device=cfg.get("device", "cpu"),
            half=cfg.get("half", False),
            track_high_thresh=cfg.get("track_high_thresh", 0.5),
            track_low_thresh=cfg.get("track_low_thresh", 0.1),
            new_track_thresh=cfg.get("new_track_thresh", 0.6),
            track_buffer=cfg.get("track_buffer", 30),
            match_thresh=cfg.get("match_thresh", 0.8),
            proximity_thresh=cfg.get("proximity_thresh", 0.5),
            appearance_thresh=cfg.get("appearance_thresh", 0.25),
            cmc_method=cfg.get("cmc_method", "ecc"),
            frame_rate=cfg.get("frame_rate", 30),
            fuse_first_associate=cfg.get("fuse_first_associate", False),
            with_reid=cfg.get("with_reid", True),
            **cfg.get("kwargs", {})
        )
    raise ValueError(f"Unknown tracker: {name}")