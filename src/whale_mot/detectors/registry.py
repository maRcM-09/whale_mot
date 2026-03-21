from whale_mot.detectors.ultralytics_detector import UltralyticsDetector


def build_detector(cfg: dict):
    name = cfg["name"].lower()

    if name == "ultralytics":
        return UltralyticsDetector(
            model_name=cfg["model_name"],
            device=cfg.get("device", "cpu"),
            conf=cfg.get("conf", 0.85),
        )

    raise ValueError(f"Unknown detector: {name}")