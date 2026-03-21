import pandas as pd

def mot_txt_to_detection_gt(txt_path, output_path):
    """
    Convert MOT .txt (CVAT export) to detection-style gt.csv with columns:
    frame,x1,y1,x2,y2,confidence,class_id
    """

    # Read MOT file
    df = pd.read_csv(txt_path, header=None)

    # Assign MOT column names
    df.columns = [
        "frame", "id", "x", "y", "w", "h",
        "confidence", "class_id", "visibility"
    ]

    # Convert frame indexing (1-based → 0-based)
    df["frame"] = df["frame"] - 1
    df["class_id"] = df["class_id"] - 1  # Ensure class_id is int

    # Convert to xyxy format
    df["x1"] = df["x"]
    df["y1"] = df["y"]
    df["x2"] = df["x"] + df["w"]
    df["y2"] = df["y"] + df["h"]

    # Select detection-style columns (drop track id)
    detection_gt = df[[
        "frame", "x1", "y1", "x2", "y2",
        "confidence", "class_id"
    ]].copy()

    # Optional: ensure consistent types
    detection_gt["frame"] = detection_gt["frame"].astype(int)

    # Save
    detection_gt.to_csv(output_path, index=False)

    print(f"Saved detection GT to: {output_path}")

if __name__ == "__main__":
    # Example usage
    mot_txt_to_detection_gt("/home/marcm/Documents/EPFL/MA4/whale_project/whale_mot/data/gt_mot/poc_testSet (1)/gt/gt.txt", 
                            "/home/marcm/Documents/EPFL/MA4/whale_project/whale_mot/data/gt_mot/detection_gt.csv")