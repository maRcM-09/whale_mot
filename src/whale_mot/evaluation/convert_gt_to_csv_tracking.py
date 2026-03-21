import pandas as pd

import pandas as pd

def mot_txt_to_track_gt(txt_path, output_path):
    """
    Convert MOT .txt (CVAT export) to track_gt.csv with columns:
    frame,id,x1,y1,x2,y2,confidence,class_id
    """

    # Read MOT file
    df = pd.read_csv(txt_path, header=None)

    # Assign MOT column names
    df.columns = [
        "frame", "id", "x", "y", "w", "h",
        "confidence", "class_id", "visibility"
    ]

    # Convert to corner format
    df["frame"] = df["frame"] - 1
    df["x1"] = df["x"]
    df["y1"] = df["y"]
    df["x2"] = df["x"] + df["w"]
    df["y2"] = df["y"] + df["h"]

    # Select final columns
    track_gt = df[[
        "frame", "id", "x1", "y1", "x2", "y2",
        "confidence", "class_id"
    ]].copy()

    # Optional: enforce types
    track_gt["frame"] = track_gt["frame"].astype(int)
    track_gt["id"] = track_gt["id"].astype(int)

    # Save
    track_gt.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    mot_txt_to_track_gt("/home/marcm/Documents/EPFL/MA4/whale_project/whale_mot/data/gt_mot/poc_testSet (1)/gt/gt.txt", 
                        "/home/marcm/Documents/EPFL/MA4/whale_project/whale_mot/data/gt_mot/track_gt.csv")