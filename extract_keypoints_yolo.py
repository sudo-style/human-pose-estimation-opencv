from prompt_toolkit.key_binding.key_bindings import key_binding
from ultralytics import YOLO
import os
import json
import torch
import sys
import numpy as np
import time
from pathlib import Path

def get_keypoints(model, vid_path, gesture_id=0):
    results = model.predict(
        source=vid_path,
        save=False,
        show=False,
        conf=0.5,
        verbose=False
    )

    kps = []

    video_entry = {
        "vid_path": vid_path,
        "gesture_id": gesture_id,
        "keypoints": []
    }

    for frame_id, result in enumerate(results):
        # skip empty frames
        if result.keypoints is None:
            continue

        # convert tensor to numpy
        kp = result.keypoints.data.cpu().numpy()

        # if more than one person, skip this video/frame
        if len(kp) > 1:
            return video_entry

        # squeeze out the first dimension if there's only one person
        kp = kp.squeeze(axis=0)  # now shape is (num_keypoints, 3)
        kps.append(kp)

    # convert each numpy array inside the list to a list
    video_entry["keypoints"] = [kp.tolist() for kp in kps]
    return video_entry


def main():
    start_time = time.time()

    # load YOLO 11 pose model
    model = YOLO("yolo11n-pose.pt", verbose=False)

    # path to folder wanting to get poses
    folder_path = "Images/small"

    # get list of gesture folders
    gesture_names = [g for g in os.listdir(folder_path) if g != ".DS_Store"]

    # output JSON file
    output_json = Path("pose_data.json")

    videos_done = 0

    # find the number of videos
    num_videos = sum(
        len([vid for vid in os.listdir(os.path.join(folder_path, gesture_name)) if vid != '.DS_Store'])
        for gesture_name in gesture_names
    )

    # load existing data if file already exists
    if output_json.exists():
        with open(output_json, "r") as f:
            all_entries = json.load(f)
    else:
        all_entries = []

    for gesture_id, gesture in enumerate(gesture_names):
        gesture_path = os.path.join(folder_path, gesture)
        vids = [v for v in os.listdir(gesture_path) if v != ".DS_Store"]

        for vid_id, vid in enumerate(vids):
            vid_path = os.path.join(gesture_path, vid)
            print(f"Processing: gesture={gesture}, video={vid}")

            pose_sequence = get_keypoints(
                model=model,
                vid_path=vid_path,
                gesture_id=gesture_id
            )

            # append each video entry to the list
            all_entries.append(pose_sequence)

            # save the data to json for later use
            with open(output_json, "w") as f:
                json.dump(all_entries, f, indent=2)

            print(f"Added {vid_path} to {output_json}\n")
            videos_done += 1

            percent_done = videos_done/num_videos
            print(f"Percentage: {percent_done * 100:.2f}%")

    print("All pose data saved.")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()