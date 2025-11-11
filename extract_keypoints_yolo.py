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
        verbose=False,
        stream=True,
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
    folder_path = "Images/body segmentation and gesturere cognition cleaned"

    # get list of gesture folders
    gesture_names = [g for g in os.listdir(folder_path) if g != ".DS_Store"]

    # output JSON file
    output_json = Path("pose_data.jsonl")

    videos_done = 0

    # find the number of videos
    num_videos = sum(
        len([vid for vid in os.listdir(os.path.join(folder_path, gesture_name)) if vid != '.DS_Store'])
        for gesture_name in gesture_names
    )

    batch_entries = []

    # load existing data if file already exists
    if output_json.exists():
        with open(output_json, "r") as f:
            all_entries = json.load(f)
    else: all_entries = []

    for gesture_id, gesture in enumerate(gesture_names):
        gesture_path = os.path.join(folder_path, gesture)
        vids = [v for v in os.listdir(gesture_path) if v != ".DS_Store"]

        for vid_id, vid in enumerate(vids):
            vid_path = os.path.join(gesture_path, vid)
            print(f"Processing: gesture={gesture}, video={vid}")

            try:
                pose_sequence = get_keypoints(
                    model=model,
                    vid_path=vid_path,
                    gesture_id=gesture_id
                )
                # append each video entry to the list
                batch_entries.append(pose_sequence)
            except Exception as e:
                print(f"An error occurred while getting keypoints: {e}")

            # save the data to json for later use
            batch_size = 50
            if batch_size < len(batch_entries):
                with open(output_json, "a", encoding="utf-8") as f:
                    for entry in batch_entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                videos_done += batch_size
                print(f"videos_done: {videos_done}")
                batch_entries = []

            percent_done = videos_done/num_videos
            print(f"videos done: {videos_done}, num videos: {num_videos}, Percentage: {percent_done * 100:.2f}%")

    print("All pose data saved.")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()