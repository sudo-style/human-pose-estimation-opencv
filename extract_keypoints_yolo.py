from prompt_toolkit.key_binding.key_bindings import key_binding
from ultralytics import YOLO
import os
import json
import torch
import sys

def get_keypoints(model, vid_path,vid_id = 0, gesture_id = 0):
    results = model.predict(
        source=vid_path,
        save=False,
        show=False,
        conf=0.5,
        verbose=False
    )

    pose_data = []
    for frame_id, result in enumerate(results):
        if result.keypoints is None:
            continue

        # convert tensor to numpy
        kp = result.keypoints.data.cpu().numpy()

        # remove person if more than one person
        if len(kp) > 1:
            return pose_data

        for person_id, person_keypoints in enumerate(kp):
            frame_entry = {
                "vid_path": vid_path,
                "gesture_id": gesture_id,
                "vid_id": vid_id,
                "frame_id": frame_id,
                "person_id": person_id,
                "keypoints": person_keypoints.tolist(),
            }
            pose_data.append(frame_entry)


    return pose_data

def main():
    # log how long it takes to run
    import time
    start_time = time.time()

    # load YOLO 11 pose model
    model = YOLO("yolo11n-pose.pt")

    # path to folder wanting to get poses
    folder_path = "Images/small"

    pose_data = []

    # number of videos
    num_videos = 0

    gesture_names = sorted(os.listdir(folder_path))
    gesture_names.remove('.DS_Store')
    for gesture_id, gesture in enumerate(gesture_names):
        gesture_path = os.path.join(folder_path, gesture)

        vids = os.listdir(gesture_path)
        vids.remove('.DS_Store')


        pose_sequence = []

        for vid_id, vid in enumerate(vids):
            vid_path = os.path.join(gesture_path, vid)
            pose_sequence = get_keypoints(model=model, vid_path=vid_path, vid_id=vid_id, gesture_id=gesture_id)


            if pose_sequence:
                pose_data.append(pose_sequence)
            else:
                frame_entry = {
                    "vid_path": vid_path,
                    "gesture_id": gesture_id,
                    "vid_id": vid_id,
                    "frame_id": 0,
                    "person_id": 0,
                    "keypoints": [],
                }
                pose_data.append([frame_entry])

        # save the data to json for later use
        with open("pose_data.json", "w") as f:
            json.dump(pose_data, f, indent=2)

    print("pose data saved to pose_data.json")
    end_time = time.time()
    print("time taken:", end_time - start_time)

if __name__ == "__main__":
    main()