from prompt_toolkit.key_binding.key_bindings import key_binding
from ultralytics import YOLO
import os
import json
import torch
import sys

# load YOLO 11 pose model
model = YOLO("yolo11n-pose.pt")

# path to folder wanting to get poses
folder_path = "Images/small/hand_fand/hand_fand_Subject1_Tuan_cam-1-index_2.mp4"

pose_data = []

# number of videos
num_videos = 0


#print(os.listdir(folder_path))


def get_keypoints(path,frame_id = 0,img_id = 0):
    model = YOLO("yolo11n-pose.pt")
    results = model.predict(
        source=path,
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

        # remove person id > 0
        if len(kp) > 1:
            return pose_data

        for person_id, person_keypoints in enumerate(kp):
            frame_entry = {
                "frame_id": frame_id,
                "person_id": person_id,
                "img_id": img_id,
                "keypoints": person_keypoints.tolist()
            }
            pose_data.append(frame_entry)


    return pose_data

keypoints = get_keypoints(folder_path)

print(len(keypoints))

for i, keypoint in enumerate(keypoints):
    poses = keypoint["keypoints"]
    print(i)
    for pose in poses:
        print(pose)


def main():
    pass

if __name__ == "__main__":
    main()

'''
for gest_id, gesture in enumerate(os.listdir(folder_path)):
    vids = sorted(os.listdir(folder_path))
    vids.remove('.DS_Store')
    for img_id, img in enumerate(vids):
        frame_id = 0

        # asked gpt for how to make a progress bar
        bar_length = 30
        filled_length = int(bar_length * num_videos // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\rProgress: |{bar}| {num_videos/len(vids) * 100:.2f}%')
        sys.stdout.flush()

        num_videos += 1
        print(num_videos)

        img_path = os.path.join(folder_path, img)





    # save the data to json for later use
    with open("pose_data.json", "w") as f:
        json.dump(pose_data, f, indent=2)

    print("pose data saved to pose_data.json")
'''




