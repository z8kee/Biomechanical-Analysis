#used to extract features from videos using mediapipe pose landmarker and save them as npy files
#the npy files can then be used for phase classification models
#this will be used everytime new videos are added to the dataset

import os, cv2, csv, numpy as np, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#----------settings-----------
window = 30
stride = 15
video_dir = 'data/video'
out_dir = 'data/npy'
m_path = 'models/pose_landmarker_full.task'
meta_path = 'data/metadata.csv'
os.makedirs(out_dir, exist_ok=True)

#----------make a model for mediapipe---------
baseOptions = python.BaseOptions
poseLandmarker = vision.PoseLandmarker
poseLandmarkerOptions = vision.PoseLandmarkerOptions
visionRunningMode = vision.RunningMode
meta_file = open(meta_path, 'w', newline='')
writer = csv.writer(meta_file)
writer.writerow(['sample_id', 'video_file', 'start_frame', 'end_frame'])
option = poseLandmarkerOptions(
    base_options=baseOptions(model_asset_path=m_path),
    running_mode = visionRunningMode.VIDEO,
    num_poses=1
)

sample_id = 0

#----------------process videos and frames from videos into npy file dir--------------------

for vid in os.listdir(video_dir): # for every vid in video directory, add to frames and save to npy file
    print(f"Processing video: {vid}")
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=m_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    vidcap = cv2.VideoCapture(os.path.join(video_dir, vid))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    frames = []
    frame_idx = 0

    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        ts_ms = int((frame_idx / fps) * 1000)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processing frame {frame_idx} of video {vid}")
        result = pose_landmarker.detect_for_video(mp_img, ts_ms)

        if not result.pose_landmarks:
            continue

        landmarks = []
        for landmark in result.pose_landmarks[0]:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        frames.append(landmarks)

    vidcap.release()
    frames = np.array(frames)
    print(f"Finished {vid}, total frames used: {len(frames)}")

    #-------------windowing time---------------------
    if len(frames) >= window:
        for i in range(0, len(frames) - window+1, stride):
            window_data = frames[i:i + window]
            np.save(f"{out_dir}/sample{sample_id}.npy", window_data)
            writer.writerow([sample_id, vid, i, i + window])

            sample_id += 1
    else:
        print(f"Video {vid} skipped due to insufficient frames.")
