import cv2
import torch
import subprocess
import numpy as np


def read_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    video_bgr = []

    while True:
        ret, frame = cap.read()  # ret is a boolean (True if frame is read correctly), frame is the image array in BGR
        
        # If the frame was not read correctly (e.g., end of video), break the loop
        if not ret:
            break
        
        video_bgr.append(frame)

    cap.release()

    video_bgr = np.asarray(video_bgr)
    video_bgr = torch.as_tensor(video_bgr, dtype=torch.float32)

    return video_bgr


def bgr_to_grayscale(video_bgr):
    bgr_to_grayscale_weights = torch.tensor([0.114, 0.587, 0.299], dtype=torch.float32, device=video_bgr.device)
    bgr_to_grayscale_weights = bgr_to_grayscale_weights.view(1,1,1,3)

    weighted_video  = video_bgr * bgr_to_grayscale_weights
    video_grayscale = torch.sum(weighted_video, dim=-1)

    return video_grayscale


def write_video(video, name):
    video = video.tobytes()

    with open(f"{name}.raw", "wb") as f:
        f.write(video)

    subprocess.run(f"ffmpeg -loglevel quiet -f rawvideo -pixel_format bgr24 -video_size 352x288 -framerate 30 -i {name}.raw -c:v rawvideo -pix_fmt bgr24 {name}.avi", shell=True, check=True)

    subprocess.run(f"rm {name}.raw", shell=True)