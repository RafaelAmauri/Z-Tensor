import cv2
import torch
import numpy as np


def read_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    video_rgb = []

    while True:
        ret, frame = cap.read()  # ret is a boolean (True if frame is read correctly), frame is the image array in BGR
        
        # If the frame was not read correctly (e.g., end of video), break the loop
        if not ret:
            break
        
        frame = frame[:, :, ::-1] # opencv reads in BGR, so here we convert from BGR to RGB
        video_rgb.append(frame)

    cap.release()

    video_rgb = np.asarray(video_rgb)
    video_rgb = torch.as_tensor(video_rgb, dtype=torch.float32)

    return video_rgb


def rgb_to_grayscale(video_rgb):
    rgb_to_grayscale_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=video_rgb.device)
    rgb_to_grayscale_weights = rgb_to_grayscale_weights.view(1,1,1,3)

    weighted_video  = video_rgb * rgb_to_grayscale_weights
    video_grayscale = torch.sum(weighted_video, dim=-1)

    return video_grayscale