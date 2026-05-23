import torch
import typing
import zstandard

import numpy as np
import torch.nn.functional as F

from ztensor.effects import chroma, quantization
from ztensor.codec import serialization


def decode_video(compressed_bytes: bytes, device: torch.device) -> torch.Tensor:
    planes, i_frame_indices, pixel_format = serialization.deserialize_payload(compressed_bytes, device)

    print(planes)
    raise NotImplementedError
    scene_boundaries = i_frame_indices.tolist()
    scene_boundaries.append(len(planes[0])) # Append the idx of the last frame to the list to use it as the scene_end variable for the last iteration of the loop below

    planes_decoded = []
    for plane in planes:
        for i in range(len(scene_boundaries)-1):
            scene_start = scene_boundaries[i]
            scene_end   = scene_boundaries[i+1]

            plane[scene_start : scene_end] = torch.cumsum(plane[scene_start : scene_end], dim=0)

        planes_decoded.append(plane)
    
    if pixel_format in ['I422', 'I420']: # This means the video is chroma subsampled, so we need to interpolate the U and V channels to be the same dimension as the Y channel
        y_tensor   = planes_decoded[0].unsqueeze(1).float()
        target_res = (y_tensor.shape[2], y_tensor.shape[3])

        u_tensor   = planes_decoded[1].unsqueeze(1).float()
        u_upscaled = F.interpolate(u_tensor, size=(target_res), mode='bilinear', align_corners=False)

        v_tensor   = planes_decoded[2].unsqueeze(1).float()
        v_upscaled = F.interpolate(v_tensor, size=(target_res), mode='bilinear', align_corners=False)

        video = torch.cat([y_tensor, u_upscaled, v_upscaled], dim=1)
        video = video.permute(0, 2, 3, 1)

        video = chroma.yuv2bgr(video)


    elif pixel_format == 'RGB3': # This means the video is RGB, so we just concatenate the channels along the last axis to form the RGB video.
        planes_decoded[0] = planes_decoded[0].unsqueeze(-1)
        planes_decoded[1] = planes_decoded[1].unsqueeze(-1)
        planes_decoded[2] = planes_decoded[2].unsqueeze(-1)

        video = torch.cat([planes_decoded[0], planes_decoded[1], planes_decoded[2]], dim=-1)
    
    else:
        raise ValueError(f"Unsupported format: {pixel_format}")

    video = video.clip(0,255).to(torch.uint8)

    return video

