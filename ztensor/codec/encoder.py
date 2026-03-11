import torch
import zstandard

import numpy as np


def encode_video(video, i_frame_indices, compression_factor, num_threads):

    # Cast to int16 to calculate P-frames. Since the P-frames are only integer values, int16 will do just fine.
    video = video.to(torch.int16)

    # Calculate the P-frames
    p_frames = torch.diff(video, dim=0)
    # Because torch.diff reduces the dimension in 1, we squeeze back an empty frame back into the P-frames vector so it mainstains the same
    # shape as the video
    p_frames = torch.cat([torch.empty_like(video[0]).unsqueeze(0), p_frames])
   
    # tells me which frames are P-frames
    p_frame_mask                  = torch.full(size=(video.shape[0],), fill_value=True)
    p_frame_mask[i_frame_indices] = False
    
    # Substitutes the frames in the p_frame_mask for the P-frames calculated with diff
    video[p_frame_mask] = p_frames[p_frame_mask]

    array_bytes_compressed = compress_video(video, i_frame_indices, compression_factor, num_threads)
    
    return array_bytes_compressed


def compress_video(video, i_frame_indices, compression_factor, num_threads):
    video_cpu = video.cpu().numpy()

    # Get the info needed for the header.
    # This info is crucial for decoding the video
    header_video_shape   = np.array(video_cpu.shape,      dtype=np.int32).tobytes()
    header_i_frame_count = np.array(len(i_frame_indices), dtype=np.int32).tobytes()
    header_i_frame_data  = i_frame_indices.cpu().numpy().astype(np.int32).tobytes()

    # Convert video to bytes and attach all the header information to it
    raw_video_bytes = video_cpu.tobytes()
    full_payload    = header_video_shape + header_i_frame_count + header_i_frame_data + raw_video_bytes

    # The compression step has to run on GPU.
    compressor = zstandard.ZstdCompressor(level=compression_factor, threads=num_threads)
    array_bytes_compressed = compressor.compress(full_payload)

    return array_bytes_compressed