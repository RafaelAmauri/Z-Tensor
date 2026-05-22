import torch
import typing
import numpy as np


def serialize_header(pixel_format: str, 
                    quantization_parameter: int,
                    i_frame_indices: torch.Tensor,
                    block_width: int,
                    num_frames: int,
                    num_planes: int
                     ) -> bytes:
    """Constructs the header for the .ztensor file format. 

    Args:
        pixel_format (str): The pixel format. Either RGB3, I422 or I420
        quantization_parameter (int): The parameter that stores which quantization option was used for the video
        i_frame_indices (torch.Tensor): The indices of the i_frames
        block_width (int): The with of the motion block. They're square, so no need to pass the height as it is the same as the width. 
        num_frames (int): The number of frames in the video
        num_planes (int): The number of planes in the video

    Returns:
        bytes: The header
    """
    header  = bytes()

    header += pixel_format.encode('ascii')
    header += quantization_parameter.to_bytes(1, signed=False)            # uint8  value for the quantization parameter. 1 = Linear (less aggresive)
    header += len(i_frame_indices).to_bytes(4, signed=False)              # uint32 the number of i-frames in the video
    header += i_frame_indices.cpu().numpy().astype(np.uint32).tobytes()   # uint32 indices of the i-frames
    header += block_width.to_bytes(4, signed=False)                       # uint32  the size of the motion blocks
    
    header += num_planes.to_bytes(4,  signed=False)                       # uint32 the number of planes in the video.
    header += num_frames.to_bytes(4,  signed=False)                       # uint32 the number of frames

    return header


def serialize_payload(motion_blocks, plane, i_frame_indices, original_plane_h, original_plane_w):
    payload = bytes()

    payload += original_plane_w.to_bytes(4,  signed=False)             # uint32 the height of the video
    payload += original_plane_h.to_bytes(4,  signed=False)             # uint32 the width of the video

    for frame_idx, frame in enumerate(plane):
        # If frame is an I-frame, store it as-is
        if frame_idx in i_frame_indices:
            payload += frame.to(torch.uint8).cpu().numpy().tobytes()

        # If not, store its motion blocks
        else:
            block_movements = motion_blocks[frame_idx]
            payload += len(block_movements).to_bytes(4, signed=False) # uint32 the number of blocks in the frame
            for block in block_movements:
                dx, dy, residual = block
                
                payload += int(dx).to_bytes(1, signed=True)  # int8 the block's horizontal motion. int8 is fine because the 
                                                             # search_window parameter is small (usually < 10). So this value is always in 
                                                             # the [-10, 10] interval

                payload += int(dy).to_bytes(1, signed=True)  # uint8 the block's vertical motion.
                payload += residual.to(torch.uint8).cpu().numpy().tobytes() # uint8 the stored residuals

    return payload
