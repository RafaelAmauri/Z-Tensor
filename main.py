import cv2
import torch
import numpy as np

from ztensor.utils import parser, video
from ztensor.codec import encoder, decoder, i_frames
from ztensor.effects import histogram, blur, edge_detect


args = parser.make_parser().parse_args()
parser.validate_args(args)


if args.encode:
    print("Encoding video...")
    video_rgb         = video.read_video(args.input_video).to(args.device)
    video_grayscale   = video.rgb_to_grayscale(video_rgb)

    blurred_grayscale = blur.blur_video(video_grayscale)
    blurred_histogram = histogram.video_histogram(blurred_grayscale, args.mem)

    video_edges       = edge_detect.sobel(blurred_grayscale)

    troi_slices       = histogram.temporal_region_of_interest(blurred_histogram)
    i_frame_indices   = i_frames.select_i_frames(video_edges, troi_slices)

    encoded_video = encoder.encode_video(video_rgb, i_frame_indices, args.compression_factor, args.threads)

    # Writing encoded video
    with open(f"{args.output_video}.ztensor", "wb") as f:
        f.write(encoded_video)

elif args.decode:
    print("Decoding video...")
    with open(args.input_video, "rb") as f:
        bytes_data = f.read()

    decoded_video = decoder.decode(bytes_data)

    # Writing decoded video
    with open(f"{args.output_video}.raw", "wb") as f:
        decoded_video = decoded_video.cpu().numpy()
        f.write(decoded_video.tobytes())