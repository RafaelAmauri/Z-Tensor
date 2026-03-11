from ztensor.utils import parser, video
from ztensor.codec import encoder, decoder, i_frames
from ztensor.effects import histogram, blur, edge_detect, bgr2yuv

def encode_pipeline(input_path, device, memory_budget, compression_factor, num_threads):
    video_bgr         = video.read_video(input_path).to(device)
    video_grayscale   = video.bgr_to_grayscale(video_bgr)

    blurred_grayscale = blur.blur_video(video_grayscale)
    blurred_histogram = histogram.video_histogram(blurred_grayscale, memory_budget)

    video_edges       = edge_detect.sobel(blurred_grayscale)

    troi_slices       = histogram.temporal_region_of_interest(blurred_histogram)
    i_frame_indices   = i_frames.select_i_frames(video_edges, troi_slices)

    encoded_video = encoder.encode_video(video_bgr, i_frame_indices, compression_factor, num_threads)

    return video_bgr, encoded_video


def decode_pipeline(bytes_data):

    decoded_video = decoder.decode(bytes_data)

    return decoded_video

