import os
import warnings
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from ztensor.pipeline import pipeline

def run_fidelity_check(video_path, device, memory_budget, compression_factor, num_threads):
    original_video, encoded_video = pipeline.encode_pipeline(   video_path, 
                                                                device, 
                                                                memory_budget, 
                                                                compression_factor, 
                                                                num_threads
                                                                )
    

    decoded_video = pipeline.decode_pipeline(encoded_video)
    
    original_video = original_video.cpu().numpy().astype(np.uint8)
    decoded_video  = decoded_video.cpu().numpy().astype(np.uint8)


    psnr_values = []
    ssim_values = []

    for i in range(len(original_video)):
        original_frame = original_video[i]
        decoded_frame  = decoded_video[i]

        # In lossless compression, the MSE component of the PSNR metric is always 0, and this triggers a division by zero warning.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            p = psnr(original_frame, decoded_frame, data_range=255)

        psnr_values.append(p)

        s = ssim(original_frame, decoded_frame, channel_axis=2, data_range=255)
        ssim_values.append(s)

    return np.mean(psnr_values), np.mean(ssim_values)


def test_codec_fidelity(args):

    if args.input_video:
        videos = [ args.input_video ]
    else:
        test_dir = "./test_videos/"
        videos   = [ os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.avi'))]


    print(f"{'Video Source':<30} | {'PSNR score':<20} | {'SSIM score':<20}")
    print("-" * 70)


    for video_path in videos:

        avg_psnr, avg_ssim = run_fidelity_check(video_path, 
                                                args.device, 
                                                args.mem, 
                                                args.compression_factor, 
                                                args.threads
                                                )

        if np.isinf(avg_psnr):
            print(f"{os.path.basename(video_path):<30} | {'Lossless':<20} | {avg_ssim:<20.4f}")
        else:
            print(f"{os.path.basename(video_path):<30} | {avg_psnr:<20} | {avg_ssim:<20.4f}")
