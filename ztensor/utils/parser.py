import os
import psutil
import torch

from argparse import ArgumentParser

def make_parser():
    parser = ArgumentParser(description='Define the program parameters')

    parser.add_argument('-i', '--input-video', type=str,
                        help="The input video")
    
    parser.add_argument('-n', '--name', type=str,
                        help="The name of the processed video.")
    
    parser.add_argument('-e', '--encode', action='store_true', default=False,
                        help="Encode the input video")
    
    parser.add_argument('-d', '--decode', action='store_true', default=False,
                        help="Decode the input video back into uncompressed video")
    
    parser.add_argument('--test', action='store_true', default=False,
                        help="Run encode/decode and calculate PSNR and SSIM metrics.")
    
    parser.add_argument('-cf', '--compression-factor', type=int, default=16,
                        help="The compression factor for zstandard. Higher values lead to better compression, but increase encode time. Accepted values go from 1 to 20.")
    
    parser.add_argument('-t', '--threads', type=int, default=-1,
                        help="The number of threads zstandard is allowed to use for compression. The default uses all your CPU threads.")
    
    parser.add_argument('-mem', type=str, default='2G',
                        help="The amount of memory that the motion estimation algorithm is allowed to use. Use G for GB and M for MB. The default is 2G. The codec respects this memory limit regardless of the value for \'-device\'")

    parser.add_argument('-device', type=str, default='0',
                        help="Use \"cpu\" to run on the CPU and numbers to select which GPU to use.")
    

    return parser


def validate_args(args) -> None:

    if sum([args.encode, args.decode, args.test]) != 1:
        raise ValueError(f"Choose exactly one operation: --encode, --decode, or --test.")
    
    if not (1 <= args.compression_factor <= 20):
        raise ValueError(f"The compression factor has to be between 1 and 20.")

    if args.encode or args.decode:
        if not args.input_video:
            raise ValueError(f"Argument -i/--input-video is required for encoding or decoding")
        if not args.name:
            raise ValueError(f"Argument -n/--name is required for encoding or decoding")
    
    if args.input_video and not os.path.isfile(args.input_video):
        raise ValueError(f"Input video \'{args.input_video}\' does not exist!")

    if args.threads > os.cpu_count():
        args.threads = -1 # -1 means use all threads.
    elif args.threads < 1:
        args.threads = 1


    max_mem = get_max_mem(args.device)
    if max_mem == 0:
        args.device = "cpu"
        max_mem     = psutil.virtual_memory().total
    else:
        args.device = int(args.device)

    mem_str = args.mem.strip().upper()
    try:
        if mem_str.endswith('G'):
            mem_bytes = float(mem_str[ : -1]) * (1024**3) 
        elif mem_str.endswith('M'):
            mem_bytes = float(mem_str[ : -1]) * (1024**2)
        else:
            raise ValueError(f"Invalid format: \'{args.mem}\'. Use \'G\' or \'M\' (example: \'2G\', \'500M\')")
    except ValueError:
        raise ValueError(f"Could not parse memory amount from \'{args.mem}\'")
    
    if mem_bytes > 0.8 * max_mem:
        print(f"Requested memory amount leaves almost no headroom for your PC. Capping the usage at 80% of the maximum memory available")
        mem_bytes = 0.8 * max_mem

    args.mem = int(mem_bytes)


def get_max_mem(device_id):
    if device_id == 'cpu':
        return 0
    else:
        device_id = int(device_id)
        if torch.cuda.is_available():
            total_memory_bytes = torch.cuda.get_device_properties(device_id).total_memory
            return total_memory_bytes

        else:
            print(f"No CUDA-enabled GPU detect. Falling back to CPU")
            return 0