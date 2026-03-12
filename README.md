# Z-Tensor: Custom-Made Hardware-Accelerated Video Codec!

### Z-Tensor is not a wrapper around FFmpeg :)

### It is a completely custom, mathematically lossless video compressor built from scratch using PyTorch Tensors and Zstandard compression. It avoids CPU bottlenecks by keeping all the heavy spatio-temporal math strictly on the GPU, achieving good compression ratios without losing any data.

### Z-Tensor gets its name from its two main components: "Z" from Zstandard, a modern state-of-the-art compression algorithm, and "Tensor" from all of the code being GPU accelerated using PyTorch Tensors!


## 🚀 Features

*   **Scene Change Detection:** Uses Z-scores on color histograms to detect sudden scene changes. Each scene change is used to outline regions where I-Frames should be used to maximize compression gains.
*   **Hardware-Accelerated:** Uses the GPU to perform motion-estimation, select I-Frames and calculate P-Frame deltas.
*   **Respects VRAM budgets:** Not enough VRAM? No problem! The code allows the user to specify how much VRAM they have available, and dynamically batches operations to respect that limit!
*   **Choose between CPU and GPU!** The code allows the user to easily select if they want to use the GPU or the CPU for encoding / decoding.
## 🛠️ Installation

```bash
git clone https://github.com/RafaelAmauri/Z-Tensor.git
cd Z-Tensor
pip install -r requirements.txt
```

## 📖 Usage

### Encoding
Encode a standard video file into the `.ztensor` format using the GPU with a memory limit of 4GB:
```bash
python main.py -i input.avi -n processed_video -e -mem 4G -device 0
```
Encode a video file with a compression factor of 18 on the CPU using 16 threads and 3 GB of RAM:
```bash
python main.py -i input.avi -n processed_video -e -cf 18 -device cpu --threads 16 -mem 3GB
```

### Decoding
Decompress a `.ztensor` file back into a playable raw stream:
```bash
python main.py -i processed_video.ztensor -n output_video -d
```

### Tests
Test the codec's quality on the uncompressed videos in test_videos:
```bash
python main.py --test
```