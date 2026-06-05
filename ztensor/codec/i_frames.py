import torch
import typing


def select_i_frames(edges_video: torch.Tensor,
                    troi_slices: typing.List[torch.Tensor],
                    max_pframe_chain: int = 48,
                    anchor_window: int = 8) -> torch.Tensor:
    """
    Places one I-frame near the start of every scene (the "anchor"), then inserts
    periodic resets so no P-frame chain runs longer than max_pframe_chain.

    The anchor is the sharpest frame within the first `anchor_window` frames of the
    scene. This (a) gives the new scene a fresh reference near its start, so its
    frames stop predicting across the cut from the previous scene, and (b) skips
    bloom / fade / flash frames automatically, since those have low edge variance.

    Args:
        edges_video:      per-frame Sobel edge maps, shape (T, H, W)
        troi_slices:      list of (start, end) scene bounds, end inclusive
        max_pframe_chain: longest run of P-frames allowed before a forced reset
        anchor_window:    how many frames after a cut to search for the scene anchor
    """
    frame_variances = torch.var(edges_video, dim=(1, 2))

    def sharpest(low: int, high: int) -> int:
        # index of the highest-variance (sharpest) frame in [lo, hi)
        return low + int(torch.argmax(frame_variances[low:high]).item())

    i_frame_indices = {0}  # frame 0 must be an I-frame: nothing precedes it

    for scene in troi_slices:
        start = int(scene[0].item())
        end   = int(scene[1].item())  # inclusive

        # 1) Anchor the scene near its start, on the sharpest non-transition frame.
        anchor = sharpest(start, min(start + anchor_window, end + 1))
        i_frame_indices.add(anchor)

        # 2) Cap drift forward to the end of the scene: no gap > max_pframe_chain,
        #    and each reset is itself the sharpest frame in its window.
        cursor = anchor
        while end - cursor > max_pframe_chain:
            reset = sharpest(cursor + 1, min(cursor + 1 + max_pframe_chain, end + 1))
            i_frame_indices.add(reset)
            cursor = reset

    return torch.as_tensor(sorted(i_frame_indices),
                           device=edges_video.device, dtype=torch.int64)