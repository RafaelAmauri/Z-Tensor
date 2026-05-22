import torch


def pad_plane(plane: torch.Tensor, block_width: int) -> torch.Tensor:
    """Pads the current plane so it is perfectly covered by 
    block_width x block_width blocks. Dimensions that are already
    perfectly divisible by block_width are ignored

    Args:
        plane (torch.Tensor): A (T, H, W) plane
        block_width (int): The block's width. Since it's a perfect square, the height is the same as the width

    Returns:
        torch.Tensor: The padded plane
    """
    _, h, w = plane.shape

    pad_h = (block_width - h % block_width) % block_width
    pad_w = (block_width - w % block_width) % block_width

    # Pad only the right and the bottom of the plane.
    # This makes the cropping easier when decoding. 
    # just do plane = plane[ : H, : W] !
    plane = torch.nn.functional.pad(plane, (0, pad_w, 0, pad_h), 'reflect')

    return plane
