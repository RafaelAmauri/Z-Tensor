import torch

# Converts patchIds into pixel coordinates.
def patchId2Coords(patchId, block_width, blocks_in_plane_width):
    return torch.as_tensor([patchId % blocks_in_plane_width * block_width, 
                           patchId // blocks_in_plane_width * block_width
                           ], dtype=torch.int32)

# Converts coordinates to patchIds taking into account how many blocks each row in the frame supports. 
def coords2PatchId(coords, block_width, blocks_in_plane_width):
    x, y = coords
    return (y // block_width) * blocks_in_plane_width + (x//block_width)

def sad(block_a, block_b):
    """Sum of absolute differences

    Args:
        block_a (_type_): _description_
        block_b (_type_): _description_

    Returns:
        _type_: _description_
    """
    scores = (block_a - block_b).abs().sum(dim=-1)
    return scores

def block_matching(plane, block_width):
    block_width     = 8
    plane           = plane.to(torch.float32)
    h, w            = plane[0].shape 
    blocks_in_plane_width  = w // block_width
    blocks_in_plane_height = h // block_width

    # These two are different. Both divide the plane into patches with dimensions block_width x block_width, but the
    # first one has a stride of block_width because it is referencing all 8x8 patches that cover the image. TODO: check if we need padding in case the 
    # frame isn't perfectly divisible by block_width x block_width patches!
    unfold_window   = torch.nn.Unfold(kernel_size=(block_width,block_width), stride=block_width)
    unfold_stride_1 = torch.nn.Unfold(kernel_size=(block_width,block_width), stride=1)


    motion_vectors_patches = {}

    for idx in range(1, len(plane[0])):
        # Stores the dx and dy motion vectors and the residuals that will be used for reconstruction.
        # This is what gets returned by the function and will be serialized.
        motion_vectors_patches[idx] = []

        plane0 = plane[idx-1]
        plane1 = plane[idx]

        blocks_plane0 = unfold_stride_1(plane0.unsqueeze(0).unsqueeze(0)) # (1, 1 * ∏(kernel_size), totalNumberOfBlocks). Since we're dealing with individual planes, C=1. And B = 1 too.
        blocks_plane0 = blocks_plane0.squeeze(0).permute(1, 0)            # (B, C * ∏(kernel_size), totalNumberOfBlocks) -> (C * ∏(kernel_size), totalNumberOfBlocks) -> (totalNumberOfBlocks, ∏(kernel_size))
        
        blocks_plane1 = unfold_window(plane1.unsqueeze(0).unsqueeze(0)) # (1, 1 * ∏(kernel_size), totalNumberOfBlocks). Again, same dimensions. The number of blocks now is different because it uses a larger stride of <block_width>
        blocks_plane1 = blocks_plane1.squeeze(0).permute(1, 0)          # (totalNumberOfBlocks, ∏(kernel_size))

        print(f"Plane shape: {plane.shape}, Blocks Plane0 shape: {blocks_plane0.shape}, Blocks Plane1 shape: {blocks_plane1.shape}")
        print(f"Blocks in plane width: {blocks_in_plane_width}, Blocks in plane height: {blocks_in_plane_height}")
        # Loop over every block_width x block_width patch with stride block_width
        for patchId in range(len(blocks_plane1)):
            # pixel coordinates are always the TOP LEFT CORNER pixel! It's a simple 2-digit tuple because ince all blocks have width <block_width> 
            # and also height <block_width>, we can always get the full patch by adding + <block_width> to x and y.
            current_coords                     = patchId2Coords(patchId, block_width, blocks_in_plane_width)
            current_x, current_y               = current_coords

            # For each patch, we explore the <block_width> neighborhood and store the patchIds of each neighboring patch.
            # Note that these patchIds refer to the patches with stride 1! Not the ones with stride <block_width>!
            # The idea is to slide the window centered in the current coordinates in all neighboring directions moving 1 pixel at a time,
            # and then store the IDs of these neighboring patches. They will later be compared to the original patch centered in the current
            # coordinates to find the neighboring 8x8 patch that is the most similar to the current one.
            patchIds_compare_to_in_prev_frame  = []
            print(f"Current coords: {current_coords}, Patch: {patchId}")
            for i in range(block_width):
                for j in range(block_width):
                    # The rightmost pixel index has to be less than w, and the bottommost one has to be less than h.
                    # < instead of <= because we're using indexes, and if the rightmost index == w, this will cause an out of bounds error
                    if current_x+i+block_width < w and current_y+j+block_width < h:
                        patchIds_compare_to_in_prev_frame.append(coords2PatchId([current_x+i, current_y+j], 1, w-(block_width))) # This -(block_width) in w-(block_width) is necessary because we currently don't support
                                                                                                                                 # videos with dimensions that aren't perfectly divisible by block_width x block_width

                    # Skip patches that start at a negative position and also skip the ones where i + j = 0, because that one was already added in the line above. It's a base one that always gets added.
                    if (current_x - i >= 0 and current_y - j >= 0) and (i + j != 0):
                        patchIds_compare_to_in_prev_frame.append(coords2PatchId([current_x-i, current_y-j], 1, w-(block_width)))


            patchIds_compare_to_in_prev_frame = torch.as_tensor(patchIds_compare_to_in_prev_frame, dtype=torch.int32, device=plane0.device)

            print(f"Patch Ids to compare: {patchIds_compare_to_in_prev_frame}")
            sad_scores    = sad(blocks_plane1[patchId], blocks_plane0[patchIds_compare_to_in_prev_frame])

            # Get coords of the candidate patch with lowest SAD score, which is the best patch (maybe change variable name to candidate_patches?).
            best_patchid            = patchIds_compare_to_in_prev_frame[torch.argmin(sad_scores)] 
            coords_patch_lowest_sad = patchId2Coords(best_patchid, 1, w-(block_width))

            # these are the motion vectors that tell us how to get to the candidate patch starting from the current patch.
            # since the block sizes are all equal, we can use the motion vectors to tell us how to reconstruct the current frame's patch
            # using just relative motion vectors from the previous frame.
            dx, dy  = coords_patch_lowest_sad - current_coords
            # prev are the best patch's pixel values in the previous frame
            prev    = plane0[current_y + dy: current_y + dy + block_width, current_x + dx: current_x + dx + block_width].flatten()
            
            print(f"Patch Id lowest SAD: {best_patchid}")
            print(current_y + dy, current_y + dy + block_width, current_x + dx, current_x + dx + block_width, prev.shape)

            # Get the residue between the current patch and the best patch in the previous frame and
            # store the motion vector and the residue
            residue = blocks_plane1[patchId] - prev
            motion_vectors_patches[idx].append([dx, dy, residue])

            # Done? I think? Just serialize motion_vectors_patches and that should be it. For serialization, the format matters. Maybe do something like:
            # <residual grid dimension> uint8, something like 8. Since it's a square, just 1 number does it and then we reconstruct 8x8 patches
            # <num frames> int32
            # <dx, dy> int8
            # <residue> uint8. Has grid_dimension * grid_dimension elements. 


        # Mock decode
        for patch_idx, info in enumerate(motion_vectors_patches[idx]):
            dx, dy, residue      = info
            # Get coords of the current patch
            current_x, current_y = patchId2Coords(patch_idx,  block_width, blocks_in_plane_width)
            # Slice the previous frame by moving the current coords using the motion vectors
            patch_prev_frame     = plane0[current_y + dy: current_y + dy + block_width, current_x + dx: current_x + dx + block_width].flatten()

            # Sanity check if the previous frame's patch + residue == the current patch
            print(torch.equal(patch_prev_frame + residue, blocks_plane1[patch_idx]))

    
        raise NotImplementedError