import torch

def bgr2yuv(bgr_video):
    '''
    https://www.computerlanguage.com/results.php?definition=YUV%2FRGB+conversion+formulas

    From RGB to YUV

    Y = 0.299R + 0.587G + 0.114B
    U = 0.492 (B-Y)
    V = 0.877 (R-Y)

    '''
    bgr_weights = torch.Tensor([0.114, 0.587, 0.299]).to(bgr_video.device)
    Y = torch.sum(bgr_video * bgr_weights, dim=-1).unsqueeze(-1)
    U = (0.492 * (bgr_video[:, :, :, 0].unsqueeze(-1) - Y))
    V = (0.877 * (bgr_video[:, :, :, 2].unsqueeze(-1) - Y))
    
    yuv_video = torch.cat([Y, U, V], dim=-1)

    return yuv_video


def yuv2bgr(yuv_video):
    '''
    From YUV to RGB

    R = Y + 1.140V
    G = Y - 0.395U - 0.581V
    B = Y + 2.032U
    '''
    R = yuv_video[:,:,:, 0] + (1.140 * yuv_video[:,:,:, 2])
    G = yuv_video[:,:,:, 0] - (0.395 * yuv_video[:,:,:, 1]) - (0.581 * yuv_video[:,:,:, 2])
    B = yuv_video[:,:,:, 0] + (2.032 * yuv_video[:,:,:, 1])

    R = R.unsqueeze(-1)
    G = G.unsqueeze(-1)
    B = B.unsqueeze(-1)

    bgr_video = torch.cat([B, G, R], dim=-1)

    return bgr_video