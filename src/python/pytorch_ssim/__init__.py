# Author: Po-Hsun Su (https://github.com/Po-Hsun-Su/pytorch-ssim)
# Modified by: Bingyao Huang
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    sigma = 1.5 # in matlab ssim, it is radius
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    # use replicate padding to be consistent with matlab SSIM function
    padSize = window_size // 2
    img1 = F.pad(img1, (padSize, padSize, padSize, padSize), mode='replicate')
    img2 = F.pad(img2, (padSize, padSize, padSize, padSize), mode='replicate')
    mu1 = F.conv2d(img1, window, groups=channel) # conv with each channel using the same 2D Gaussian filter,
    mu2 = F.conv2d(img2, window, groups=channel)

    # mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    # mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    # sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    # sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        if mask is not None:
            return ssim_map[mask].mean()
        else:
            return ssim_map.mean()
    else:
        if mask is not None:
            return ssim_map[mask].mean(1).mean(1).mean(1)
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, mask=None):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, mask)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
