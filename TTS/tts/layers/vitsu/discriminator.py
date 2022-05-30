import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import Conv1d

# from TTS.vocoder.models.hifigan_discriminator import DiscriminatorP, MultiPeriodDiscriminator
from TTS.utils.audio import TorchSTFT
from TTS.vocoder.models.hifigan_discriminator import DiscriminatorP

LRELU_SLOPE = 0.1


# class DiscriminatorS(torch.nn.Module):
#     """HiFiGAN Scale Discriminator. Channel sizes are different from the original HiFiGAN.

#     Args:
#         use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
#     """

#     def __init__(self, use_spectral_norm=False):
#         super().__init__()
#         norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
#         self.convs = nn.ModuleList(
#             [
#                 norm_f(Conv1d(1, 16, 15, 1, padding=7)),
#                 norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
#                 norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
#                 norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
#                 norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
#                 norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
#             ]
#         )
#         self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

#     def forward(self, x):
#         """
#         Args:
#             x (Tensor): input waveform.

#         Returns:
#             Tensor: discriminator scores.
#             List[Tensor]: list of features from the convolutiona layers.
#         """
#         feat = []
#         for l in self.convs:
#             x = l(x)
#             x = torch.nn.functional.leaky_relu(x, 0.1)
#             feat.append(x)
#         x = self.conv_post(x)
#         feat.append(x)
#         x = torch.flatten(x, 1, -1)
#         return x, feat


class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, fft_size=1024, hop_length=120, win_length=600, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.weight_norm if use_spectral_norm is False else nn.utils.spectral_norm
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft = TorchSTFT(fft_size, hop_length, win_length)
        self.discriminators = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ]
        )

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):

        fmap = []
        with torch.no_grad():
            y = y.squeeze(1)
            y = self.stft(y)
        y = y.unsqueeze(1)
        for _, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap


class VitsuDiscriminator(nn.Module):
    """VITS+Univnet (VITSU) discriminator wrapping one Scale Discriminator and a stack of Period Discriminator.

    ::
        waveform -> ScaleDiscriminator() -> scores_sd, feats_sd --> append() -> scores, feats
               |--> MultiPeriodDiscriminator() -> scores_mpd, feats_mpd ^

    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        periods = [2, 3, 5, 7, 11]
        fft_sizes = [1024, 2048, 512]
        hop_sizes = [120, 240, 50]
        win_lengths = [600, 1200, 240]
        window = "hann_window"

        self.nets = nn.ModuleList(
            [
                SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
                SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
                SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window),
            ]
        )
        self.nets.extend([DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods])

    def forward(self, x, x_hat=None):
        """
        Args:
            x (Tensor): ground truth waveform.
            x_hat (Tensor): predicted waveform.

        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of list of features from each layers of each discriminator.
        """
        x_scores = []
        x_hat_scores = [] if x_hat is not None else None
        x_feats = []
        x_hat_feats = [] if x_hat is not None else None
        for net in self.nets:
            x_score, x_feat = net(x)
            x_scores.append(x_score)
            x_feats.append(x_feat)
            if x_hat is not None:
                x_hat_score, x_hat_feat = net(x_hat)
                x_hat_scores.append(x_hat_score)
                x_hat_feats.append(x_hat_feat)
        return x_scores, x_feats, x_hat_scores, x_hat_feats
