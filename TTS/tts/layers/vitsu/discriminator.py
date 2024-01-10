import torch
import torch.nn.functional as F
from torch import nn

from TTS.vocoder.models.univnet_discriminator import MultiResSpecDiscriminator
from TTS.vocoder.models.hifigan_discriminator import DiscriminatorP


class MultiPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN Multi-Period Discriminator (MPD)
    Wrapper for the `PeriodDiscriminator` to apply it in different periods.
    Periods are suggested to be prime numbers to reduce the overlap between each discriminator.
    """

    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm=False):
        super().__init__()
        self.discriminators = nn.ModuleList()
        self.discriminators.extend([DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods])

    def forward(self, x):
        """
        Args:
            x (Tensor): input waveform.

        Returns:
        [List[Tensor]]: list of scores from each discriminator.
            [List[List[Tensor]]]: list of list of features from each discriminator's each convolutional layer.

        Shapes:
            x: [B, 1, T]
        """
        scores = []
        feats = []
        for _, d in enumerate(self.discriminators):
            score, feat = d(x)
            scores.append(score)
            feats.append(feat)
        return scores, feats


class VitsuDiscriminator(nn.Module):
    """VITS+Univnet (VITSU) discriminator wrapping
       - a stack of Univnet Scale Discriminators and
       - a stack of HiFiGAN Period Discriminators.

    ::
        waveform -> MultiResSpecDiscriminator() -> scores_sd, feats_sd --> append() -> scores, feats
               |--> MultiPeriodDiscriminator() -> scores_mpd, feats_mpd ^

    Args:
        periods (list)
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
        fft_sizes
        hop_sizes
        win_lengths
    """

    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm=False,
                 fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
        super().__init__()
        self.msd = MultiResSpecDiscriminator(fft_sizes, hop_sizes, win_lengths, use_spectral_norm)
        self.mpd = MultiPeriodDiscriminator(periods, use_spectral_norm)


    def forward(self, x, x_hat=None):
        """
        Args:
            x (Tensor): ground truth waveform.
            x_hat (Tensor): predicted waveform.

        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of list of features from each layers of each discriminator.
        """
        scores_msd, feats_msd = self.msd(x)
        scores_mpd, feats_mpd = self.mpd(x)
        hat_scores_msd, hat_feats_msd = self.msd(x_hat) if x_hat is not None else (None, None)
        hat_scores_mpd, hat_feats_mpd = self.mpd(x_hat) if x_hat is not None else (None, None)
        if x_hat is not None:
            return scores_msd+scores_mpd, feats_msd+feats_mpd, hat_scores_msd+hat_scores_mpd, hat_feats_msd+hat_feats_mpd
        else:
            return scores_msd+scores_mpd, feats_msd+feats_mpd, None, None
