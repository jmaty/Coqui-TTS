import math

import torch
from torch import nn
from torch.nn import functional as F

from TTS.tts.layers.generic.normalization import LayerNorm, LayerNorm2
from TTS.tts.layers.glow_tts.transformer import FeedForwardNetwork
from TTS.tts.layers.vits.s4 import S4

class S4Layer(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.2, measure="legs", transposed=True,
                 lr=0.001, mode="nplr", n_ssm=None):
        """ Layer based on S4 model.

        https://github.com/state-spaces/s4/tree/main/models/s4

        Args:
            d_model (or H): (from S4 docstring:) Number of independent 
                SSM copies; controls the size of the model. Should be 
                same as hidden_size of the model.
            d_state (or N): (from S4 docstring:) State size (dimensionality 
                of parameters A, B, C). Generally shouldn't need to be 
                adjusted and doesn't affect speed much.
            dropout: (from S4 docstring:) standard dropout argument
            measure: Options for initialization of (A, B).
                For NPLR mode, recommendations are "legs", "fout", "hippo" (combination of both).
                For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin)
            transposed: (from S4 docstring:) choose backbone axis ordering 
                of (B, L, H) (if False) or (B, H, L) (if True) 
                [B=batch size, L=sequence length, H=hidden dimension]
            lr: (from S4 docstring:) Passing in a number (e.g. 0.001) sets 
                attributes of SSM parameers (A, B, dt). A custom optimizer 
                hook is needed to configure the optimizer to set the 
                learning rates appropriately for these parameters.
            mode: Which kernel algorithm to use.
                'nplr' is the full S4 model;
                'diag' is the simpler S4D;
                'slow' is a dense version for testing
            n_ssm: Number of independent trainable (A, B) SSMs, e.g.
                n_ssm=1 means all A/B parameters are tied across the H different instantiations of C;
                n_ssm=None means all H SSMs are completely independent.
                Generally, changing this option can save parameters but doesn't affect performance or speed much.
                This parameter must divide H
        """
        super().__init__()
        self.s4 = S4(
            d_model=d_model,
            d_state=d_state,
            dropout=dropout,
            measure=measure,
            transposed=transposed,
            lr=lr,
            mode=mode,
            n_ssm=n_ssm,
        )

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor]:
        # print(f"hidden_states: {hidden_states.shape}")
        output, _ = self.s4(hidden_states)
        # print(f"output: {output.shape}")
        return output


class RelativePositionTransformerS4(nn.Module):
    """Transformer with Relative Potional Encoding.
    https://arxiv.org/abs/1803.02155

    Args:
        in_channels (int): number of channels of the input tensor.
        out_chanels (int): number of channels of the output tensor.
        hidden_channels (int): model hidden channels.
        d_state (or N): (from S4 docstring:) State size (dimensionality
            of parameters A, B, C). Generally shouldn't need to be adjusted and doesn't affect speed much.
        measure: Options for initialization of (A, B).
            For NPLR mode, recommendations are "legs", "fout", "hippo" (combination of both).
            For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin)
        lr: (from S4 docstring:) Passing in a number (e.g. 0.001) sets 
            attributes of SSM parameers (A, B, dt). A custom optimizer 
            hook is needed to configure the optimizer to set the 
            learning rates appropriately for these parameters.
        mode: Which kernel algorithm to use.
            'nplr' is the full S4 model;
            'diag' is the simpler S4D;
            'slow' is a dense version for testing
        n_ssm: Number of independent trainable (A, B) SSMs, e.g.
            n_ssm=1 means all A/B parameters are tied across the H different instantiations of C;
            n_ssm=None means all H SSMs are completely independent.
            Generally, changing this option can save parameters but doesn't affect performance or speed much.
            This parameter must divide H
        hidden_channels_ffn (int): hidden channels of FeedForwardNetwork.
        num_layers (int): number of transformer layers.
        kernel_size (int, optional): kernel size of feed-forward inner layers. Defaults to 1.
        dropout_p (float, optional): dropout rate for self-attention/S4 and feed-forward inner layers_per_stack. Defaults to 0.
        layer_norm_type (str, optional): type "1" uses torch tensor operations and type "2" uses torch layer_norm
            primitive. Use type "2", type "1: is for backward compat. Defaults to "1".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        d_state: int,
        measure: str,
        lr: float,
        mode: str,
        n_ssm: int,
        hidden_channels_ffn: int,
        num_layers: int,
        kernel_size=1,
        dropout_p=0.0,
        layer_norm_type: str = "1",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_channels_ffn = hidden_channels_ffn
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(dropout_p)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for idx in range(self.num_layers):
            self.attn_layers.append(
                S4Layer(
                    d_model=hidden_channels if idx != 0 else in_channels,
                    d_state=d_state,
                    dropout=dropout_p,
                    measure=measure,
                    transposed=True,
                    lr=lr,
                    mode=mode,
                    n_ssm=n_ssm,
                )
            )
            if layer_norm_type == "1":
                self.norm_layers_1.append(LayerNorm(hidden_channels))
            elif layer_norm_type == "2":
                self.norm_layers_1.append(LayerNorm2(hidden_channels))
            else:
                raise ValueError(" [!] Unknown layer norm type")

            if hidden_channels != out_channels and (idx + 1) == self.num_layers:
                self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

            self.ffn_layers.append(
                FeedForwardNetwork(
                    hidden_channels,
                    hidden_channels if (idx + 1) != self.num_layers else out_channels,
                    hidden_channels_ffn,
                    kernel_size,
                    dropout_p=dropout_p,
                )
            )

            if layer_norm_type == "1":
                self.norm_layers_2.append(LayerNorm(hidden_channels if (idx + 1) != self.num_layers else out_channels))
            elif layer_norm_type == "2":
                self.norm_layers_2.append(LayerNorm2(hidden_channels if (idx + 1) != self.num_layers else out_channels))
            else:
                raise ValueError(" [!] Unknown layer norm type")

    def forward(self, x, x_mask):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
        """
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.num_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.dropout(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.dropout(y)

            if (i + 1) == self.num_layers and hasattr(self, "proj"):
                x = self.proj(x)

            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x
