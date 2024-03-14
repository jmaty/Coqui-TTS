
from typing import Dict,Tuple,List

import numpy as np
import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast

from coqpit import Coqpit

from TTS.tts.models.vits import Vits,wav_to_mel
from TTS.tts.utils.helpers import segment, sequence_mask
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio.processor import AudioProcessor

class VitsHDS(Vits):


    def __init__(self, config:Coqpit, ap:AudioProcessor = None, tokenizer:TTSTokenizer = None, **kws):
        """ Custom VITS constructor. Add additional parameters as required.
        """
        super().__init__(config, ap = ap, tokenizer = tokenizer)


    def train_step(self, batch: dict, criterion: List[str], optimizer_idx: int) -> Tuple[Dict, Dict]:
        """Perform a single training step. Run the model forward pass and compute losses.

        Args:
            batch (Dict): Input tensors.
            criterion (List): Loss layer designed for the model.
            optimizer_idx (int): Index of optimizer to use. 0 for the generator and 1 for the discriminator networks.

        Returns:
            Tuple[Dict, Dict]: Model outputs and computed losses.
        """

        print('!!!! Volám DUMMY implementaci train_step() - vrací nesmyslnou hodnotu loss.')

        # Build completely meaningless loss value
        outputs = self.forward(batch["tokens"], batch["token_lens"], batch["spec"], batch["spec_lens"], batch["waveform"])
        loss = torch.mean(outputs["model_outputs"])

        # Get the minimum of the required data
        return {'waveform' : batch["waveform"],
                'model_outputs' : batch["waveform"],
                'alignments' : outputs["alignments"],
               },\
               {"loss" : loss,
               }

