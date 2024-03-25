
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


    # --------------
    def __init__(self, config:Coqpit, ap:AudioProcessor = None, tokenizer:TTSTokenizer = None, **kws):
        """ Custom VITS constructor. Add additional parameters as required.
        """
        super().__init__(config, ap = ap, tokenizer = tokenizer)

    # --------------
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


        # Discriminator
        if optimizer_idx == 0 :

           # Synthesize
           # NOTE: may be cached for the future use
           outputs = self.forward(batch["tokens"], batch["token_lens"], batch["spec"], batch["spec_lens"], batch["waveform"])

           # Generated and real speech segments
           speech_gener = outputs['model_outputs']
           speech_real  = outputs['waveform_seg']

           # Use discriminator
           # Detach the generated speech, as we do not want update weights of the generator
           scores_disc_fake,_,scores_disc_real,_ = self.disc(speech_gener.detach(), speech_real)

           # Compute the loss. This loss will affect the (discriminator) DNN, as it uses data PROVIDED by it (outputs of the discriminator).
           loss  = self.sample_loss1(scores_disc_fake, scores_disc_real)
           loss *= 2 # A weight

           # Get the minimum of the required data
           return {'waveform_seg' : batch["waveform"],
                   'model_outputs' : batch["waveform"],
                   'alignments' : outputs["alignments"],
                  },\
                  {"loss" : loss,
                  }


        # Generator
        else :

           # Synthesize
           outputs = self.forward(batch["tokens"], batch["token_lens"], batch["spec"], batch["spec_lens"], batch["waveform"])

           # Generated and real speech segments
           speech_gener = outputs['model_outputs']
           speech_real  = outputs['waveform_seg']

           # Compute the loss. This loss will affect the DNN, as it uses data PROVIDED by the DNN (generated speech samples).
           loss  = self.sample_loss2(speech_real, speech_gener)
           loss *= 5 # A weight

           # Get the minimum of the required data
           return {'waveform_seg' : batch["waveform"],
                   'model_outputs' : batch["waveform"],
                   'alignments' : outputs["alignments"],
                  },\
                  {"loss" : loss,
                  }


    # --------------
    def sample_loss1(self, scores_disc_fake:torch.tensor, scores_disc_real:torch.tensor) -> float:
        """Sample loss implementation. It computes

        Args:
            scores_disc_fake: the scores of the discriminator when recognising generated (fake) data
            scores_disc_real: the scores of the discriminator when recognising real data

        Returns:
            Sum of absolute differences of fake/real scores
        """
        loss = 0
        for sr,sf in zip(scores_disc_real, scores_disc_fake) :
            loss += torch.mean(torch.abs(sf-sr))
        return loss

    # --------------
    def sample_loss2(self, waveform_real:torch.tensor, waveform_gen:torch.tensor) -> float:
        """Sample loss implementation. It computes MSE between real and generated samples

        Args:
            waveform_real: the real waweform segments
            waveform_gen: the generated waweform segments

        Returns:
            RMS value between real and generated speech samples
        """
        # MSE (mean-square error)
        return torch.mean((waveform_real-waveform_gen)**2)
