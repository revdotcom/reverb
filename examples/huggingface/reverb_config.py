# Following https://huggingface.co/docs/transformers/en/custom_models
import math
from typing import Dict, List, Optional
from transformers import PretrainedConfig
import numpy as np
import yaml
from pyctcdecode import build_ctcdecoder


def cmvn(means: List[float], variance: List[float], count: int):
    """ Calculate cmvn from stats

    Returns:
        a numpy array of [means, vars]
    """
    for i in range(len(means)):
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])
    return [means, variance]


class ReverbConfig(PretrainedConfig):
    model_type = "reverb_asr"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set default special tokens if not provided
        if not hasattr(self, 'special_tokens'):
            self.special_tokens = {
                "<blank>": 0,
                "<sos>": 2,
                "<eos>": 2,
                "<unk>": 1,
            }
            
        # Calculate CMVN if the required stats are provided
        if hasattr(self, 'cmvn_mean_stat') and hasattr(self, 'cmvn_var_stat') and hasattr(self, 'cmvn_frame_num'):
            self.cmvn_mean, self.cmvn_istd = cmvn(
                self.cmvn_mean_stat, 
                self.cmvn_var_stat, 
                self.cmvn_frame_num
            )
            
        # Set default ratio if not provided
        if not hasattr(self, 'inputs_to_logits_ratio'):
            self.inputs_to_logits_ratio = 1
            
        # Tokenizer configuration
        if not hasattr(self, 'tokenizer_path'):
            self.tokenizer_path = "path/to/tokenizer.model"
        if not hasattr(self, 'units_path'):
            self.units_path = "path/to/units.txt"
        if not hasattr(self, 'decoder_beam_width'):
            self.decoder_beam_width = 8
        if not hasattr(self, 'decoder_token_min_logp'):
            self.decoder_token_min_logp = -10
            
        # Load units and build decoder
        self._load_units_and_build_decoder()
        
    def _load_units_and_build_decoder(self):
        """Load units from file and build the CTC decoder."""
        decoder_ids = []
        with open(self.units_path, 'r') as units_file:
            for line in units_file:
                token = line.split()[0]
                if len(token) == 0:
                    continue
                if token == '<blank>':
                    token = ''
                decoder_ids.append(token)
        self.decoder = build_ctcdecoder(decoder_ids)

    @classmethod
    def from_yaml_file(cls, yaml_file: str) -> "ReverbConfig":
        """Load a ReverbConfig from a YAML file.
        
        Args:
            yaml_file: Path to the YAML file containing the configuration
            
        Returns:
            A ReverbConfig instance loaded from the file
        """
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)