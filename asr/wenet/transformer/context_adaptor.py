# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

import logging
from typing import Tuple, List, Optional

import torch
from typeguard import check_argument_types

from wenet.utils.mask import (subsequent_mask, make_pad_mask)


class ContextAdaptor(torch.nn.Module):
    """ContextAdaptor: https://assets.amazon.science/43/13/104c968c45ea9ed02cffaa1448e0/personalization-of-ctc-speech-recognition-models.pdf
    Args:
        vocab_size: subword vocab size
        embedding_dim: size of subword embeddings
        encoder_output_size: dimension of attention
        num_layers: the number of bilstm layers
        dropout_rate: dropout rate
        attention_heads: the number of heads of multi head attention
        attention_dropout_rate: dropout rate for attention
    """
    def __init__(
        self,
        vocab_size: int,
        output_size: int = 512,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        attention_heads: int = 1,
        attention_dropout_rate: float = 0.0,
    ):
        assert check_argument_types()
        super().__init__()

        self.vocab_size = vocab_size
        # embedding layer (subword unit --> embedding)
        self.embed = torch.nn.Embedding(vocab_size+1, embedding_dim)

        # bidirectional LSTM -- output size will be doubled
        lstm_output_size = int(output_size/2)
        self.encoder = torch.nn.LSTM(embedding_dim,
                                     lstm_output_size,
                                     num_layers,
                                     batch_first = True,
                                     dropout = dropout_rate,
                                     bidirectional = True)

        # attention mechanism - ASR encoder outputs vs. context terms
        self.attention = torch.nn.MultiheadAttention(output_size, attention_heads,
                                                     attention_dropout_rate,
                                                     batch_first=True)

    def forward(
        self,
        encoder_layer_outs: List[torch.Tensor],
        cv_encoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """Forward just attention piece of contextual adaptor
        Args:
            encoder_layer_outs: list of outputs of ASR encoder layers. each one is (batch, maxlen, output_size)
            cv_encoder_out: output of cv encoder (1, n_cv_terms, output_size)
        Returns:
            x: decoded token score before softmax (batch, maxlen, output_size) 
        """
        combined_encoder_layer_outs = self.combine_layers(encoder_layer_outs)
        cv_encoder_out = cv_encoder_out.expand(combined_encoder_layer_outs.shape[0], -1, -1)

        x, y = self.attention(combined_encoder_layer_outs, cv_encoder_out, cv_encoder_out)
        # x = batch x frame x embedding
        # y = batch x frame x CV term
        assert y is not None
        #mask = y[:, :, 0] > 0.5
        mask = torch.argmax(y, dim=2) == 0
        x[mask.unsqueeze(2).expand(-1, -1, x.shape[2])] = 0.
        # JDF: uncomment for CV detection during decoding
        #for i in range(y.shape[1]):
        #    if y[0,i,0] <= 0.5:
        #        logging.info(str(i) + " " + str(y[0, i, :]))
        return x

    def encode_cv(
        self,
        cv: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """Encode context terms - separated from main forward step so that it can be done just once per audio file at inference time
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
        Returns:
            x: decoded token score before softmax (batch, maxlen_out,
                vocab_size) 
        """
        blank_token = torch.zeros(1, cv.shape[1], dtype=torch.int32)
        blank_token[0,0] = self.vocab_size
        if cv.get_device() >= 0:
            blank_token = blank_token.to(cv.get_device(), non_blocking=True)
        blank_length = torch.ones(1, dtype=torch.int32)

        if lengths.get_device() >= 0:
            blank_length = blank_length.to(lengths.get_device(), non_blocking=True)
        
        cv = torch.cat([blank_token, cv])

        lengths = torch.cat([blank_length, lengths])
        # pack_padded_sequence requires lengths to be on CPU
        lengths = lengths.to('cpu') 
        
        # subwords --> embeddings
        x = self.embed(cv) # nTerms x maxlen x embdding_dim
        # padding
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # lstm on each CV term, pull out last hidden state
        _, (x,_) = self.encoder(x) # (nLayers x 2) x nTerms x output_dim
        x = x.view(-1, 2, x.shape[1], x.shape[2])

        # concat forward and backward from last layer
        x = torch.cat([x[-1, 0, :, :], x[-1, 1, :, :]], dim=1).unsqueeze(0) # nTerms x 1 x output_dim*2

        return x

    def combine_layers(
        self,
        layer_outs: List[torch.Tensor]
    ) -> torch.Tensor:
        # in https://assets.amazon.science/43/13/104c968c45ea9ed02cffaa1448e0/personalization-of-ctc-speech-recognition-models.pdf
        # they use a weighted sum of the 6th, 12th, and 20th layers out of a 20-layer encoder
        # but they don't say what the weights are :/
        return 0.5*layer_outs[-1] + 0.25*layer_outs[-9] + 0.25*layer_outs[-15]
    

