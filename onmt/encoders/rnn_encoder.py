"""Define RNN-based encoders."""
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory

import torch


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, wals_model, rnn_type, bidirectional, num_layers,
                 hidden_size, wals_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.embeddings = embeddings
        self.wals_model = wals_model
        self.num_layers = num_layers
        self.wals_size = wals_size

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        if self.wals_model == 'WalstoSource_Target' or self.wals_model == 'WalstoSource_Both':

            self.rnn_wals, self.no_pack_padded_seq_wals = \
                rnn_factory(rnn_type,
                            input_size=hidden_size+2*self.wals_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, input, lengths=None, hidden=None, wals_features=None, check_LSTM=False, is_LSTM=False):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        if check_LSTM: # Condition to check whether rnn is LSTM or GRU under Model A.

            packed_emb = emb

            if lengths is not None and not self.no_pack_padded_seq:

                # Lengths data is wrapped inside a Tensor.
                lengths = lengths.view(-1).tolist()
                packed_emb = pack(emb, lengths)

            memory_bank, encoder_final = self.rnn(packed_emb, hidden)

            if lengths is not None and not self.no_pack_padded_seq:

                memory_bank = unpack(memory_bank)[0]

        else: # If we are not testing this condition ...

            if self.wals_model == 'EncInitHidden_Target' or self.wals_model == 'EncInitHidden_Both':

                if is_LSTM:
                    hidden = hidden[0]
                    hidden = hidden.repeat(self.num_layers, batch, 1) # enc_layers x batch_size x rnn_size
                    hidden = (hidden, hidden)
                else:
                    hidden = hidden.repeat(self.num_layers, batch, 1) # enc_layers x batch_size x rnn_size

                packed_emb = emb

                if lengths is not None and not self.no_pack_padded_seq:

                    # Lengths data is wrapped inside a Tensor.
                    lengths = lengths.view(-1).tolist()
                    packed_emb = pack(emb, lengths)

                memory_bank, encoder_final = self.rnn(packed_emb, hidden)

                if lengths is not None and not self.no_pack_padded_seq:

                    memory_bank = unpack(memory_bank)[0]

            if self.wals_model == 'DecInitHidden_Target' or self.wals_model == 'DecInitHidden_Both' or self.wals_model == 'WalstoTarget_Target' or self.wals_model == 'WalstoTarget_Both' or self.wals_model == 'WalstoDecHidden_Target' or self.wals_model == 'WalstoDecHidden_Both' or self.wals_model == 'WalsDoublyAttentive_Target' or self.wals_model == 'WalsDoublyAttentive_Both':

                packed_emb = emb

                if lengths is not None and not self.no_pack_padded_seq:

                    # Lengths data is wrapped inside a Tensor.
                    lengths = lengths.view(-1).tolist()
                    packed_emb = pack(emb, lengths)

                memory_bank, encoder_final = self.rnn(packed_emb, hidden)

                if lengths is not None and not self.no_pack_padded_seq:

                    memory_bank = unpack(memory_bank)[0]

            if self.wals_model == 'WalstoSource_Target' or self.wals_model == 'WalstoSource_Both':

                wals_features = wals_features.repeat(s_len,batch,1) # s_len x batch_size x wals_size

                emb = torch.cat((wals_features, emb, wals_features), 2)
                packed_emb = emb

                if lengths is not None and not self.no_pack_padded_seq_wals:

                    # Lengths data is wrapped inside a Tensor.
                    lengths = lengths.view(-1).tolist()
                    packed_emb = pack(emb, lengths)

                memory_bank, encoder_final = self.rnn_wals(packed_emb, hidden)

                if lengths is not None and not self.no_pack_padded_seq_wals:

                    memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)

        return encoder_final, memory_bank

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs
