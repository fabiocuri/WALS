""" Onmt NMT Model base class definition """
import torch.nn as nn
from collections import defaultdict
import torch

def get_local_features(EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt, MLP_target_or_both=None, MLPFeatureTypes=None):

    features_row, features_concat_per_type, featuretypes_after_MLP = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    all_featuretypes, featuretypes_concat, mean_all_features, out = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))

    for Language in SimulationLanguages:

        l_pertype_MLP, l_all_features = [], []

        for FeatureType in FeatureTypes:

            l_pertype = []

            for Feature in FeatureType[1]:

                # Obtain rows of features in its original embeddings, given the feature value.
                features_row[Language][Feature] = EmbeddingFeatures[Feature](FeatureValues[Language][Feature]) 
                features_row[Language][Feature] = features_row[Language][Feature].view(1,len(features_row[Language][Feature]))
                l_pertype.append(features_row[Language][Feature])

            # Concatenate features of the same type.
            features_concat_per_type[Language][FeatureType[0]] = torch.cat(l_pertype, dim=1) # size: 1 x "feature type"

	    # Concatenate all feature embeddings for all feature types.
            l_all_features.append(features_concat_per_type[Language][FeatureType[0]])  
 
            if model_opt.wals_model == 'WalsDoublyAttentive_Target' or model_opt.wals_model == 'WalsDoublyAttentive_Both':

                # Run a MLP, for each feature type.
                featuretypes_after_MLP[Language][FeatureType[0]] = MLPFeatureTypes[FeatureType[0]](features_concat_per_type[Language][FeatureType[0]]) # size: 1 x wals_size
                l_pertype_MLP.append(featuretypes_after_MLP[Language][FeatureType[0]])

        if model_opt.wals_model == 'EncInitHidden_Target' or model_opt.wals_model == 'EncInitHidden_Both' or model_opt.wals_model == 'DecInitHidden_Target' or model_opt.wals_model == 'DecInitHidden_Both' or model_opt.wals_model == 'WalstoSource_Target' or model_opt.wals_model == 'WalstoSource_Both' or model_opt.wals_model == 'WalstoTarget_Target' or model_opt.wals_model == 'WalstoTarget_Both' or model_opt.wals_model == 'WalstoDecHidden_Target' or model_opt.wals_model == 'WalstoDecHidden_Both':

            # Concatenate all embeddings of the features.
            out[Language] = torch.cat(l_all_features, dim=1) # size: 1 x 200

        if model_opt.wals_model == 'WalsDoublyAttentive_Target' or model_opt.wals_model == 'WalsDoublyAttentive_Both':

            # Concatenate all 11 vectors of size 1 x wals_size, per language.
            out[Language] = torch.cat(l_pertype_MLP, dim=0) # size: 11 x wals_size

    # Select WALS for both languages or target only.
    if model_opt.wals_model == 'EncInitHidden_Target' or model_opt.wals_model == 'DecInitHidden_Target' or model_opt.wals_model == 'WalstoSource_Target' or model_opt.wals_model == 'WalstoTarget_Target' or model_opt.wals_model == 'WalsDoublyAttentive_Target' or model_opt.wals_model == 'WalstoDecHidden_Target':

        wals_features = out[SimulationLanguages[1]] # [1 x 200] or [11 x wals_size]
        wals_features = MLP_target_or_both(wals_features) # [1 x rnn_size] or [1 x wals_size] or [11 x rnn_size]

    if model_opt.wals_model == 'EncInitHidden_Both' or model_opt.wals_model == 'DecInitHidden_Both' or model_opt.wals_model == 'WalstoSource_Both' or model_opt.wals_model == 'WalstoTarget_Both' or model_opt.wals_model == 'WalsDoublyAttentive_Both' or model_opt.wals_model == 'WalstoDecHidden_Both':

        wals_features = torch.cat((out[SimulationLanguages[0]], out[SimulationLanguages[1]]) , 1) # [1 x 2.200] or [11 x 2.wals_size]
        wals_features = MLP_target_or_both(wals_features) # [1 x rnn_size] or [1 x wals_size] or [11 x rnn_size]

    return wals_features


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state

class EncoderInitialization(nn.Module):

    """
    Model A: uses WALS features to initialize encoder's hidden state.

    Encoder: RNNEncoder
    Decoder: StdRNNDecoder
    """
    def __init__(self, wals_model, encoder, decoder, MLP_target_or_both, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt, multigpu=False):

        self.multigpu = multigpu
        super(EncoderInitialization, self).__init__()
        self.wals_model = wals_model
        self.encoder = encoder  
        self.decoder = decoder  
        self.MLP_target_or_both = MLP_target_or_both
        self.EmbeddingFeatures = EmbeddingFeatures  
        self.FeatureValues = FeatureValues 
        self.FeatureTypes = FeatureTypes
        self.SimulationLanguages = SimulationLanguages
        self.model_opt = model_opt
        self.is_tuple_hidden = None

    def evaluate_is_tuple_hidden(self, src, lengths):

        enc_hidden, context = self.encoder(input=src, lengths=lengths, check_LSTM=True)

        if isinstance(enc_hidden, tuple): # LSTM
            self.is_tuple_hidden = True
        else: # GRU
            self.is_tuple_hidden = False

        ret = self.is_tuple_hidden

        return ret

    def forward(self, src, tgt, lengths, dec_state=None):

        wals_features = get_local_features(self.EmbeddingFeatures, self.FeatureValues, self.FeatureTypes, self.SimulationLanguages, self.model_opt, self.MLP_target_or_both) # 1 x rnn_size

        dim0, dim1 = wals_features.size()
        wals_features = wals_features.view(1, dim0, dim1) # 1 x 1 x rnn_size

        tgt = tgt[:-1]  # exclude last target from inputs

        # create initial hidden state differently for GRU/LSTM
        if self.evaluate_is_tuple_hidden(src, lengths):
            enc_init_state = (wals_features, wals_features)
            is_tuple = True
        else:
            enc_init_state = wals_features
            is_tuple = False


        enc_hidden, context = self.encoder(input=src, lengths=lengths, hidden=enc_init_state, is_LSTM=is_tuple)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             lengths)

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        return out, attns, dec_state

class DecoderInitialization(nn.Module):

    """
    Model B: adds WALS features to the encoder's output to initialize decoder's hidden state.

    Encoder: RNNEncoder
    Decoder: StdRNNDecoder
    """

    def __init__(self, wals_model, encoder, decoder, MLP_target_or_both, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt, multigpu=False):

        self.multigpu = multigpu
        super(DecoderInitialization, self).__init__()
        self.wals_model = wals_model
        self.encoder = encoder  
        self.decoder = decoder  
        self.MLP_target_or_both = MLP_target_or_both
        self.EmbeddingFeatures = EmbeddingFeatures  
        self.FeatureValues = FeatureValues 
        self.FeatureTypes = FeatureTypes
        self.SimulationLanguages = SimulationLanguages
        self.model_opt = model_opt

    def combine_enc_hidden_wals_proj(self, enc_hidden, wals_features):
    
        dec_init_state = []
        if isinstance(enc_hidden, tuple):
            for e in enc_hidden:
                # e.size() = enc_layers x batch_size x rnn_size
                dec_init_state.append(e + wals_features)
            dec_init_state = tuple(dec_init_state)
        else:
            dec_init_state = enc_hidden + wals_features

        return dec_init_state

    def forward(self, src, tgt, lengths, dec_state=None):

        wals_features = get_local_features(self.EmbeddingFeatures, self.FeatureValues, self.FeatureTypes, self.SimulationLanguages, self.model_opt, self.MLP_target_or_both) # 1 x rnn_size

        dim0, dim1 = wals_features.size()
        wals_features = wals_features.view(1, dim0, dim1) # 1 x 1 x rnn_size

        # Find mini-batch size and get the right size.
        dim0, dim1, dim2 = src.size()

        wals_features_modelB = wals_features.repeat(self.model_opt.dec_layers,dim1,1) # dec_layers x batch_size x rnn_size

        tgt = tgt[:-1]  # exclude last target from inputs

        enc_hidden, context = self.encoder(input=src, lengths=lengths)
        dec_init_state = self.combine_enc_hidden_wals_proj(enc_hidden, wals_features_modelB)
        enc_state = self.decoder.init_decoder_state(src, context, dec_init_state)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             lengths)

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        return out, attns, dec_state

class CombineWalsSourceWords(nn.Module):

    """
    Model C: concatenates WALS features to source words embeddings.

    Encoder: RNNEncoder
    Decoder: StdRNNDecoder
    """

    def __init__(self, wals_model, encoder, decoder, MLP_target_or_both, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt, multigpu=False):

        self.multigpu = multigpu
        super(CombineWalsSourceWords, self).__init__()
        self.wals_model = wals_model
        self.encoder = encoder  
        self.decoder = decoder  
        self.MLP_target_or_both = MLP_target_or_both
        self.EmbeddingFeatures = EmbeddingFeatures  
        self.FeatureValues = FeatureValues 
        self.FeatureTypes = FeatureTypes
        self.SimulationLanguages = SimulationLanguages
        self.model_opt = model_opt

    def forward(self, src, tgt, lengths, dec_state=None):

        wals_features = get_local_features(self.EmbeddingFeatures, self.FeatureValues, self.FeatureTypes, self.SimulationLanguages, self.model_opt, self.MLP_target_or_both) # 1 x wals_size

        dim0, dim1 = wals_features.size()
        wals_features = wals_features.view(1, dim0, dim1) # 1 x 1 x wals_size

        tgt = tgt[:-1]  # exclude last target from inputs

        enc_hidden, context = self.encoder(input=src, lengths=lengths, wals_features=wals_features)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             lengths)

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        return out, attns, dec_state

class CombineWalsTargetWords(nn.Module):

    """
    Model D: concatenates WALS features to target words embeddings.

    Encoder: RNNEncoder
    Decoder: StdRNNDecoder
    """

    def __init__(self, wals_model, encoder, decoder, MLP_target_or_both, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt, multigpu=False):

        self.multigpu = multigpu
        super(CombineWalsTargetWords, self).__init__()
        self.wals_model = wals_model
        self.encoder = encoder  
        self.decoder = decoder  
        self.MLP_target_or_both = MLP_target_or_both
        self.EmbeddingFeatures = EmbeddingFeatures  
        self.FeatureValues = FeatureValues 
        self.FeatureTypes = FeatureTypes
        self.SimulationLanguages = SimulationLanguages
        self.model_opt = model_opt

    def forward(self, src, tgt, lengths, dec_state=None):

        wals_features = get_local_features(self.EmbeddingFeatures, self.FeatureValues, self.FeatureTypes, self.SimulationLanguages, self.model_opt, self.MLP_target_or_both) # 1 x wals_size

        dim0, dim1 = wals_features.size()
        wals_features = wals_features.view(1, dim0, dim1) # 1 x 1 x wals_size

        tgt = tgt[:-1]  # exclude last target from inputs

        enc_hidden, context = self.encoder(input=src, lengths=lengths)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             lengths, wals_features)

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        return out, attns, dec_state


class WalsDoublyAttention(nn.Module):

    """
    Model E: the WALS features are incorporated as an additional attention mechanism.

    Encoder: RNNEncoder
    Decoder: StdRNNDecoderDoublyAttentive
    """

    def __init__(self, wals_model, encoder, decoder, MLP_target_or_both, MLPFeatureTypes, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt, multigpu=False):

        self.multigpu = multigpu
        super(WalsDoublyAttention, self).__init__()
        self.wals_model = wals_model
        self.encoder = encoder  
        self.decoder = decoder  
        self.MLP_target_or_both = MLP_target_or_both
        self.MLPFeatureTypes = MLPFeatureTypes
        self.EmbeddingFeatures = EmbeddingFeatures  
        self.FeatureValues = FeatureValues 
        self.FeatureTypes = FeatureTypes
        self.SimulationLanguages = SimulationLanguages
        self.model_opt = model_opt

    def forward(self, src, tgt, lengths, dec_state=None):

        wals_features = get_local_features(self.EmbeddingFeatures, self.FeatureValues, self.FeatureTypes, self.SimulationLanguages, self.model_opt, self.MLP_target_or_both, self.MLPFeatureTypes) # 11 x rnn_size

        dim0, dim1 = wals_features.size()
        dim0_src, dim1_src, dim2_src = src.size()

        wals_features = wals_features.view(1, dim0, dim1) # 1 x 11 x rnn_size
        wals_features = wals_features.transpose(0,1) # 11 x 1 x rnn_size

        wals_features = wals_features.repeat(1, dim1_src, 1) # 11 x batch_size x rnn_size

        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(input=src, lengths=lengths)
        enc_state = self.decoder.init_decoder_state(src, context, wals_features, enc_hidden)
        out, out_wals, dec_state, attns = self.decoder(tgt, context,
                                                       enc_state if dec_state is None
                                                       else dec_state,
                                                       lengths, wals_features)

        out_total = out + out_wals

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        return out_total, attns, dec_state



class WalstoDecHidden(nn.Module):

    """
    Model F: the WALS features are concatenated to the decoder's hidden state.

    Encoder: RNNEncoder
    Decoder: StdRNNDecoder
    """

    def __init__(self, wals_model, encoder, decoder, MLP_target_or_both, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt, multigpu=False):

        self.multigpu = multigpu
        super(WalstoDecHidden, self).__init__()
        self.wals_model = wals_model
        self.encoder = encoder  
        self.decoder = decoder  
        self.MLP_target_or_both = MLP_target_or_both
        self.EmbeddingFeatures = EmbeddingFeatures  
        self.FeatureValues = FeatureValues 
        self.FeatureTypes = FeatureTypes
        self.SimulationLanguages = SimulationLanguages
        self.model_opt = model_opt

    def forward(self, src, tgt, lengths, dec_state=None):

        wals_features = get_local_features(self.EmbeddingFeatures, self.FeatureValues, self.FeatureTypes, self.SimulationLanguages, self.model_opt, self.MLP_target_or_both) # 1 x wals_size

        dim0, dim1 = wals_features.size()
        wals_features = wals_features.view(1, dim0, dim1) # 1 x 1 x wals_size

        tgt = tgt[:-1]  # exclude last target from inputs

        enc_hidden, context = self.encoder(input=src, lengths=lengths)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             lengths, wals_features)

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        return out, attns, dec_state
