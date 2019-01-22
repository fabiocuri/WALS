"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder

from onmt.decoders.decoder import InputFeedRNNDecoder, StdRNNDecoder, StdRNNDecoderDoublyAttentive
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder

from onmt.modules import Embeddings, CopyGenerator
from onmt.modules.embeddings import FeatureEmbedding, FeatureMLP, MLP2RNNHiddenTarget, MLP2RNNHiddenBoth, MLP2WalsHiddenTarget, MLP2WalsHiddenBoth, MLPAttentionTarget, MLPAttentionBoth
from onmt.models.model import EncoderInitialization, DecoderInitialization, CombineWalsSourceWords, CombineWalsTargetWords, WalsDoublyAttention, WalstoDecHidden
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from operator import itemgetter

def build_feature_embeddings(gpu, FeatureTensors, FeaturesList, FeatureNames, Feature):

    idx_feature = FeatureNames.index(Feature)
    embedding_dim = FeaturesList[idx_feature][2] # Result of a linear transformation (check wals.py)
    num_embeddings = len(FeatureTensors[Feature]) 
    dic = FeatureTensors[Feature]

    if gpu:
        dic = dic.cuda()
    
    return FeatureEmbedding(embedding_dim,
                      num_embeddings,
                      dic)

def build_mlp_feature_type(opt, FTInfos, FeatureTypesNames, FeatureType):
    
    idx_featuretype = FeatureTypesNames.index(FeatureType)
    num_embeddings = FTInfos[idx_featuretype][3]
    
    return FeatureMLP(num_embeddings,
                      opt.wals_size)

def build_mlp2rnnhiddensize_target(opt, FTInfos):

    num_embeddings=0
    for FT in FTInfos:
        num_embeddings+=FT[3]

    return MLP2RNNHiddenTarget(num_embeddings, opt)

def build_mlp2rnnhiddensize_both(opt, FTInfos):

    num_embeddings=0
    for FT in FTInfos:
        num_embeddings+=FT[3]

    return MLP2RNNHiddenBoth(num_embeddings, opt)

def build_mlp2walshiddensize_target(opt, FTInfos):

    num_embeddings=0
    for FT in FTInfos:
        num_embeddings+=FT[3]

    return MLP2WalsHiddenTarget(num_embeddings, opt)

def build_mlp2walshiddensize_both(opt, FTInfos):

    num_embeddings=0
    for FT in FTInfos:
        num_embeddings+=FT[3]

    return MLP2WalsHiddenBoth(num_embeddings, opt)

def build_doublyattentive_target(opt):

    return MLPAttentionTarget(opt)

def build_doublyattentive_both(opt):

    return MLPAttentionBoth(opt)


def build_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Build an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[inputters.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[inputters.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings,
                      sparse=opt.optim == "sparseadam")


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """

    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.heads, opt.transformer_ff,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(opt.wals_model, opt.rnn_type, opt.brnn, opt.enc_layers,
                          opt.rnn_size, opt.wals_size, opt.dropout, embeddings,
                          opt.bridge)


def build_decoder(opt, embeddings, dec_size):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """

    if opt.wals_model == 'WalsDoublyAttentive_Target' or opt.wals_model == 'WalsDoublyAttentive_Both':

        return StdRNNDecoderDoublyAttentive(opt.rnn_type, opt.brnn,
                                 opt.dec_layers, dec_size,
                                 opt.global_attention,
                                 opt.global_attention_function,
                                 opt.coverage_attn,
                                 opt.context_gate,
                                 opt.copy_attn,
                                 opt.dropout,
                                 embeddings,
                                 opt.reuse_copy_attn)

    else:

        if opt.decoder_type == "transformer":
            return TransformerDecoder(opt.dec_layers, dec_size,
                                      opt.heads, opt.transformer_ff,
                                      opt.global_attention, opt.copy_attn,
                                      opt.self_attn_type,
                                      opt.dropout, embeddings)
        elif opt.decoder_type == "cnn":
            return CNNDecoder(opt.dec_layers, dec_size,
                              opt.global_attention, opt.copy_attn,
                              opt.cnn_kernel_width, opt.dropout,
                              embeddings)
        elif opt.input_feed:
            return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                       opt.dec_layers, dec_size,
                                       opt.global_attention,
                                       opt.global_attention_function,
                                       opt.coverage_attn,
                                       opt.context_gate,
                                       opt.copy_attn,
                                       opt.dropout,
                                       embeddings,
                                       opt.reuse_copy_attn)
        else:
            return StdRNNDecoder(opt.wals_model, opt.rnn_type, opt.brnn,
                                 opt.dec_layers, dec_size,
                                 opt.wals_size, opt.global_attention,
                                 opt.global_attention_function,
                                 opt.coverage_attn,
                                 opt.context_gate,
                                 opt.copy_attn,
                                 opt.dropout,
                                 embeddings,
                                 opt.reuse_copy_attn)


def load_test_model(opt, dummy_opt, FeatureValues, FeatureTensors, FeatureTypes, FeaturesList, FeatureNames, FTInfos, FeatureTypesNames, SimulationLanguages, model_path=None):

    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    fields = inputters.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    model = build_base_model(model_opt, fields, use_gpu(opt), FeatureValues, FeatureTensors, FeatureTypes, FeaturesList, FeatureNames, FTInfos, FeatureTypesNames, SimulationLanguages, checkpoint)
    model.eval()
    model.generator.eval()

    return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, FeatureValues, FeatureTensors, FeatureTypes, FeaturesList, FeatureNames, FTInfos, FeatureTypesNames, SimulationLanguages, checkpoint=None):

    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        WALS info
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Build encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = inputters.collect_feature_vocabs(fields, 'src')
        src_embeddings = build_embeddings(model_opt, src_dict, feature_dicts)
        encoder = build_encoder(model_opt, src_embeddings)
    elif model_opt.model_type == "img":
        if ("image_channel_size" not in model_opt.__dict__):
            image_channel_size = 3
        else:
            image_channel_size = model_opt.image_channel_size

        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               image_channel_size)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Build decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = inputters.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = build_embeddings(model_opt, tgt_dict,
                                      feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    if model_opt.wals_model == 'WalstoDecHidden_Target' or model_opt.wals_model == 'WalstoDecHidden_Both':
        #dec_size = model_opt.rnn_size + 2*model_opt.wals_size
        dec_size = model_opt.rnn_size
    else:
        dec_size = model_opt.rnn_size

    decoder = build_decoder(model_opt, tgt_embeddings, dec_size)



    # Wals

    print('Building embeddings for each WALS feature and MLP models for each feature type...')

    embeddings_list, embeddings_keys, mlp_list, mlp_keys = [], [], [], []

    for FeatureType in FeatureTypes:

        list_features = FeatureType[1]

        for Feature in list_features:

            globals()['embedding_%s' % Feature] = build_feature_embeddings(gpu, FeatureTensors, FeaturesList, FeatureNames, Feature)   # 192 embedding structures, one for each feature.
            embeddings_keys.append(Feature)
            embeddings_list.append(globals()['embedding_%s' % Feature])
        globals()['mlp_%s' % FeatureType[0]] = build_mlp_feature_type(model_opt, FTInfos, FeatureTypesNames, FeatureType[0]) # 11 MLPs, one for each feature type.
        mlp_keys.append(FeatureType[0])
        mlp_list.append(globals()['mlp_%s' % FeatureType[0]])

    embeddings_dic_keys = dict(zip(embeddings_keys, embeddings_list))
    EmbeddingFeatures = nn.ModuleDict(embeddings_dic_keys)

    mlp_dic_keys = dict(zip(mlp_keys, mlp_list))

    # Build NMTModel(= encoder + decoder).
    device = torch.device("cuda" if gpu else "cpu")

    if model_opt.wals_model == 'EncInitHidden_Target':

        MLP2RNNHiddenSize_Target = build_mlp2rnnhiddensize_target(model_opt, FTInfos)
        print('Embeddings for WALS features and MLP models are built!')
        model = EncoderInitialization(model_opt.wals_model, encoder, decoder, MLP2RNNHiddenSize_Target, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt)
        print("Model created: uses WALS features from the target language to initialize encoder's hidden state.")

    elif model_opt.wals_model == 'EncInitHidden_Both':
    
        MLP2RNNHiddenSize_Both = build_mlp2rnnhiddensize_both(model_opt, FTInfos)
        print('Embeddings for WALS features and MLP models are built!')
        model = EncoderInitialization(model_opt.wals_model,encoder, decoder, MLP2RNNHiddenSize_Both, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt)
        print("Model created: uses WALS features from the source and target languages to initialize encoder's hidden state.")

    elif model_opt.wals_model == 'DecInitHidden_Target':

        MLP2RNNHiddenSize_Target = build_mlp2rnnhiddensize_target(model_opt, FTInfos)
        print('Embeddings for WALS features and MLP models are built!')
        model = DecoderInitialization(model_opt.wals_model,encoder, decoder, MLP2RNNHiddenSize_Target, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt)
        print("Model created: adds WALS features from the target language to the encoder's output to initialize decoder's hidden state.")

    elif model_opt.wals_model == 'DecInitHidden_Both':
    
        MLP2RNNHiddenSize_Both = build_mlp2rnnhiddensize_both(model_opt, FTInfos)
        print('Embeddings for WALS features and MLP models are built!')
        model = DecoderInitialization(model_opt.wals_model,encoder, decoder, MLP2RNNHiddenSize_Both, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt)
        print("Model created: adds WALS features from the source and target languages to the encoder's output to initialize decoder's hidden state.")

    elif model_opt.wals_model == 'WalstoSource_Target':

        MLP2WALSHiddenSize_Target = build_mlp2walshiddensize_target(model_opt, FTInfos)
        print('Embeddings for WALS features and MLP models are built!')
        model = CombineWalsSourceWords(model_opt.wals_model,encoder, decoder, MLP2WALSHiddenSize_Target, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt)
        print("Model created: concatenates WALS features from the target language to source words embeddings.")

    elif model_opt.wals_model == 'WalstoSource_Both':
    
        MLP2WALSHiddenSize_Both = build_mlp2walshiddensize_both(model_opt, FTInfos)
        print('Embeddings for WALS features and MLP models are built!')
        model = CombineWalsSourceWords(model_opt.wals_model,encoder, decoder, MLP2WALSHiddenSize_Both, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt)
        print("Model created: concatenates WALS features from the source and target languages to source words embeddings.")

    elif model_opt.wals_model == 'WalstoTarget_Target':

        MLP2WALSHiddenSize_Target = build_mlp2walshiddensize_target(model_opt, FTInfos)
        print('Embeddings for WALS features and MLP models are built!')
        model = CombineWalsTargetWords(model_opt.wals_model,encoder, decoder, MLP2WALSHiddenSize_Target, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt)
        print("Model created: concatenates WALS features from the target language to target words embeddings.")

    elif model_opt.wals_model == 'WalstoTarget_Both':
    
        MLP2WALSHiddenSize_Both = build_mlp2walshiddensize_both(model_opt, FTInfos)
        print('Embeddings for WALS features and MLP models are built!')
        model = CombineWalsTargetWords(model_opt.wals_model,encoder, decoder, MLP2WALSHiddenSize_Both, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt)
        print("Model created: concatenates WALS features from the source and target languages to target words embeddings.")

    elif model_opt.wals_model == 'WalsDoublyAttentive_Target':

        MLPFeatureTypes = nn.ModuleDict(mlp_dic_keys)
        MLP_AttentionTarget = build_doublyattentive_target(model_opt)
        print('Embeddings for WALS features and MLP models are built!')
        model = WalsDoublyAttention(model_opt.wals_model,encoder, decoder, MLP_AttentionTarget, MLPFeatureTypes, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt)
        print("Model created: the WALS features from the target language are incorporated as an additional attention mechanism.")

    elif model_opt.wals_model == 'WalsDoublyAttentive_Both': 

        MLPFeatureTypes = nn.ModuleDict(mlp_dic_keys)
        MLP_AttentionBoth = build_doublyattentive_both(model_opt)
        print('Embeddings for WALS features and MLP models are built!')
        model = WalsDoublyAttention(model_opt.wals_model,encoder, decoder, MLP_AttentionBoth, MLPFeatureTypes, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt)
        print("Model created: the WALS features from the source and target languages are incorporated as an additional attention mechanism.")

    elif model_opt.wals_model == 'WalstoDecHidden_Target':
        
        MLP2WALSHiddenSize_Target = build_mlp2walshiddensize_target(model_opt, FTInfos)   
        print('Embeddings for WALS features and MLP models are built!') 
        model = WalstoDecHidden(model_opt.wals_model, encoder, decoder, MLP2WALSHiddenSize_Target, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt)
        print("Model created: concatenates WALS features from the target language to decoder hidden state.")

    elif model_opt.wals_model == 'WalstoDecHidden_Both':
        
        MLP2WALSHiddenSize_Both = build_mlp2walshiddensize_both(model_opt, FTInfos)   
        print('Embeddings for WALS features and MLP models are built!') 
        model = WalstoDecHidden(model_opt.wals_model, encoder, decoder, MLP2WALSHiddenSize_Both, EmbeddingFeatures, FeatureValues, FeatureTypes, SimulationLanguages, model_opt)
        print("Model created: concatenates WALS features from the target language to decoder hidden state.")

    else:
        raise Exception("WALS model type not yet implemented: %s"%(
                        opt.wals_model))

    model.model_type = model_opt.model_type

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)), gen_func
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator
    model.to(device)

    return model


def build_model(model_opt, opt, fields, checkpoint, FeatureValues, FeatureTensors, FeatureTypes, FeaturesList, FeatureNames, FTInfos, FeatureTypesNames, SimulationLanguages):
    """ Build the Model """
    logger.info('Building model...')
    model = build_base_model(model_opt, fields,
                             use_gpu(opt), FeatureValues, FeatureTensors, FeatureTypes, FeaturesList, FeatureNames, FTInfos, FeatureTypesNames, SimulationLanguages, checkpoint)
    logger.info(model)
    return model
