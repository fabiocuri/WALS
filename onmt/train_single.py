#!/usr/bin/env python
"""
    Training on a single process
"""
from __future__ import division

import argparse
import os
import random
import torch

import onmt.opts as opts

from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, _collect_report_features
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger

import sqlite3
from collections import defaultdict
import numpy as np

def get_feat_values(SimulationLanguages, WalsValues, FeaturesList, ListLanguages, FeatureTypes, FeatureNames) :

    FeatureValues, FeatureTensors = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))

    for Language in SimulationLanguages: # For each language in the simulation...
        idx_language = ListLanguages.index(Language) 
        for FeatureType in FeatureTypes: # tuple
            for Feature in FeatureType[1]: # For each feature...
                idx_feature = FeatureNames.index(Feature)
                FeatureValues[Language][Feature] = WalsValues[idx_language][idx_feature+1] 
                FeatureTensors[Feature] = torch.from_numpy(np.array(range(FeaturesList[idx_feature][1] + 1)))

    return FeatureValues, FeatureTensors


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


def training_opt_postprocessing(opt, device_id):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    opt.brnn = (opt.encoder_type == "brnn")

    if opt.rnn_type == "SRU" and not opt.gpu_ranks:
        raise AssertionError("Using SRU requires -gpu_ranks set.")

    if torch.cuda.is_available() and not opt.gpu_ranks:
        logger.info("WARNING: You have a CUDA device, \
                    should run with -gpu_ranks")

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)

    return opt


def main(opt, device_id):

    SimulationLanguages = [opt.wals_src, opt.wals_tgt]

    print('Loading WALS features from databases...')

    cwd = os.getcwd()

    db = sqlite3.connect(cwd + '/onmt/WalsValues.db')
    cursor = db.cursor()
    cursor.execute('SELECT * FROM WalsValues')
    WalsValues = cursor.fetchall()

    db = sqlite3.connect(cwd + '/onmt/FeaturesList.db')
    cursor = db.cursor()
    cursor.execute('SELECT * FROM FeaturesList')
    FeaturesList = cursor.fetchall()

    db = sqlite3.connect(cwd + '/onmt/FTInfos.db')
    cursor = db.cursor()
    cursor.execute('SELECT * FROM FTInfos')
    FTInfos = cursor.fetchall()

    db = sqlite3.connect(cwd + '/onmt/FTList.db')
    cursor = db.cursor()
    cursor.execute('SELECT * FROM FTList')
    FTList = cursor.fetchall()

    ListLanguages = []
    for i in WalsValues:
        ListLanguages.append(i[0])

    FeatureTypes = []
    for i in FTList:
        FeatureTypes.append((i[0], i[1].split(',')))

    FeatureNames = []
    for i in FeatureTypes:
        FeatureNames += i[1]

    FeatureTypesNames = []
    for i in FeatureTypes:
        FeatureTypesNames.append(i[0])

    FeatureValues, FeatureTensors = get_feat_values(SimulationLanguages, WalsValues, FeaturesList, ListLanguages, FeatureTypes, FeatureNames) 

    print('WALS databases loaded!')

    # FeatureValues: defaultdict with feature values, per language.
    # FeatureTensors: tensor of possible outputs, per feature.

    opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
    else:
        checkpoint = None
        model_opt = opt

    # Peek the first dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train", opt))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = _load_fields(first_dataset, data_type, opt, checkpoint)

    # Report src/tgt features.

    src_features, tgt_features = _collect_report_features(fields)
    for j, feat in enumerate(src_features):
        logger.info(' * src feature %d size = %d'
                    % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        logger.info(' * tgt feature %d size = %d'
                    % (j, len(fields[feat].vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint, FeatureValues, FeatureTensors, FeatureTypes, FeaturesList, FeatureNames, FTInfos, FeatureTypesNames, SimulationLanguages)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(opt, device_id, model, fields,
                            optim, data_type, model_saver=model_saver)

    def train_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("train", opt), fields, opt)

    def valid_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("valid", opt), fields, opt, is_train=False)

    # Do training.
    trainer.train(train_iter_fct, valid_iter_fct, opt.train_steps,
                  opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
