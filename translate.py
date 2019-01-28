#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts

import os
import sqlite3
from collections import defaultdict
import numpy as np

from onmt.train_single import get_feat_values

def main(opt):
    
    #TODO: delete all lines related to WALS
    #begin
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
    #end

    #TODO: load wals features from command-line (wals.npz)
    #TODO: remove all parameters related to WALS features and include four numpy vectors that describe WALS
    translator = build_translator(opt, FeatureValues, FeatureTensors, FeatureTypes, FeaturesList, FeatureNames, FTInfos, FeatureTypesNames, SimulationLanguages, report_score=True)
    translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
