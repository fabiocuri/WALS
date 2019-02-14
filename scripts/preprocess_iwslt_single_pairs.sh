#!/usr/bin/bash

SRCS=(en de nl it ro)
TGTS=(en de nl it ro)

for src in ${SRCS[@]}; do
    for tgt in ${TGTS[@]}; do
        [ "$src" == "$tgt" ] && continue

        python preprocess.py \
            -train_src data/iwslt17/single-pair/train.tags.${src}-${tgt}.lc.tok.norm.clean.bpe.30000.${src} \
            -train_tgt data/iwslt17/single-pair/train.tags.${src}-${tgt}.lc.tok.norm.clean.bpe.30000.${tgt} \
            -valid_src data/iwslt17/single-pair/IWSLT17.TED.dev2010.${src}-${tgt}.tok.lc.norm.bpe.30000.${src} \
            -valid_tgt data/iwslt17/single-pair/IWSLT17.TED.dev2010.${src}-${tgt}.tok.lc.norm.bpe.30000.${tgt} \
            -save_data data/iwslt17/single-pair/iwslt17.${src}-${tgt}.bpe.30000 \
            -share_vocab
    done
done
