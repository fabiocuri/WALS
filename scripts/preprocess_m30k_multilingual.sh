#!/bin/bash

# This script creates the multilingual Multi30k used in experiments with NMT models that use WALS features.
# It expects that the data set is already downloaded, tokenized, normalised, and lowercased.
#
# Rationale: The idea is to demonstrate that NMT models that use WALS features are
# better than Google's multilingual NMT (Johnson et al., 2016).
# Google's multilingual system is a standard NMT model and only the data is changed by
# adding a token to the source sentence stating what is the target language, i.e. <2en> or <2fr> .
#
# This script will create a single training file, train a BPE model, and create train/validation sets for:
# French  -> {English}
# German  -> {English, French}
# English -> {French}
# English -> {German}***
#
# *** Note that the English->German is the only language pair that includes German as a target language.
#     However, German is seen as a source language with all two other languages.
#     We will create three versions of the training data set, where we include:
#     (i)   1K English->German examples;
#     (ii)  10K English->German examples;
#     (iii) 29K English->German examples (all examples available in the M30k training data for the language pair);

SUBWORD_HOME="/data/icalixto/subword-nmt/subword_nmt"
PATH_TO_DATA="./data/m30k/multilingual"
[ -d "${PATH_TO_DATA}" ] || (echo "Not found: ${PATH_TO_DATA}" && exit)

# in case we wish to include Czech, do as below (having French as the low-resource target language).
#SRCS=('en' 'en' 'de' 'de' 'fr' 'fr' 'fr' 'cs' 'cs')
#TGTS=('de' 'cs' 'en' 'cs' 'en' 'de' 'cs' 'en' 'de')
#SRCS_WALS=('eng' 'eng' 'ger' 'ger' 'fre' 'fre' 'fre' 'cze' 'cze')
#TGTS_WALS=('ger' 'cze' 'eng' 'cze' 'eng' 'ger' 'cze' 'eng' 'ger')

# not including en->de in the list below
SRCS=('fr' 'de' 'de' 'en')
TGTS=('en' 'en' 'fr' 'fr')
SRCS_WALS=('__fre__' '__ger__' '__ger__' '__eng__')
TGTS_WALS=('__eng__' '__eng__' '__fre__' '__fre__')
IDXS=(0 1 2 3)
# output files' prefixes
TRAIN_OUT="$PATH_TO_DATA/train.lc.norm.tok.all-languages"
VALID_OUT="$PATH_TO_DATA/val.lc.norm.tok.all-languages"
BPE_SUFFIX="bpe.10000"


function process_train_valid_test()
{
    [ -f "${TRAIN_OUT}.src" ] && rm ${TRAIN_OUT}.src
    [ -f "${TRAIN_OUT}.tgt" ] && rm ${TRAIN_OUT}.tgt
    [ -f "${VALID_OUT}.src" ] && rm ${VALID_OUT}.src
    [ -f "${VALID_OUT}.tgt" ] && rm ${VALID_OUT}.tgt

    [ -f "${TRAIN_OUT}.src.langs" ] && rm ${TRAIN_OUT}.src.langs
    [ -f "${TRAIN_OUT}.tgt.langs" ] && rm ${TRAIN_OUT}.tgt.langs
    [ -f "${VALID_OUT}.src.langs" ] && rm ${VALID_OUT}.src.langs
    [ -f "${VALID_OUT}.tgt.langs" ] && rm ${VALID_OUT}.tgt.langs

    echo -ne "Concatenating training/validation data into a single file...\n"
    # concatenate training/validation data into single file
    for idx in ${IDXS[@]}; do
        SRC=${SRCS[${idx}]}
        TGT=${TGTS[${idx}]}
        SRC_WALS=${SRCS_WALS[${idx}]}
        TGT_WALS=${TGTS_WALS[${idx}]}

        TRAIN_SRC="${PATH_TO_DATA}/train.lc.norm.tok.${SRC}"
        TRAIN_TGT="${PATH_TO_DATA}/train.lc.norm.tok.${TGT}"
        VALID_SRC="${PATH_TO_DATA}/val.lc.norm.tok.${SRC}"
        VALID_TGT="${PATH_TO_DATA}/val.lc.norm.tok.${TGT}"

        echo -ne "$SRC $TGT \n"
        [ -f "${TRAIN_SRC}" ] || (echo "Not found: ${TRAIN_SRC}" && exit 1)
        [ -f "${TRAIN_TGT}" ] || (echo "Not found: ${TRAIN_TGT}" && exit 1)
        [ -f "${VALID_SRC}" ] || (echo "Not found: ${VALID_SRC}" && exit 1)
        [ -f "${VALID_TGT}" ] || (echo "Not found: ${VALID_TGT}" && exit 1)

        # training set
        ##############
        # get #lines in single-pair training file
        nlines_train=$(wc -l $TRAIN_SRC | cut -d" " -f1)
        cat $TRAIN_SRC >> "${TRAIN_OUT}.src"
        cat $TRAIN_TGT >> "${TRAIN_OUT}.tgt"
        # write it to '.langs' files to know language splits in concatenated file
        echo -e "${SRC_WALS}\t${nlines_train}" >> "${TRAIN_OUT}.src.langs"
        echo -e "${TGT_WALS}\t${nlines_train}" >> "${TRAIN_OUT}.tgt.langs"

        # validation set
        ################
        # get #lines in single-pair validation file
        nlines_valid=$(wc -l $VALID_SRC | cut -d" " -f1)
        cat $VALID_SRC >> "${VALID_OUT}.src"
        cat $VALID_TGT >> "${VALID_OUT}.tgt"
        # write it to '.langs' files to know language splits in concatenated file
        echo -e "${SRC_WALS}\t${nlines_valid}" >> "${VALID_OUT}.src.langs"
        echo -e "${TGT_WALS}\t${nlines_valid}" >> "${VALID_OUT}.tgt.langs"
    done

    # concatenate English->German
    # create three copies of the training set, one for each number of English->German examples (1K, 10K, 29K)
    cp "${TRAIN_OUT}.src" "${TRAIN_OUT}.de-1K.src"
    cp "${TRAIN_OUT}.src" "${TRAIN_OUT}.de-10K.src"
    cp "${TRAIN_OUT}.src" "${TRAIN_OUT}.de-29K.src"
    cp "${TRAIN_OUT}.src.langs" "${TRAIN_OUT}.de-1K.src.langs"
    cp "${TRAIN_OUT}.src.langs" "${TRAIN_OUT}.de-10K.src.langs"
    cp "${TRAIN_OUT}.src.langs" "${TRAIN_OUT}.de-29K.src.langs"
    cp "${TRAIN_OUT}.tgt" "${TRAIN_OUT}.de-1K.tgt"
    cp "${TRAIN_OUT}.tgt" "${TRAIN_OUT}.de-10K.tgt"
    cp "${TRAIN_OUT}.tgt" "${TRAIN_OUT}.de-29K.tgt"
    cp "${TRAIN_OUT}.tgt.langs" "${TRAIN_OUT}.de-1K.tgt.langs"
    cp "${TRAIN_OUT}.tgt.langs" "${TRAIN_OUT}.de-10K.tgt.langs"
    cp "${TRAIN_OUT}.tgt.langs" "${TRAIN_OUT}.de-29K.tgt.langs"
    # same for valid set
    #cp "${VALID_OUT}.src" "${VALID_OUT}.de-1K.src"
    #cp "${VALID_OUT}.src" "${VALID_OUT}.de-10K.src"
    #cp "${VALID_OUT}.src" "${VALID_OUT}.de-29K.src"
    #cp "${VALID_OUT}.tgt" "${VALID_OUT}.de-1K.tgt"
    #cp "${VALID_OUT}.tgt" "${VALID_OUT}.de-10K.tgt"
    #cp "${VALID_OUT}.tgt" "${VALID_OUT}.de-29K.tgt"
    #cp "${VALID_OUT}.src.langs" "${VALID_OUT}.de-1K.src.langs"
    #cp "${VALID_OUT}.src.langs" "${VALID_OUT}.de-10K.src.langs"
    #cp "${VALID_OUT}.src.langs" "${VALID_OUT}.de-29K.src.langs"
    #cp "${VALID_OUT}.tgt.langs" "${VALID_OUT}.de-1K.tgt.langs"
    #cp "${VALID_OUT}.tgt.langs" "${VALID_OUT}.de-10K.tgt.langs"
    #cp "${VALID_OUT}.tgt.langs" "${VALID_OUT}.de-29K.tgt.langs"

    # single-pair English->German data
    TRAIN_SRC="${PATH_TO_DATA}/train.lc.norm.tok.en"
    TRAIN_TGT="${PATH_TO_DATA}/train.lc.norm.tok.de"
    VALID_SRC="${PATH_TO_DATA}/val.lc.norm.tok.en"
    VALID_TGT="${PATH_TO_DATA}/val.lc.norm.tok.de"

    # 29K English->German
    # training set
    ##############
    # get #lines in single-pair training file
    nlines_29K=$(wc -l $TRAIN_SRC | cut -d" " -f1)
    cat $TRAIN_SRC >> "${TRAIN_OUT}.de-29K.src"
    cat $TRAIN_TGT >> "${TRAIN_OUT}.de-29K.tgt"
    # write it to '.langs' files to know language splits in concatenated file
    echo -e "__eng__\t${nlines_29K}" >> "${TRAIN_OUT}.de-29K.src.langs"
    echo -e "__ger__\t${nlines_29K}" >> "${TRAIN_OUT}.de-29K.tgt.langs"

    # 10K English->German
    # training set
    ##############
    # upsample the low-resource English-German (add it three times ~30K examples)
    nlines_10K=10000
    total_nlines_10K=0
    for _ in $(seq 1 3); do
        head -n${nlines_10K} $TRAIN_SRC >> "${TRAIN_OUT}.de-10K.src"
        head -n${nlines_10K} $TRAIN_TGT >> "${TRAIN_OUT}.de-10K.tgt"
        let total_nlines_10K=${total_nlines_10K}+${nlines_10K}
    done
    # write it to '.langs' files to know language splits in concatenated file
    echo -e "__eng__\t${total_nlines_10K}" >> "${TRAIN_OUT}.de-10K.src.langs"
    echo -e "__ger__\t${total_nlines_10K}" >> "${TRAIN_OUT}.de-10K.tgt.langs"

    # 1K English->German
    # training set
    ##############
    # upsample the low-resource English-German (add it twenty-nine times 29K examples)
    nlines_1K=1000
    total_nlines_1K=0
    for _ in $(seq 1 29); do
        head -n${nlines_1K} $TRAIN_SRC >> "${TRAIN_OUT}.de-1K.src"
        head -n${nlines_1K} $TRAIN_TGT >> "${TRAIN_OUT}.de-1K.tgt"
        let total_nlines_1K=${total_nlines_1K}+${nlines_1K}
    done
    # write it to '.langs' files to know language splits in concatenated file
    echo -e "__eng__\t${total_nlines_1K}" >> "${TRAIN_OUT}.de-1K.src.langs"
    echo -e "__ger__\t${total_nlines_1K}" >> "${TRAIN_OUT}.de-1K.tgt.langs"

    ################
    # validation set
    ################
    # get #lines in single-pair validation file
    nlines_valid=$(wc -l $VALID_SRC | cut -d" " -f1)
    cat $VALID_SRC >> "${VALID_OUT}.src"
    cat $VALID_TGT >> "${VALID_OUT}.tgt"
    # write it to '.langs' files to know language splits in concatenated file
    echo -e "__eng__\t${nlines_valid}" >> "${VALID_OUT}.src.langs"
    echo -e "__ger__\t${nlines_valid}" >> "${VALID_OUT}.tgt.langs"
    #echo -e "__eng__\t${nlines_valid}" >> "${VALID_OUT}.de-1K.src.langs"
    #echo -e "__ger__\t${nlines_valid}" >> "${VALID_OUT}.de-1K.tgt.langs"
    #echo -e "__eng__\t${nlines_valid}" >> "${VALID_OUT}.de-10K.src.langs"
    #echo -e "__ger__\t${nlines_valid}" >> "${VALID_OUT}.de-10K.tgt.langs"
    #echo -e "__eng__\t${nlines_valid}" >> "${VALID_OUT}.de-29K.src.langs"
    #echo -e "__ger__\t${nlines_valid}" >> "${VALID_OUT}.de-29K.tgt.langs"
}

function train_bpe()
{
    # train a BPE model using the concatenated files
    LANGS=('en' 'de' 'fr')
    CONCAT_FILES=('de-1K' 'de-10K' 'de-29K')
    for concat_file in ${CONCAT_FILES[@]}; do
        TRAIN_SRC="${TRAIN_OUT}.${concat_file}.src"
        TRAIN_TGT="${TRAIN_OUT}.${concat_file}.tgt"

        [ -f "${TRAIN_SRC}" ] || (echo "Not found: ${TRAIN_SRC}" && exit)
        [ -f "${TRAIN_TGT}" ] || (echo "Not found: ${TRAIN_TGT}" && exit)

        echo -ne "Training BPE model with $TRAIN_SRC $TRAIN_TGT \n"
        cat $TRAIN_SRC $TRAIN_TGT | python $SUBWORD_HOME/learn_bpe.py -s 30000 -o "${TRAIN_OUT}.${concat_file}.codes-file"

        echo -ne "Applying BPE on train/valid...\n"
        python $SUBWORD_HOME/apply_bpe.py -c ${TRAIN_OUT}.${concat_file}.codes-file < $TRAIN_SRC | python $SUBWORD_HOME/get_vocab.py > ${TRAIN_SRC}.${concat_file}.vocab
        python $SUBWORD_HOME/apply_bpe.py -c ${TRAIN_OUT}.${concat_file}.codes-file < $TRAIN_TGT | python $SUBWORD_HOME/get_vocab.py > ${TRAIN_TGT}.${concat_file}.vocab
        # train
        python $SUBWORD_HOME/apply_bpe.py -c ${TRAIN_OUT}.${concat_file}.codes-file --vocabulary ${TRAIN_SRC}.${concat_file}.vocab < ${TRAIN_SRC} > ${TRAIN_OUT}.${concat_file}.${BPE_SUFFIX}.src
        python $SUBWORD_HOME/apply_bpe.py -c ${TRAIN_OUT}.${concat_file}.codes-file --vocabulary ${TRAIN_TGT}.${concat_file}.vocab < ${TRAIN_TGT} > ${TRAIN_OUT}.${concat_file}.${BPE_SUFFIX}.tgt
        # valid
        python $SUBWORD_HOME/apply_bpe.py -c ${TRAIN_OUT}.${concat_file}.codes-file --vocabulary ${TRAIN_SRC}.${concat_file}.vocab < ${VALID_OUT}.src > ${VALID_OUT}.${concat_file}.${BPE_SUFFIX}.src
        python $SUBWORD_HOME/apply_bpe.py -c ${TRAIN_OUT}.${concat_file}.codes-file --vocabulary ${TRAIN_TGT}.${concat_file}.vocab < ${VALID_OUT}.tgt > ${VALID_OUT}.${concat_file}.${BPE_SUFFIX}.tgt

        # apply BPE
        for src in ${LANGS[@]}; do
            for tgt in ${LANGS[@]}; do
                [ "$src" == "$tgt" ] && continue

                TEST_2016_PREFIX="${PATH_TO_DATA}/test_2016_flickr.lc.norm.tok"
                TEST_2017_PREFIX="${PATH_TO_DATA}/test_2017_flickr.lc.norm.tok"

                # test 2016
                python $SUBWORD_HOME/apply_bpe.py -c ${TRAIN_OUT}.${concat_file}.codes-file --vocabulary ${TRAIN_SRC}.${concat_file}.vocab < ${TEST_2016_PREFIX}.${src} > ${TEST_2016_PREFIX}.${concat_file}.${BPE_SUFFIX}.${src}
                python $SUBWORD_HOME/apply_bpe.py -c ${TRAIN_OUT}.${concat_file}.codes-file --vocabulary ${TRAIN_TGT}.${concat_file}.vocab < ${TEST_2016_PREFIX}.${tgt} > ${TEST_2016_PREFIX}.${concat_file}.${BPE_SUFFIX}.${tgt}
                # test 2017
                python $SUBWORD_HOME/apply_bpe.py -c ${TRAIN_OUT}.${concat_file}.codes-file --vocabulary ${TRAIN_SRC}.${concat_file}.vocab < ${TEST_2017_PREFIX}.${src} > ${TEST_2017_PREFIX}.${concat_file}.${BPE_SUFFIX}.${src}
                python $SUBWORD_HOME/apply_bpe.py -c ${TRAIN_OUT}.${concat_file}.codes-file --vocabulary ${TRAIN_TGT}.${concat_file}.vocab < ${TEST_2017_PREFIX}.${tgt} > ${TEST_2017_PREFIX}.${concat_file}.${BPE_SUFFIX}.${tgt}
            done
        done
    done
}

# pre-process train/valid sets
#process_train_valid_test

# train bpe
#train_bpe

# shuffle minibatches
#echo -ne "Shuffling minibatches...\n"
#python scripts/create_sorted_batches_multilingual.py --m30k

# create preprocessed pytorch files using concatenated, BPE-processed, shuffled/sorted files
LANGS=('en' 'de' 'fr')
CONCAT_FILES=('de-1K' 'de-10K' 'de-29K')
for concat_file in ${CONCAT_FILES[@]}; do
    TRAIN_SRC="${TRAIN_OUT}.${concat_file}.${BPE_SUFFIX}.src.shuf"
    TRAIN_TGT="${TRAIN_OUT}.${concat_file}.${BPE_SUFFIX}.tgt.shuf"
    VALID_SRC="${VALID_OUT}.${concat_file}.${BPE_SUFFIX}.src.shuf"
    VALID_TGT="${VALID_OUT}.${concat_file}.${BPE_SUFFIX}.tgt.shuf"

    [ -f "${TRAIN_SRC}" ] || (echo "Not found: ${TRAIN_SRC}" && exit)
    [ -f "${TRAIN_TGT}" ] || (echo "Not found: ${TRAIN_TGT}" && exit)
    [ -f "${VALID_SRC}" ] || (echo "Not found: ${VALID_SRC}" && exit)
    [ -f "${VALID_TGT}" ] || (echo "Not found: ${VALID_TGT}" && exit)

    echo -ne "Creating pytorch files with 'preprocess.py'\n"
    python preprocess.py \
        -train_src "${TRAIN_SRC}" \
        -train_tgt "${TRAIN_TGT}" \
        -valid_src "${VALID_SRC}" \
        -valid_tgt "${VALID_TGT}" \
        -save_data "${TRAIN_OUT}.pytorch.${concat_file}.bpe" \
        -share_vocab

done

