#!/bin/bash
SUBWORD_HOME="/data/icalixto/subword-nmt/subword_nmt"
PATH_TO_DATA="/data/icalixto/multi30k_multilingual"
OUTPUT_DATA="./data/m30k"
[ -d "${PATH_TO_DATA}" ] || (echo "Not found: ${PATH_TO_DATA}" && exit)
[ -d "${OUTPUT_DATA}" ] || (echo "Not found: ${OUTPUT_DATA}" && exit)

SRCS=('en' 'en' 'en' 'de' 'de' 'de' 'fr' 'fr' 'fr' 'cs' 'cs' 'cs')
TGTS=('de' 'cs' 'fr' 'en' 'cs' 'fr' 'en' 'de' 'cs' 'en' 'de' 'fr')
IDXS=(0 1 2 3 4 5 6 7 8 9 10 11)

for idx in ${IDXS[@]}; do
    SRC=${SRCS[${idx}]}
    TGT=${TGTS[${idx}]}
    TRAIN_PREFIX="$PATH_TO_DATA/train.lc.norm.tok"
    VALID_PREFIX="$PATH_TO_DATA/val.lc.norm.tok"
    TEST2016_PREFIX="$PATH_TO_DATA/test_2016_flickr.lc.norm.tok"
    TEST2017_PREFIX="$PATH_TO_DATA/test_2017_flickr.lc.norm.tok"
    CODES_FILE="${TRAIN_PREFIX}.codes-file"
    TRAIN_SRC="${TRAIN_PREFIX}.${SRC}"
    TRAIN_TGT="${TRAIN_PREFIX}.${TGT}"
    VALID_SRC="${VALID_PREFIX}.${SRC}"
    VALID_TGT="${VALID_PREFIX}.${TGT}"
    TEST2016_SRC="${TEST2016_PREFIX}.${SRC}"
    TEST2016_TGT="${TEST2016_PREFIX}.${TGT}"
    TEST2017_SRC="${TEST2017_PREFIX}.${SRC}"
    TEST2017_TGT="${TEST2017_PREFIX}.${TGT}"

    echo -ne "$SRC $TGT \n"
    [ -f "${TRAIN_SRC}" ] || (echo "Not found: ${TRAIN_SRC}" && exit)
    [ -f "${TRAIN_TGT}" ] || (echo "Not found: ${TRAIN_TGT}" && exit)
    [ -f "${VALID_SRC}" ] || (echo "Not found: ${VALID_SRC}" && exit)
    [ -f "${VALID_TGT}" ] || (echo "Not found: ${VALID_TGT}" && exit)
    [ -f "${TEST2016_SRC}" ] || (echo "Not found: ${TEST2016_SRC}" && exit)
    [ -f "${TEST2016_TGT}" ] || (echo "Not found: ${TEST2016_TGT}" && exit)
    [ -f "${TEST2017_SRC}" ] || (echo "Not found: ${TEST2017_SRC}" && exit)
    [ -f "${TEST2017_TGT}" ] || (echo "Not found: ${TEST2017_TGT}" && exit)

    echo -ne "Training BPE model for $SRC $TGT \n"
    cat $TRAIN_SRC $TRAIN_TGT | python $SUBWORD_HOME/learn_bpe.py -s 10000 -o "${CODES_FILE}"
    echo -ne "Applying BPE on train/valid...\n"
    python $SUBWORD_HOME/apply_bpe.py -c ${CODES_FILE} < $TRAIN_SRC | python $SUBWORD_HOME/get_vocab.py > ${TRAIN_PREFIX}.vocab.${SRC}-${TGT}.$SRC
    python $SUBWORD_HOME/apply_bpe.py -c ${CODES_FILE} < $TRAIN_TGT | python $SUBWORD_HOME/get_vocab.py > ${TRAIN_PREFIX}.vocab.${SRC}-${TGT}.$TGT
    # train
    python $SUBWORD_HOME/apply_bpe.py -c ${CODES_FILE} --vocabulary ${TRAIN_PREFIX}.vocab.${SRC}-${TGT}.$SRC < ${TRAIN_SRC} > ${TRAIN_PREFIX}.single-pair.bpe.${SRC}-${TGT}.$SRC
    python $SUBWORD_HOME/apply_bpe.py -c ${CODES_FILE} --vocabulary ${TRAIN_PREFIX}.vocab.${SRC}-${TGT}.$TGT < ${TRAIN_TGT} > ${TRAIN_PREFIX}.single-pair.bpe.${SRC}-${TGT}.$TGT
    # valid
    python $SUBWORD_HOME/apply_bpe.py -c ${CODES_FILE} --vocabulary ${TRAIN_PREFIX}.vocab.${SRC}-${TGT}.$SRC < ${VALID_SRC} > ${VALID_PREFIX}.single-pair.bpe.${SRC}-${TGT}.$SRC
    python $SUBWORD_HOME/apply_bpe.py -c ${CODES_FILE} --vocabulary ${TRAIN_PREFIX}.vocab.${SRC}-${TGT}.$TGT < ${VALID_TGT} > ${VALID_PREFIX}.single-pair.bpe.${SRC}-${TGT}.$TGT
    # test 2016
    python $SUBWORD_HOME/apply_bpe.py -c ${CODES_FILE} --vocabulary ${TRAIN_PREFIX}.vocab.${SRC}-${TGT}.$SRC < ${TEST2016_SRC} > ${TEST2016_PREFIX}.single-pair.bpe.${SRC}-${TGT}.$SRC
    python $SUBWORD_HOME/apply_bpe.py -c ${CODES_FILE} --vocabulary ${TRAIN_PREFIX}.vocab.${SRC}-${TGT}.$TGT < ${TEST2016_TGT} > ${TEST2016_PREFIX}.single-pair.bpe.${SRC}-${TGT}.$TGT
    # test 2017
    python $SUBWORD_HOME/apply_bpe.py -c ${CODES_FILE} --vocabulary ${TRAIN_PREFIX}.vocab.${SRC}-${TGT}.$SRC < ${TEST2017_SRC} > ${TEST2017_PREFIX}.single-pair.bpe.${SRC}-${TGT}.$SRC
    python $SUBWORD_HOME/apply_bpe.py -c ${CODES_FILE} --vocabulary ${TRAIN_PREFIX}.vocab.${SRC}-${TGT}.$TGT < ${TEST2017_TGT} > ${TEST2017_PREFIX}.single-pair.bpe.${SRC}-${TGT}.$TGT
    echo -ne "Finished applying BPE.\n"

    python preprocess.py \
        -train_src "${TRAIN_PREFIX}.single-pair.bpe.${SRC}-${TGT}.$SRC" \
        -train_tgt "${TRAIN_PREFIX}.single-pair.bpe.${SRC}-${TGT}.$TGT" \
        -valid_src "${VALID_PREFIX}.single-pair.bpe.${SRC}-${TGT}.$SRC" \
        -valid_tgt "${VALID_PREFIX}.single-pair.bpe.${SRC}-${TGT}.$TGT" \
        -save_data "$OUTPUT_DATA/single-pair.bpe.${SRC}-${TGT}" \
        -share_vocab
done

