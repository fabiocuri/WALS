#!/usr/bin/env bash

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# set this to your moses and subword paths
MOSES_HOME=/data/icalixto/mosesdecoder
SUBWORD_HOME=/data/icalixto/subword-nmt/subword_nmt
DATASET=/data/icalixto/iwslt2017-multilingual/DeEnItNlRo-DeEnItNlRo

# used to concatenate training and validation data
PADDING_TOKEN="<blank>"

# file to hold training data for all language pair combinations
TRAIN_ALL_PREFIX=train.tags.lc.tok.norm.clean.all-languages
TRAIN_ALL_SRC=${TRAIN_ALL_PREFIX}.src
TRAIN_ALL_TGT=${TRAIN_ALL_PREFIX}.tgt
# file to hold validation data for all language pair combinations
VALID_ALL_PREFIX=IWSLT17.TED.dev2010.lc.tok.norm.all-languages
VALID_ALL_SRC=${VALID_ALL_PREFIX}.src
VALID_ALL_TGT=${VALID_ALL_PREFIX}.tgt

TRAIN_MBATCH_SIZE=100
VALID_MBATCH_SIZE=8

function process_train()
{
    # choose src and target languages
    SRC=$1
    TGT=$2
    PREFIX="train.tags.$SRC-$TGT"
    echo "Downloading IWSLT 2017 ${SRC}-${TGT} data..."

    echo "Removing xml from train files..."
    grep '^[[:blank:]]*[^[:blank:]<]' $DATASET/${PREFIX}.${SRC} > $DATASET/${PREFIX}.raw.${SRC} &
    grep '^[[:blank:]]*[^[:blank:]<]' $DATASET/${PREFIX}.${TGT} > $DATASET/${PREFIX}.raw.${TGT} &
    wait

    echo "Tokenizing..."
    ${MOSES_HOME}/scripts/tokenizer/tokenizer.perl -q -l ${SRC} -threads 4 < $DATASET/${PREFIX}.raw.${SRC} > $DATASET/${PREFIX}.tok.${SRC} &
    ${MOSES_HOME}/scripts/tokenizer/tokenizer.perl -q -l ${TGT} -threads 4 < $DATASET/${PREFIX}.raw.${TGT} > $DATASET/${PREFIX}.tok.${TGT} &
    wait

    echo "Cleaning..."
    ${MOSES_HOME}/scripts/training/clean-corpus-n.perl $DATASET/${PREFIX}.tok ${SRC} ${TGT} "$DATASET/${PREFIX}.tok.clean" 1 80
    wc -l $DATASET/${PREFIX}.tok.clean.*

    echo "Lowercasing..."
    ${MOSES_HOME}/scripts/tokenizer/lowercase.perl < $DATASET/${PREFIX}.tok.clean.${SRC} > $DATASET/${PREFIX}.lc.tok.clean.${SRC} &
    ${MOSES_HOME}/scripts/tokenizer/lowercase.perl < $DATASET/${PREFIX}.tok.clean.${TGT} > $DATASET/${PREFIX}.lc.tok.clean.${TGT} &
    wait;

    echo "Normalising punctuation..."
    ${MOSES_HOME}/scripts/tokenizer/normalize-punctuation.perl -l $SRC < $DATASET/${PREFIX}.lc.tok.clean.${SRC} > $DATASET/${PREFIX}.lc.tok.norm.clean.${SRC} &
    ${MOSES_HOME}/scripts/tokenizer/normalize-punctuation.perl -l $SRC < $DATASET/${PREFIX}.lc.tok.clean.${TGT} > $DATASET/${PREFIX}.lc.tok.norm.clean.${TGT} &
    wait;
}

function process_dev_test()
{
    # choose src and target languages
    SRC=$1
    TGT=$2
    # dev prefix already includes validation (IWSLT17.TED.dev2010) and test (IWSLT17.TED.tst2010)
    DEV_PREFIX="IWSLT17.TED"
    echo "Downloading IWSLT 2017 ${SRC}-${TGT} data..."

    echo "Removing xml from dev/test files..."
    for f in $DATASET/${DEV_PREFIX}*.xml; do
            cat ${f} | ${MOSES_HOME}/scripts/ems/support/input-from-sgm.perl > ${f%.xml}
    done

    echo "Tokenizing dev/test data sets..."
    for f in $DATASET/${DEV_PREFIX}*.??-??.${SRC}; do
            ${MOSES_HOME}/scripts/tokenizer/tokenizer.perl -q -l ${SRC} -threads 4 < ${f} > ${f%.*}.tok.${SRC} &
    done
    for f in $DATASET/${DEV_PREFIX}*.??-??.${TGT}; do
            ${MOSES_HOME}/scripts/tokenizer/tokenizer.perl -q -l ${TGT} -threads 4 < ${f} > ${f%.*}.tok.${TGT} &
    done
    wait

    echo "Lowercasing dev/test data sets..."
    for f in $DATASET/${DEV_PREFIX}*.??-??.tok.${SRC}; do
            ${MOSES_HOME}/scripts/tokenizer/lowercase.perl < ${f} > ${f%.*}.lc.${SRC} & \
    done
    for f in $DATASET/${DEV_PREFIX}*.??-??.tok.${TGT}; do
            ${MOSES_HOME}/scripts/tokenizer/lowercase.perl < ${f} > ${f%.*}.lc.${TGT} &
    done
    wait;

    echo "Normalising punctuation dev/test data sets..."
    for f in $DATASET/${DEV_PREFIX}*.??-??.tok.lc.${SRC}; do
            ${MOSES_HOME}/scripts/tokenizer/lowercase.perl < ${f} > ${f%.*}.norm.${SRC} & \
    done
    for f in $DATASET/${DEV_PREFIX}*.??-??.tok.lc.${TGT}; do
            ${MOSES_HOME}/scripts/tokenizer/lowercase.perl < ${f} > ${f%.*}.norm.${TGT} &
    done
    wait;
}


function process_bpe_one_language_pair()
{
    SRC=$1
    TGT=$2
    echo -ne "Train a BPE model using ${SRC} ${TGT}\n"

    MERGE_OPS=30000
    # /data/icalixto/iwslt2017-multilingual/DeEnItNlRo-DeEnItNlRo/train.tags.de-en.lc.tok.norm.clean.en
    PREFIX="train.tags.${SRC}-${TGT}.lc.tok.norm.clean"
    # dev prefix already includes validation (IWSLT17.TED.dev2010) and test (IWSLT17.TED.tst2010)
    DEV_PREFIX="IWSLT17.TED"
    OUTPUT_DIR="./data/iwslt17/bpe"
    #echo ${OUTPUT_DIR}
    #echo "Writing to ${OUTPUT_DIR}. To change this, give OUTPUT_DIR as argument."
    mkdir -p ${OUTPUT_DIR}
    #exit 1;

    if [ -f "${OUTPUT_DIR}/bpe.${SRC}-${TGT}.${MERGE_OPS}.codes-file" ]; then
        echo "Codes file found ($SRC-$TGT)."
    else
        echo "Learning joint (bilingual) BPE with ${MERGE_OPS} merges. This may take a while..."
        cat ${DATASET}/${PREFIX}.${SRC} ${DATASET}/${PREFIX}.${TGT} | python ${SUBWORD_HOME}/learn_bpe.py \
                    -s ${MERGE_OPS} \
                    -o ${OUTPUT_DIR}/bpe.${SRC}-${TGT}.${MERGE_OPS}.codes-file
    fi
    if [ -f "${DATASET}/${PREFIX}.bpe.${MERGE_OPS}.${SRC}" ]; then
        echo "BPE-preprocessed file found: ${DATASET}/${PREFIX}.bpe.${MERGE_OPS}.${SRC}"
    else
        echo "Applying BPE..."
        python ${SUBWORD_HOME}/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${SRC}-${TGT}.${MERGE_OPS}.codes-file" < ${DATASET}/${PREFIX}.${SRC} > ${DATASET}/${PREFIX}.bpe.${MERGE_OPS}.${SRC} &
        python ${SUBWORD_HOME}/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${SRC}-${TGT}.${MERGE_OPS}.codes-file" < ${DATASET}/${PREFIX}.${TGT} > ${DATASET}/${PREFIX}.bpe.${MERGE_OPS}.${TGT} &
        wait
    fi

    for f in ${DATASET}/${DEV_PREFIX}*.${SRC}-${TGT}.tok.lc.norm.${SRC}; do
        if [ -f "${f%.*}.bpe.${MERGE_OPS}.${SRC}" ]; then
            echo "BPE-preprocessed file found: ${f%.*}.bpe.${MERGE_OPS}.${SRC}"
        else
            echo "Applying BPE on dev/test data sets (src)..."
            # /data/icalixto/iwslt2017-multilingual/DeEnItNlRo-DeEnItNlRo/IWSLT17.TED.dev2010.de-en.tok.lc.norm.en
            # /data/icalixto/iwslt2017-multilingual/DeEnItNlRo-DeEnItNlRo/IWSLT17.TED.tst2010.en-it.tok.lc.norm.en
            ${SUBWORD_HOME}/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${SRC}-${TGT}.${MERGE_OPS}.codes-file" < ${f} > ${f%.*}.bpe.${MERGE_OPS}.${SRC}
        fi
    done
    for f in ${DATASET}/${DEV_PREFIX}*.${SRC}-${TGT}.tok.lc.norm.${TGT}; do
        if [ -f "${f%.*}.bpe.${MERGE_OPS}.${TGT}" ]; then
            echo "BPE-preprocessed file found: ${f%.*}.bpe.${MERGE_OPS}.${TGT}"
        else
            echo "Applying BPE on dev/test data sets (tgt)..."
            # /data/icalixto/iwslt2017-multilingual/DeEnItNlRo-DeEnItNlRo/IWSLT17.TED.dev2010.de-en.tok.lc.norm.en
            # /data/icalixto/iwslt2017-multilingual/DeEnItNlRo-DeEnItNlRo/IWSLT17.TED.tst2010.en-it.tok.lc.norm.en
            ${SUBWORD_HOME}/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${SRC}-${TGT}.${MERGE_OPS}.codes-file" < ${f} > ${f%.*}.bpe.${MERGE_OPS}.${TGT}
        fi
    done

    if [ -f "${OUTPUT_DIR}/vocab.bpe.${SRC}-${TGT}.${MERGE_OPS}.${SRC}" ]; then
        continue
    else
        echo "Creating vocabularies"
        cat ${DATASET}/${PREFIX}.bpe.${MERGE_OPS}.${SRC} | ${SUBWORD_HOME}/get_vocab.py \
            | cut -f1 -d ' ' >> "${OUTPUT_DIR}/vocab.bpe.${SRC}-${TGT}.${MERGE_OPS}.${SRC}"
    fi
    if [ -f "${OUTPUT_DIR}/vocab.bpe.${SRC}-${TGT}.${MERGE_OPS}.${TGT}" ]; then
        continue
    else
        cat ${DATASET}/${PREFIX}.bpe.${MERGE_OPS}.${TGT} | ${SUBWORD_HOME}/get_vocab.py \
            | cut -f1 -d ' ' >> "${OUTPUT_DIR}/vocab.bpe.${SRC}-${TGT}.${MERGE_OPS}.${TGT}"
    fi

    echo "Done"
}

function process_bpe_all_languages()
{
    echo -ne "Train one joint BPE model using all languages\n"

    MERGE_OPS=30000
    #PATH_TO_DATA="/data/icalixto/iwslt2017-multilingual/DeEnItNlRo-DeEnItNlRo"
    #PREFIX="train.tags.lc.tok.norm.clean.all-languages"
    LOW_RESOURCE_SIZES=(10000 100000 200000)
    DEV_PREFIX="IWSLT17.TED"
    OUTPUT_DIR="./data/iwslt17/all-languages"
    #OUTPUT_DIR="${1:-iwslt17_deen}"
    echo ${OUTPUT_DIR}

    echo "Writing to ${OUTPUT_DIR}. To change this, give OUTPUT_DIR as argument."

    mkdir -p ${OUTPUT_DIR}
    #cd ${OUTPUT_DIR}
    #exit 1;

    for size in ${LOW_RESOURCE_SIZES[@]};
    do
        echo -ne "Training BPE models for low-resource data (size $size) ...\n"
        PREFIX="train.tags.lc.tok.norm.clean.all-languages.low-resource-${size}"

        echo "Learning BPE with ${MERGE_OPS} merges. This may take a while..."
        cat ${DATASET}/${PREFIX}.src ${DATASET}/${PREFIX}.tgt | python ${SUBWORD_HOME}/learn_bpe.py \
                    -s ${MERGE_OPS} \
                    -o ${OUTPUT_DIR}/bpe.all-languages.low-resource-${size}.${MERGE_OPS}.codes-file

        echo "Applying BPE..."
        python ${SUBWORD_HOME}/apply_bpe.py -c "${OUTPUT_DIR}/bpe.all-languages.low-resource-${size}.${MERGE_OPS}.codes-file" \
            < ${DATASET}/${PREFIX}.src \
            > ${DATASET}/${PREFIX}.bpe.${MERGE_OPS}.src &
        python ${SUBWORD_HOME}/apply_bpe.py -c "${OUTPUT_DIR}/bpe.all-languages.low-resource-${size}.${MERGE_OPS}.codes-file" \
            < ${DATASET}/${PREFIX}.tgt \
            > ${DATASET}/${PREFIX}.bpe.${MERGE_OPS}.tgt &

        #${SUBWORD_HOME}/apply_bpe.py -c "bpe.${MERGE_OPS}.${SRC}" < ${PREFIX}.lc.tok.norm.clean.${SRC} > ${PREFIX}.bpe.${MERGE_OPS}.${SRC} &
        #${SUBWORD_HOME}/apply_bpe.py -c "bpe.${MERGE_OPS}.${TGT}" < ${PREFIX}.lc.tok.norm.clean.${TGT} > ${PREFIX}.bpe.${MERGE_OPS}.${TGT} &
        wait

        echo "Applying BPE on dev/test data sets..."
        #IWSLT17.TED.dev2010.lc.tok.norm.all-languages.src
        for f in ${DATASET}/${DEV_PREFIX}*.lc.tok.norm.all-languages.src; do
                ${SUBWORD_HOME}/apply_bpe.py -c "${OUTPUT_DIR}/bpe.all-languages.low-resource-${size}.${MERGE_OPS}.codes-file" \
                    < ${f} \
                    > ${f%.*}.low-resource-${size}.bpe.${MERGE_OPS}.src
        done
        for f in ${DATASET}/${DEV_PREFIX}*.lc.tok.norm.all-languages.tgt; do
                ${SUBWORD_HOME}/apply_bpe.py -c "${OUTPUT_DIR}/bpe.all-languages.low-resource-${size}.${MERGE_OPS}.codes-file" \
                    < ${f} \
                    > ${f%.*}.low-resource-${size}.bpe.${MERGE_OPS}.tgt
        done

        echo "Creating vocabularies"
        #echo -e "<unk>\n<s>\n</s>" > "vocab.bpe.${MERGE_OPS}.src"
        #echo -e "<unk>\n<s>\n</s>" > "vocab.bpe.${MERGE_OPS}.tgt"

        cat ${DATASET}/${PREFIX}.bpe.${MERGE_OPS}.src | ${SUBWORD_HOME}/get_vocab.py \
            | cut -f1 -d ' ' >> "${OUTPUT_DIR}/vocab.low-resource-${size}.bpe.${MERGE_OPS}.src"
        cat ${DATASET}/${PREFIX}.bpe.${MERGE_OPS}.tgt | ${SUBWORD_HOME}/get_vocab.py \
            | cut -f1 -d ' ' >> "${OUTPUT_DIR}/vocab.low-resource-${size}.bpe.${MERGE_OPS}.tgt"
    done

    echo "Done"
}

function _pad_training_data()
{
    src=$1
    tgt=$2

    # if pre-processed file has an odd number of lines
    # create the missing lines with one single padding token (to have a file with length multiple of 100)
    nlines=$(wc -l "$DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.${src}" | cut -d" " -f1)
    nover=$((nlines%TRAIN_MBATCH_SIZE))
    nmissing=$((100-nover))

    start=1
    end=$nmissing
    PADDING=""
    for ((i=$start; i<=$end; i++)); do
        PADDING="${PADDING_TOKEN}\n$PADDING"
    done
    #echo -ne "$src $tgt $nlines $nover $nmissing\n"

    cat $DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.${src} > $DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.padded.${src}
    echo -ne "$PADDING" >> $DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.padded.${src}
    cat $DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.${tgt} > $DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.padded.${tgt}
    echo -ne "$PADDING" >> $DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.padded.${tgt}

    nlines_before=$(wc -l "$DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.${src}" | cut -d" " -f1)
    nlines_after=$(wc -l "$DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.padded.${src}" | cut -d" " -f1)
    echo -ne "$src $tgt : $nlines_before / $nlines_after (before/after padding)\n"
}

function _pad_valid_data()
{
    src=$1
    tgt=$2

    # if pre-processed file has an odd number of lines
    # create the missing lines with one single padding token (to have a file with length multiple of 8)
    VAL_SRC=IWSLT17.TED.dev2010.$src-$tgt.tok.lc.norm.${src}
    VAL_TGT=IWSLT17.TED.dev2010.$src-$tgt.tok.lc.norm.${tgt}
    nlines=$(wc -l "$DATASET/$VAL_SRC" | cut -d" " -f1)
    nover=$((nlines%VALID_MBATCH_SIZE))
    nmissing=$((100-nover))

    start=1
    end=$nmissing
    PADDING=""
    for ((i=$start; i<=$end; i++)); do
        PADDING="${PADDING_TOKEN}\n$PADDING"
    done
    #echo -ne "$src $tgt $nlines $nover $nmissing\n"

    VAL_SRC_PADDED=IWSLT17.TED.dev2010.$src-$tgt.tok.lc.norm.padded.${src}
    VAL_TGT_PADDED=IWSLT17.TED.dev2010.$src-$tgt.tok.lc.norm.padded.${tgt}

    cat $DATASET/$VAL_SRC > $DATASET/${VAL_SRC_PADDED}
    echo -ne "$PADDING" >> $DATASET/${VAL_SRC_PADDED}
    cat $DATASET/$VAL_TGT > $DATASET/${VAL_TGT_PADDED}
    echo -ne "$PADDING" >> $DATASET/${VAL_TGT_PADDED}

    nlines_before=$(wc -l "$DATASET/${VAL_SRC}" | cut -d" " -f1)
    nlines_after=$(wc -l "$DATASET/${VAL_SRC_PADDED}" | cut -d" " -f1)
    echo -ne "$src $tgt : $nlines_before / $nlines_after (before/after padding)\n"
}

function pad_data()
{
    # pad both "full" and low-resource language pairs

    # iterate through "full" language pair combinations
    for idx in ${IDXS[@]}; do
        src=${SRCS[$idx]}
        tgt=${TGTS[$idx]}

        echo -ne "Padding (full-resource) $src $tgt ...\n"

        # add paddings to training/validation data (make sure data set size is a multiple of minibatch size)
        _pad_training_data $src $tgt
        _pad_valid_data $src $tgt
    done

    # iterate through low-resource language pair combinations
    for idx in ${IDXS_LR[@]}; do
        src=${SRCS_LR[$idx]}
        tgt=${TGTS_LR[$idx]}

        echo -ne "Padding (low-resource) $src $tgt ...\n"

        # add paddings to training/validation data (make sure data set size is a multiple of minibatch size)
        _pad_training_data $src $tgt
        _pad_valid_data $src $tgt
    done
}

function concatenate_data()
{
    # TRAIN_ALL_PREFIX=train.tags.lc.tok.norm.clean.all-languages
    # /data/icalixto/iwslt2017-multilingual/DeEnItNlRo-DeEnItNlRo/train.tags.lc.tok.norm.clean.all-languages.???.langs
    #if ls "$DATASET/${TRAIN_ALL_PREFIX}*" 1> /dev/null 2>&1; then rm $DATASET/${TRAIN_ALL_PREFIX}*; fi
    #if ls "$DATASET/${VALID_ALL_PREFIX}*" 1> /dev/null 2>&1; then rm $DATASET/${VALID_ALL_PREFIX}*; fi
    # check if concatenated training data already exists and delete it if it is the case
    for f in $DATASET/${TRAIN_ALL_PREFIX}*; do
        # if at least one file exists, the test below will return 
        [ -e "$f" ] && (echo "Training files do exist" && rm $DATASET/${TRAIN_ALL_PREFIX}*) || echo "Training files do not exist"
        break
    done
    # check if concatenated valid data already exists and delete it if it is the case
    for f in $DATASET/${VALID_ALL_PREFIX}*; do
        # if at least one file exists, the test below will return 
        [ -e "$f" ] && (echo "Valid files do exist" && rm $DATASET/${VALID_ALL_PREFIX}*) || echo "Valid files do not exist"
        break
    done

    # iterate through all language pair combinations allowed
    for idx in ${IDXS[@]}; do
        src=${SRCS[$idx]}
        tgt=${TGTS[$idx]}

        echo -ne "Concatenate (full-resource) padded $src $tgt ...\n"

        # concatenate padded training/validation data to final training/valid set
        cat $DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.padded.${src} >> $DATASET/$TRAIN_ALL_SRC
        cat $DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.padded.${tgt} >> $DATASET/$TRAIN_ALL_TGT
        VAL_SRC_PADDED=IWSLT17.TED.dev2010.$src-$tgt.tok.lc.norm.padded.${src}
        VAL_TGT_PADDED=IWSLT17.TED.dev2010.$src-$tgt.tok.lc.norm.padded.${tgt}
        cat $DATASET/${VAL_SRC_PADDED} >> $DATASET/$VALID_ALL_SRC
        cat $DATASET/${VAL_TGT_PADDED} >> $DATASET/$VALID_ALL_TGT

        # concatenate number of lines each language pair takes in ".langs" file
        language_codes_suffix=".langs"
        nlines_train=$(wc -l "$DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.padded.${src}" | cut -d" " -f1)
        nlines_valid=$(wc -l "$DATASET/${VAL_SRC_PADDED}" | cut -d" " -f1)
        lang_token_src="__${src}__"
        lang_token_tgt="__${tgt}__"
        echo -ne "${lang_token_src}\t${nlines_train}\n" >> "$DATASET/${TRAIN_ALL_SRC}${language_codes_suffix}"
        echo -ne "${lang_token_tgt}\t${nlines_train}\n" >> "$DATASET/${TRAIN_ALL_TGT}${language_codes_suffix}"
        echo -ne "${lang_token_src}\t${nlines_valid}\n" >> "$DATASET/${VALID_ALL_SRC}${language_codes_suffix}"
        echo -ne "${lang_token_tgt}\t${nlines_valid}\n" >> "$DATASET/${VALID_ALL_TGT}${language_codes_suffix}"
    done

    echo -ne "Concatenated training data found in:\n\t$TRAIN_ALL_SRC\n\t$TRAIN_ALL_TGT\n"
    echo -ne "Concatenated validation data found in:\n\t$VALID_ALL_SRC\n\t$VALID_ALL_TGT\n"
}

function concatenate_data_low_resource()
{
    # idxs used to concomitantly iterate (i.e. zip) the arrays below (LOW_RESOURCE_SIZES, CONCAT_N_TIMES)
    IDXS_INTERNAL=(0 1 2)
    # number of sentences from the original training data to select
    LOW_RESOURCE_SIZES=(10000 100000 200000)
    # how many times to concatenate selected sentences? (up-sampling)
    CONCAT_N_TIMES=(20 2 1)

    language_codes_suffix=".langs"

    for size in ${LOW_RESOURCE_SIZES[@]}; do
        echo -ne "Copying training data for low-resource size $size ...\n"
        # create a copy of the original training file for the low-resource size
        cp $DATASET/$TRAIN_ALL_SRC $DATASET/${TRAIN_ALL_PREFIX}.low-resource-${size}.src
        cp $DATASET/$TRAIN_ALL_TGT $DATASET/${TRAIN_ALL_PREFIX}.low-resource-${size}.tgt
        # create a copy of the file with the current training language codes and line numbers for each low-resource copy
        cp "$DATASET/${TRAIN_ALL_SRC}${language_codes_suffix}" "$DATASET/${TRAIN_ALL_PREFIX}.low-resource-${size}.src${language_codes_suffix}"
        cp "$DATASET/${TRAIN_ALL_TGT}${language_codes_suffix}" "$DATASET/${TRAIN_ALL_PREFIX}.low-resource-${size}.tgt${language_codes_suffix}"
    done

    # iterate through low-resource language pair combinations
    for idx in ${IDXS_LR[@]}; do
        src=${SRCS_LR[$idx]}
        tgt=${TGTS_LR[$idx]}

        echo -ne "Concatenate low-resource (cropped) $src $tgt ...\n"

        ###############################
        # CONCATENATE LOW-RESOURCE DATA
        ###############################

        # concatenate full validation data to final valid set
        VAL_SRC=IWSLT17.TED.dev2010.$src-$tgt.tok.lc.norm.${src}
        VAL_TGT=IWSLT17.TED.dev2010.$src-$tgt.tok.lc.norm.${tgt}
        cat $DATASET/${VAL_SRC} >> $DATASET/$VALID_ALL_SRC
        cat $DATASET/${VAL_TGT} >> $DATASET/$VALID_ALL_TGT

        # concatenate number of lines each language pair takes in ".langs" file
        lang_token_src="__${src}__"
        lang_token_tgt="__${tgt}__"
        nlines_valid=$(wc -l "$DATASET/${VAL_SRC}" | cut -d" " -f1)
        echo -ne "${lang_token_src}\t${nlines_valid}\n" >> "$DATASET/${VALID_ALL_SRC}${language_codes_suffix}"
        echo -ne "${lang_token_tgt}\t${nlines_valid}\n" >> "$DATASET/${VALID_ALL_TGT}${language_codes_suffix}"

        #for size in ${LOW_RESOURCE_SIZES[@]}; do
        for idx_internal in ${IDXS_INTERNAL[@]}; do
            size=${LOW_RESOURCE_SIZES[$idx_internal]}
            ntimes=${CONCAT_N_TIMES[$idx_internal]}

            echo -ne "Creating training data for low-resource size $size ...\n"

            # sanity check
            nlines_full=$(wc -l "$DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.${src}" | cut -d" " -f1)
            [ "$nlines_full" -ge "$size" ] || (echo -ne "ERROR: Number of lines in original training file ($nlines_full) smaller than low-resource/cropped size ($size)!" && exit 1)

            # variable to hold number of lines added to the concatenated file
            nlines_train_final=0

            # upsampling? concatenate training data `$ntimes`
            for _ in $(seq 1 $ntimes);
            do
                # concatenate trimmed training data to final training set
                head -n$size $DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.${src} >> $DATASET/${TRAIN_ALL_PREFIX}.low-resource-${size}.src
                head -n$size $DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.${tgt} >> $DATASET/${TRAIN_ALL_PREFIX}.low-resource-${size}.tgt
                # update counter with number of lines concatenated insofar
                nlines_train=$(head -n$size "$DATASET/train.tags.$src-$tgt.lc.tok.norm.clean.${src}" | wc -l | cut -d" " -f1)
                let nlines_train_final=${nlines_train_final}+${nlines_train}
            done

            # concatenate number of lines each language pair takes in ".langs" file
            echo -ne "${lang_token_src}\t${nlines_train_final}\n" >> "$DATASET/${TRAIN_ALL_PREFIX}.low-resource-${size}.src${language_codes_suffix}"
            echo -ne "${lang_token_tgt}\t${nlines_train_final}\n" >> "$DATASET/${TRAIN_ALL_PREFIX}.low-resource-${size}.tgt${language_codes_suffix}"
        done
    done

    echo -ne "Concatenated low-resource training data found in:\n\t$TRAIN_ALL_SRC\n\t$TRAIN_ALL_TGT\n"
    echo -ne "Concatenated low-resource validation data found in:\n\t$VALID_ALL_SRC\n\t$VALID_ALL_TGT\n"
}

function create_pytorch_files_preprocess()
{
    # create preprocessed pytorch files using concatenated, BPE-processed, shuffled/sorted files
    BPE_SUFFIX="bpe.30000"
    # number of sentences from the original training data to select
    LOW_RESOURCE_SIZES=(10000 100000 200000)
    for size in ${LOW_RESOURCE_SIZES[@]}; do
        # $DATASET/${TRAIN_ALL_PREFIX}.low-resource-${size}.src
        # $DATASET/${TRAIN_ALL_PREFIX}.low-resource-${size}.src${language_codes_suffix}
        TRAIN_SRC="$DATASET/${TRAIN_ALL_PREFIX}.low-resource-${size}.${BPE_SUFFIX}.src.shuf"
        TRAIN_TGT="$DATASET/${TRAIN_ALL_PREFIX}.low-resource-${size}.${BPE_SUFFIX}.tgt.shuf"
        VALID_SRC="$DATASET/${VALID_ALL_PREFIX}.low-resource-${size}.${BPE_SUFFIX}.src.shuf"
        VALID_TGT="$DATASET/${VALID_ALL_PREFIX}.low-resource-${size}.${BPE_SUFFIX}.tgt.shuf"

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
            -save_data "$DATASET/${TRAIN_ALL_PREFIX}.pytorch.low-resource-${size}.${BPE_SUFFIX}" \
            -share_vocab

    done
}

# source-target pairs
SRCS=(de en nl it ro)
TGTS=(de en nl it ro)

# TRAINING DATA
# iterate through all possible language pair combinations and pre-process (no BPE)
for src in ${SRCS[@]}; do
    for tgt in ${TGTS[@]}; do
        if [ "$src" != "$tgt" ]; then
            continue
            # pre-process the data: tokenize, lowercase, normalise punctuation, clean
            #echo -ne "Processing $src $tgt (train) ...\n"
            #process_train $src $tgt
            
        fi
    done
done

# DEV DATA
# iterate through all possible language pair combinations and pre-process (no BPE)
for src in ${SRCS[@]}; do
    for tgt in ${TGTS[@]}; do
        if [ "$src" != "$tgt" ]; then
            continue
            # pre-process the data: tokenize, lowercase, normalise punctuation (NOT cleaning)
            #echo -ne "Processing $src $tgt (dev/test) ...\n"
            #process_dev_test $src $tgt
        fi
    done
done

# fully used language-pairs
SRCS=(de de en en ro ro nl nl nl it it it)
TGTS=(en ro de ro en de en de ro en de ro)
IDXS=(0 1 2 3 4 5 6 7 8 9 10 11)

# concatenate the preprocessed data
#pad_data

# concatenate the preprocessed, padded data
#concatenate_data

# low-resource language pairs (used with limitations)
SRCS_LR=(en en)
TGTS_LR=(nl it)
IDXS_LR=(0 1)

# concatenate the preprocessed data
#concatenate_data_low_resource

# train a BPE model using all languages
#process_bpe_all_languages


# train multiple BPE models for each language pair
# iterate through all possible language pair combinations and pre-process (no BPE)
LANGUAGES=(en de ro it nl)
for src in ${LANGUAGES[@]}; do
    for tgt in ${LANGUAGES[@]}; do
        [ "$src" == "$tgt" ] && continue
        echo -ne "Training BPE models for each language pair $src $tgt ...\n"
        process_bpe_one_language_pair $src $tgt
    done
done

# shuffle minibatches
#echo -ne "Shuffling minibatches...\n"
#ipython scripts/create_sorted_batches_multilingual.py --iwslt17

# create pytorch train/valid/vocab files (with `preprocess.py`)
#create_pytorch_files_preprocess

