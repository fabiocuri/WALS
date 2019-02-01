
SRCS=('en' 'fr' 'en' 'de' 'de' 'fr')
TGTS=('fr' 'en' 'de' 'en' 'fr' 'de')
IDXS=(0 1 2 3 4 5)

for idx in ${IDXS[@]}; do

    SRC=${SRCS[${idx}]}
    TGT=${TGTS[${idx}]}

    PATH_TO_DATA="./data/m30k"
    TRAIN_SRC="$PATH_TO_DATA/train.lc.norm.tok.bpe.${SRC}"
    TRAIN_TGT="$PATH_TO_DATA/train.lc.norm.tok.bpe.${TGT}"
    VALID_SRC="$PATH_TO_DATA/val.lc.norm.tok.bpe.${SRC}"
    VALID_TGT="$PATH_TO_DATA/val.lc.norm.tok.bpe.${TGT}"

    echo -ne "$SRC $TGT \n"
    [ -d "${PATH_TO_DATA}" ] || (echo "Not found: ${PATH_TO_DATA}" && exit)
    [ -f "${TRAIN_SRC}" ] || (echo "Not found: ${TRAIN_SRC}" && exit)
    [ -f "${TRAIN_TGT}" ] || (echo "Not found: ${TRAIN_TGT}" && exit)
    [ -f "${VALID_SRC}" ] || (echo "Not found: ${VALID_SRC}" && exit)
    [ -f "${VALID_TGT}" ] || (echo "Not found: ${VALID_TGT}" && exit)

    python preprocess.py \
        -train_src "${TRAIN_SRC}" \
        -train_tgt "${TRAIN_TGT}" \
        -valid_src "${VALID_SRC}" \
        -valid_tgt "${VALID_TGT}" \
        -save_data "$PATH_TO_DATA/single-pair.${SRC}-${TGT}" \
        -share_vocab

done

