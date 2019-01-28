# how to call `preprocess.py` (with concatenated training/validation data in both directions).
python3 preprocess.py \
    -train_src data/train.lc.norm.tok.bpe.en-fr.en-fr \
    -train_tgt data/train.lc.norm.tok.bpe.en-fr.fr-en \
    -valid_src data/endefr.val.en-fr.en-fr \
    -valid_tgt data/endefr.val.en-fr.fr-en \
    -save_data data/demo-train-concat-en-fr \
    -share_vocab

# how to call `train.py` (with concatenated training/validation data in both directions).
python3 train.py \
    -data data/demo-train-concat-en-fr \
    -save_model model_snapshots/En-Fr/en-fr.DecInitHidden_Target \
    -wals_src eng -wals_tgt fre \
    -wals_model DecInitHidden_Target -wals_function 'tanh' -wals_size 10 \
    -input_feed 0 -gpu_ranks 0 \
    -save_checkpoint_steps 1000 \
    -train_steps 50000 \
    -valid_steps 1000 \
    -optim 'adam' -learning_rate 0.002 \
    -share_embeddings

# how to call `translate_model_selection.py` (with NORMAL validation data).
python translate_model_selection.py \
    --data data/ \
    --model model_snapshots/En-Fr/en-fr.DecInitHidden_Target_step_ \
    --wals_src eng --wals_tgt fre \
    --wals_function tanh --wals_model_type DecInitHidden_Target \
    --src_language en --tgt_language fr \
    --delete_model_files 'no' \
    --ngpus 2

# end

