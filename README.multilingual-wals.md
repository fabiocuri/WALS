# Single-pair baselines

# how we called `preprocess.py` (no need to do it again).
python3 preprocess.py \
    -train_src data/m30k/train.lc.norm.tok.bpe.en \
    -train_tgt data/m30k/train.lc.norm.tok.bpe.fr \
    -valid_src data/m30k/val.lc.norm.tok.bpe.en \
    -valid_tgt data/m30k/val.lc.norm.tok.bpe.fr \
    -save_data data/m30k/single-pair.en-fr \
    -share_vocab

# how to call `train.py`.
python3 train.py \
    -data data/m30k/single-pair.en-fr \
    -save_model model_snapshots/m30k/En-Fr/single-pair.en-fr \
    -input_feed 0 -gpu_ranks 0 \
    -save_checkpoint_steps 1000 \
    -train_steps 50000 \
    -valid_steps 1000 \
    -optim 'adam' -learning_rate 0.002 \
    -batch_size 100 \
    -share_embeddings

# how to call `translate_model_selection.py`. Still not implemented for baselines.
python translate_baseline_model_selection.py \
    --data data/m30k/ \
    --model model_snapshots/m30k/En-Fr/single-pair.en-fr_step_ \
    --delete_model_files 'all-but-best' \
    --valid_src data/m30k/val.lc.norm.tok.bpe.en \
    --valid_ref data/m30k/val.lc.norm.tok.bpe.fr \
    --test_src data/m30k/test_2016_flickr.lc.norm.tok.bpe.en \
    --test_ref data/m30k/test_2016_flickr.lc.norm.tok.bpe.fr \
    --ngpus 2

# end

