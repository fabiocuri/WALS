##############################################################################################En-Fr#################################################################################################

##############EncInitHidden##############

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_enctgt -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/relu_adam_0002_enctgt -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/none_adam_0002_enctgt -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_encboth -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/relu_adam_0002_encboth -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/none_adam_0002_encboth -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

##############DecInitHidden##############

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_dectgt -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/relu_adam_0002_dectgt -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/none_adam_0002_dectgt -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_decboth -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/relu_adam_0002_decboth -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/none_adam_0002_decboth -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

##############WalstoSource##############

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_tanh_adam_0002_wals2srctgt -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_tanh_adam_0002_wals2srctgt -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_tanh_adam_0002_wals2srctgt -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_tanh_adam_0002_wals2srctgt -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_relu_adam_0002_wals2srctgt -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_relu_adam_0002_wals2srctgt -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_relu_adam_0002_wals2srctgt -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_relu_adam_0002_wals2srctgt -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_none_adam_0002_wals2srctgt -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_none_adam_0002_wals2srctgt -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_none_adam_0002_wals2srctgt -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_none_adam_0002_wals2srctgt -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_tanh_adam_0002_wals2srcboth -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_tanh_adam_0002_wals2srcboth -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_tanh_adam_0002_wals2srcboth -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_tanh_adam_0002_wals2srcboth -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_relu_adam_0002_wals2srcboth -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_relu_adam_0002_wals2srcboth -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_relu_adam_0002_wals2srcboth -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_relu_adam_0002_wals2srcboth -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_none_adam_0002_wals2srcboth -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_none_adam_0002_wals2srcboth -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_none_adam_0002_wals2srcboth -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_none_adam_0002_wals2srcboth -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

##############WalstoTarget##############

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_tanh_adam_0002_wals2tgttgt -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_tanh_adam_0002_wals2tgttgt -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_tanh_adam_0002_wals2tgttgt -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_tanh_adam_0002_wals2tgttgt -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_relu_adam_0002_wals2tgttgt -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_relu_adam_0002_wals2tgttgt -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_relu_adam_0002_wals2tgttgt -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_relu_adam_0002_wals2tgttgt -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_none_adam_0002_wals2tgttgt -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_none_adam_0002_wals2tgttgt -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_none_adam_0002_wals2tgttgt -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_none_adam_0002_wals2tgttgt -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_tanh_adam_0002_wals2tgtboth -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_tanh_adam_0002_wals2tgtboth -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_tanh_adam_0002_wals2tgtboth -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_tanh_adam_0002_wals2tgtboth -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_relu_adam_0002_wals2tgtboth -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_relu_adam_0002_wals2tgtboth -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_relu_adam_0002_wals2tgtboth -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_relu_adam_0002_wals2tgtboth -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_none_adam_0002_wals2tgtboth -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_none_adam_0002_wals2tgtboth -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_none_adam_0002_wals2tgtboth -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_none_adam_0002_wals2tgtboth -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

##############WalsDoublyAttentive##############

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_tanh_adam_0002_doublytgt -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_tanh_adam_0002_doublytgt -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_tanh_adam_0002_doublytgt -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_tanh_adam_0002_doublytgt -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_relu_adam_0002_doublytgt -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_relu_adam_0002_doublytgt -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_relu_adam_0002_doublytgt -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_relu_adam_0002_doublytgt -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_none_adam_0002_doublytgt -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_none_adam_0002_doublytgt -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_none_adam_0002_doublytgt -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_none_adam_0002_doublytgt -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_tanh_adam_0002_doublyboth -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_tanh_adam_0002_doublyboth -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_tanh_adam_0002_doublyboth -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_tanh_adam_0002_doublyboth -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_relu_adam_0002_doublyboth -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_relu_adam_0002_doublyboth -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_relu_adam_0002_doublyboth -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_relu_adam_0002_doublyboth -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -input_feed 0 -wals_function relu -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_none_adam_0002_doublyboth -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_none_adam_0002_doublyboth -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 20

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_none_adam_0002_doublyboth -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 50

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_none_adam_0002_doublyboth -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -input_feed 0 -optim adam -learning_rate 0.002 -train_steps 50000 -save_checkpoint_steps 10000 -wals_size 100
