# Train

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_enctgt -wals_src eng -wals_tgt fre -train_steps 100 -save_checkpoint_steps 100 -wals_model EncInitHidden_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_00022_encboth -wals_src eng -wals_tgt fre -train_steps 100 -save_checkpoint_steps 100 -wals_model EncInitHidden_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_dectgt -wals_src eng -wals_tgt fre -train_steps 100 -save_checkpoint_steps 100 -wals_model DecInitHidden_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_decboth -wals_src eng -wals_tgt fre -train_steps 100 -save_checkpoint_steps 100 -wals_model DecInitHidden_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_10_wals2srctgt -wals_src eng -wals_tgt fre -train_steps 100 -save_checkpoint_steps 100 -wals_model WalstoSource_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -wals_size 10

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_10_wals2srcboth -wals_src eng -wals_tgt fre -train_steps 100 -save_checkpoint_steps 100 -wals_model WalstoSource_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -wals_size 10

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_10_wals2tgttgt -wals_src eng -wals_tgt fre -train_steps 100 -save_checkpoint_steps 100 -wals_model WalstoTarget_Target -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -wals_size 10

python3 train.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_10_wals2tgtboth -wals_src eng -wals_tgt fre -train_steps 100 -save_checkpoint_steps 100 -wals_model WalstoTarget_Both -input_feed 0 -wals_function tanh -optim adam -learning_rate 0.002 -wals_size 10

# Translate

python3 translate.py -model data/En-Fr/tanh_adam_0002_enctgt_step_100.pt -src data/test_src.en -output data/pred_A1.fr -replace_unk -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Target -wals_function tanh

python3 translate.py -model data/En-Fr/tanh_adam_00022_encboth_step_100.pt -src data/test_src.en -output data/pred_A2.fr -replace_unk -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Both -wals_function tanh

python3 translate.py -model data/En-Fr/tanh_adam_0002_dectgt_step_100.pt -src data/test_src.en -output data/pred_B1.fr -replace_unk -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Target -wals_function tanh

python3 translate.py -model data/En-Fr/tanh_adam_0002_decboth_step_100.pt -src data/test_src.en -output data/pred_B2.fr -replace_unk -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Both -wals_function tanh

python3 translate.py -model data/En-Fr/tanh_adam_0002_10_wals2srctgt_step_100.pt -src data/test_src.en -output data/pred_C1.fr -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -wals_function tanh -wals_size 10

python3 translate.py -model data/En-Fr/tanh_adam_0002_10_wals2srcboth_step_100.pt -src data/test_src.en -output data/pred_C2.fr -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -wals_function tanh -wals_size 10

python3 translate.py -model data/En-Fr/tanh_adam_0002_10_wals2tgttgt_step_100.pt -src data/test_src.en -output data/pred_D1.fr -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -wals_function tanh -wals_size 10

python3 translate.py -model data/En-Fr/tanh_adam_0002_10_wals2tgtboth_step_100.pt -src data/test_src.en -output data/pred_D2.fr -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -wals_function tanh -wals_size 10
