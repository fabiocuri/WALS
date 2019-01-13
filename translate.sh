##############################################################################################En-Fr#################################################################################################

##############EncInitHidden##############

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_enctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/tanh_adam_0002_enctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Target -wals_function tanh

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/relu_adam_0002_enctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/relu_adam_0002_enctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Target -wals_function relu

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/none_adam_0002_enctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/none_adam_0002_enctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Target

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_encboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/tanh_adam_0002_encboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Both -wals_function tanh

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/relu_adam_0002_encboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/relu_adam_0002_encboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Both -wals_function relu

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/none_adam_0002_encboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/none_adam_0002_encboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model EncInitHidden_Both

##############DecInitHidden##############

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_dectgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/tanh_adam_0002_dectgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Target -wals_function tanh

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/relu_adam_0002_dectgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/relu_adam_0002_dectgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Target -wals_function relu

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/none_adam_0002_dectgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/none_adam_0002_dectgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Target

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/tanh_adam_0002_decboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/tanh_adam_0002_decboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Both -wals_function tanh

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/relu_adam_0002_decboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/relu_adam_0002_decboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Both -wals_function relu

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/none_adam_0002_decboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/none_adam_0002_decboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model DecInitHidden_Both

##############WalstoSource##############

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_tanh_adam_0002_wals2srctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_tanh_adam_0002_wals2srctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -wals_function tanh

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_tanh_adam_0002_wals2srctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_tanh_adam_0002_wals2srctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -wals_function tanh -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_tanh_adam_0002_wals2srctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_tanh_adam_0002_wals2srctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -wals_function tanh -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_tanh_adam_0002_wals2srctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_tanh_adam_0002_wals2srctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -wals_function tanh -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_relu_adam_0002_wals2srctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_relu_adam_0002_wals2srctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -wals_function relu

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_relu_adam_0002_wals2srctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_relu_adam_0002_wals2srctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -wals_function relu -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_relu_adam_0002_wals2srctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_relu_adam_0002_wals2srctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -wals_function relu -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_relu_adam_0002_wals2srctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_relu_adam_0002_wals2srctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -wals_function relu -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_none_adam_0002_wals2srctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_none_adam_0002_wals2srctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_none_adam_0002_wals2srctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_none_adam_0002_wals2srctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_none_adam_0002_wals2srctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_none_adam_0002_wals2srctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_none_adam_0002_wals2srctgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_none_adam_0002_wals2srctgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Target -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_tanh_adam_0002_wals2srcboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_tanh_adam_0002_wals2srcboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -wals_function tanh

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_tanh_adam_0002_wals2srcboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_tanh_adam_0002_wals2srcboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -wals_function tanh -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_tanh_adam_0002_wals2srcboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_tanh_adam_0002_wals2srcboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -wals_function tanh -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_tanh_adam_0002_wals2srcboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_tanh_adam_0002_wals2srcboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -wals_function tanh -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_relu_adam_0002_wals2srcboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_relu_adam_0002_wals2srcboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -wals_function relu

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_relu_adam_0002_wals2srcboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_relu_adam_0002_wals2srcboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -wals_function relu -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_relu_adam_0002_wals2srcboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_relu_adam_0002_wals2srcboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -wals_function relu -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_relu_adam_0002_wals2srcboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_relu_adam_0002_wals2srcboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -wals_function relu -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_none_adam_0002_wals2srcboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_none_adam_0002_wals2srcboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_none_adam_0002_wals2srcboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_none_adam_0002_wals2srcboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_none_adam_0002_wals2srcboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_none_adam_0002_wals2srcboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_none_adam_0002_wals2srcboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_none_adam_0002_wals2srcboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoSource_Both -wals_size 100

##############WalstoTarget##############

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_tanh_adam_0002_wals2tgttgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_tanh_adam_0002_wals2tgttgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -wals_function tanh

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_tanh_adam_0002_wals2tgttgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_tanh_adam_0002_wals2tgttgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -wals_function tanh -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_tanh_adam_0002_wals2tgttgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_tanh_adam_0002_wals2tgttgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -wals_function tanh -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_tanh_adam_0002_wals2tgttgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_tanh_adam_0002_wals2tgttgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -wals_function tanh -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_relu_adam_0002_wals2tgttgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_relu_adam_0002_wals2tgttgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -wals_function relu

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_relu_adam_0002_wals2tgttgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_relu_adam_0002_wals2tgttgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -wals_function relu -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_relu_adam_0002_wals2tgttgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_relu_adam_0002_wals2tgttgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -wals_function relu -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_relu_adam_0002_wals2tgttgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_relu_adam_0002_wals2tgttgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -wals_function relu -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_none_adam_0002_wals2tgttgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_none_adam_0002_wals2tgttgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_none_adam_0002_wals2tgttgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_none_adam_0002_wals2tgttgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_none_adam_0002_wals2tgttgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_none_adam_0002_wals2tgttgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_none_adam_0002_wals2tgttgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_none_adam_0002_wals2tgttgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Target -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_tanh_adam_0002_wals2tgtboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_tanh_adam_0002_wals2tgtboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -wals_function tanh

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_tanh_adam_0002_wals2tgtboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_tanh_adam_0002_wals2tgtboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -wals_function tanh -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_tanh_adam_0002_wals2tgtboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_tanh_adam_0002_wals2tgtboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -wals_function tanh -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_tanh_adam_0002_wals2tgtboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_tanh_adam_0002_wals2tgtboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -wals_function tanh -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_relu_adam_0002_wals2tgtboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_relu_adam_0002_wals2tgtboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -wals_function relu

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_relu_adam_0002_wals2tgtboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_relu_adam_0002_wals2tgtboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -wals_function relu -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_relu_adam_0002_wals2tgtboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_relu_adam_0002_wals2tgtboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -wals_function relu -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_relu_adam_0002_wals2tgtboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_relu_adam_0002_wals2tgtboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -wals_function relu -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_none_adam_0002_wals2tgtboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_none_adam_0002_wals2tgtboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_none_adam_0002_wals2tgtboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_none_adam_0002_wals2tgtboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_none_adam_0002_wals2tgtboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_none_adam_0002_wals2tgtboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_none_adam_0002_wals2tgtboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_none_adam_0002_wals2tgtboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalstoTarget_Both -wals_size 100

##############WalsDoublyAttentive##############

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_tanh_adam_0002_doublytgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_tanh_adam_0002_doublytgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -wals_function tanh

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_tanh_adam_0002_doublytgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_tanh_adam_0002_doublytgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -wals_function tanh -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_tanh_adam_0002_doublytgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_tanh_adam_0002_doublytgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -wals_function tanh -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_tanh_adam_0002_doublytgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_tanh_adam_0002_doublytgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -wals_function tanh -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_relu_adam_0002_doublytgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_relu_adam_0002_doublytgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -wals_function relu

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_relu_adam_0002_doublytgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_relu_adam_0002_doublytgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -wals_function relu -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_relu_adam_0002_doublytgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_relu_adam_0002_doublytgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -wals_function relu -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_relu_adam_0002_doublytgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_relu_adam_0002_doublytgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -wals_function relu -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_none_adam_0002_doublytgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_none_adam_0002_doublytgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_none_adam_0002_doublytgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_none_adam_0002_doublytgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_none_adam_0002_doublytgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_none_adam_0002_doublytgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_none_adam_0002_doublytgt_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_none_adam_0002_doublytgt_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Target -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_tanh_adam_0002_doublyboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_tanh_adam_0002_doublyboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -wals_function tanh

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_tanh_adam_0002_doublyboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_tanh_adam_0002_doublyboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -wals_function tanh -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_tanh_adam_0002_doublyboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_tanh_adam_0002_doublyboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -wals_function tanh -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_tanh_adam_0002_doublyboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_tanh_adam_0002_doublyboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -wals_function tanh -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_relu_adam_0002_doublyboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_relu_adam_0002_doublyboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -wals_function relu

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_relu_adam_0002_doublyboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_relu_adam_0002_doublyboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -wals_function relu -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_relu_adam_0002_doublyboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_relu_adam_0002_doublyboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -wals_function relu -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_relu_adam_0002_doublyboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_relu_adam_0002_doublyboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -wals_function relu -wals_size 100

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/10_none_adam_0002_doublyboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/10_none_adam_0002_doublyboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/20_none_adam_0002_doublyboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/20_none_adam_0002_doublyboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -wals_size 20

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/50_none_adam_0002_doublyboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/50_none_adam_0002_doublyboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -wals_size 50

python3 translate.py -data data/En-Fr/bpe_endefr -save_model data/En-Fr/100_none_adam_0002_doublyboth_step_10000.pt -src data/endefr.val.en -output data/En-Fr/100_none_adam_0002_doublyboth_step_10000.en -replace_unk -wals_src eng -wals_tgt fre -wals_model WalsDoublyAttentive_Both -wals_size 100
