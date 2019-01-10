# OpenNMT-py: Open-Source Neural Machine Translation with WALS features

[![Build Status](https://travis-ci.org/OpenNMT/OpenNMT-py.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-py)
[![Run on FH](https://img.shields.io/badge/Run%20on-FloydHub-blue.svg)](https://floydhub.com/run?template=https://github.com/OpenNMT/OpenNMT-py)

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system. It is designed to be research friendly to try out new ideas in translation, summary, image-to-text, morphology, and many other domains.

Codebase is relatively stable, but PyTorch is still evolving. We currently only support PyTorch 0.4 and recommend forking if you need to have stable code.

OpenNMT-py is run as a collaborative open-source project. It is maintained by [Sasha Rush](http://github.com/srush) (Cambridge, MA), [Ben Peters](http://github.com/bpopeters) (Lisbon), and [Jianyu Zhan](http://github.com/jianyuzhan) (Shanghai). The original code was written by [Adam Lerer](http://github.com/adamlerer) (NYC). 
We love contributions. Please consult the Issues page for any [Contributions Welcome](https://github.com/OpenNMT/OpenNMT-py/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22) tagged post. 

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>


Table of Contents
=================
  * [Full Documentation](http://opennmt.net/OpenNMT-py/)
  * [Requirements](#requirements)
  * [Features](#features)
  * [Quickstart](#quickstart)
  * [Run on FloydHub](#run-on-floydhub)
  * [Citation](#citation)

## Requirements

Pre-requirements: pip, git, python.

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

Note that we currently only support PyTorch 0.4.

## Features

The following OpenNMT features are implemented:

- [data preprocessing](http://opennmt.net/OpenNMT-py/options/preprocess.html)
- [Inference (translation) with batching and beam search](http://opennmt.net/OpenNMT-py/options/translate.html)
- [Multiple source and target RNN (lstm/gru) types and attention (dotprod/mlp) types](http://opennmt.net/OpenNMT-py/options/train.html#model-encoder-decoder)
- [TensorBoard](http://opennmt.net/OpenNMT-py/options/train.html#logging)
- [Source word features](http://opennmt.net/OpenNMT-py/options/train.html#model-embeddings)
- [Pretrained Embeddings](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)
- [Copy and Coverage Attention](http://opennmt.net/OpenNMT-py/options/train.html#model-attention)
- [Image-to-text processing](http://opennmt.net/OpenNMT-py/im2text.html)
- [Speech-to-text processing](http://opennmt.net/OpenNMT-py/speech2text.html)
- ["Attention is all you need"](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model)
- [Multi-GPU](http://opennmt.net/OpenNMT-py/FAQ.html##do-you-support-multi-gpu)
- Inference time loss functions.

Beta Features (committed):
- Structured attention
- [Conv2Conv convolution model]
- SRU "RNNs faster than CNN" paper

## Quickstart

[Full Documentation](http://opennmt.net/OpenNMT-py/)


### Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Run the WALS script to build the linguistic databases.

```bash
python onmt/wals.py
```

The following databases are created:

WalsValues.db: WALS values for each feature, per language.

FeaturesList.db: Number of possible classes and embedding dimension, per feature.

FTInfos.db: Number of features, classes and total embedding dimension, per feature type.

FTList.db: Features present, per feature type.

### Step 3: Train the model with the WALS model

#### Go to wals.info and download the following files: codes.csv, language.csv, parameters.csv and values.csv.

```bash
python3 train.py -data data/demo -save_model demo-model -wals_src src_language -wals_tgt tgt_language -wals_model model -wals_function tanh -wals_size 10
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder. You
can also add `-gpuid 1` to use (say) GPU 1.

The -wals_src and -wals_tgt options should be language names as listed under the WALS dataset (see https://wals.info/languoid for codes).

#### Possible WALS models (-wals_model) :

##### Model EncInitHidden: uses WALS features to initialize encoder's hidden state.

EncInitHidden_Target (WALS information only from target language)

EncInitHidden_Both (WALS information from both languages)

##### Model DecInitHidden: adds WALS features to the encoder's output to initialize decoder's hidden state.

DecInitHidden_Target (WALS information only from target language)

DecInitHidden_Both (WALS information from both languages)

##### Model WalstoSource: concatenates WALS features to source words embeddings.

WalstoSource_Target (WALS information only from target language)

WalstoSource_Both (WALS information from both languages)

##### Model WalstoTarget: concatenates WALS features to target words embeddings.

WalstoTarget_Target (WALS information only from target language)

WalstoTarget_Both (WALS information from both languages)

##### Model WalsDoublyAttentive: the WALS features are incorporated as an additional attention mechanism.

WalsDoublyAttentive_Target (WALS information only from target language)

WalsDoublyAttentive_Both (WALS information from both languages)



### Step 4: Translate

#### Models EncInitHidden and DecInitHidden:

```bash
python translate.py -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose -wals_src src_language -wals_tgt tgt_language -wals_model model -wals_function function
```

#### Models WalstoSource and WalstoTarget:

```bash
python translate.py -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose -wals_src src_language -wals_tgt tgt_language -wals_model model -wals_function function -wals_size size
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.

Make sure that the right source and target languages are specified when calling translate.py, as well as the WALS model and wals size that were used for training!

!!! note "Note"
    The predictions are going to be quite terrible, as the demo dataset is small. Try running on some larger datasets! For example you can download millions of parallel sentences for [translation](http://www.statmt.org/wmt16/translation-task.html) or [summarization](https://github.com/harvardnlp/sent-summary).

## Alternative: Run on FloydHub

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run?template=https://github.com/OpenNMT/OpenNMT-py)

Click this button to open a Workspace on [FloydHub](https://www.floydhub.com/?utm_medium=readme&utm_source=opennmt-py&utm_campaign=jul_2018) for training/testing your code.


## Pretrained embeddings (e.g. GloVe)

Go to tutorial: [How to use GloVe pre-trained embeddings in OpenNMT-py](http://forum.opennmt.net/t/how-to-use-glove-pre-trained-embeddings-in-opennmt-py/1011)

## Pretrained Models

The following pretrained models can be downloaded and used with translate.py.

http://opennmt.net/Models-py/

## Citation

[OpenNMT: Neural Machine Translation Toolkit](https://arxiv.org/pdf/1805.11462)


[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```

