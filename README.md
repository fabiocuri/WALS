# OpenNMT-py: Open-Source Neural Machine Translation with WALS features

Author: Fabio Curi Paixao 

E-mail: fcuri91@gmail.com

Date: June 2019

## Requirements

Pre-requirements: pip, git, python.

All dependencies can be installed via:

```bash
pip3 install -r requirements.txt
```

### Step 1: Preprocess the data

```bash
python3 preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
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
cd onmt
python3 wals.py
```

The following databases are created:

WalsValues.db: WALS values for each feature, per language.

FeaturesList.db: Number of possible classes and embedding dimension, per feature.

FTInfos.db: Number of features, classes and total embedding dimension, per feature type.

FTList.db: Features present, per feature type.

### Step 3: Train the model with the WALS model

#### Go to wals.info and download the following files: codes.csv, language.csv, parameters.csv and values.csv.

```bash
python3 train.py -data data/demo -save_model demo-model -wals_src src_language -wals_tgt tgt_language -wals_model model -wals_function function -wals_size size -input_feed 0
```

The -wals_src and -wals_tgt options should be language names as listed under the WALS dataset (see https://wals.info/languoid for codes). Here below, you can find the list of all possible -wals_model

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

##### Model WalstoDecHidden: the WALS features are concatenated to the decoder's hidden state.

WalstoDecHidden_Target (WALS information only from target language)

WalstoDecHidden_Both (WALS information from both languages)


The WALS features can be projected into a -wals_size dimension, which is an integer. If not specified, -wals_size is 10. Note that this parameter is not required for models EncInitHidden and DecInitHidden, as they are projected into the RNN size, directly. If not specified for the other models, its standard value is 10.

In respect to -wals_function, the user can either call `tanh`, `relu` or do not call it, which will not enable any function to be run upon the WALS features. 


### Step 4: Translate

```bash
python3 translate.py -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose -wals_src src_language -wals_tgt tgt_language -wals_model model -wals_function function -wals_size size
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.

Make sure that the right -wals_src and -wals_tgt languages are specified when calling translate.py, as well as the -wals_model, -wals_size and wals_function that were used for training!

!!! note "Note"
    The predictions are going to be quite terrible, as the demo dataset is small. Try running on some larger datasets! For example you can download millions of parallel sentences for [translation](http://www.statmt.org/wmt16/translation-task.html) or [summarization](https://github.com/harvardnlp/sent-summary).
