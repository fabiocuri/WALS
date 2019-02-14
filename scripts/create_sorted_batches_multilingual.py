import sys, os
import codecs
import argparse
import numpy
from operator import itemgetter

def get_file_length(fname):
    """ Returns number of lines in file.
        Args:
            fname (str)     : File name to check for number of lines.
    """
    with codecs.open(fname, 'r', 'utf8') as fh:
        for idx,_ in enumerate(fh):
            pass
        print("#lines: %s"%str(idx+1))
        return idx+1


def process_split(mbatch_size, src, tgt, bpe_suffix=None, fname_src_lang=None, fname_tgt_lang=None, langs='.langs', suffix='.shuf'):
    """ Shuffle minibatches for a split (valid, train) and write shuffled minibatch to disk (src, tgt).

        Args:
        mbatch_size (int)       : Minibatch size.
        src (str)               : Path to source side of the split.
        tgt (str)               : Path to target side of the split.
    """
    # load language code for each example in each minibatch (from {file_name}.langs file)
    langs_src, langs_tgt = dict(), dict()

    if fname_src_lang is None:
        src_prebpe=src.replace(bpe_suffix, '').replace('..', '.')
        fname_src_lang = src_prebpe+langs
        print("source before: ", src)
        print("source after: ", src_prebpe)
    if fname_tgt_lang is None:
        tgt_prebpe=tgt.replace(bpe_suffix, '').replace('..', '.')
        fname_tgt_lang = tgt_prebpe+langs
        print("target before: ", tgt)
        print("target after: ", tgt_prebpe)

    with codecs.open(fname_src_lang, 'r', 'utf8') as fh_src:
        with codecs.open(fname_tgt_lang, 'r', 'utf8') as fh_tgt:
            minibatch = []
            last_idx_src, last_idx_tgt = 1, 1
            for idx, (line_src, line_tgt) in enumerate(zip(fh_src, fh_tgt)):
                line_src = line_src.strip()
                line_tgt = line_tgt.strip()

                print(line_src.split("\t"), line_tgt.split("\t"))

                # we accept information on the number of lines per language in two ways:
                #
                # the first one we provide first and last lines with a certain language (as below)
                # eng 1 2000
                # fre 2001 4000
                #
                # the second one we provide only the total number of lines for a certain language (as below)
                # eng 2000
                # fre 2000
                #
                # the two examples above mean the exact same thing: first 2000 lines consist of english sentences,
                # and last 2000 lines consist of french sentences.
                try:
                    lang_code_src, from_idx_src, to_idx_src= line_src.split("\t")
                    lang_code_tgt, from_idx_tgt, to_idx_tgt = line_tgt.split("\t")
                    total_lines_src, total_lines_tgt = None, None
                except:
                    try:
                        lang_code_src, total_lines_src = line_src.split("\t")
                        lang_code_tgt, total_lines_tgt = line_tgt.split("\t")
                        from_idx_src, to_idx_src = None, None
                        from_idx_tgt, to_idx_tgt = None, None
                    except:
                        raise Exception("Language information not recognized:\nsource:\t%s\ntarget:\t%s\n"%(str(line_src),str(line_tgt)))

                # format one
                if not from_idx_src is None and not to_idx_src is None and total_lines_src is None:
                    for idx in range(int(from_idx_src), int(to_idx_src)+1):
                        langs_src[idx] = lang_code_src
                    for idx in range(int(from_idx_tgt), int(to_idx_tgt)+1):
                        langs_tgt[idx] = lang_code_tgt
                else:
                    # format two
                    # if from_idx is None and to_idx is None and not total_lines is None:
                    for _ in range(int(total_lines_src)):
                        langs_src[last_idx_src] = lang_code_src
                        last_idx_src+=1
                    for _ in range(int(total_lines_tgt)):
                        langs_tgt[last_idx_tgt] = lang_code_tgt
                        last_idx_tgt+=1

                    assert(last_idx_src==last_idx_tgt), 'Source/target idxs do not match! %i and %i'%(last_idx_src,last_idx_tgt)

    # sanity check: make sure every example has one and only one corresponding language code
    # 1 -> 116000
    keys = range(1, get_file_length(src) +1)
    for i,j in zip(keys, list(langs_src.keys())):
        if not i==j:
            print("ERROR: Different idxs %s, %s"%(str(i), str(j)))
            sys.exit(1)
    assert(len(list(langs_src.keys())) == len(keys)), 'Source lengths do not match! %i and %i'%(
            len(list(langs_src.keys())), len(keys))

    keys = range(1, get_file_length(tgt) +1)
    for i,j in zip(keys, list(langs_tgt.keys())):
        if not i==j:
            print("ERROR: Different idxs %s, %s"%(str(i), str(j)))
            sys.exit(1)
    assert(len(list(langs_tgt.keys())) == len(keys)), 'Target lengths do not match! %i and %i'%(
            len(list(langs_tgt.keys())), len(keys))

    # create minibatches
    minibatches = []
    with codecs.open(src, 'r', 'utf8') as fh_src:
        with codecs.open(tgt, 'r', 'utf8') as fh_tgt:
            minibatch = []
            for idx, (line_src, line_tgt) in enumerate(zip(fh_src, fh_tgt)):
                line_src = line_src.strip()
                line_tgt = line_tgt.strip()
                minibatch.append( (langs_src[idx+1], line_src, langs_tgt[idx+1], line_tgt) )
                if (idx+1) % mbatch_size == 0:
                    minibatches.append(minibatch)
                    minibatch = []

    print('..number of minibatches: %i'%len(minibatches))
    # create vector with minibatches' idxs
    idxs = numpy.array(range(len(minibatches)), dtype='uint')
    # create a random permutation of these idxs and shuffle minibatch
    shuffled_idxs = numpy.random.permutation( idxs )
    shuffled_minibatches = itemgetter(*shuffled_idxs)(minibatches)

    # write shuffled minibatches to disk
    src_out = src+suffix
    tgt_out = tgt+suffix
    with codecs.open(src_out, 'w', 'utf8') as fh_src:
        with codecs.open(tgt_out, 'w', 'utf8') as fh_tgt:
            for minibatch in shuffled_minibatches:
                for lang_src, example_src, lang_tgt, example_tgt in minibatch:
                    #print( "example_src: ", example_src, "example_tgt: ", example_tgt )
                    fh_src.write(lang_src+" "+example_src+"\n")
                    fh_tgt.write(lang_tgt+" "+example_tgt+"\n")

def m30k():
    """ Shuffle multi30k data set keeping minibatches intact
    """
    path_to_data="./data/m30k/multilingual"
    datasets=['de-1K', 'de-10K', 'de-29K']
    bpe_suffix="bpe.10000"
    lang_suffix="langs"

    for dataset in datasets:
        train_src=path_to_data+"/train.lc.norm.tok.all-languages."+dataset+"."+bpe_suffix+".src"
        train_tgt=path_to_data+"/train.lc.norm.tok.all-languages."+dataset+"."+bpe_suffix+".tgt"

        train_src_langs=path_to_data+"/train.lc.norm.tok.all-languages."+dataset+".src."+lang_suffix
        train_tgt_langs=path_to_data+"/train.lc.norm.tok.all-languages."+dataset+".tgt."+lang_suffix

        assert(not os.path.isfile(train_src+".shuf") and not os.path.isfile(train_tgt+".shuf")), \
                       'ERROR: At least one shuffled file was found! Please delete it manually and try again.'

        train_mbatch_size = 100
        valid_mbatch_size = 8

        print('Processing training set (minibatch size %i)...'%train_mbatch_size)
        process_split(train_mbatch_size, train_src, train_tgt,
                fname_src_lang=train_src_langs,
                fname_tgt_lang=train_tgt_langs)

        valid_src=path_to_data+"/val.lc.norm.tok.all-languages."+dataset+"."+bpe_suffix+".src"
        valid_tgt=path_to_data+"/val.lc.norm.tok.all-languages."+dataset+"."+bpe_suffix+".tgt"
        valid_src_langs=path_to_data+"/val.lc.norm.tok.all-languages.src."+lang_suffix
        valid_tgt_langs=path_to_data+"/val.lc.norm.tok.all-languages.tgt."+lang_suffix

        assert(not os.path.isfile(valid_src+".shuf") and not os.path.isfile(valid_tgt+".shuf")), \
                       'ERROR: At least one shuffled file was found! Please delete it manually and try again.'

        print('Processing valid set (minibatch size %i)...'%valid_mbatch_size)
        process_split(valid_mbatch_size, valid_src, valid_tgt,
                fname_src_lang=valid_src_langs,
                fname_tgt_lang=valid_tgt_langs)
                
        print('Shuffled data found in %s/{original_file_name}.shuf'%(str(path_to_data)))

def iwslt17():
    """ Shuffle IWSLT17 data set keeping minibatches intact
    """
    #path_to_data="./data/iwslt17"
    #train_src=path_to_data+"/train.tags.lc.tok.norm.clean.all-languages.bpe.30000.src"
    #train_tgt=path_to_data+"/train.tags.lc.tok.norm.clean.all-languages.bpe.30000.tgt"
    #valid_src=path_to_data+"/IWSLT17.TED.dev2010.lc.tok.norm.all-languages.bpe.30000.src"
    #valid_tgt=path_to_data+"/IWSLT17.TED.dev2010.lc.tok.norm.all-languages.bpe.30000.tgt"
    path_to_data="/data/icalixto/iwslt2017-multilingual/DeEnItNlRo-DeEnItNlRo"
    low_resource_sizes = [10000, 100000, 200000]
    bpe_suffix = "bpe.30000"
    lang_suffix = "langs"

    # /data/icalixto/iwslt2017-multilingual/DeEnItNlRo-DeEnItNlRo/IWSLT17.TED.dev2010.lc.tok.norm.all-languages.src.langs
    valid_src_langs=path_to_data+"/IWSLT17.TED.dev2010.lc.tok.norm.all-languages.src."+lang_suffix
    valid_tgt_langs=path_to_data+"/IWSLT17.TED.dev2010.lc.tok.norm.all-languages.tgt."+lang_suffix

    for size in low_resource_sizes:
        train_src=path_to_data+"/train.tags.lc.tok.norm.clean.all-languages.low-resource-"+str(size)+"."+bpe_suffix+".src"
        train_tgt=path_to_data+"/train.tags.lc.tok.norm.clean.all-languages.low-resource-"+str(size)+"."+bpe_suffix+".tgt"
        valid_src=path_to_data+"/IWSLT17.TED.dev2010.lc.tok.norm.all-languages.low-resource-"+str(size)+"."+bpe_suffix+".src"
        valid_tgt=path_to_data+"/IWSLT17.TED.dev2010.lc.tok.norm.all-languages.low-resource-"+str(size)+"."+bpe_suffix+".tgt"
        # /data/icalixto/iwslt2017-multilingual/DeEnItNlRo-DeEnItNlRo/train.tags.lc.tok.norm.clean.all-languages.low-resource-100000.tgt.langs
        train_src_langs=path_to_data+"/train.tags.lc.tok.norm.clean.all-languages.low-resource-"+str(size)+".src."+lang_suffix
        train_tgt_langs=path_to_data+"/train.tags.lc.tok.norm.clean.all-languages.low-resource-"+str(size)+".tgt."+lang_suffix

        assert(not os.path.isfile(train_src+".shuf") and not os.path.isfile(train_tgt+".shuf") and
               not os.path.isfile(valid_src+".shuf") and not os.path.isfile(valid_tgt+".shuf")), \
                       'ERROR: At least one shuffled file was found! Please delete it manually and try again.'

        train_mbatch_size = 100
        valid_mbatch_size = 8

        print('Processing training set (minibatch size %i)...'%train_mbatch_size)
        process_split(train_mbatch_size, train_src, train_tgt,
                fname_src_lang=train_src_langs,
                fname_tgt_lang=train_tgt_langs)
        print('Processing valid set (minibatch size %i)...'%valid_mbatch_size)
        process_split(valid_mbatch_size, valid_src, valid_tgt,
                fname_src_lang=valid_src_langs,
                fname_tgt_lang=valid_tgt_langs)
        print('Shuffled data found in %s/{original_file_name}.shuf'%(str(path_to_data)))

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--m30k', action="store_true", required=False, help="Created sorted batches for the Multi30k data set?")
    p.add_argument('--iwslt17', action="store_true", required=False, help="Created sorted batches for the IWSLT17 data set?")
    args = p.parse_args()

    if not args.m30k and not args.iwslt17:
        p.error('No action requested, please choose at least one of --m30k --iwslt17')

    if args.m30k:
        m30k()
    if args.iwslt17:
        iwslt17()
    print('Finished sorting minibatches.')
