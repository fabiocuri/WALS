import sys, os
from subprocess import Popen, PIPE
from glob import glob
from threading import Thread
import time
from queue import Queue
from multiprocessing.dummy import Pool
import numpy
import pickle
import tempfile

def valid_set_size_match(valid_set_translations, raise_exception=False, verbose=False):
    """ Sanity check:
        confirm that validation set size and translations of the validation set have the same size.
    """
    assert(os.path.isfile(valid_set_translations)), 'Validation set translation not found: %s'%str(valid_set_translations)
    if verbose:
        print("Valid set translations: %s"%str(valid_set_translations))
    # obtain number of sentences in validation set (references)
    p = Popen(['wc', '-l', VALID_REF], stderr=fnull, stdout=PIPE)
    p2 = Popen(['cut', '-d', ' ', '-f', '1'], stdin=p.stdout, stderr=fnull, stdout=PIPE)
    valid_set_size = int(p2.communicate()[0])

    # obtain number of sentences in validation set (translation hypotheses)
    p = Popen(['wc', '-l', valid_set_translations], stderr=fnull, stdout=PIPE)
    p2 = Popen(['cut', '-d', ' ', '-f', '1'], stdin=p.stdout, stderr=fnull, stdout=PIPE)
    valid_translations_size = int(p2.communicate()[0])

    if not valid_set_size==valid_translations_size and raise_exception:
        raise Exception("Validation set has %i sentences, but translation file has %i lines!\nTranslation file name: %s"%(
                valid_set_size, valid_translations_size, valid_set_translations))

    return False if not valid_set_size==valid_translations_size else True


def test_set_size_match(test_set_translations, raise_exception=False, verbose=False):
    """ Sanity check:
        confirm that test set size and translations of the test set have the same size.
    """
    assert(os.path.isfile(test_set_translations)), 'Test set translation not found: %s'%str(test_set_translations)
    if verbose:
        print("Test set translations: %s"%str(test_set_translations))
    # obtain number of sentences in test set (references)
    p = Popen(['wc', '-l', TEST_REF], stderr=fnull, stdout=PIPE)
    p2 = Popen(['cut', '-d', ' ', '-f', '1'], stdin=p.stdout, stderr=fnull, stdout=PIPE)
    test_set_size = int(p2.communicate()[0])

    # obtain number of sentences in test set (translation hypotheses)
    p = Popen(['wc', '-l', test_set_translations], stderr=fnull, stdout=PIPE)
    p2 = Popen(['cut', '-d', ' ', '-f', '1'], stdin=p.stdout, stderr=fnull, stdout=PIPE)
    test_translations_size = int(p2.communicate()[0])

    if not test_set_size==test_translations_size and raise_exception:
        raise Exception("Test set has %i sentences, but translation file has %i lines!\nTranslation file name: %s"%(
                test_set_size, test_translations_size, test_set_translations))
    return False if not test_set_size==test_translations_size else True


def translate_with_pool(source_fname, model_fname, hypfname_out, beam_size, wals_src, wals_tgt, wals_function, wals_model_type, gpuid=None, test_or_valid='valid', verbose=False):
    """ Translate a job expressed by: (source_fname, model fname, output hypotheses fname, beam size, gpuid, wals_src, wals_tgt, wals_function, wals_model_type, 'valid'/'test2016').
        The Pool implements the parallelisation logic.
    """
    assert(test_or_valid in ['test2016','valid']), 'test_or_valid must be set to one of [\'test2016\', \'valid\'].'

    if verbose:
        print("Model file name: %s\nHypotheses file name (to be created): %s\nbeam size: %i  -  gpuid: %i  -  split: %s"%(
            str(model_fname), str(hypfname_out), beam_size, gpuid, test_or_valid))

    # call external python script to translate validation/test set
    job_cmd = ['python', 'translate.py',
               '-src', source_fname,
               '-model', model_fname,
               '-batch_size', str(1),
               '-beam_size', str(beam_size),
               '-gpu', str(gpuid),
               '-wals_src', str(wals_src),
               '-wals_tgt', str(wals_tgt),
               '-wals_function', str(wals_function),
               '-wals_model', str(wals_model_type),
               '-output', hypfname_out]
    print("job command: %s"%str(job_cmd))
    p = Popen(job_cmd, stderr=fnull, stdout=PIPE)
    out = p.communicate()[0]



def compute_meteors(hypotheses_fname, references_fname, split='valid'):
    """ This function computes METEOR for all translations one by one without threading/queuing.
        It first converts subwords back into words before computing METEOR scores.
    """
    # java -Xmx2G -jar /data/icalixto/meteor-1.5/meteor-1.5.jar /data/icalixto/exp/variational-multimodal-nmt/model_snapshots/VI-NMT-Model1_RUN-1_adam_batch-size-40_lrate-0.002_z-dim-32_posterior-img-feats_conditional_acc_65.82_ppl_9.84_e14.pt.translations-valid /data/icalixto/multi30k/val.lc.norm.tok.de -l de -norm
    assert(split in ['valid', 'test2016']), 'Must compute METEOR for either valid or test set test2016!'
    # compute METEOR scores for each of the translations of the validation set
    curr_model_idx=0
    model_meteors, model_names, translation_files = [], [], []

    # post-process reference translations
    with tempfile.NamedTemporaryFile() as temp_ref:
        # call another python script to compute scores
        pcat  = Popen(['cat', references_fname], stdout=PIPE, stderr=fnull)
        # convert translations from subwords into words
        psubword = Popen(['sed', '-r', 's/(@@ )|(@@ ?$)//g'], stdin=pcat.stdout, stdout=temp_ref)
        # write translations after BPE-to-word post-processing to temporary file
        psubword.communicate()

        for hypfile in glob(hypotheses_fname):
            # post-process hypothesis translations
            with tempfile.NamedTemporaryFile() as temp_hyp:
                # call another python script to compute scores
                pcat  = Popen(['cat', hypfile], stdout=PIPE, stderr=fnull)
                # convert translations from subwords into words
                psubword = Popen(['sed', '-r', 's/(@@ )|(@@ ?$)//g'], stdin=pcat.stdout, stdout=temp_hyp)
                # write translations after BPE-to-word post-processing to temporary file
                psubword.communicate()

                # compute METEOR using `meteor-1.5.jar`
                pmeteor = Popen(['java', '-Xmx2G', '-jar', METEOR_SCRIPT, temp_hyp.name, temp_ref.name, "-l", "de", "-norm"], stdout=PIPE, stderr=fnull)
                # extract METEOR scores from output string
                # grep -w "Final score:[[:space:]]" | tr -s ' ' | cut -d' ' -f3
                pgrep = Popen(['grep', '-w', "Final score:[[:space:]]"], stdin=pmeteor.stdout,  stdout=PIPE, stderr=fnull)
                #print(pgrep.communicate()[0].strip().decode('utf8'))
                #sys.exit(1)
                ptr = Popen(['tr', '-s', ' '], stdin=pgrep.stdout,  stdout=PIPE, stderr=fnull)
                pcut = Popen(['cut', '-d', ' ', '-f', '3'], stdin=ptr.stdout,  stdout=PIPE, stderr=fnull)
                #pcut1 = Popen(['cut', '-d', ',', '-f1'], stdin=pbleu.stdout, stdout=PIPE, stderr=fnull)
                #pcut2 = Popen(['cut', '-d', ' ', '-f3'], stdin=pcut1.stdout, stdout=PIPE, stderr=fnull)

                # add it to array of BLEUs
                final_meteor=pcut.communicate()[0].strip().decode('utf8')
                final_meteor=float(final_meteor)*100
                model_meteors.append(final_meteor)
                model_names.append(hypfile.replace('.pt.translations-%s'%split, '.pt'))
                translation_files.append(hypfile)

                #print("Computed METEOR: %s"%final_meteor)

                curr_model_idx += 1
                #print("final bleu: %s"%str(final_bleu))
    assert(curr_model_idx == len(model_meteors)), 'Problem detected while computing METEORs.'
    return model_names, model_meteors, translation_files


def compute_bleus(hypotheses_fname, references_fname, split='valid', bleu_script="./multi-bleu.perl"):
    """ This function computes BLEU for all translations one by one without threading/queuing.
        It first converts subwords back into words before compute BLEU scores.
    """
    assert(split in ['valid', 'test2016']), 'Must compute BLEU for either valid or test set test2016!'

    # compute BLEU scores for each of the translations of the validation set
    curr_model_idx=0
    model_bleus, model_names, translation_files = [], [], []
    # post-process reference translations
    with tempfile.NamedTemporaryFile() as temp_ref:
        pcat  = Popen(['cat', references_fname], stdout=PIPE, stderr=fnull)
        # convert translations from subwords into words
        psubword = Popen(['sed', '-r', 's/(@@ )|(@@ ?$)//g'], stdin=pcat.stdout, stdout=temp_ref)
        # write translations after BPE-to-word post-processing to temporary file
        psubword.communicate()

        for hypfile in glob(hypotheses_fname):
            # call another python script to compute validation set BLEU scores
            pcat  = Popen(['cat', hypfile], stdout=PIPE, stderr=fnull)
            # convert translations from subwords into words
            psubword = Popen(['sed', '-r', 's/(@@ )|(@@ ?$)//g'], stdin=pcat.stdout, stdout=PIPE)
            # compute BLEU using `multi-bleu.perl`
            pbleu = Popen([bleu_script, temp_ref.name], stdin=psubword.stdout, stdout=PIPE, stderr=fnull)
            #pbleu = Popen([BLEU_SCRIPT, references_fname], stdin=psubword.stdout, stdout=PIPE, stderr=fnull)
            # extract BLEU scores from output string
            pcut1 = Popen(['cut', '-d', ',', '-f1'], stdin=pbleu.stdout, stdout=PIPE, stderr=fnull)
            pcut2 = Popen(['cut', '-d', ' ', '-f3'], stdin=pcut1.stdout, stdout=PIPE, stderr=fnull)

            # add it to array of BLEUs
            final_bleu=pcut2.communicate()[0].strip().decode('utf8')
            model_bleus.append(final_bleu)
            model_names.append(hypfile.replace('.pt.translations-%s'%split, '.pt'))
            translation_files.append(hypfile)

            curr_model_idx += 1
            #print("final bleu: %s"%str(final_bleu))
    assert(curr_model_idx == len(model_bleus)), 'Problem detected while computing BLEUs.'
    return model_names, model_bleus, translation_files


def parse_cmd_line():
    import argparse

    parser = argparse.ArgumentParser(prog='translate_model_selection')
    parser.description = 'Model selection of NMT models'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--src_language', type=str, required=True, help='WALS source language code (three letters)')
    parser.add_argument('--tgt_language', type=str, required=True, help='WALS target language code (three letters)')
    #parser.add_argument(
    #    '--bleu-script', type=str, required=True,
    #    help="Path to BLEU evaluation script multi-bleu.perl")
    #parser.add_argument(
    #    '--meteor-script', type=str, required=True,
    #    help="Path to METEOR (meteor-1.5.jar)")
    parser.add_argument('-v', '--verbose', required=False, default=False, action='store_true')
    parser.add_argument(
        '--data', type=str, required=True, 
        help="Path to datasets")
    parser.add_argument(
        '--model', type=str, required=True,
        help="Path to model as a prefix up to N_TRAINING_STEPS number (this is the prefix in *_steps_10000.pt)")
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output directory (it will be created if necessary)')
    parser.add_argument("--wals_src", type=str, required=True,
            help="""WALS source language code (three letters).""")
    parser.add_argument("--wals_tgt", type=str, required=True,
            help="""WALS target language code (three letters).""")
    parser.add_argument("--wals_function", type=str, required=True, choices=['tanh', 'relu'],
            help="""WALS features activation function.""")
    parser.add_argument("--wals_model_type", type=str, required=True,
            choices=['EncInitHidden_Target', 'EncInitHidden_Both',
                'DecInitHidden_Target', 'DecInitHidden_Both',
                'WalstoSource_Target', 'WalstoSource_Both',
                'WalstoTarget_Target', 'WalstoTarget_Both',
                'WalsDoublyAttentive_Target', 'WalsDoublyAttentive_Both'],
            help="""WALS model type.""")
    parser.add_argument('--delete_model_files', required=True, type=str, choices=['no', 'all-but-best', 'all'],
            help='Delete model files after generating translations?')

    return parser.parse_args()


if __name__=="__main__":
    args = parse_cmd_line()

    #BLEU_SCRIPT=args.bleu_script if args.bleu_script.endswith("multi-bleu.perl") else '%s/multi-bleu.perl'%args.bleu_script
    BLEU_SCRIPT="./tools/multi-bleu.perl"
    #METEOR_SCRIPT='%s/meteor-1.5.jar' % args.meteor_script

    OUTPUT_DIR = '{}/{}'.format(args.output, os.path.basename(args.model))
    #os.makedirs(OUTPUT_DIR, exist_ok=True)

    # validation set files
    VALID_SRC='%s/endefr.val.%s' % (args.data, args.src_language)
    VALID_REF='%s/endefr.val.%s' % (args.data, args.tgt_language)
    #VALID_WALS='%s/flickr30k_valid_wals_features.hdf5' % args.data
    # test set files
    TEST_SRC='%s/endefr.test2016.%s' % (args.data, args.src_language)
    TEST_REF='%s/endefr.test2016.%s' % (args.data, args.tgt_language)
    #TEST_WALS='%s/flickr30k_test2016_wals_features.hdf5' % args.data

    # /dev/null
    fnull = open(os.devnull, 'w')

    #summaries = []
    # set n_runs=1 and uncomment the MODEL_PREFIX you want to generate translations for (when only 1 run is available)
    # translate validation set with each of the model snapshots saved during training
    #MODEL_PREFIX="{}_RUN-{}_{}*".format(args.model, run_, args.hparams)
    MODEL_PREFIX="{}*".format(args.model)
    # make sure model prefix is valid (there is at least one model file that can be obtained with it
    assert(len(glob("%s*.pt"%str(MODEL_PREFIX)))>0), \
            'Model prefix did not result in any files: %s'%MODEL_PREFIX

    # translation options
    gpuid=None
    verbose=args.verbose
    overwrite_previous_translations = False
    beam_size = 1
    num_threads = 5

    #model_names, validation_bleus, validation_meteors, translation_files = [], [], [], []
    model_names, validation_bleus, translation_files = [], [], []
    # translate the validation set with all model snapshots obtained during training
    print("Translating validation set with a pool of %i workers..."%(num_threads))
    with Pool(num_threads) as p:
        # create list of translation jobs
        translation_jobs = []

        source_fname = VALID_SRC
        #img_feats_fname = TEST_IMG
        for model_fname in glob("%s*.pt"%str(MODEL_PREFIX)):
            hyp_fname = "%s.translations-valid"%str(model_fname)

            if not os.path.isfile(hyp_fname) or overwrite_previous_translations:
                if gpuid==1:
                    gpuid = 0
                else:
                    gpuid = 1

                #print("gpuid: %i"%gpuid)
                translation_jobs.append( (source_fname, model_fname, hyp_fname, beam_size, args.wals_src, args.wals_tgt, args.wals_function, args.wals_model_type, gpuid, 'valid', verbose) )
            else:
                # check that all jobs were translated as expected
                valid_set_size_match( hyp_fname, raise_exception=True, verbose=verbose )
                #if not valid_set_size_match( hyp_fname, raise_exception=False, verbose=verbose ):
                #    # translate once again
                #    if gpuid==1:
                #        gpuid = 0
                #    else:
                #        gpuid = 1

                #    #print("gpuid: %i"%gpuid)
                #    translation_jobs.append( (source_fname, model_fname, hyp_fname, beam_size, args.wals_src, args.wals_tgt, args.wals_function, args.wals_model_type, gpuid, 'valid', verbose) )



        #print("translation jobs: ",translation_jobs)
        #sys.exit(1)

        # run translation jobs in parallel using a Pool object
        p.starmap(translate_with_pool, translation_jobs)
        p.close()
        p.join()
        print("Finished all jobs in the pool!")
        print()

        print("Computing metrics (BLEU, METEOR) on validation set with a pool of %i workers..."%(num_threads))
        # compute BLEU and METEOR scores for all translations
        for hyp_fname in glob("%s*.pt.translations-valid"%str(MODEL_PREFIX)):
            ref_fname = VALID_REF
            #model_name_, validation_meteor_, translation_file_ = compute_meteors(hyp_fname, ref_fname, 'valid')
            model_name_, validation_bleu_,   translation_file_ = compute_bleus(hyp_fname, ref_fname, 'valid', BLEU_SCRIPT)
            model_names.append(model_name_[0])
            validation_bleus.append(validation_bleu_[0])
            #validation_meteors.append(validation_meteor_[0])
            translation_files.append(translation_file_[0])

            #print("model names "+str(model_names))
            #print("validation bleus "+str(validation_bleus))
            #print("translation files "+str(translation_files))
        print("Finished!")

        # check that all jobs were translated as expected
        for hyp_fname in glob("%s*.pt.translations-valid"%str(MODEL_PREFIX)):
            valid_set_size_match( hyp_fname, raise_exception=True, verbose=verbose )

    # get max and min validation BLEU
    validation_bleus = numpy.array(validation_bleus, dtype='float32')
    if verbose:
        print("validation_bleus: %s"%str(validation_bleus))

    # if there is a tie, get the latest best model
    occurrences = numpy.where(validation_bleus == validation_bleus.max())
    idx_max, idx_min = occurrences[-1][0], numpy.argmin(validation_bleus)
    best_model, worst_model = model_names[idx_max], model_names[idx_min]
    best_model_translations, worst_model_translations = translation_files[idx_max], translation_files[idx_min]
    best_bleu, worst_bleu   = validation_bleus[idx_max], validation_bleus[idx_min]
    print("Best model on validation set: %s"%(str(best_model)))
    print("Validation BLEU: %.2f"%(best_bleu))
    print("---")

    # translate the test set with the best model snapshot on the validation set
    beam_size = 10
    num_threads = 1
    overwrite_previous_translations = False
    print("Translating test set with a pool of %i workers..."%(num_threads))
    with Pool(num_threads) as p:
        model_fname = best_model
        hyp_fname   = "%s.translations-test2016"%str(best_model)
        source_fname = TEST_SRC
        #img_feats_fname = TEST_IMG
        if not os.path.isfile(hyp_fname) or overwrite_previous_translations:
            gpuid = 0 if gpuid is None or gpuid==1 else 1
            p.starmap(translate_with_pool, [(source_fname, model_fname, hyp_fname, beam_size, args.wals_src, args.wals_tgt, args.wals_function, args.wals_model_type, gpuid, 'test2016', verbose)])
            p.close()
            p.join()
            print("Finished all jobs in the pool!")
            print()
        else:
            test_set_size_match( hyp_fname, raise_exception=True, verbose=verbose )

    test_hypotheses_fname = "%s*.translations-test2016"%(str(best_model))
    ref_fname = TEST_REF
    #model_names, test_meteors, translation_files = compute_meteors(test_hypotheses_fname, ref_fname, 'test2016')
    model_names, test_bleus,   translation_files = compute_bleus(test_hypotheses_fname, ref_fname, 'test2016', BLEU_SCRIPT)

    print("Best model on test2016 set: %s"%str(model_names))
    print("BLEU: %.2f"%(numpy.array(test_bleus, dtype='float32')))
    #print("METEOR: %.2f"%(numpy.array(test_meteors, dtype='float32')))
    print("Translations found in: %s"%str(translation_files))

    summary = {'model_name': model_names[0],
       'valid': {
           'epoch': idx_max+1,
           'bleu': numpy.array(validation_bleus, dtype='float32'),
           #'meteor': numpy.array(validation_meteors, dtype='float32')
        },
       'test': {
           'bleu': numpy.array(test_bleus, dtype='float32'),
           #'meteor': numpy.array(test_meteors, dtype='float32')
        }
    }

    # save summary
    #pickle.dump( summary, open("%s.bleus-and-meteors.pkl"%str(model_names[0]), 'wb') )
    pickle.dump( summary, open("%s.bleus.pkl"%str(model_names[0]), 'wb') )
    #summaries.append(summary)

    # delete model files?
    if not args.verbose:
        print("Delete model files? %s"%args.delete_model_files)

    if not args.delete_model_files == 'no':
        if args.delete_model_files == 'all':
            # delete all files
            for model_fname in glob("%s*.pt"%str(MODEL_PREFIX)):
                if args.verbose:
                    print("Deleting model file: %s"%model_fname)
                os.remove(model_fname)

        elif args.delete_model_files == 'all-but-best':
            # delete all models but the one selected
            for model_fname in glob("%s*.pt"%str(MODEL_PREFIX)):
                if model_fname == best_model:
                    if args.verbose:
                        print("NOT deleting model file: %s"%model_fname)
                    continue
                else:
                    if args.verbose:
                        print("Deleting model file: %s"%model_fname)
                    os.remove(model_fname)

        else:
            raise Exception("Not implemented!")


