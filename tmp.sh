#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 0-3:00:00
#SBATCH -p gpu


TIMESTAMP=`date +%s`

# Details of the training: this is the stuff you can change
SRC_TGT="En-De"
WALS_SRC="eng"
WALS_TGT="ger"
SRC="en"
TGT="de"
#WALS_MODELTYPE="EncInitHidden_Target"
#WALS_FUNC="tanh"
#WALS_SIZE=10
#MODEL_CONFIG=${SRC_TGT}."enc-init-hidden-target".${WALS_FUNC}.${WALS_SIZE}
COPY_OUTPUT_DIR=${HOME}/experiments/wals/${SRC_TGT}
PATH_DATA=${HOME}/data/wals/
TRAINER=${HOME}/git/WALS/train.py
TRANSLATE_MS=${HOME}/git/WALS/translate_model_selection.py
VENV=${HOME}/envs/wals_env


# Get a working directory on scratch space
# share data for all gpus
mkdir -p ${TMPDIR}/data
# to save models
mkdir -p ${TMPDIR}/0/${SRC_TGT}/model_snapshots
mkdir -p ${TMPDIR}/1/${SRC_TGT}/model_snapshots
mkdir -p ${TMPDIR}/2/${SRC_TGT}/model_snapshots
mkdir -p ${TMPDIR}/3/${SRC_TGT}/model_snapshots

# Load modules and python environment
module load python/3.5.0
module load gcc/5.4.0
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SURFSARA_LIBRARY_PATH
# Source python venv
source ${VENV}/bin/activate
# copy to scracht
cp -r ${PATH_DATA} ${TMPDIR}/data/

# Run training: on lisa each machine has 4 GPUs, thus we run 4 jobs

WALS_MODEL_TYPE=("EncInitHidden_Target" "EncInitHidden_Both")
WALS_FUNC=("tanh" "relu")
WALS_SIZE=("10" "100")
pids=()
gpuid=0
ngpus=4

# iterate the lists with different hyperparameter options (grid search?)
for i in ${WALS_MODEL_TYPE[@]}; do
    for j in ${WALS_FUNC[@]}; do
        for k in ${WALS_SIZE[@]}; do

            # call to the runner
            MODEL_CONFIG=${SRC_TGT}.$i.$j.$k
            
            python3 ${TRAINER} -data ${TMPDIR}/data/${SRC_TGT}/bpe_endefr -save_model ${TMPDIR}/$gpuid/${SRC_TGT}/model_snapshots/${MODEL_CONFIG} -wals_src ${WALS_SRC} -wals_tgt ${WALS_TGT} -wals_model ${i} -wals_function ${j} -wals_size ${k} -input_feed 0 -gpu_ranks $gpuid -save_checkpoint_steps 1000 -train_steps 1000 -optim 'adam' -learning_rate 0.002 \
                    &> ${TMPDIR}/$gpuid/${SRC_TGT}/log.${SRC_TGT}.${MODEL_CONFIG} && \
            python3 ${TRANSLATE_MS} --data ${TMPDIR}/data/ --model ${MODEL_CONFIG} --output ${TMPDIR}/$gpuid/${SRC_TGT}/model_snapshots/ --wals_src ${WALS_SRC} --wals_tgt ${WALS_TGT} --wals_function ${j} --wals_model_type ${i} --src_language $SRC --tgt_language $TGT --delete_model_files 'all-but-best' &

            # switch gpu id between 0 and 3
            echo "gpuid: $gpuid"
            (( $gpuid < $ngpus - 1 )) && let gpuid=$gpuid+1 || gpuid=0

            # make sure we are only running a maximum of `ngpus` jobs in parallel
            pids+=($!)
            if (( ${#pids[@]} >= ${ngpus} )); then
                echo "waiting... pids: ${pids[@]}"
                wait ${pids[@]}
                pids=()
                #python3 ${TRANSLATE_MS} --data ${TMPDIR}/data/ --model ${MODEL_CONFIG} --output ${TMPDIR}/$gpuid/${SRC_TGT}/model_snapshots/ --wals_src ${WALS_SRC} --wals_tgt ${WALS_TGT} --wals_function ${j} --wals_model_type ${i} --src_language $SRC --tgt_language $TGT --delete_model_files 'all-but-best'
            fi
        done
    done
done

# wait for remaining jobs (if there are any)
wait
                
echo "You will eventually find the results in: TMPDIR"
# TODO copy from tmpdir to your local 
# TODO define ${COPY_OUTPUT_DIR}
# copy all output from scratch to our home dir
# if we used a dependent job we will also have a copy in the archive
#delete extra data
rm -rf ${TMP_DIR}/data
mkdir -p ${COPY_OUTPUT_DIR}
rsync -vat ${TMPDIR} ${COPY_OUTPUT_DIR} &

wait
