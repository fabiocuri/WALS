#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 0-10:00:00
#SBATCH -p gpu


TIMESTAMP=`date +%s`

# Details of the training: this is the stuff you can change
SRC_TGT="En-De"
WALS_SRC="eng"
WALS_TGT="ger"
WALS_MODELTYPE="EncInitHidden_Target"
WALS_FUNC="tanh"
WALS_SIZE=10
MODEL_CONFIG=${SRC_TGT}."enc-init-hidden-target".${WALS_FUNC}.${WALS_SIZE}
COPY_OUTPUT_DIR=${HOME}/experiments/wals/${TIMESTAMP}
PATH_DATA=${HOME}/data/wals/${SRC_TGT}
SCRIPTSDIR=${HOME}/OpenNMT/OpenNMTWals/
TRAINER=${SCRIPTSDIR}/train.py

# Get a working directory on scratch space
mkdir -p ${TMPDIR}/f0
mkdir -p ${TMPDIR}/f0/model_snapshots/${SRC_TGT}
#mkdir -p ${TMPDIR}/f1
#mkdir -p ${TMPDIR}/f2
#mkdir -p ${TMPDIR}/f3

# Load modules and python environment
module gcc/4.7.1
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SURFSARA_LIBRARY_PATH

echo $LD_LIBRARY_PATH
echo $SURFSARA_LIBRARY_PATH
# copy to scracht
cp -r ${PATH_DATA} ${TMPDIR}/f0

# Run training: on lisa each machine has 4 GPUs, thus we run 4 jobs
python3 ${TRAINER} -data ${TMPDIR}/f0/bpe_endefr -save_model ${TMPDIR}/f0/model_snapshots/${SRC_TGT}/${MODEL_CONFIG} -wals_src ${WALS_SRC} -wals_tgt ${WALS_TGT} -wals_model ${WALS_MODELTYPE} -wals_function ${WALS_FUNC} -wals_size ${WALS_SIZE} -input_feed 0 -gpu_ranks 0 -save_checkpoint_steps 1000 -train_steps 1000 -optim 'adam' -learning_rate 0.002 \
    &> ${TMPDIR}/f0/log.${SRC_TGT}.${MODEL_CONFIG} &


wait
                
echo "You will eventually find the results in: TMPDIR"
# TODO copy from tmpdir to your local 
# TODO define ${COPY_OUTPUT_DIR}
# copy all output from scratch to our home dir
# if we used a dependent job we will also have a copy in the archive
mkdir -p ${COPY_OUTPUT_DIR}
rsync -vat ${TMPDIR} ${COPY_OUTPUT_DIR} &

wait
