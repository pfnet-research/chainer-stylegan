#!/bin/bash

set -e

SUBPROJ_NAME="stylegan-ffhq"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
RUN_ID=$RANDOM

EXPR_ID=$1
EXPR_ID=${EXPR_ID:=0}

if [ $EXPR_ID -lt 1 ]; then
  echo "Specify the EXPR_ID that is > 0"
  exit 0
fi

SCRATCH_ROOT="$SCRIPT_DIR/../../scratch/"
RUN_SCRIPT="$SCRIPT_DIR/train.py"
runner_prefix=""
python_version="python3"

DATASET_CONFIG="$SCRIPT_DIR/../dataset_config/ffhq.json"

if [ $EXPR_ID -eq 1 ]; then
    runner="$runner_prefix "
    hps_device="--gpu 0"
    mpi="--use_mpi=False"
    run_iter="--dynamic_batch_size 256,256,256,128,128,64,64,32,32,8,8,4,4,2,2,1,1 --max_stage 13 --stage_interval 1250000"
    eval_iter="--evaluation_sample_interval 500 --display_interval 10 --snapshot_interval 5000"
elif [ $EXPR_ID -eq 2 ]; then
    nb_gpu="8"
    runner="$runner_prefix mpiexec -n $nb_gpu"
    hps_device=""
    mpi="--use_mpi=True --comm_name single_node"
    run_iter="--dynamic_batch_size 512,256,256,128,128,64,64,32,32,8,8,4,4,2,2,1,1 --max_stage 17 --stage_interval 1250000"
    eval_iter="--evaluation_sample_interval 500 --display_interval 10 --snapshot_interval 5000"
fi

hps_training_dynamics="$eval_iter $mpi $run_iter"
hps_lr="--adam_alpha_g 0.001 --adam_alpha_d 0.001 --adam_beta1 0.0 --adam_beta2 0.999"
hps_hyperparameters="--lambda_gp 5.0 --smoothing 0.999 --keep_smoothed_gen=True"
hps_dataset="--dataset_config $DATASET_CONFIG --dataset_worker_num 16"
hps_output="--out $SCRATCH_ROOT/$SUBPROJ_NAME/$EXPR_ID"
hps_resume="--auto_resume"

$runner $python_version $RUN_SCRIPT \
    $hps_lr \
    $hps_training_dynamics \
    $hps_hyperparameters \
    $hps_dataset \
    $hps_device \
    $hps_output \
    $hps_resume \
    ;

