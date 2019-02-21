#!/bin/bash

set -e

# Parse Experiment ID

EXPR_ID=$1
EXPR_ID=${EXPR_ID:=0}

if [ $EXPR_ID -lt 1 ]; then
  echo "Specify the EXPR_ID that is > 0"
  exit 0
fi

# Dirs
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$SCRIPT_DIR/../../data/"

# Common hps across Experiments
hps_config_common="--num_threads 8 --buffer_size_mb 8000"

################
# Experiments #
###############

if [ $EXPR_ID -ge 1 -a $EXPR_ID -le 1 ]; then
  declare -A _map=([1]=1024)
  image_size=${_map[$EXPR_ID]}
  min_input_image_size=$((200))
  echo "Case ${EXPR_ID}: FFHQ, multisize h5 with $image_size resolution."

  DATASET_DIR="$2"
  DATASET_H5="$ROOT/ffhq-hdf5/"
  hps_io="--folder_path $DATASET_DIR --h5_filename $DATASET_H5/data.h5"
  hps_config="--image_size ${image_size} ${hps_config_common} --min_input_image_size ${min_input_image_size}"

  if [ -f $DATASET_H5 ] ; then
    echo "Skip since $DATASET_H5 exists."
  else
    python3 folder_to_multisize_hdf5.py $hps_io $hps_config
  fi
fi
