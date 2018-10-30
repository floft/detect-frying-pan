#!/bin/bash
#
# Run evaluation constantly while training (if desired)
#
. config.sh
cd models/research

export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/slim"
export CUDA_VISIBLE_DEVICES= # Run evaluation on CPU since GPU used by training
python3 object_detection/model_main.py \
        --model_dir="../../$models" \
        --checkpoint_dir="../../$models" \
        --pipeline_config_path="../../$config"
