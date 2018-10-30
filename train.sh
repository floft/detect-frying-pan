#!/bin/bash
#
# Run training (and evaluation apparently)
#
. config.sh
cd models/research

protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/slim"
python3 object_detection/model_main.py \
        --model_dir="../../$models" \
        --pipeline_config_path="../../$config"
