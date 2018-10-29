#!/bin/bash
#
# Run training
#
cd models/research

protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/slim"
python3 object_detection/model_main.py \
        --noeval_training_data \
        --model_dir="../../object_detection_models" \
        --pipeline_config_path="../../ssd_mobilenet_v1_coco.config"
