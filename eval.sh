#!/bin/bash
#
# Run evaluation (while training)
#
cd models/research

export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/slim"
python3 object_detection/model_main.py \
        --checkpoint_dir="../../object_detection_models" \
        --pipeline_config_path="../../ssd_mobilenet_v1_coco.config"
