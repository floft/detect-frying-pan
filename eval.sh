#!/bin/bash
#
# Run evaluation constantly while training (if desired)
#
cd models/research

export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/slim"
export CUDA_VISIBLE_DEVICES= # Run evaluation on CPU since GPU used by training
python3 object_detection/model_main.py \
        --model_dir="../../object_detection_models" \
        --checkpoint_dir="../../object_detection_models" \
        --pipeline_config_path="../../ssdlite_mobilenet_v1_coco.config"
