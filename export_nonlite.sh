#!/bin/bash
#
# Export model, but freeze the graph rather than exporting for TF Lite
#
. config.sh
cd models/research

export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/slim"
export CUDA_VISIBLE_DEVICES=

# Export latest checkpoint
lastCheckpoint="$(ls ../../$models/model.ckpt-* | \
        sort | tail -n 1 | sed -r 's#\.[^\.]+$##g')"

python3 object_detection/export_inference_graph.py \
    --pipeline_config_path="../../$config" \
    --trained_checkpoint_prefix="$lastCheckpoint" \
    --output_directory="../../$exported_graph"
