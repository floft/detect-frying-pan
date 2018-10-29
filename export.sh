#1/bin/bash
#
# Export model
#
cd models/research

export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/slim"
export CUDA_VISIBLE_DEVICES=

lastCheckpoint="$(ls ../../object_detection_models/model.ckpt-* | \
        sort | tail -n 1 | sed -r 's#\.[^\.]+$##g')"

python3 object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path="../../ssdlite_mobilenet_v2_coco.config" \
    --trained_checkpoint_prefix="$lastCheckpoint" \
    --output_directory="../../exported_models" \
    --add_postprocessing_op=true
