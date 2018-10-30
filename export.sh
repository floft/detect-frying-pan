#!/bin/bash
#
# Export model
#
. config.sh
cd models/research

export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/slim"
export CUDA_VISIBLE_DEVICES=

# Export latest checkpoint
lastCheckpoint="$(ls ../../$models/model.ckpt-* | \
        sort | tail -n 1 | sed -r 's#\.[^\.]+$##g')"

python3 object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path="../../$config" \
    --trained_checkpoint_prefix="$lastCheckpoint" \
    --output_directory="../../$exported" \
    --add_postprocessing_op=true

if (( $quantize )); then
    echo "Outputting quantized model"
    toco \
        --graph_def_file="../../$exported/tflite_graph.pb" \
        --output_file="../../$exported/detect.tflite" \
        --input_shapes=1,300,300,3 \
        --input_arrays=normalized_input_image_tensor \
        --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
        --inference_type=QUANTIZED_UINT8 \
        --mean_values=128 \
        --std_dev_values=128 \
        --change_concat_input_ranges=false \
        --allow_custom_ops
else
    echo "Outputting floating point model"
    toco \
        --graph_def_file="../../$exported/tflite_graph.pb" \
        --output_file="../../$exported/detect.tflite" \
        --input_shapes=1,300,300,3 \
        --input_arrays=normalized_input_image_tensor \
        --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
        --inference_type=FLOAT \
        --allow_custom_ops
fi
