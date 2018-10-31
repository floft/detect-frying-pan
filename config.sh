#
# Options
#
# For timing testing, also output the frozen graph (i.e. not using TF Lite)
exported_graph="exported_models.graph"

# Floating point
#config="ssd_mobilenet_v1_coco.config"
#models="object_detection_models.float"
#exported="exported_models.float"
#quantize=0

# Quantized
config="ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync.config"
models="object_detection_models.quantized"
exported="exported_models.quantized"
quantize=1

# PPN
#config="ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync.config"
#models="object_detection_models.ppn"
#exported="exported_models.ppn"
#quantize=0
