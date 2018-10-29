Detect Frying Pan
=================
Use an object detection neural network to detect frying pans in images. The
purpose of this is for use in an autonomous drone egg drop entry.

# Instructions
Get this code:

    git clone --recursive https://github.com/floft/detect-frying-pan
    cd detect-frying-pan

## Collect images for dataset
### Download images from Google

    sudo pacman -S python-selenium chromium
    ./google-images-download/google_images_download/google_images_download.py \
        -cf google_image_config.json

Delete all those you don't like from the *google_images/* folder, e.g. if
they're not black, have watermarks, are clipart, have a white background, have
low depth of field, etc. Look for images that are mostly top-down mostly empty
if possible.

Since there will likely be duplicates, you can remove some of these (if they're
exact duplicates):

    jdupes google_images/ -r -d # select which you want, e.g. 1 for all duplicates

### Take your own images
Buy a skillet or frying pan and go take top-down pictures from a variety of
angles with many different backgrounds and lighting conditions. Maybe with a
variety of cameras too. Put them all in a *my_images_large/* folder.

Shrink them. Outputs to *my_images/*:

    ./shrink_images.sh

## Labeling images
Then, to label them in Sloth (see my Arch
[PKGBUILD](https://github.com/floft/PKGBUILDs/tree/master/python-sloth)):

    ./gen_sloth.sh
    ./annotate.sh # 'f' to label as frying pan, space for next, Ctrl+S to save

Convert TensorFlow {tftrain,tfvalid,tftest}.record files:

    ./sloth2tf.py

## Get pre-trained TensorFlow network

    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
    tar xaf ssd_mobilenet_v1_coco_2018_01_28.tar.gz

## Fix bug
To fix [a bug](https://github.com/tensorflow/models/issues/4996#issuecomment-410640308), in *models/research/object_detection/metrics/coco_tools.py* change

    results.dataset['categories'] = copy.deepcopy(self.dataset['categories'])

to

    results.dataset['categories'] = self.dataset['categories']

## Training
Install dependencies:

    sudo pacman -S cython
    pip install --user pycocotools

Then run training and monitor the results:

    ./train.sh
    tensorboard --logdir object_detection_models/

## Export model
Export the model for TensorFlow Lite:

    ./export.sh

