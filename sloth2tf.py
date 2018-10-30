#!/usr/bin/env python3
"""
Sloth to TensorFlow Object Detection

Convert the .json file generated with Sloth to be in the format for TF.

Note: this is a subset of what is contained in the RAS version:
https://github.com/WSU-RAS/ras-object-detection
"""
import os
import random
from PIL import Image
import tensorflow as tf
from models.research.object_detection.utils import dataset_util

from sloth_common import getJson, uniqueClasses, predefinedClasses, \
    imgInfo, mapLabel, splitData

# Make this repeatable
random.seed(0)

def loadImage(filename):
    """
    TensorFlow needs the encoded data
    """
    with tf.gfile.GFile(filename, 'rb') as f:
        encoded = f.read()

    return encoded

def remove_transparency(im, bg_colour=(255, 255, 255)):
    """
    Remove alpha channel of image

    Taken from: https://stackoverflow.com/a/35859141
    """
    # Only process if image has transparency
    # (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        # Need to convert to RGBA if LA format due to a bug in PIL
        # (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').getchannel('A')

        # Create a new background image of our matt color. Must be RGBA because
        # paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632 and
        # http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im

def resize_image(filename, max_size=600):
    """
    Resize if any dimension over a certain amount. This is used because without
    it I'm CPU bound trying to do all the data augmentation to huge images.

    Taken from: https://stackoverflow.com/a/28453021
    Results: find -name "*_resized.jpg"
    """
    image = Image.open(filename)
    original_size = max(image.size[0], image.size[1])

    # Skip RGBA images
    if image.mode == "RGBA":
        return

    # Too big -> shrink
    if original_size > max_size:
        resized_file = open(os.path.splitext(filename)[0] + '_resized.jpg', "w")
        if (image.size[0] > image.size[1]):
            resized_width = max_size
            resized_height = int(round((max_size/float(image.size[0]))*image.size[1]))
        else:
            resized_height = max_size
            resized_width = int(round((max_size/float(image.size[1]))*image.size[0]))

        image = image.resize((resized_width, resized_height), Image.ANTIALIAS)
        image.save(resized_file, 'JPEG')
        new_fn = resized_file.name

        print("Shrinking", filename, "since", original_size, ">", max_size,
                "new size:", str(resized_width)+"x"+str(resized_height),
                "new name:", resized_file.name)

    # No change
    else:
        new_fn = filename

    return new_fn

def bounds(x):
    """
    TensorFlow errors if we have a value less than 0 or more than 1. This
    occurs if in Sloth you draw a bounding box slightly out of the image. We'll
    just cut the bounding box at the edges of the images.
    """
    return max(min(x, 1), 0)

def create_tf_example(labels, filename, annotations, debug=False):
    """
    Based on:
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
    """
    if debug:
        print(filename)

    # Skip RGBA images
    orig_width, orig_height, _, orig_mode = imgInfo(filename)

    if orig_mode == "RGBA":
        print("Warning: skipping", filename, "since RGBA")
        return

    filename = resize_image(filename) # Resize if too big
    new_width, new_height, imgformat, _ = imgInfo(filename)
    encoded_image_data = loadImage(filename) # Encoded image bytes

    if debug:
        print(filename, str(width)+"x"+str(height), imgformat)

    if imgformat == 'PNG':
        image_format = b'png' # b'jpeg' or b'png'
    elif imgformat == 'JPEG':
        image_format = b'jpeg'
    else:
        print("Warning: skipping", filename, "since only supports PNG or JPEG images")

    # Calculate the annotations based on the original width/height since that's
    # what was annotated (i.e. before we resize)
    xmins = []        # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []        # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []        # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []        # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = []      # List of integer class id of bounding box (1 per box)

    for a in annotations:
        # Numeric and text class labels
        classes.append(mapLabel(labels, a['class']))
        classes_text.append(a['class'].encode())

        # Scaled min/maxes
        xmins.append(bounds(a['x']/orig_width))
        ymins.append(bounds(a['y']/orig_height))
        xmaxs.append(bounds((a['x']+a['width'])/orig_width))
        ymaxs.append(bounds((a['y']+a['height'])/orig_height))

        # We got errors: maximum box coordinate value is larger than 1.010000
        valid = lambda x: x >= 0 and x <= 1
        assert valid(xmins[-1]) and valid(ymins[-1]) and valid(xmaxs[-1]) and valid(ymaxs[-1]), \
                "Invalid values for "+filename+": "+ \
                str(xmins[-1])+","+str(ymins[-1])+","+str(xmaxs[-1])+","+str(ymaxs[-1])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(new_height),
        'image/width': dataset_util.int64_feature(new_width),
        'image/filename': dataset_util.bytes_feature(filename.encode()),
        'image/source_id': dataset_util.bytes_feature(filename.encode()),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def splitJsonData(data, trainPercent=0.8, validPercent=0.2, shuffle=True):
    """
    Split the JSON data so we can get a training, validation, and testing file

    Returns pairs of (img filename, annotations) for each set
    """
    results = []

    for image in data:
        # Skip if we don't have any labels for this image
        if not len(image['annotations']) > 0:
            continue

        results.append((image['filename'], image['annotations']))

    return splitData(results, trainPercent, validPercent, shuffle=shuffle)

def tfRecord(folder, labels, output, data):
    """
    Output to TF record file
    """
    with tf.python_io.TFRecordWriter(output) as writer:
        for (img, annotations) in data:
            filename = os.path.join(folder, img)
            tf_example = create_tf_example(labels, filename, annotations)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())

def tfLabels(labels, output):
    """ Write tf_label_map.pbtxt file containing the labels/integer mapping """
    with open(output, 'w') as f:
        for i, label in enumerate(labels):
            f.write('item {\n'+
                    '  id: '+str(i+1)+'\n'+
                    '  name: \''+label+'\'\n'+
                    '}\n')

def main(_):
    # Get JSON data
    folder = "."
    data = getJson(os.path.join(folder, "sloth.json"))
    labels = uniqueClasses(data)
    #labels = predefinedClasses()

    # Save labels
    tfLabels(labels, os.path.join(folder, "tf_label_map.pbtxt"))

    # Split, e.g. 80% training, 20% validation, and 0% testing
    training_data, validate_data, testing_data = splitJsonData(data)

    # Save the record files
    print("Saving train")
    tfRecord(folder, labels, os.path.join(folder, "tftrain.record"), training_data)
    print("Saving valid")
    tfRecord(folder, labels, os.path.join(folder, "tfvalid.record"), validate_data)
    print("Saving test")
    tfRecord(folder, labels, os.path.join(folder, "tftest.record"), testing_data)

    for f, a in training_data:
        print(f)

if __name__ == "__main__":
    tf.app.run()
