import argparse
import csv
import random

import numpy as np
import cv2
import tensorflow as tf

import utils

def create_tf_example(image, file, bounding_boxes):
    """ Creates a tf.Example proto hand image.

    Args:
        image: The raw image opened via opencv.
        file: The file name as a string.
        bounding_boxes: A list of bounding boxes. Each element is a list [x, y, w, h]

    Returns:
        example: The created tf.Example.
    """

    filename = bytes(file, 'utf-8')
    height, width, _ = np.shape(image)
    encoded_image_string = cv2.imencode('.jpg', image)[1].tostring()
    image_format = b'jpg'

    xmins, xmaxs, ymins, ymaxs = [[] for i in range(4)]
    for box in bounding_boxes:
        x, y, w, h = box
        xmins.append(x / width)
        xmaxs.append((x+w) / width)
        ymins.append(y / height)
        ymaxs.append((y+h) / height)

    num_boxes = len(bounding_boxes)
    classes_text = [b'fingertip' for i in range(num_boxes)]
    classes = [1 for i in range(num_boxes)]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': utils.int64_feature(height),
        'image/width': utils.int64_feature(width),
        'image/filename': utils.bytes_feature(filename),
        'image/source_id': utils.bytes_feature(filename),
        'image/encoded': utils.bytes_feature(encoded_image_string),
        'image/format': utils.bytes_feature(image_format),
        'image/object/bbox/xmin': utils.float_list_feature(xmins),
        'image/object/bbox/xmax': utils.float_list_feature(xmaxs),
        'image/object/bbox/ymin': utils.float_list_feature(ymins),
        'image/object/bbox/ymax': utils.float_list_feature(ymaxs),
        'image/object/class/text': utils.bytes_list_feature(classes_text),
        'image/object/class/label': utils.int64_list_feature(classes),
    }))
    return tf_example

def write_sharded_tfrecord(example_list, output_filebase, num_shards=10):
    """ Writes tf.Example data to sharded TFRecord files.

    Args:
        example_list: A list of tf.Example elements.
        output_filebase: The destination path and base filename for output files.
        num_shards: Number of output file pieces.
    """
    length = len(example_list)
    sharded_examples = [example_list[i*length//num_shards: (i+1)*length//num_shards] for i in range(num_shards)]
    for i in range(num_shards):
        shard = sharded_examples[i]
        output_path = "{0}-{1:05d}-of-{2:05d}".format(output_filebase, i, num_shards)
        writer = tf.python_io.TFRecordWriter(output_path)

        for tf_example in shard:
            writer.write(tf_example.SerializeToString())
        writer.close()

if __name__ == "__main__":
    dataset_path = "11k_hands/"
    input_path = dataset_path + "fingertip_labels.csv"

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--split", type=float, required=False, default=2/3, help="fraction of data for training")
    args = ap.parse_args()
    assert 0 < args.split < 1

    example_list = []
    with open(input_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            file = row.pop(0)
            raw = cv2.imread(file)
            bounding_boxes = []
            for box in np.reshape(row, [-1, 4]):
                bounding_boxes.append(box)

            tf_example = create_tf_example(raw, file, bounding_boxes)
            example_list.append(tf_example)

    print("{0} examples created.".format(len(example_list)))
    random.seed(2018)
    random.shuffle(example_list)
    train_list, val_list = np.split(example_list, [int(len(example_list) * args.split)])
    print("writing TFRecord files...")
    write_sharded_tfrecord(train_list, output_filebase=dataset_path+"fingertips_train.record")
    write_sharded_tfrecord(val_list, output_filebase=dataset_path+"fingertips_val.record")
    print("done.")
