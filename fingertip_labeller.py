""" Single-Camera-Touchpad/fingertip_labeller.py

    Semi-automatical labeller for fingertips on the 11k hands dataset.
    Only images of the dorsal aspect are used.
    Convex hull vertices are used to guess fingertip locations.
    The user has the option to manually edit bounding box locations.
    Labels are exported in TFRecord format.
"""

import argparse
import csv
import random

import numpy as np
import cv2
import tensorflow as tf

import utils

def find_fingertips(image, threshold=5, finger_angle=5, tip_radius=25):
    """ Returns candidate bounding boxes after pre-processing and convex hull calculations.

    Args:
        image: The raw image of the hand as a numpy array of BGR values.
        threshold: The value for masking away the white background.
            Grayscale values in the range [255 - threshold, 255] are excluded.
        finger_angle: The number of degrees for vertices to be considered the same finger.
            Angles are calculated with respect to the center of the top row of pixels.
        tip_radius: The distance around the vertex to be placed in a bounding box.

    Returns:
        image: The hand image post-processing.
        bounding_boxes: A list of bounding boxes. Each element is a list [x, y, w, h]
    """

    image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 255-threshold, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    img, cnts, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    cv2.drawContours(image, contours, 0, (0, 0, 255), 1)

    contours = np.vstack(contours)
    hull = cv2.convexHull(contours)
    cv2.drawContours(image, [hull], 0, (0, 255, 0), 1)

    height, width, _ = np.shape(image)
    origin = [width//2, 0]
    dist = lambda a,b: (sum((ai-bi)**2 for ai, bi in zip(a, b))) ** 0.5
    def angle(point):
        if point[0] == origin[0]:
            return 90
        theta = np.arctan(point[1]/(origin[0] - point[0]))
        if theta < 0:
            theta += np.pi
        return theta * 180 / np.pi

    current_id = -1
    candidates = {}
    longest_length = 0
    middle_finger = -1

    for h in hull:
        point = h[0]
        d = dist(origin, point)
        t = angle(point)

        if current_id == -1:
            check_id = -1
        else:
            if abs(t - angle(candidates[current_id])) < finger_angle:
                check_id = current_id
            elif abs(t - angle(candidates[0])) < finger_angle:
                check_id = 0
            else:
                check_id = -1
        if check_id == -1: # new finger found
            current_id += 1
            candidates[current_id] = point
        else: # compare point with previous max
            if d > dist(origin, candidates[check_id]):
                candidates[check_id] = point
                current_id = check_id

        if d > longest_length:
            longest_length = d
            middle_finger = current_id

    for i in range(current_id+1):
        tip = tuple(candidates[i])
        cv2.circle(image, tip, 2, (255, 0, 0), -1)
    
    fingertips = []
    for n in range(-2, 3):
        finger_id = (middle_finger + n) % (current_id + 1)
        fingertips.append(candidates[finger_id])

    cnt_indices = [[] for x in fingertips]
    for i in range(len(contours)):
        point = contours[i][0]
        for j in range(len(fingertips)):
            tip = fingertips[j]
            if dist(tip, point) < tip_radius:
                cnt_indices[j].append(i)

    bounding_boxes = []
    for indices in cnt_indices:
        cnt = contours[indices]
        x, y, w, h = cv2.boundingRect(cnt)
        if h < w:
            y -= (w - h)
            h = w
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 1)
        bounding_boxes.append([x, y, w, h])

    return image, bounding_boxes

def manual_edit(image, bounding_boxes):
    """ Creates an interactive window for the user to manually move bounding boxes.

    Args:
        image: The raw image of the hand as a numpy array of BGR values.
        bounding_boxes: A list of bounding boxes. Each element is a list [x, y, w, h]

    Returns:
        bounding_boxes: The modified list of bounding boxes.
    """

    image = image.copy()
    def click_and_crop(event, x, y, flags, param):
        nonlocal refPt, moving, sel_rect_endpoint

        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            moving = True
            sel_rect_endpoint = []
     
        elif event == cv2.EVENT_LBUTTONUP:
            refPt.append((x, y))
            moving = False

        elif event == cv2.EVENT_MOUSEMOVE and moving:
            sel_rect_endpoint = [(x, y)]

    clone = image.copy()
    original_boxes = bounding_boxes.copy()
    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 1)

    refPt = []
    moving = False
    sel_rect_endpoint = []
    dragging_box = -1

    cv2.namedWindow("manual_edit")
    cv2.setMouseCallback("manual_edit", click_and_crop)

    while True:
        if not moving:
            if len(refPt) == 2 and dragging_box != -1:
                x, y, w, h = bounding_boxes[dragging_box]
                x += refPt[1][0] - refPt[0][0]
                y += refPt[1][1] - refPt[0][1]
                bounding_boxes[dragging_box] = [x, y, w, h]
                dragging_box = -1

                image = clone.copy()
                for box in bounding_boxes:
                    x, y, w, h = box
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 1)
            cv2.imshow("manual_edit", image)
        elif moving:
            for i in range(len(bounding_boxes)):
                x, y, w, h = bounding_boxes[i]
                px, py = refPt[0]
                if x <= px <= x+w and y <= py <= y+h:
                    dragging_box = i

            if sel_rect_endpoint and dragging_box != -1:
                rect_cpy = image.copy()
                x, y, w, h = bounding_boxes[dragging_box]
                x += sel_rect_endpoint[0][0] - refPt[0][0]
                y += sel_rect_endpoint[0][1] - refPt[0][1]
                cv2.rectangle(rect_cpy, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.imshow("manual_edit", rect_cpy)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            image = clone.copy()
            refPt = []
            bounding_boxes = original_boxes.copy()
            for box in bounding_boxes:
                x, y, w, h = box
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 1)
        elif key == 13: # enter key to finish
            break
    return bounding_boxes

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

    classes_text = [b'fingertip']
    classes = [1]

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
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, required=False, default=2, help="starting image index")
    ap.add_argument("-s", "--split", type=float, required=False, default=2/3, help="fraction of data for training")
    args = ap.parse_args()
    assert args.num >= 2
    assert 0 < args.split < 1

    dataset_path = "11k_hands/"
    filenames = []
    start = False
    with open(dataset_path + "HandInfo.csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) # skip header
        for row in reader:
            file = row[7]
            index = int(file[len("Hand_"): -len(".jpg")])
            if index >= args.num:
                start = True
            if start:
                aspect = row[6]
                if "dorsal" in aspect:
                    filenames.append(dataset_path + "Hands/" + file)

    example_list = []
    print("q to quit, e to edit, any other key for next image")
    print("editing mode: drag boxes with mouse. r to reset, enter to submit changes")
    for file in filenames:
        raw = cv2.imread(file)
        raw = utils.resize(raw, width=400)
        image, bounding_boxes = find_fingertips(raw)

        cv2.imshow(file, image)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord('q'):
            break
        elif key == ord('e'):
            bounding_boxes = manual_edit(raw, bounding_boxes)

        tf_example = create_tf_example(raw, file, bounding_boxes)
        example_list.append(tf_example)

    random.seed(2018)
    random.shuffle(example_list)
    train_list, val_list = np.split(example_list, [int(len(example_list) * args.split)])
    print("writing TFRecord files...")
    write_sharded_tfrecord(train_list, output_filebase=dataset_path+"fingertips_train.record")
    write_sharded_tfrecord(val_list, output_filebase=dataset_path+"fingertips_val.record")
    print("done.")

