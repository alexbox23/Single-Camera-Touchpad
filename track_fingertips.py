import argparse
import time

import numpy as np
import cv2
import tensorflow as tf

import imutils
from imutils.video import FPS
# comment out line below when not running on raspberry pi
#from imutils.video.pivideostream import PiVideoStream 

def run_inference_for_single_image(image, graph):
    """ from tensorflow repo: object_detection/object_detection_tutorial.ipynb

    Args:
        image: The raw image opened via opencv.
        graph: 

    Returns:
        output_dict: detection data
    """
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def detect_fingertip(frame, margin=5):
    PATH_TO_FROZEN_GRAPH = "tensorflow/touch_inference_graph/frozen_inference_graph.pb"
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    print("[INFO] detection score: " + str(output_dict['detection_scores'][0]))

    ymin, xmin, ymax, xmax = output_dict['detection_boxes'][0]
    xmin = int(xmin * width) - margin//2
    ymin = int(ymin * height)
    xmax = int(xmax * width) + margin//2
    ymax = int(ymax * height) + margin
    box = (xmin, ymin, xmax - xmin, ymax - ymin)

    return box

def calibrate_touch_color(roi):
    cv2.namedWindow("Sliders")
    placeholder = np.zeros([1,500])
    channels = ["B", "G", "R"]
    rgb_bounds = {}
    for ch in channels:
        rgb_bounds[ch + "_low"] = 0
        rgb_bounds[ch + "_high"] = 255
    for label in rgb_bounds.keys():
        cv2.createTrackbar(label, "Sliders", rgb_bounds[label], 255, lambda x: None)
    cv2.imshow("Sliders", placeholder)

    while True:
        for label in rgb_bounds.keys():
            rgb_bounds[label] = cv2.getTrackbarPos(label, "Sliders")
        lows = tuple([rgb_bounds[ch + "_low"] for ch in channels])
        highs = tuple([rgb_bounds[ch + "_high"] for ch in channels])
        signal = cv2.inRange(roi, lows, highs)

        height, width, _ = np.shape(roi)
        window = np.zeros([height, width*4, 3], dtype=np.uint8)
        for i in range(4):
            if i == 0:
                section = cv2.bitwise_and(roi, roi, mask=signal)
            else:
                section = np.copy(roi[:,:,3-i])
                ch = channels[3-i]
                threshold_indices = section < rgb_bounds[ch + "_low"]
                threshold_indices |= section > rgb_bounds[ch + "_high"]
                section[threshold_indices] = 0
                section = cv2.cvtColor(section, cv2.COLOR_GRAY2BGR)
                for j in range(3):
                    if j != 3-i:
                        section[:,:,j] = 0

            window[:,i*width:(i+1)*width] = section
        window = imutils.resize(window, height=height*2)

        cv2.imshow("ROI", window)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break

    cv2.destroyWindow("ROI")
    cv2.destroyWindow("Sliders")
    return lows, highs

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
        help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="csrt",
        help="OpenCV object tracker type")
    args = ap.parse_args()

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    trackers = cv2.MultiTracker_create()

    if not args.video:
        print("[INFO] starting rpi video stream...")
        vs = PiVideoStream().start()
        time.sleep(2.0)
    else:
        vs = cv2.VideoCapture(args.video)

    detected = False
    calibrated = False
    while True:
        frame = vs.read()
        frame = frame[1] if args.video else frame
        if frame is None:
            break
     
        frame = imutils.resize(frame, width=400)
        height, width, _ = np.shape(frame)
        (success, boxes) = trackers.update(frame)
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            roi = np.copy(frame[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if calibrated:
                signal = cv2.inRange(roi, tuple(lows), tuple(highs))
                touch_count = np.count_nonzero(signal)
                print(touch_count)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
     
        if key == ord("d"):
            box = detect_fingertip(frame)
            detected = True

            tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
            trackers.add(tracker, frame, box)

        elif key == ord("c") and detected:
            lows, highs = calibrate_touch_color(roi)
            calibrated = True
            fps = FPS().start()

        elif key == ord("q"):
            break

        if calibrated:
            fps.update()
     
    if not args.video:
        vs.stop()
    else:
        vs.release()

    if calibrated:
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        
    cv2.destroyAllWindows()