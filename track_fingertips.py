import argparse
import time
import csv

import numpy as np
import cv2

import imutils
from imutils.video import FPS
# comment out line below when not running on raspberry pi
#from imutils.video.pivideostream import PiVideoStream 

def adjust_gamma(image, gamma=0.8):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def detect_fingertip(frame, margin=5):
    print("[INFO] starting inference graph...")
    frozen = "tensorflow/touch_inference_graph/frozen_inference_graph.pb"
    text_graph = "tensorflow/graph.pbtxt"
    cvNet = cv2.dnn.readNetFromTensorflow(frozen, text_graph)

    img = imutils.resize(frame, width=400)
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(400, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    rows = img.shape[0]
    cols = img.shape[1]
    box = None
    best_score = 0
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.1 and score > best_score:
            best_score = score
            left = detection[3] * cols - margin//2
            top = detection[4] * rows
            right = detection[5] * cols + margin//2
            bottom = detection[6] * rows + margin
            box = (left, top, right - left, bottom - top)

    print("[INFO] detection score: " + str(best_score))
    return box

def calibrate_touch_color(roi):
    print("[INFO] entering calibration mode...")
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
    ap.add_argument("--display", dest="display", action='store_true')
    ap.set_defaults(display=False)
    ap.add_argument("-v", "--video", type=str,
        help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="csrt",
        help="OpenCV object tracker type")
    ap.add_argument("-o", "--output", required=True,
        help="path to output video file")
    ap.add_argument("-f", "--fps", type=int, default=4,
        help="fps for saving video file")
    ap.add_argument("-c", "--codec", type=str, default="MJPG",
        help="codec of output video")
    ap.add_argument("-d", "--data", type=str, required=True,
        help="path to output csv file")
    args = ap.parse_args()

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    if not args.video:
        print("[INFO] starting rpi video stream...")
        vs = PiVideoStream().start()
        time.sleep(2.0)
    else:
        vs = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = None

    csvfile = open(args.data, 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(["x", "y", "touch_count"])

    detected = False
    calibrated = False
    touch_count = 0
    headless_init = False
    edit_flag = False
    while True:
        if not edit_flag:
            frame = vs.read()
            frame = frame[1] if args.video else frame
        else:
            edit_flag = False
        if frame is None:
            break
     
        frame = imutils.resize(frame, width=400)
        frame = adjust_gamma(frame)
        height, width, _ = np.shape(frame)

        if writer is None:
            writer = cv2.VideoWriter(args.output, fourcc, args.fps, 
                (width, height), True)

        vis = np.copy(frame)
        if detected:
            (success, box) = tracker.update(frame)
            (x, y, w, h) = [int(v) for v in box]
            if x >= 0 and y >= 0:
                roi = np.copy(frame[y:y+h, x:x+h])
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if calibrated:
                signal = cv2.inRange(roi, tuple(lows), tuple(highs))
                touch_count = np.count_nonzero(signal)

        if args.display:
            cv2.imshow("Frame", vis)
            key = cv2.waitKey(0) & 0xFF
         
            if key == ord("d"):
                box = detect_fingertip(frame)
                if box:
                    tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
                    tracker.init(frame, box)
                    detected = True
                else:
                    print("[INFO] no fingertip detected")

            elif key == ord("c") and detected:
                lows, highs = calibrate_touch_color(roi)
                calibrated = True
                lows = (35, 51, 80)
                highs = (67, 81, 129)
                fps = FPS().start()

            elif key == ord("s"):
                box = cv2.selectROI("Frame", frame, fromCenter=False,
                    showCrosshair=True)
                tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
                tracker.init(frame, box)
                detected = True
                edit_flag = True

            elif key == ord("q"):
                break
        else:
            if not headless_init:
                headless_init = True
                headless_count = 0
                box = detect_fingertip(frame)
                if box:
                    detected = True
                    tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
                    tracker.init(frame, box)
                else:
                    print("[INFO] no fingertip detected")
                lows = [75, 70, 111]
                highs = [111, 124, 171]
                calibrated = True
                fps = FPS().start()
            headless_count += 1
            if headless_count > 100:
                break

        if calibrated:
            fps.update()

        if not edit_flag:
            writer.write(vis)
            if detected and calibrated:
                csvwriter.writerow([x+w//2, y+h, touch_count])
     
    if not args.video:
        vs.stop()
    else:
        vs.release()
    writer.release()

    csvfile.close()

    if calibrated:
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        
    cv2.destroyAllWindows()