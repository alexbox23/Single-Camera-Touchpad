import argparse
import csv

import cv2
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
        help="path to input video file")
    ap.add_argument("-o", "--output", required=True,
        help="path to output video file")
    ap.add_argument("-f", "--fps", type=int, default=4,
        help="fps for saving video file")
    ap.add_argument("-c", "--codec", type=str, default="MJPG",
        help="codec of output video")
    ap.add_argument("-d", "--data", type=str, required=True,
        help="path to output csv file")
    args = ap.parse_args()

    vs = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = None

    cam_pts = []
    top_pts = [[200, 225], [300, 75], [100, 75], [100, 225], [300, 225]]
    curr_line = []

    with open(args.data, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) # skip header
        frame = vs.read()[1] # skip frame before detection

        for row in reader:
            x, y, touch_count = [int(r) for r in row]
            frame = vs.read()[1]
            height, width, _ = np.shape(frame)

            if writer is None:
                writer = cv2.VideoWriter(args.output, fourcc, args.fps, 
                    (2 * width, height), True)

                cam_view = np.zeros([height, width, 3], dtype=np.uint8)
                cam_view[:,:] = [255, 255, 255]
                top_view = np.copy(cam_view)

            print(touch_count)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('t'):
                cv2.circle(cam_view, (x, y), 2, (0, 255, 0), -1)
                curr_line.append((x, y))
                cv2.polylines(cam_view, np.array([curr_line]), False, (0, 255, 0), 1)
                if len(cam_pts) == len(top_pts):
                    top_view = cv2.warpPerspective(cam_view, h, (width, height))

            elif key == ord('c'):
                cv2.circle(cam_view, (x, y), 2, (255, 0, 0), -1)
                cv2.circle(top_view, tuple(top_pts[len(cam_pts)]), 2, (0, 0, 0), -1)
                cam_pts.append([x, y])
                if len(cam_pts) == len(top_pts):
                    h, status = cv2.findHomography(np.array(cam_pts), np.array(top_pts))
                    top_view = cv2.warpPerspective(cam_view, h, (width, height))

            elif key == ord('q'):
                break

            if key != ord('t'):
                curr_line = []

            vis = np.hstack((frame, top_view))
            cv2.imshow("homography", vis)
            writer.write(vis)

    vs.release()
    writer.release()
    cv2.destroyAllWindows()