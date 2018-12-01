# semi-automatically label candidates on 11k hands
# dorsal view only
# convex hull to find candidates
# missing candidates => save for the end, prompt user to label

import csv
import numpy as np
import cv2
import argparse
import utils


def find_fingertips(image, threshold=5, finger_angle=5, tip_radius=25):
    # finds convex hull of contours after thresholding
    # fingertips are the five vertices furthest from the center of the top row
    #   threshold: grayscale values in (255 - threshold, 255) considered white background
    #   finger_angle: number of degrees for vertices to be considered the same finger
    #       with respect to center of the top row
    #   tip_radius: distance from fingertip to be placed in bounding box

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, required=False, default=2, help="starting image index")
    args = ap.parse_args()
    assert args.num >= 2
    start = False

    filenames = []
    with open('11k_hands/HandInfo.csv', newline='') as csvfile:
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
                    filenames.append("11k_hands/Hands/" + file)

    print("q to quit, e to edit, any other button for next image")
    print("editing mode: drag boxes with mouse. r to reset, enter to submit changes")
    for file in filenames:
        raw = cv2.imread(file)
        raw = utils.resize(raw, width=400)
        image, bounding_boxes = find_fingertips(raw.copy())

        cv2.imshow(file, image)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord('q'):
            break
        elif key == ord('e'):
            bounding_boxes = manual_edit(raw.copy(), bounding_boxes)

        # store fingers data

