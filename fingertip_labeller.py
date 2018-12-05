""" Single-Camera-Touchpad/fingertip_labeller.py

    Semi-automatical labeller for fingertips on the 11k hands dataset.
    Only images of the dorsal aspect are used.
    Convex hull vertices are used to guess fingertip locations.
    The user has the option to manually edit bounding box locations.
    Labels are exported in csv format.
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

        if h < tip_radius:
            y -= (tip_radius - h)
            h = tip_radius
        elif h > tip_radius * 3//2:
            h = tip_radius * 3//2
        w = max(tip_radius, min(tip_radius * 3//2, w))
        x = max(0, min(width - w, x))
        y = max(0, min(height - h, y))

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
        nonlocal refPt, moving, resizing, new_pos, new_size, sel_rect_endpoint

        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            moving = True
            sel_rect_endpoint = []
     
        elif event == cv2.EVENT_LBUTTONUP:
            refPt.append((x, y))
            moving = False
            new_pos = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            refPt = [(x, y)]
            resizing = True
            sel_rect_endpoint = []

        elif event == cv2.EVENT_RBUTTONUP:
            refPt.append((x, y))
            resizing = False
            new_size = True

        elif event == cv2.EVENT_MOUSEMOVE and (resizing or moving):
            sel_rect_endpoint = [(x, y)]

    clone = image.copy()
    original_boxes = bounding_boxes.copy()
    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 1)

    refPt = []
    moving = False
    resizing = False
    new_pos = False
    new_size = False
    sel_rect_endpoint = []
    dragging_box = -1

    cv2.namedWindow("manual_edit")
    cv2.setMouseCallback("manual_edit", click_and_crop)

    while True:
        if not moving and not resizing:
            if len(refPt) == 2 and dragging_box != -1:
                x, y, w, h = bounding_boxes[dragging_box]
                dx = refPt[1][0] - refPt[0][0]
                dy = refPt[1][1] - refPt[0][1]
                if new_pos:
                    x += dx
                    y += dy
                    new_pos = False
                elif new_size:
                    w += dx
                    h += dy
                    new_size = False
                bounding_boxes[dragging_box] = [x, y, w, h]
                dragging_box = -1

                image = clone.copy()
                for box in bounding_boxes:
                    x, y, w, h = box
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 1)
            cv2.imshow("manual_edit", image)
        else:
            for i in range(len(bounding_boxes)):
                x, y, w, h = bounding_boxes[i]
                px, py = refPt[0]
                if x <= px <= x+w and y <= py <= y+h:
                    dragging_box = i
            if dragging_box != -1 and sel_rect_endpoint:
                rect_cpy = image.copy()
                x, y, w, h = bounding_boxes[dragging_box]
                dx = sel_rect_endpoint[0][0] - refPt[0][0]
                dy = sel_rect_endpoint[0][1] - refPt[0][1]
                if moving:
                    x += dx
                    y += dy
                elif resizing:
                    w += dx
                    h += dy
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
    dataset_path = "11k_hands/"
    output_path = dataset_path + "fingertip_labels.csv"

    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="resume from last entry of "+output_path)
    args = ap.parse_args()

    start = not args.resume
    last_labelled = None
    if args.resume:
        try:
            with open(output_path, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    last_labelled = row[0]
                if not last_labelled:
                    start = True
        except FileNotFoundError:
            start = True

    mode = 'w' if start else 'a'
    filenames = []
    with open(dataset_path + "HandInfo.csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) # skip header
        for row in reader:
            file = dataset_path + "Hands/" + row[7]
            #index = int(file[len("Hand_"): -len(".jpg")])
            if start:
                aspect = row[6]
                if "dorsal" in aspect:
                    filenames.append(file)
            elif file == last_labelled:
                start = True

    with open(output_path, mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        print("q to quit, e to edit, any other key for next image")
        print("editing mode: left click to move, right click to resize, r to reset, enter to submit changes")
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

            row = [file]
            for box in bounding_boxes:
                row.extend(box)
            writer.writerow(row)

    print("saved labels in " + output_path)


