# semi-automatically label candidates on 11k hands
# dorsal view only
# convex hull to find candidates
# missing candidates => save for the end, prompt user to label

import csv
import numpy as np
import cv2

import utils


def find_fingertips(image, threshold=5):
    # find convex hull of contours after thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 255-threshold, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    img, cnts, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    cv2.drawContours(image, contours, 0, (0, 0, 255), 3)

    contours = np.vstack(contours)
    hull = cv2.convexHull(contours)
    cv2.drawContours(image, [hull], 0, (0, 255, 0), 3)

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
    tolerance = 5 # degrees for points to be considered same finger
    longest_length = 0
    middle_finger = -1

    for h in hull:
        point = h[0]
        d = dist(origin, point)
        t = angle(point)

        if current_id == -1:
            check_id = -1
        else:
            if abs(t - angle(candidates[current_id])) < tolerance:
                check_id = current_id
            elif abs(t - angle(candidates[0])) < tolerance:
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
        cv2.circle(image, tip, 10, (255, 0, 0), -1)
    
    fingertips = []
    for n in range(-2, 3):
        finger_id = (middle_finger + n) % (current_id + 1)
        fingertips.append(candidates[finger_id])

    radius = 100 # radius within fingertip to boxed

    cnt_indices = [[] for x in fingertips]
    for i in range(len(contours)):
        point = contours[i][0]
        for j in range(len(fingertips)):
            tip = fingertips[j]
            if dist(tip, point) < radius:
                cnt_indices[j].append(i)

    for indices in cnt_indices:
        cnt = contours[indices]
        x, y, w, h = cv2.boundingRect(cnt)
        l = max(w, h)
        cv2.rectangle(image, (x, y), (x+l, y+l), (0, 0, 0), 5)

    return image, fingertips




def manual_edit(image, bounding_boxes):

    return bounding_boxes


if __name__ == "__main__":
    filenames = []
    with open('11k_hands/HandInfo.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) # skip header
        for row in reader:
            aspect = row[6]
            if "dorsal" in aspect:
                filenames.append("11k_hands/Hands/"+row[7])

    for file in filenames:
        print(file)
        raw = cv2.imread(file)
        image, bounding_boxes = find_fingertips(raw)

        cv2.imshow(file, utils.resize(image, width=400))
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord('q'):
            break
        elif key == ord('n'):
            bounding_boxes = manual_edit(image, bounding_boxes)

        # store fingers data

