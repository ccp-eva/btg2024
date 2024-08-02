import numpy as np
import cv2


def calc_angle(p1, p2, p3):
    """" Function predicting angle between 3 points (point 2 is centre) """
    v12 = p1[:2] - p2[:2]
    v32 = p3[:2] - p2[:2]
    angl = np.arccos(np.dot(v12, v32) / (np.linalg.norm(v12) * np.linalg.norm(v32)))
    return np.degrees(angl)


def plot_segment(img, mask):
    """" Function adding segmentation as polylines on image """
    img = cv2.polylines(img, [mask], True, (125, 255, 125), 2)
    return img


def plot_kpoints(img, kpoints, thresh=0.7):
    """" Function adding keypoint detections above threshold to an image """
    for i in range(kpoints.shape[1]):
        if np.array(kpoints.conf.cpu()).squeeze()[i] >= thresh:
            x = np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[i][0]
            y = np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[i][1]
            img = cv2.circle(img, (x, y), radius=4, color=(0, i * (255 / 17), 255), thickness=-1)

            # img = cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    return img


def plot_skeleton(img, skeleton, kpoints, thresh=0.7):
    """" Function adding skeleton lines from keypoint detections above threshold to an image """
    for p1, p2 in skeleton:
        if (np.array(kpoints.conf.cpu()).squeeze()[p1] >= thresh and
                np.array(kpoints.conf.cpu()).squeeze()[p2] >= thresh):
            p1xy = (np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[p1][0],
                    np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[p1][1])
            p2xy = (np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[p2][0],
                    np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[p2][1])

            img = cv2.line(img, p1xy, p2xy, color=(255, 0, 255), thickness=2)
    return img


def plot_angle(img, points, kpoints, thresh=0.7):
    """" Function adding angle between keypoint detections above threshold to an image (points is list of three
    points where p2 is the centre """
    for p1, p2, p3 in points:
        if (np.array(kpoints.conf.cpu()).squeeze()[p1] >= thresh and
                np.array(kpoints.conf.cpu()).squeeze()[p2] >= thresh and
                np.array(kpoints.conf.cpu()).squeeze()[p3] >= thresh):
            ngl = calc_angle(np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[p1],
                             np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[p2],
                             np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[p3])

            img = cv2.putText(img, str(ngl), (np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[p2][0],
                                              np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[p2][1]),
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    return img


def plot_sit_stand(img, points, kpoints, box, angle_thresh=90, thresh=0.7):
    """" Function adding label for sit/stand dependent on keypoint and angle threshold to an image """
    for p1, p2, p3 in points:
        if (np.array(kpoints.conf.cpu()).squeeze()[p1] >= thresh and
                np.array(kpoints.conf.cpu()).squeeze()[p2] >= thresh and
                np.array(kpoints.conf.cpu()).squeeze()[p3] >= thresh):

            ngl = calc_angle(np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[p1],
                             np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[p2],
                             np.array(kpoints.xy.cpu()).astype(np.int32).squeeze()[p3])

            # get bbox position and add text
            bx = min(np.array(box.xyxy.cpu()).astype(np.int32).squeeze()[0] + 50, img.shape[1])
            by = np.array(box.xyxy.cpu()).astype(np.int32).squeeze()[1]
            if ngl > angle_thresh:
                img = cv2.putText(img, "Standing", (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                  (255, 255, 0), 2, cv2.LINE_AA)
            else:
                img = cv2.putText(img, "Sitting", (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                  (0, 255, 255), 2, cv2.LINE_AA)

    return img


def max_conf_id(id_preds):
    """" Function assigning identities according to the highest confidence """
    ids = {}
    ids_conf = []
    while id_preds.sum() > 0:
        row, col = np.unravel_index(np.argmax(id_preds), id_preds.shape)
        ids[row] = col.item()
        ids_conf.append(id_preds[row, col].item())
        id_preds[:, col] = 0

    return ids, ids_conf


def plot_id(img, id, box):
    """" Function plotting the assigned identity to  an image"""
    bx0 = np.array(box).astype(np.int32)[0]
    by0 = np.array(box).astype(np.int32)[1]
    bx1 = np.array(box).astype(np.int32)[2]
    by1 = np.array(box).astype(np.int32)[3]
    col = int(255 / (id + 1))
    img = cv2.rectangle(img, (bx0, by0), (bx1, by1), (col, 255, 125), 2)
    img = cv2.putText(img, str(id), (bx0, by0), cv2.FONT_HERSHEY_SIMPLEX, 2,
                      (col, 255, 125), 2, cv2.LINE_AA)

    return img


def none_check(obj):
    """" Function checking if an object(here arrays from yolo predictions) is None otherwise converting to list"""
    if obj is not None:
        return obj.tolist()
    else:
        return None
