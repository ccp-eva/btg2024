import cv2
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from ultralytics import YOLO

from utils import *

# define paths to files
id_model = tf.keras.models.load_model('./identification_macaques_ft.keras')
seg_model = YOLO("./macpose_l_seg.pt")
pose_model = YOLO("./macpose_l_pose.pt")

vid_path = "./youtube_clip_trim.mp4"
out_csv_path = "./pipeline_pred_youtube_clip_trim.csv"
out_json_path = "./pipeline_pred_youtube_clip_trim.json"
# initialise reader for the video file
cap = cv2.VideoCapture(vid_path)

# define skeleton used for plotting pose
custom_skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12),
                   (11, 13), (12, 14), (13, 15), (14, 16)]

# initialise empty list and dataframe for predicitons
out_json_ls = []
out_csv = pd.DataFrame()

fnum = 0  # initiate frame number

# Loop through the video frame by frame
while cap.isOpened():

    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # predict using segmentation model
        res = seg_model.track(frame, tracker="./cstm_tracker_bots.yaml")

        # create list for ID-predictions
        id_preds = []
        # Iterate detection results
        for r in res:
            # Copy the image
            img = np.copy(r.orig_img)

            # Iterate each object contours
            for ci, c in enumerate(r):
                label = c.names[c.boxes.cls.tolist().pop()]

                b_mask = np.zeros(img.shape[:2], np.uint8)

                # Create contour mask
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)

                _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                # Isolate object with black background
                mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                isolated = cv2.bitwise_and(mask3ch, img)

                id_input = cv2.resize(isolated, (256, 256))[None, :, :, :]
                # use this image to predict ID
                id_preds.append(id_model.predict(id_input)[0])

                # use segmented image  to predict pose (could have multiple instances on the same frame (overlapping
                # monkeys) so we only use the one with the highest confidence)
                prs = pose_model.predict(isolated)
                if not all([p.keypoints.conf is None for p in prs]):
                    kypt_id = np.mean(
                        np.array([np.array(p.keypoints.conf.cpu()).squeeze() if p.keypoints.conf is not None
                                  else np.array(0) for p in prs]), axis=-1)
                    if len(kypt_id) == 0:
                        continue
                    kypts = prs[0].keypoints[np.argmax(kypt_id)]
                    # plot keypoints and skeleton on image
                    plot_kpoints(img, kypts)
                    plot_skeleton(img, custom_skeleton, kypts)
                    # plot_sit_stand(img, [(12, 14, 16), (11, 13, 15)], kypts, c.boxes)

                # plot the segmentation mask on image
                plot_segment(img, contour)

            id_preds = np.array(id_preds)

            ids, ids_conf = max_conf_id(id_preds)
            boxes = r.boxes.xyxy.cpu()
            for bi in ids.keys():
                plot_id(img, ids[bi], boxes[bi])

            # checks for none types in masks
            if r.masks is None:
                masks_list = np.repeat(None, len(r.boxes.cls)).tolist()
            else:
                masks_list = [m.tolist() for m in r.masks.xy]

            # stores results in list
            out_json_ls.append({'frame_number': fnum,
                                'track_id': none_check(r.boxes.id),
                                'conf': r.boxes.conf.tolist(),
                                'identity': list(ids.values()),
                                'id_conf': ids_conf,
                                'bbox': r.boxes.xyxy.tolist(),
                                'mask': masks_list})

            # checks for None types for iteration to fill dataframe
            ids_list = list(ids.values())
            if len(r.boxes.cls) != len(ids_list):
                while len(r.boxes.cls) > len(ids_list):
                    ids_list.append(None)
            if len(r.boxes.cls) != len(ids_conf):
                while len(r.boxes.cls) > len(ids_conf):
                    ids_conf.append(None)
            if none_check(r.boxes.id) is None:
                track_ids_list = np.repeat(None, len(r.boxes.cls)).tolist()
            else:
                track_ids_list = none_check(r.boxes.id)

            # stores results in Dataframe
            for i in range(len(r.boxes.cls)):
                out_csv = pd.concat([out_csv,
                                     pd.DataFrame({'frame_number': [fnum],
                                                   'track_id': track_ids_list[i],
                                                   'conf': r.boxes.conf[i].item(),
                                                   'identity': ids_list[i],
                                                   'id_conf': ids_conf[i],
                                                   'x_min': int(r.boxes.xyxy[i][0]),
                                                   'y_min': int(r.boxes.xyxy[i][1]),
                                                   'x_max': int(r.boxes.xyxy[i][2]),
                                                   'y_max': int(r.boxes.xyxy[i][3]),
                                                   }
                                                  )], ignore_index=True)



        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", img)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        fnum += 1  # update frame number
    else:
        # Break the loop if the end of the video is reached
        break

# writing output in JSON format
with open(out_json_path, "w") as final:
    json.dump(out_json_ls, final)

# writing output in csv format
out_csv.to_csv(out_csv_path, index=False)

cap.release()
cv2.destroyAllWindows()
