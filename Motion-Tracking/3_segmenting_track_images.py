import cv2
import os
import pandas as pd
import json
import numpy as np
import tqdm

# define paths to files
trcks = pd.read_csv("./pred_youtube_clip_trim.csv")  # loads predictions from segmentation
txs = json.load(open("./pred_youtube_clip_trim.json"))  # loads predictions from segmentation
vid_path = "./youtube_clip_trim.mp4"  # path to video file
min_len_frames = 90  # threshold to identify minimum length of track in number of frames
step_frame_num = 15  # step length between frames to save
outvid_path = "./pred_track_id_clip.avi"  # if None - no output video created

out_dir = "./out_tracks/"  # folder where images are saved
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# initialise reader for the video file
cap = cv2.VideoCapture(vid_path)

# loop through found tracks
for trk in tqdm.tqdm(trcks.track_id.unique()):
    frms = trcks.loc[trcks['track_id'] == trk].frame_number.tolist()

    # only take tracks of with a minimum length of frames
    if len(frms) >= min_len_frames:
        if not os.path.exists(out_dir + str(trk) + "/"):
            os.mkdir(out_dir + str(int(trk)) + "/")
        sl_frms = frms[::step_frame_num]
        for sf in sl_frms:
            tx = [tx for tx in txs if tx['frame_number'] == sf][0]
            msk = tx['mask'][tx['track_id'].index(trk)]

            cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
            ret, frame = cap.read()
            # Crop the bounding rect
            pts = np.array(msk).astype(np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            croped = frame[y:y + h, x:x + w].copy()

            # create the mask
            pts = pts - pts.min(axis=0)
            mask = np.zeros(croped.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

            # do bit-wise operation
            dst = cv2.bitwise_and(croped, croped, mask=mask)

            # write the image
            cv2.imwrite(out_dir + str(int(trk)) + "/" + "monkey_" + str(int(trk)) + "_" + str(sf) + ".png", dst)

# creates list of track_ids that where extracted
list_extr_trks = [int(i) for i in os.listdir("./out_tracks/")]

if outvid_path is not None:
    # initialise reader for the video file
    cap = cv2.VideoCapture(vid_path)

    print("Writing video:", outvid_path)
    outvid = cv2.VideoWriter(outvid_path,
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                             )
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fm in tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        tx = [tx for tx in txs if tx['frame_number'] == fm]
        if len(tx) == 1:
            tx = tx[0]
            idxs = [i for i in range(len(tx['track_id'])) if tx['track_id'][i] in list_extr_trks]

            for idx in idxs:
                x1, y1, x2, y2 = np.array(tx['bbox'][idx]).astype(np.int32)
                cv2.rectangle(frame, (x1, y1),
                              (x2, y2), (0, 255, 255), 2)

                cv2.putText(frame, str(tx['track_id'][idx]), (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        outvid.write(frame)

    outvid.release()

cap.release()
