import cv2
import pandas as pd
import json
import os
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("./macpose_l_seg.pt")
# Open the video file
video_path = "./youtube_clip_trim.mp4"
cap = cv2.VideoCapture(video_path)

# Tracker
trkr = "./cstm_tracker_bots.yaml"

# create paths to save files
out_json_path = "./pred_" + video_path.split("/")[-1].split(".")[0] + ".json"
out_csv_path = "./pred_" + video_path.split("/")[-1].split(".")[0] + ".csv"

out_json_ls = []
out_csv = pd.DataFrame()

fnum = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, tracker=trkr, persist=True)

        for result in results:
            if result.boxes.id is None:
                continue
            out_json_ls.append({'frame_number': fnum,
                                'track_id': result.boxes.id.tolist(),
                                'seg_class': result.boxes.cls.tolist(),
                                'conf': result.boxes.conf.tolist(),
                                'bbox': result.boxes.xyxy.tolist(),
                                'mask': [m.tolist() for m in result.masks.xy]})

            for i in range(len(result.boxes.cls)):
                out_csv = pd.concat([out_csv,
                                     pd.DataFrame({'frame_number': [fnum],
                                                   'track_id': result.boxes.id[i].item(),
                                                   'seg_class': result.boxes.cls[i].item(),
                                                   'conf': result.boxes.conf[i].item(),
                                                   'x_min': int(result.boxes.xyxy[i][0]),
                                                   'y_min': int(result.boxes.xyxy[i][1]),
                                                   'x_max': int(result.boxes.xyxy[i][2]),
                                                   'y_max': int(result.boxes.xyxy[i][3]),
                                                   }
                                                  )], ignore_index=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
    fnum += 1

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

with open(out_json_path, "w") as final:
    json.dump(out_json_ls, final)

out_csv.to_csv(out_csv_path, index=False)
