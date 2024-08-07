from ultralytics import YOLO

# Load a model
model = YOLO("yolov8l-seg.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    # Train the model
    results = model.train(data="costum_macaque_pose_seg.yaml", epochs=10, plots=True)
    # all training parameter settings listed: https://docs.ultralytics.com/modes/train/#train-settings
    # training tips: https://docs.ultralytics.com/guides/model-training-tips/
    # guide to parameter tuning: https://docs.ultralytics.com/guides/hyperparameter-tuning/

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print("BBox mAP50-95: ", metrics.box.map)  # map50-95
    print("BBox mAP50: ", metrics.box.map50)  # map50
    print("BBox mAP95: ", metrics.box.map75)  # map75
    print("SMask mAP50-95: ", metrics.seg.map)  # map50-95(M)
    print("SMask mAP50: ", metrics.seg.map50)  # map50(M)
    print("SMask mAP95: ", metrics.seg.map75)  # map75(M)
