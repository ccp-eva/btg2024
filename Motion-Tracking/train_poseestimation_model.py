from ultralytics import YOLO

# Load a model
model = YOLO("yolov8l-pose.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    # Train the model
    results = model.train(data="costum_macaque_pose_kpt.yaml", epochs=100, patience=10, batch=0.8, plots=True)
    # all training parameter settings listed: https://docs.ultralytics.com/modes/train/#train-settings
    # training tips: https://docs.ultralytics.com/guides/model-training-tips/
    # guide to parameter tuning: https://docs.ultralytics.com/guides/hyperparameter-tuning/

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print("mAP50-95: ", metrics.box.map)  # map50-95
    print("mAP50: ", metrics.box.map50)  # map50
    print("mAP95: ", metrics.box.map75)  # map75
