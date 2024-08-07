# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:52:19 2024

@author: Arja Mentink
"""

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import numpy as np 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import heapq
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, f1_score
import seaborn as sns
from collections import defaultdict, Counter

    
    
def main():
    """Build and evaluate the identification model on the macaques."""

    #create the datasets
    train_data = tf.keras.utils.image_dataset_from_directory("./identification_images/train", batch_size=32, label_mode="categorical", shuffle = True, seed = 111)
    test_data = tf.keras.utils.image_dataset_from_directory("./identification_images/test", batch_size=32, label_mode="categorical")
    val_data = tf.keras.utils.image_dataset_from_directory("./identification_images/val", batch_size=32, label_mode="categorical")
    #Augment the training data
    train_aug_data = random_data_augmentation(train_data)

    #load the pre-trained ChimpACT model
    #trained on ChimpACT dataset https://shirleymaxx.github.io/ChimpACT/
    #see the publication by Ma et al. 2023: https://proceedings.neurips.cc/paper_files/paper/2023/file/57a95cd3898bf4912269848a01f53620-Paper-Datasets_and_Benchmarks.pdf
    chimp_model = tf.keras.models.load_model('identification_chimps.keras')

    #Train the ChimpACT model on the macaque data
    model = build_ft_model(chimp_model, 5, 0.02201320099345077, 0.7492046347073001, 0.059141107541367824)

    #calculate the weights for each class, as they are not balanced
    class_weight = class_weights(train_data)

    #Save the best model during training
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=f'identification_macaques.keras', monitor='val_loss',
                                    save_best_only=True, verbose=1)

    #Stop after 20 epochs of no improvement
    early_stopper=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001,
                                                restore_best_weights=True, patience=20)

    callbacks_list=[early_stopper, checkpoint]

    #Train the model
    model.fit(train_aug_data, epochs=200, batch_size=32, validation_data=val_data, callbacks=callbacks_list,  class_weight=class_weight)

    #After the dense layer is trained, train the whole model with a small learning rate.
    best_model = load_pretrained('identification_macaques.keras')
    fine_tuned_model = fine_tune(class_weight, best_model, train_aug_data, val_data)

    #Write the results to a txt-file
    results = fine_tuned_model.history
    f = open(f'identification_macaques' + ".txt", "x")
    f.write(str(0.02201320099345077) + "\n" + str(0.7492046347073001) + "\n" + str(0.059141107541367824) + "\n" + str(results.history) + "\n")
    f.close()

    fine_tuned_model.evaluate(test_data)
    #Test the model with and without the threshold:
    test_model(fine_tuned_model, test_data, True) #threshold
    test_model(fine_tuned_model, test_data, False) #no threshold


def random_data_augmentation(train_ds):
    """Function that transforms the training data with several augmentation layers"""

    augmentation_layers = [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomBrightness([0.0, 0.1]),
        keras.layers.RandomCrop(256, 256)
    ]

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, augmentation_layers), y))

    return train_ds


def data_augmentation(x, augmentation_layers):
    for layer in augmentation_layers:
        x = layer(x)
    return x


def class_weights(data):
    labels = []
    for x, y in data.take(len(data)):
        # Not one hot encoded
        # labels.append(y.numpy())

        # If one hot encoded, then apply argmax
        labels.append(np.argmax(y, axis=-1))
    labels = np.concatenate(labels, axis=0)  # Assuming dataset was batched.

    classes = [x for x in Counter(labels).keys()]  # equals to list(set(words))
    counts = [x for x in Counter(labels).values()]  # counts the elements' frequency

    total = sum(counts)

    class_weight = defaultdict(lambda: float)
    for i, label in enumerate(classes):
        class_weight[label] = float(1.0 - (counts[i] / float(total)))

    return class_weight


def build_efficientnetV2(NUM_classes, lr, momentum, weight_decay):
    """Build an EfficientNetV2M that is pre-trained on ImageNet.
        A new dense layer is added with the correct number of classes. Only this layer is trainable."""

    # load the pretrained model
    base_model = keras.applications.EfficientNetV2M(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(256, 256, 3),
        include_top=False)

    # freeze the convolutional layers
    base_model.trainable = False

    # create new model with new dense layers
    inputs = keras.Input(shape=(256, 256, 3))

    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(NUM_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr,
                                                     beta_1=momentum,
                                                     decay=weight_decay,
                                                     ),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.AUC(), keras.metrics.CategoricalAccuracy()])

    return model


def build_ft_model(ft_model, NUM_classes, lr, momentum, weight_decay):
    """Builds an EfficientNetV2M that is pre-trained on ChimpACT.
        A new dense layer is added with the correct number of classes. Only this layer is trainable."""

    # Retrieve the ChimpACT model and remove the last dense layer.
    base_model = ft_model
    base_model = keras.models.Model(inputs=ft_model.input, outputs=ft_model.layers[-2].output)

    base_model.trainable = False  # set trainable to false

    # create new model with a new dense layer
    inputs = keras.Input(shape=(256, 256, 3))

    x = base_model(inputs, training=False)

    outputs = keras.layers.Dense(NUM_classes, activation="softmax")(
        x)  # add one more dense layer with now 6 classes for the macaque individuals

    model = keras.Model(inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr,
                                                     beta_1=momentum,
                                                     decay=weight_decay,
                                                     ),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.AUC(), keras.metrics.CategoricalAccuracy()])

    return model


def fine_tune(class_weight, model, train_data, val_data):
    """After training, this function ensures that the entire model is trained, not only the dense layer, with a very small learning rate."""

    checkpoint = keras.callbacks.ModelCheckpoint(filepath=f'identification_macaques_ft.keras', monitor='val_loss',
                                                 save_best_only=True, verbose=1)

    early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001,
                                                  restore_best_weights=True, patience=2)

    callbacks_list = [checkpoint, early_stopper]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5  # small learning rate
                                                     ),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.AUC(), keras.metrics.CategoricalAccuracy()])

    model.fit(train_data, epochs=5, batch_size=32, validation_data=val_data, callbacks=callbacks_list,
              class_weight=class_weight)

    return model


def load_pretrained(model_name):
    """Load the pre-trained model and ensure that the entire model is trainable for fine-tuning."""

    chimp_model = tf.keras.models.load_model(model_name)

    chimp_model.trainable = True
    chimp_model.summary()

    return chimp_model


def test_model(model, val_data, threshold):
    """This functions calculates several evaluation metrics for the identification model.
        It also creates a confusion matrix."""

    predictions = np.array([])
    labels = np.array([])

    # The model can be evaluated with or without a confidence threshold.
    if threshold:
        t = 0
        for x, y in val_data:
            prediction_list = model.predict(x)
            for i, pred in enumerate(prediction_list):
                highest_pred = heapq.nlargest(2, pred)
                if not abs(highest_pred[0] - highest_pred[
                    1]) < 0.15:  # with a threshold, uncertain images are removed from the testset.
                    predictions = np.concatenate([predictions, np.array([np.argmax(pred, axis=-1)])])
                    labels = np.concatenate([labels, np.array([np.argmax(y[i].numpy(), axis=-1)])])
                else:
                    print("image discarded")
            t += 1

    else:
        for x, y in val_data:
            predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=-1)])
            labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

    cm = confusion_matrix(y_true=labels, y_pred=predictions, labels=[0., 1., 2., 3., 4.])
    norm_cm = cm / cm.astype('float').sum(axis=1)[:, np.newaxis]

    correct_pred = sum(cm[i][i] for i in range(len(cm)))

    accuracy = correct_pred / sum(sum(cm))
    print("accuracy = " + str(accuracy))

    print('ROC AUC score:', multiclass_roc_auc_score(labels, predictions))
    print('macro ROC AUC score:', multiclass_roc_auc_score(labels, predictions, average='macro'))
    print('F1-score:', f1_score(labels, predictions, average=None))
    print('macro F1-score:', f1_score(labels, predictions, average='macro'))

    df_cm = pd.DataFrame(cm)

    plt.figure()
    plt.title('Confusion matrix macaques')
    x_labels = ["ID0", "ID1", "ID2", "ID3", "ID4", "ID5"]

    y_labels = x_labels
    sn.heatmap(norm_cm, xticklabels=x_labels, yticklabels=y_labels, cmap=sns.color_palette("Spectral", as_cmap=True))
    plt.xlabel('Predicted')
    plt.ylabel('True')


def multiclass_roc_auc_score(labels, predictions, average='weighted'):
    """Create a plot with all the AUC-ROC scores for each individual."""

    fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))

    target = ["ID0", "ID1", "ID2", "ID3", "ID4"]

    lb = LabelBinarizer()
    lb.fit(labels)
    labels = lb.transform(labels)
    predictions = lb.transform(predictions)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(labels[:, idx].astype(int), predictions[:, idx])
        c_ax.plot(fpr, tpr, label="%s (AUC:%0.2f)" % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label='Random Guessing')

    c_ax.legend()
    c_ax.set_title("AUC-scores for the ID-model")
    c_ax.set_xlabel("False Positive Rate")
    c_ax.set_ylabel('True Positive Rate')
    plt.show()

    return roc_auc_score(labels, predictions, average=average)

if __name__ == "__main__":
    main()

