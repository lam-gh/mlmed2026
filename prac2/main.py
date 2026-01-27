import pandas as pd
import numpy as np
import cv2
from tensorflow import keras
import os
import segmentation_models as sm

def main():
    # print(os.listdir("../data/hc18"))
    # ['archive_hc18.zip', 'training_set', 'training_set_pixel_size_and_HC.csv', 'test_set', 'test_set_pixel_size.csv']

    root_data_dir = "../data/hc18/"
    train_dir = root_data_dir + "train_set"
    test_dir = root_data_dir + "test_set"
    train_csv = root_data_dir + "training_set_pixel_size_and_HC.csv"
    test_csv = root_data_dir + "test_set_pixel_size.csv"

    # Images: filename.png
    # Annotated images: filename_Annotation.png
    # shape: 800, 540

    # print(train_df)
    # print(test_df)

    # print(x_train)
    # print(y_train)
    
    # for filename, pixel_size in x_train:

    model = sm.Unet(
        'resnet34',
        input_shape=(800, 540, 1),
        classes=1,
        activation="sigmoid"
    )

    model.compile(
        optimizer="adam",
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score]
    )

    # Load images
    keras.utils.image_dataset_from_directory(
        directory="../data/hc18/",

    
    )

    #####
    #
    # NOT DONE YETTTTTTTT
    #
    #####
         
if __name__ == "__main__":
    main()
