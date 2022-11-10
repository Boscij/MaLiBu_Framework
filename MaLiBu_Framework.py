import os
import glob

from PIL.Image import FASTOCTREE
from sklearn.model_selection import train_test_split
import shutil
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping

from my_utils import order_data, split_data, create_generators

from deeplearning_models import malibu_model
import tensorflow as tf


if __name__=="__main__":

    path_to_data = 'C:\\Users\\Domaschk\\Documents\\01_Malibu\\00_MaLiBu_Framework\\Daten\\Input\\20221110_cleaned'
    path_to_sorted_data = 'C:\\Users\\Domaschk\\Documents\\01_Malibu\\00_MaLiBu_Framework\\Daten\\Preprocessed\\0_All'
    path_to_train = 'C:\\Users\\Domaschk\\Documents\\01_Malibu\\00_MaLiBu_Framework\\Daten\\Preprocessed\\1_Train'
    path_to_val = 'C:\\Users\\Domaschk\\Documents\\01_Malibu\\00_MaLiBu_Framework\\Daten\\Preprocessed\\2_Val'
    path_to_test = 'C:\\Users\\Domaschk\\Documents\\01_Malibu\\00_MaLiBu_Framework\\Daten\\Preprocessed\\3_Test'
    batch_size = 64
    epochs = 15
    lr = 0.0001

    SORT  = False
    TRAIN = True
    TEST  = True

    if SORT:
        order_data(path_to_data, path_to_sorted_data)
        split_data(path_to_sorted_data, path_to_train, path_to_val, path_to_test)
    
    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    if TRAIN:
        path_to_save_model = './Models'
        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(monitor="val_accuracy", patience=10)

        model = malibu_model(nbr_classes)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
        
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_generator,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_generator,
                callbacks=[ckpt_saver, early_stop]
                )

    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        print("Evaluating validation set:")
        model.evaluate(val_generator)
        
        print("Evaluating test set : ")
        model.evaluate(test_generator)