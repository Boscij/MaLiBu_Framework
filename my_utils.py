import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import glob
import shutil
import json
import ntpath
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def order_data(path_to_data, path_to_folder):

    folders = os.listdir(path_to_data)
    images_paths = []

    if os.path.isdir(path_to_folder):
        shutil.rmtree(path_to_folder)
    os.makedirs(path_to_folder)
    os.makedirs(path_to_folder  +  '\\1')
    os.makedirs(path_to_folder  +  '\\0')

    for folder in folders:

        path_to_samples = os.path.join(path_to_data,folder)
        with open(path_to_samples + '\\err_layer.txt') as f:
            data = f.read()
        js = json.loads(data)  
        samples = os.listdir(path_to_samples)
        samples.remove('err_layer.txt')

        for sample in samples:
            err_layer_min = js[sample + '_min']
            err_layer_max = js[sample + '_max']
            path_sample = os.path.join(path_to_data,folder,sample)
            images_path_sample = glob.glob(os.path.join(path_sample, '*.png'))

            for image in images_path_sample:
                img_name = ntpath.basename(image)
                img_layer = int(img_name[-9:-4])
                new_name = folder +'_'+ sample +'_'+ img_name
                if img_layer >= err_layer_max:
                    shutil.copy(image, path_to_folder + '\\1\\' + new_name)
                elif img_layer <= err_layer_min:
                    shutil.copy(image, path_to_folder + '\\0\\' + new_name)

def split_data(path_to_data, path_to_save_train, path_to_save_val, path_to_save_test, test_split_size=0.2, train_split_size=0.6):

    if train_split_size + test_split_size > 1:
        print('test_split_size and train split size not compatible, using default values (test_split_size = 0.2, train_split_size = 0.6)')
        test_split_size=0.2
        train_split_size=0.6
    
    folders = os.listdir(path_to_data)
    val_split_size = (1-train_split_size-test_split_size)/(1-test_split_size)

    if os.path.isdir(path_to_save_test):
        shutil.rmtree(path_to_save_test)
    os.makedirs(path_to_save_test)
    if os.path.isdir(path_to_save_train):
        shutil.rmtree(path_to_save_train)
    os.makedirs(path_to_save_train)
    if os.path.isdir(path_to_save_val):
        shutil.rmtree(path_to_save_val)
    os.makedirs(path_to_save_val)

    for folder in folders:

        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png'))
        
        X_train, X_test = train_test_split(images_paths, test_size=test_split_size, random_state=1)
        X_train, X_val  = train_test_split(X_train, test_size=val_split_size, random_state=1)

        for x in X_train:
            
            path_to_folder = os.path.join(path_to_save_train, folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

        for x in X_val:

            path_to_folder = os.path.join(path_to_save_val, folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)    

        for x in X_test:
            path_to_folder = os.path.join(path_to_save_test, folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)   

def create_generators(batch_size, train_data_path, val_data_path, test_data_path, target_size = (150,150)):

    train_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
        rotation_range=10,
        width_shift_range=0.1
    )

    test_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
    )

    val_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
    )

    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=target_size,
        color_mode='grayscale',
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = val_preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=target_size,
        color_mode="grayscale",
        shuffle=False,
        batch_size=batch_size
    )

    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=target_size,
        color_mode="grayscale",
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, test_generator


if __name__ == '__main__':
    path_to_data = 'C:\\Users\\Domaschk\\Documents\\01_Malibu\\00_MaLiBu_Framework\\Daten\\Input\\20221104'
    path_to_sorted_data = 'C:\\Users\\Domaschk\\Documents\\01_Malibu\\00_MaLiBu_Framework\\Daten\\Input\\Preprocessed\\0_All'
    path_to_train = 'C:\\Users\\Domaschk\\Documents\\01_Malibu\\00_MaLiBu_Framework\\Daten\\Preprocessed\\1_Train'
    path_to_val = 'C:\\Users\\Domaschk\\Documents\\01_Malibu\\00_MaLiBu_Framework\\Daten\\Preprocessed\\2_Val'
    path_to_test = 'C:\\Users\\Domaschk\\Documents\\01_Malibu\\00_MaLiBu_Framework\\Daten\\Preprocessed\\3_Test'
    order_data(path_to_data, path_to_sorted_data)
    split_data(path_to_sorted_data, path_to_train, path_to_val, path_to_test)
    