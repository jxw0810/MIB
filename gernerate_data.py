from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import random
import shutil
import cv2
import os

def load_cla_data(train_imagePaths, target):

    data = []
    labels = []
    images_names = []
    for imagePath in train_imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, target)
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
        _, images_name = os.path.split(imagePath)
        print(images_name)
        images_names.append(images_name)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    return data, labels, images_names

def split_train_and_test(img_dir, mask_dir, test_dir, split_size=0.25):


    print("test ratio:", split_size)

    test_mask_dir = os.path.join(test_dir, 'masks')
    test_img_dir = os.path.join(test_dir, 'images')

    if not os.path.exists:
        os.makedirs(test_mask_dir)
        os.makedirs(test_img_dir)



    trainX_list = os.listdir(img_dir)
 
    random.shuffle(trainX_list)

    number = len(trainX_list)
    test_number = int(number * split_size)
    test_img_list = trainX_list[-test_number:]

    for image_name in test_img_list:


        shutil.move(os.path.join(img_dir,image_name), test_img_dir)

        mask_name = image_name.split('.')[0] + '_mask.png'
        shutil.move(os.path.join(mask_dir, mask_name), test_mask_dir)

    print("=====================finish move======================")

def load_SEG_data(testX_dir, target=(64, 64), shuffle=True):

    class_list = os.listdir(testX_dir)
    
    data = []
    labels = []
    data_names = []
    
    for class_name in class_list:
        class_path = os.path.join(testX_dir, class_name)
        img_list = os.listdir(class_path)
        if shuffle:
            random.shuffle(img_list)
        for image_Name in img_list:

            img_path = os.path.join(class_path, image_Name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, target)
            data_names.append(image_Name)
            print(img_path)

            mask_Path = img_path.replace('images', 'masks')
            mask = cv2.imread(mask_Path, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask,target)
            mask = np.expand_dims(mask, axis=-1)
            print(mask.shape)
            print(mask_Path)
            data.append(image)
            labels.append(mask)

    data = np.array(data, dtype="float")
    labels = np.array(labels, dtype="float")
    data = data / 255.0
    labels = labels / 255.0
    labels = np.array(labels)
    if labels.shape == 3:
        np.expand_dims(labels, axis=-1)

#     onehot_mask = mask_to_onehot(labels, 2)

    return data, labels, data_names
    
def load_SEG_data_for_test(testX_dir, target=(64, 64), shuffle=False):

    class_list = os.listdir(testX_dir)
    data = []
    labels = []
    original_imgs = []
    data_names = []
    for class_name in class_list:
        class_path = os.path.join(testX_dir, class_name)
        img_list = os.listdir(class_path)
        if shuffle:
            random.shuffle(img_list)
        for image_Name in img_list:

            img_path = os.path.join(class_path, image_Name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, target)
            data_names.append(image_Name)
            print(img_path)

            mask_Path = img_path.replace('images', 'masks')
            mask = cv2.imread(mask_Path)
            mask = cv2.resize(mask, target)
            print(mask_Path)
            data.append(image)
            labels.append(mask)

    data = np.array(data, dtype="float")
    labels = np.array(labels, dtype="float")
    original_imgs =data
    data = data / 255.0
    labels = labels / 255.0
    if labels.shape == 3:
        np.expand_dims(labels, axis=-1)

    onehot_mask = mask_to_onehot(labels, 2)
    return data, onehot_mask, data_names, original_imgs

def load_clas_seg_data(testX_dir, target=(64, 64), shuffle = True):

    class_list = os.listdir(testX_dir)
    data = []
    labels = []
    masks = []
    for class_name in class_list:
        class_path = os.path.join(testX_dir, class_name)
        img_list = os.listdir(class_path)
        if shuffle:
            random.shuffle(img_list)
        for image_Name in img_list:

            img_path = os.path.join(class_path, image_Name)
#             print(img_path)
            image = cv2.imread(img_path)
            image = cv2.resize(image, target)
         

            label = img_path.split(os.path.sep)[-2]
            labels.append(label)

            mask_Path = img_path.replace('images', 'masks')
#             print(mask_Path)
            mask = cv2.imread(mask_Path)
            mask = cv2.resize(mask, target)
            
            data.append(image)
            masks.append(mask)

    data = np.array(data, dtype="float")
    labels = np.array(labels)
    masks = np.array(masks, dtype="float")
    data = data / 255.0     
    masks = masks / 255.0
    masks = np.array(masks)
    if masks.shape == 3:
        np.expand_dims(masks, axis=-1)
    onehot_mask = mask_to_onehot(masks, 2)

    return data, labels, onehot_mask

def load_clas_seg_data_for_test(testX_dir, target=(64, 64), shuffle = False):

    class_list = os.listdir(testX_dir)
    data = []
    imgs = []
    labels = []
    masks = []
    for class_name in class_list:
        class_path = os.path.join(testX_dir, class_name)
        img_list = os.listdir(class_path)
        if shuffle:
            random.shuffle(img_list)
        for image_Name in img_list:

            img_path = os.path.join(class_path, image_Name)
            image = cv2.imread(img_path)
            imgs.append(image)
            image = cv2.resize(image, target)
#             print(img_path)

            label = img_path.split(os.path.sep)[-2]
            labels.append(label)

            mask_Path = img_path.replace('images', 'masks')
            mask = cv2.imread(mask_Path)
            mask = cv2.resize(mask, target)
#             print(mask_Path)
            data.append(image)
            masks.append(mask)

    data = np.array(data, dtype="float")
    labels = np.array(labels)
    masks = np.array(masks, dtype="float")
    
    imgs = data
    data = data / 255.0
    masks = masks / 255.0
    masks = np.array(masks)
    if masks.shape == 3:
        np.expand_dims(masks, axis=-1)
    onehot_mask = mask_to_onehot(masks, 2)

    return data, labels, onehot_mask, img_list, imgs

def load_img(testX_dir, target=(64, 64), shuffle = True):

    class_list = os.listdir(testX_dir)
    data = []
    labels = []
    masks = []
    for class_name in class_list:
        class_path = os.path.join(testX_dir, class_name)
        img_list = os.listdir(class_path)
        if shuffle:
            random.shuffle(img_list)
        for image_Name in img_list:

            img_path = os.path.join(class_path, image_Name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, target)
            print(img_path)

            label = img_path.split(os.path.sep)[-2]
            labels.append(label)

            mask_Path = img_path.replace('images', 'masks')
            mask = cv2.imread(mask_Path)
            mask = cv2.resize(mask, target)
            print(mask_Path)
            data.append(image)
            masks.append(mask)

    data = np.array(data, dtype="float")
    labels = np.array(labels)
    masks = np.array(masks, dtype="float")
    data = data / 255.0
    masks = masks / 255.0
    masks = np.array(masks)
    if masks.shape == 3:
        np.expand_dims(masks, axis=-1)
    onehot_mask = mask_to_onehot(masks, 2)

    return data, labels, onehot_mask

def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in range(palette):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map

def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x
