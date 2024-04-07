import os
import cv2
import shutil

def rename_SEG_data(testX_dir):
    class_list = os.listdir(testX_dir)

    for class_name in class_list:
        class_path = os.path.join(testX_dir, class_name)
        img_list = os.listdir(class_path)
        for image_Name in img_list:
            img_path = os.path.join(class_path, image_Name)
            print('img path:', img_path)
            mask_Path = img_path.replace('images', 'masks')
            mask_Path, _ = mask_Path.split('.')
            old_mask = mask_Path + '_mask.png'
            if os.path.exists(old_mask):
                print('old path:', old_mask)
                new_mask = mask_Path + '.png'
                print('new path:', new_mask)
                try:
                    shutil.move(old_mask, old_mask)
                except Exception as e:
                    print(e)

def rename_SEG_data2(mask_dir):
    class_list = os.listdir(mask_dir)

    for class_name in class_list:
        class_path = os.path.join(mask_dir, class_name)
        img_list = os.listdir(class_path)
        for image_Name in img_list:
            old_Path = os.path.join(class_path, image_Name)
            new_Name = image_Name.replace("_mask.png", ".png")
            new_Path = os.path.join(class_path, new_Name)

            if os.path.exists(old_Path):
                print('old path:', old_Path)
                print('new path:', new_Path)
                try:
                    shutil.move(old_Path, new_Path)
                except Exception as e:
                    print(e)

train_path = '../dataset/Dataset_BUSI/train/masks/'
test_path = train_path.replace('train', 'test')
val_path = train_path.replace('train', 'val')

rename_SEG_data2(train_path)
rename_SEG_data2(test_path)
rename_SEG_data2(test_path)



