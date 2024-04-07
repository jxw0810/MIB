from gernerate_data import split_train_and_test
import shutil
import random
import os
import glob

img_dir = "../dataset/ICIS/train/images"
mask_dir = "../dataset/ICIS/train/masks"
contour_dir = "../dataset/ICIS/train/contour"
test_dir = "../dataset/ICIS/test"
# seg_dir = "../dataset/ICIS/test"


def copy_segmentation(img_dir, mask_dir, contour_dir, test_dir):
    test_mask_dir = os.path.join(test_dir, 'masks')
    test_img_dir = os.path.join(test_dir, 'images')
    test_contour_dir = os.path.join(test_dir, 'contour')


    if not os.path.exists:
        os.mkdir(test_mask_dir)
        os.mkdir(test_img_dir)
        os.mkdir(test_contour_dir)

    # 读取数据和标签
    print("------开始读取数据------")
    maks_list = os.listdir(mask_dir)

    number = len(maks_list)
    print("number:", number)

    for maks_name in maks_list:
        # copy mask
        shutil.copy(os.path.join(mask_dir, maks_name), test_mask_dir)

        # copy images
        img_name = maks_name.split('label')[0] + '.jpg'
        shutil.copy(os.path.join(img_dir, img_name), test_img_dir)

        # copy contour
        contour_name = maks_name.split('label')[0] + 'contour.png'
        shutil.copy(os.path.join(contour_dir, contour_name), test_contour_dir)

    print("=====================finish move======================")

def split_train_and_test(img_dir, mask_dir,contour_dir, test_dir, split_size=0.25):

    print("======================开始分离数据=====================")
    print("test ratio:", split_size)

    test_mask_dir = os.path.join(test_dir, 'masks')
    test_contour_dir = os.path.join(test_dir, 'contour')
    test_img_dir = os.path.join(test_dir, 'images')

    if not os.path.exists:
        os.mkdir(test_mask_dir)
        os.mkdir(test_img_dir)
        os.mkdir(test_contour_dir)

    # 读取数据和标签
    print("------开始读取数据------")
    trainX_list = os.listdir(img_dir)
    # 随机打乱
    random.shuffle(trainX_list)

    number = len(trainX_list)
    test_number = int(number * split_size)
    print(test_number)
    test_img_list = trainX_list[-test_number:]

    for image_name in test_img_list:
        # 移动图像
        shutil.move(os.path.join(img_dir, image_name), test_img_dir)
        # 移动contour
        contour_name = image_name.split('.')[0] + 'contour.png'
        shutil.move(os.path.join(contour_dir, contour_name), test_contour_dir)
        # 移动mask
        mask_name = image_name.split('.')[0] + 'label.png'
        shutil.move(os.path.join(mask_dir, mask_name), test_mask_dir)

    print("=====================finish move======================")

# 3级目录，图像目录结构train/images/normal_images/  图像掩码目录结构train/masks/normal_masks/
def split_train_and_test2(train_dir, test_split_size=0.1, val_split_size=0.1):

    # 对每个类别下的数据：
    #     首先读取每个类别下的图像名，
    #     随机打乱，分出测试集，验证集图像列表，
    #     递归测试集列表：
    #          根据测试集图像名获取得对应的mask名
    #          移动图像和mask
    #     递归验证集列表：
    #          根据测试集图像名获取得对应的mask名
    #          移动图像和mask

    print("======================开始分离数据=====================")
    print("test ratio:", test_split_size)
    print("val ratio:", val_split_size)

    train_img_dir = os.path.join(train_dir, 'images')

    class_path_list = glob.glob(train_img_dir+'*\\*')
    for class_path in class_path_list:
        all_images_list = glob.glob(class_path + '*\\*')
        all_masks_list = str(all_images_list).replace('images', 'masks').replace('.jpg', '_mask.png')

        all_number = len(all_images_list)
        test_number = int(all_number * test_split_size)
        val_number = int(all_number * val_split_size)

        val_test_img_list = random.sample(all_images_list, test_number + val_number)
        test_img_list = val_test_img_list[-test_number:]
        val_img_list = val_test_img_list[:-test_number]

        # 处理测试集
        for image_name in test_img_list:
            new_image_path = image_name.replace('train', 'test')
            shutil.move(image_name, new_image_path)
            mask_name = image_name.replace('images', 'masks').replace('.jpg', 'mask.png')
            # if not os.path.exists(mask_name):
            #     print("image_name:", image_name)
            #     print("mask_name:", mask_name)
            new_mask_path = mask_name.replace('train', 'test')
            shutil.move(mask_name, new_mask_path)

        # 处理验证集
        for image_name in val_img_list:
            new_image_path = image_name.replace('train', 'val')
            shutil.move(image_name, new_image_path)
            mask_name = image_name.replace('images', 'masks').replace('.jpg', 'mask.png')
            # if not os.path.exists(mask_name):
            #     print("image_name:", image_name)
            #     print("mask_name:", mask_name)
            new_mask_path = mask_name.replace('train', 'val')
            shutil.move(mask_name, new_mask_path)


    print("=====================finish move======================")













# 从原始数据集中找出带有mask的图像，并复制到test_dir中
# copy_segmentation(img_dir, mask_dir, contour_dir, test_dir)

# 分出测试集到test文件夹中
# split_train_and_test(img_dir, mask_dir, contour_dir, test_dir, split_size=0.25)

# 从训练集每个类别中分出验证集和测试集图像和对应的mask
train_dir = '..\\dataset\\ICIS\\train\\'
split_train_and_test2(train_dir, test_split_size=0.1, val_split_size=0.1)
