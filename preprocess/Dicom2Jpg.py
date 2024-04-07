import os
import pydicom       #用于读取DICOM(DCOM)文件
import argparse
import scipy.misc
# import scipy.misc    #用imageio替代
import imageio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin', type=str, default='F:\\1_数据集\\公开乳腺癌数据集\\INbreast\\AllDICOMs', help='train photos')
    parser.add_argument('--JPG', type=str, default='F:\\1_数据集\\公开乳腺癌数据集\\INbreast\\JPG', help='test photos')
    opt=parser.parse_args()
    print(opt)

    #imgway_1为源文件夹
    #imgway_2为jpg文件夹
    imgway_1=opt.origin
    imgway_2 = opt.JPG

    i=0

    for filename in os.listdir(r"%s" % imgway_1):

        # name = str(i)
        name=filename[:-4]
        ds = pydicom.read_file("%s/%s" % (imgway_1, filename))          #读取文件
        img = ds.pixel_array

        # path1 = os.path.join(imgway_1, filename)
        short_name = name.split("_")[0]+'.jpg'
        path2 = os.path.join(imgway_2, short_name)

        imageio.imwrite(path2, img)

        # os.rename(path1, path2)

        # if i==410:      #转换410张
        #     break
