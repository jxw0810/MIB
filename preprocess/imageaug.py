#!usr/bin/python
# -*- coding: utf-8 -*-
import cv2
from imgaug import augmenters as iaa
import os
 
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([

    iaa.SomeOf((0, 5),
        [
                iaa.Fliplr(0.6),
                iaa.Flipud(0.6),
    			iaa.Affine(rotate=(-40, 40)),#sjq add
                # Convert some images into their superpixel representation,
                # sample between 20 and 200 superpixels per image, but do
                # not replace all superpixels with their average, only
                # some of them (p_replace).
                #sometimes(
                #   iaa.Superpixels(
                #        p_replace=(0, 1.0),
                #        n_segments=(20, 200)
                #   )
                #),
 
                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)),#sjq add
                    iaa.AverageBlur(k=(2, 3)),#sjq add
                    iaa.MedianBlur(k=(3, 5)),#sjq add
                ]),
 
                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                iaa.Sharpen(alpha=(0, 0.5), lightness=(0.9, 1.3)),
 
                # Same as sharpen, but for an embossing effect.
                #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
 
 
                # Add gaussian noise to some images.
                # In 50% of these cases, the noise is randomly sampled per
                # channel and pixel.
                # In the other 50% of all cases it is sampled once per
                # pixel (i.e. brightness change).
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.04*255)
                ),
 
                # Invert each image's chanell with 5% probability.
                # This sets each pixel value v to 255-v.5%
                #iaa.Invert(0.05, per_channel=True), # invert color channels
                #iaa.Invert(0.05, per_channel=False),# not invert color channels
 
                # Add a value of -10 to 10 to each pixel.
                #iaa.Add((-10, 10), per_channel=0.5),
                iaa.Add((-10, 10), per_channel=0),
 
                # Add random values between -40 and 40 to images, with each value being sampled per pixel:
                #iaa.AddElementwise((-40, 40)),
                iaa.AddElementwise((-10, 40)),
 
                # Change brightness of images (50-150% of original value).
                #iaa.Multiply((0.5, 1.5)),
                iaa.Multiply((0.8, 1.2)),
 
                # Multiply each pixel with a random value between 0.5 and 1.5.
                #iaa.MultiplyElementwise((0.5, 1.5)),
                iaa.MultiplyElementwise((0.8, 1.2)),
 
                # Improve or worsen the contrast of images.
                #iaa.ContrastNormalization((0.5, 2.0)),
                iaa.ContrastNormalization((0.8, 1.2)),
        ],
        # do all of the above augmentations in random order
        random_order=True
    )
],random_order=True) #apply augmenters in random order

path = 'E:\\Multi-task IBA\\dataset\INbreast\\train\\images\\benign_images\\'
savedpath = 'E:\\Multi-task IBA\\dataset\\INbreast\\train\\images\\benign_images\\'
if not os.path.exists(savedpath):
    os.mkdir(savedpath)

imglist=[]
filelist = os.listdir(path)

for i in range(len(filelist)):
    img = cv2.imread(path + filelist[i])
    imglist.append(img)

print('all the picture have been appent to imglist')

for count in range(20):
    images_aug = seq.augment_images(imglist)
    for j in range(len(filelist)):
        (imagename, extension) = os.path.splitext(filelist[j])
        filename = imagename + str(count) + '_' + str(j) + '_' + '.jpg'

        cv2.imwrite(savedpath + filename, images_aug[j])
        print('image of count%s index%s has been writen' %(count, j))