import csv
import shutil
import os

# 根据.csv文件批量给图片分类 BI-RADS分级
# csv文件
target_path = '../dataset/INbreast/'
original_path = '../dataset/INbreast/JPEG/'
with open('../dataset/INbreast/label.csv',"rt", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    for row in rows:
        if os.path.exists(target_path + row[2]):
            full_path = original_path + row[0] + '.jpg'
            shutil.move(full_path, target_path + row[2] + '/'+ row[0] + '.jpg')
        else:
            os.makedirs(target_path + row[2])
            full_path = original_path + row[0] + '.jpg'
            shutil.move(full_path, target_path + row[2] +'/' + row[0] + '.jpg')
