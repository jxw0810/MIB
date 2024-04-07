import os
import shutil

count = 0

# 将几个文件夹中的图片移动到一个文件夹中
def moveFiles(path, disdir):  # path为原始路径，disdir是移动的目标目录
    global count
    dirlist = os.listdir(path)
    for i in dirlist:
        child = os.path.join('%s\%s' % (path, i))
        if os.path.isfile(child):
            count += 1
            shutil.move(child, os.path.join(disdir, str(count) + ".jpg"))
            continue
        moveFiles(child, disdir)

if __name__ == '__main__':
    path = 'G:/毓璜顶医院乳腺能谱数据/癌症_正常_未裁剪_/normal2'
    disdir = 'G:/毓璜顶医院乳腺能谱数据/normal_images'
    moveFiles(path, disdir)