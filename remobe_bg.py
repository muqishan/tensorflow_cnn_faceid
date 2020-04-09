import sys
import dlib
from skimage import io
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

predictor_path = "shape_predictor_68_face_landmarks.dat"  # 人脸提取加载器模型
faces_path = "../faces/lzy/655.jpg"

detector = dlib.get_frontal_face_detector()  # 加载dlib人脸检测器
predictor = dlib.shape_predictor(predictor_path)


def get_landmarks(img):
    dets = detector(img, 1)
    landmarks = np.zeros((19, 2))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        for i in range(17):
            landmarks[i] = landmarks[i] = (shape.part(i).x, shape.part(i).y)
        landmarks[17] = (shape.part(24).x, shape.part(24).y - 20)
        landmarks[18] = (shape.part(19).x, shape.part(19).y - 20)
    return landmarks


def inside(x, y, region):
    j = len(region) - 1
    flag = False
    for i in range(len(region)):
        if region[i][1] < y <= region[j][1] or region[j][1] < y <= region[i][1]:
            if region[i][0] + (y - region[i][1]) / (region[j][1]
                                                    - region[i][1]) * (region[j][0] - region[i][0]) < x:
                flag = not flag
        j = i
    return flag


def remove_b(image):
    region = get_landmarks(image)
    shape = list(image.shape) + [3]  # 将图像转化为列表，便于访问
    for i in range(shape[0]):
        for j in range(shape[1]):
            if not inside(j, i, region):
                image[i, j] = (255, 0, 0)  # bgr颜色通道 与rgb相反，所以255.0.0是蓝色
    return image


def is_same(l):
    temp = l[0]
    count = 0
    for i in l:
        if (temp == i).all():
            count = 0
        else:
            count = count + 1
            if count > 7:  # 对于连续出现7次非背景色像素点时，判断这一行不能删除
                return False
    return True


def remove_border(img):
    while True:  # 重复扫描图片 行操作
        for index, value in enumerate(img):
            if is_same(value):
                img = np.delete(img, index, axis=0)  # 如果这一行颜色全部相同，则跳出循环
                break
        else:
            break  # 扫描完一遍未发现相同行颜色，则退出扫描
    while True: # 重复扫描图片 列操作
        for i in range(img.shape[1]):
            if is_same(img[:, i]):
                img = np.delete(img, i, axis=1)
                break
        else:
            break
    return img


if __name__ == '__main__':

    # paths = ['faces/zx/', 'faces/tt/', 'faces/qq/', 'faces/by/',
    #          'test_faces/zx/', 'test_faces/tt/', 'test_faces/qq/', 'test_faces/by/']
    paths = ['faces/lzy/']
    # paths = ['test_faces//', 'test_faces//', 'test_faces//',
    #          'test_faces//', 'test_faces//', 'test_faces//', ]
    for path in paths:
        for i in range(414, 415):
            image_path = path + str(i) + '.jpg'
            img_name = path + str(i) + '.jpg'
            # image = image[:, :, ::-1]  # 翻转颜色通道
            try:
                image = cv2.imread(image_path)
                # img = remove_b(image)  # 擦除背景
                img = remove_border(image)  # 劲量保证脸部最大 这两个方法一次只运行一个最佳
                cv2.imwrite(img_name, img)
            except:
                pass
            if i % 5 == 0:
                print(img_name + '已经完成')
