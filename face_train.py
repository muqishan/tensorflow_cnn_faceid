from load_dataset import load_dataset, resize_image, IMAGE_SIZE
from setting import cascade_path, face_path, labels_dict, model_path, test_path  # 导入人脸识别分类器 ,数据集存放根目录
import tensorflow as tf
import numpy as np
import cv2
from remobe_bg import remove_b, remove_border
import dlib


class Faces(object):
    def __init__(self):
        self.file_path = face_path  # 初始化训练数据集路径
        self.test_path = test_path  # 初始化测试数据集路径
        self.model_path = model_path  # 初始化模型保存位置路径
        self.train_images, self.train_labels = load_dataset(self.file_path)  # 加载训练集
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"  # 加入人脸细节提取模型path
        self.predictor = dlib.shape_predictor(self.predictor_path)  # 加载人脸细节提取模型
        self.image_size = IMAGE_SIZE  # 初始化照片大小
        self.labels = {v: k for k, v in labels_dict.items()}  # 反转字典，方便后续使用
        self.test_images, self.test_labels = load_dataset(self.test_path)  # 加载测试数据集
        self.detector = dlib.get_frontal_face_detector()

    # 定义模型
    def tf_model(self):
        self.model = tf.keras.Sequential()  # 建立顺序模型
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=self.train_images.shape[1:],
                                              activation='relu', padding='same'))

        # self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        # self.model.add(tf.keras.layers.MaxPool2D())
        # self.model.add(tf.keras.layers.Dropout(0.25))  # 抑制过拟合
        # self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        # self.model.add(tf.keras.layers.GlobalAveragePooling2D())
        # self.model.add(tf.keras.layers.Dense(6, activation='softmax'))

        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.MaxPool2D())
        self.model.add(tf.keras.layers.Dropout(0.25))  # 抑制过拟合
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

        self.model.add(tf.keras.layers.MaxPool2D())
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

        self.model.add(tf.keras.layers.MaxPool2D())
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))

        self.model.add(tf.keras.layers.GlobalAveragePooling2D())
        self.model.add(tf.keras.layers.Dense(128,activation='relu'))
        self.model.add(tf.keras.layers.Dense(8, activation='softmax'))

    # 显示模型架构
    def show_summary(self):  # 展示模型概况
        self.model.summary()

    # 重新定义需要识别的照片
    def resize_test_image(self, image_path, image=None):
        if image is None:
            image = cv2.imread(image_path)
            i = resize_image(image)  # 需要进行预测的数据也要进行等比缩放
            inp = i.reshape(1, 64, 64, 3)  # 重新确定image的形状 测试数据输入维度应当为4维 数量*IMAGESIZE*IMAGESIZE*RGB
            inp = inp.astype('float32')  # dtype需要一致 这里更该为 float32
            return inp
        else:
            i = resize_image(image)  # 需要进行预测的数据也要进行等比缩放
            inp = i.reshape(1, 64, 64, 3)  # 重新确定image的形状 测试数据输入维度应当为4维
            inp = inp.astype('float32')  # dtype需要一致 这里更该为 float32
            return inp

    # 加载已保存的模型
    def load_model(self):
        self.model = tf.keras.models.load_model('model/my_model_5.h5')

    # 训练并保存模型
    def face_fit(self):
        self.model.compile(loss='sparse_categorical_crossentropy',
                           # optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-6,
                           #                                   momentum=0.9, nesterov=True),
                           optimizer='adam',
                           metrics=['acc'])
        self.model.fit(self.train_images, self.train_labels, epochs=10)
        self.model.save(model_path)  # 保存训练好的模型

    # 评估模型
    def assess(self):
        loss, acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
        # 对于训练数据集擦掉背景，测试数据集未擦掉背景，1号模型的识别率 75.67%
        # 对于训练数据集擦掉背景，测试数据集同擦掉背景，1号模型的识别率 92.99%
        # 对于训练数据及擦掉背景边界，测试数据集同擦掉背景边界，2号模型识别率97%+
    # 获取获取人脸特征的19个坐标点
    def get_landmarks(self, img):
        dets = self.detector(img, 1)
        landmarks = np.zeros((19, 2))
        for k, d in enumerate(dets):
            shape = self.predictor(img, d)
            for i in range(17):
                landmarks[i] = landmarks[i] = (shape.part(i).x, shape.part(i).y)
            landmarks[17] = (shape.part(24).x, shape.part(24).y - 20)
            landmarks[18] = (shape.part(19).x, shape.part(19).y - 20)
        return landmarks

    def inside(self, x, y, region):
        j = len(region) - 1
        flag = False
        for i in range(len(region)):
            if region[i][1] < y <= region[j][1] or region[j][1] < y <= region[i][1]:
                if region[i][0] + (y - region[i][1]) / (region[j][1]
                                                        - region[i][1]) * (region[j][0] - region[i][0]) < x:
                    flag = not flag
            j = i
        return flag

    # 将背景颜色置位蓝色
    def remove_b(self, image):
        region = self.get_landmarks(image)
        shape = list(image.shape) + [3]  # 将图像转化为列表，便于访问
        for i in range(shape[0]):
            for j in range(shape[1]):
                if not self.inside(j, i, region):
                    image[i, j] = (255, 0, 0)  # bgr颜色通道 与rgb相反，所以255.0.0是蓝色

        return image

    def image_test(self, file_path):
        test_image = self.resize_test_image(file_path)
        test_image = self.remove_b(test_image)
        test_image = remove_border(test_image)
        chenjia = self.model.predict(test_image)
        index = np.argmax(chenjia)  # 取出最大几率值对应索引
        print(chenjia)
        print(self.labels)
        print('this is {}'.format(self.labels.get(index)))
        return self.labels.get(index)

    def video_test(self):
        color = (0, 255, 0)

        # 捕获指定摄像头的实时视频流
        origin = 0  # 改为视频路径则为识别视频中人脸
        cap = cv2.VideoCapture(origin)
        # 循环检测识别人脸
        count = 0
        while count == 0:
            count = count + 1
            ret, frame = cap.read()  # 读取一帧视频
            if ret is True:
                # 图像灰化，降低计算复杂度
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue
            # 使用人脸识别分类器，读入分类器
            cascade = cv2.CascadeClassifier(cascade_path)
            # 利用分类器识别出哪个区域为人脸
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect
                    # 截取脸部图像提交给模型识别这是谁
                    image = frame[y - 50: y + h + 50, x - 50: x + w + 50]
                    image = self.resize_test_image(image_path='', image=image)
                    chenjia = self.model.predict(image)
                    index = np.argmax(chenjia)  # 取出最大几率值对应索引
                    print(chenjia, end=' ')
                    print(self.labels, index)
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    # 文字提示是谁
                    cv2.putText(frame, self.labels.get(index),  # 获取labels中的标注
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽

            # cv2.imshow("softmax", frame)

            # 等待按键响应 10毫秒
            k = cv2.waitKey(10)
            # 如果输入q则退出循环
            if k & 0xFF == ord('q'):
                break

        # 释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()


F = Faces()
# F.tf_model()
# F.face_fit()
F.load_model()
F.assess()
# F.video_test()
