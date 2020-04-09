# 人脸识别分类器
cascade_path = "G:\\open_cv_\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml "
# 数据集存放根目录
face_path = "G:/python/py/bydesign/lzy/faces"
test_path = "G:/python/py/bydesign/lzy/test_faces"
# 分类标注字典
labels_dict = {
    'wyt': 0,
    'gxx': 1,
    'gxm': 2,
    'hkj': 3,
    'tt': 4,
    'qq': 5,
    'zx': 6,
    'by': 7,
}
# sava_model 保存地址
model_path = 'model/my_model_5.h5'
# 模型4  八个人的人脸数据集测试 95.46%
# 模型5  八个人的人脸数据集测试 97.19%
