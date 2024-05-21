import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir, Qt
import cv2
import numpy as np
import onnxruntime as ort
import os
import time


def preprocess(img):
    # 对图像进行原始变换
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # 将归一化后的图像数据类型转换为 uint8（用于 matplotlib 显示）
    # img = (img * 255).astype(np.uint8)
    img = np.transpose(img, [2, 0, 1])
    print(img.shape)
    img = np.expand_dims(img, axis=0)
    return img

def load_labels(label_file):
    label_map = {}
    with open(label_file, 'r') as file:
        for line in file:
            # 分割索引和类别名称
            parts = line.strip().split(' ')
            index = int(parts[0])  # 索引是第一个元素
            label = ' '.join(parts[1:])  # 剩下的部分是类别名称
            label_map[index] = label
    return label_map

# def classify_results(results, label_map):
#     # 假设 results 是一个包含分类索引的列表
#     classified_labels = [label_map[result] for result in results]
#     return classified_labels
# 加载标签文件
label_map = load_labels('./data/imagenet1k_label_list.txt')

# 假设的分类函数，需要替换为实际的分类逻辑
def classify_image(image_path):
    # 在这里加载你的分类模型并进行预测
    # 返回预测结果，例如：'dog', 'cat', 'bird' 等
    # 这里只是返回一个模拟结果
    image = preprocess(image_path)
    onnx_path = r"./model/mobilenetv3.onnx"
    start_time = time.time()
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    prediction = ort_session.run(
        # ort_session.get_outputs()[0],
        ['output'],
        ort_inputs)
    class_index = np.argmax(prediction[0])
    end_time = time.time()
    # 获取分类后的标签名称
    class_name = label_map[class_index]
    inference_time = (end_time - start_time) * 1000
    return class_name, int(inference_time)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("图像分类")
        self.setGeometry(100, 100, 400, 600)

        # 创建布局
        layout = QVBoxLayout()
        # 创建按钮，用于选择图片
        self.btn_select_image = QPushButton("选择图片并分类", self)
        self.btn_select_image.clicked.connect(self.selectAndClassifyImage)
        layout.addWidget(self.btn_select_image)

        # 创建标签，用于显示图像
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.imageLabel)
        layout.addSpacing(150)

        # 创建标签，用于显示分类结果
        self.resultLabel = QLabel("分类结果:", self)
        layout.addWidget(self.resultLabel)

        # 创建标签，用于显示推理时间
        self.InfertimeLabel = QLabel("推理时间:", self)
        layout.addWidget(self.InfertimeLabel)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def selectAndClassifyImage(self):
        default_dir = "./data"
        # 确保目录存在，否则使用当前工作目录
        if not os.path.exists(default_dir):
            default_dir = QDir.currentPath()
        fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", default_dir, "Image Files (*.jpg *.png)")
        if fileName:
            image = QPixmap(fileName)
            # 调整QLabel的大小以适应图片的尺寸，同时保持一定的边距
            # 显示选择的图片
            self.imageLabel.setPixmap(image.scaled(self.imageLabel.size(), Qt.KeepAspectRatio))
            # 进行图像分类
            result, inference_time = classify_image(fileName)
            self.resultLabel.setText(f'Classification Result: {result}')
            self.InfertimeLabel.setText(f'Inference Time: {inference_time} ms')
            self.resultLabel.setAlignment(Qt.AlignCenter)
        else:
            self.resultLabel.setText('Please select an image first!')

def main():
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.showMaximized()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()