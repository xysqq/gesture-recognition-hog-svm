# 使用HOG+SVM识别手势
一共识别5种手势动作

1、加速 2、减速 3、停止 4、左转 5、右转

项目文件列表如下：

data：存放五种手势图片

model:存在训练完的模型

train:保存训练集的特征向量

test:保存验证集的特征向量

add_data.py:利用OpenCV获取图片并标记，用来制作数据集，里面写好了预处理操作，保存的是轮廓图像

bulid_txt.py:生成训练需要的训练集验证集路径的txt文件

train.txt:训练集的路径集合

test.txt:验证集的路径集合

train.py:训练SVM，将训练好的模型保存在model

result.txt:保存验证集结

使用方法：

先使用add_data.py保存手势数据，键盘上的12345分别对应这五个手势，然后用build_txt.py生成训练用的txt文件，最后用train.py训练模型，查看模型效果。

测试结果：
使用自制手势数据集，可以达到百分97%的准确率
