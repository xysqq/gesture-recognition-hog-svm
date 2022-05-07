import random
import os

if __name__ == '__main__':
    length = 10  # 数据集数量
    pro = 0.75  # 训练集比例
    train_label_path = 'train.txt'
    test_label_path = 'test.txt'

    if os.path.exists(train_label_path):
        os.remove(train_label_path)
    if os.path.exists(test_label_path):
        os.remove(test_label_path)

    f1 = open('train.txt', 'a')
    f2 = open('test.txt', 'a')

    labels = ['speed', 'shift', 'stop', 'turn_left', 'turn_right']

    i_list = [i for i in range(1, length + 1)]
    random.shuffle(i_list)

    train_list = i_list[:int(length * pro)]
    test_list = i_list[int(length * pro):]

    for i in range(len(train_list)):
        for j in range(len(labels)):
            f1.write(f'{labels[j]}_{train_list[i]}.jpg {j}\n')

    for i in range(len(test_list)):
        for j in range(len(labels)):
            f2.write(f'{labels[j]}_{test_list[i]}.jpg {j}\n')

    f1.close()
    f2.close()