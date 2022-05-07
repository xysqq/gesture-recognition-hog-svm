import cv2
import os
import numpy as np

MIN_DESCRIPTOR = 32  # surprisingly enough, 2 descriptors are already enough
font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体
size = 0.5  # 设置大小

def skinMask(roi):
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
    (y, cr, cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5,5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu处理
    res = cv2.bitwise_and(roi, roi, mask=skin)
    return res

##计算傅里叶描述子
def fourierDesciptor(res):
    # Laplacian算子进行八邻域检测
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)
    contour = find_contours(Laplacian)  # 提取轮廓点坐标
    if contour:
        contour_array = contour[0][:, 0, :]  # 注意这里只保留区域面积最大的轮廓点坐标
        ret_np = np.ones(dst.shape, np.uint8)  # 创建黑色幕布
        ret = cv2.drawContours(ret_np, contour[0], -1, (255, 255, 255), 1)  # 绘制白色轮廓
        contours_complex = np.empty(contour_array.shape[:-1], dtype=complex)
        contours_complex.real = contour_array[:, 0]  # 横坐标作为实数部分
        contours_complex.imag = contour_array[:, 1]  # 纵坐标作为虚数部分
        fourier_result = np.fft.fft(contours_complex)  # 进行傅里叶变换
        # fourier_result = np.fft.fftshift(fourier_result)
        descirptor_in_use = truncate_descriptor(fourier_result)  # 截短傅里叶描述子
        # reconstruct(ret, descirptor_in_use)
        skin_area = cv2.contourArea(contour[0])
        return ret

def find_contours(Laplacian):
    # binaryimg = cv2.Canny(res, 50, 200) #二值化，canny检测
    binary, contour = cv2.findContours(Laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
    contour = sorted(binary, key=cv2.contourArea, reverse=True)  # 对一系列轮廓点坐标按它们围成的区域面积进行排序
    return contour

# 截短傅里叶描述子
def truncate_descriptor(fourier_result):
    descriptors_in_use = np.fft.fftshift(fourier_result)

    # 取中间的MIN_DESCRIPTOR项描述子
    center_index = int(len(descriptors_in_use) / 2)
    low, high = center_index - int(MIN_DESCRIPTOR / 2), center_index + int(MIN_DESCRIPTOR / 2)
    descriptors_in_use = descriptors_in_use[low:high]

    descriptors_in_use = np.fft.ifftshift(descriptors_in_use)
    return descriptors_in_use


##由傅里叶描述子重建轮廓图
def reconstruct(img, descirptor_in_use):
    # descirptor_in_use = truncate_descriptor(fourier_result, degree)
    # descirptor_in_use = np.fft.ifftshift(fourier_result)
    # descirptor_in_use = truncate_descriptor(fourier_result)
    # print(descirptor_in_use)
    contour_reconstruct = np.fft.ifft(descirptor_in_use)
    contour_reconstruct = np.array([contour_reconstruct.real,
                                    contour_reconstruct.imag])
    contour_reconstruct = np.transpose(contour_reconstruct)
    contour_reconstruct = np.expand_dims(contour_reconstruct, axis=1)
    if contour_reconstruct.min() < 0:
        contour_reconstruct -= contour_reconstruct.min()
    contour_reconstruct *= img.shape[0] / contour_reconstruct.max()
    contour_reconstruct = contour_reconstruct.astype(np.int32, copy=False)

    black_np = np.ones(img.shape, np.uint8)  # 创建黑色幕布
    black = cv2.drawContours(black_np, contour_reconstruct, -1, (255, 255, 255), 1)  # 绘制白色轮廓
    cv2.imshow("contour_reconstruct", black)
    # cv2.imwrite('recover.png',black)
    return black

if __name__ == '__main__':
    width, height = 200, 200  # 设置拍摄窗口大小
    x0, y0 = 400, 200  # 设置选取位置
    cap = cv2.VideoCapture(0)  # 开摄像头

    labels = ['speed', 'shift', 'stop', 'turn_left', 'turn_right']

    print("现在进入手势控制模式")
    print("请把右手对准摄像头")

    while 1:
        ret, frame = cap.read()  # 读取摄像头的内容
        frame = cv2.flip(frame, 2)

        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))  # 画出截取的手势框图
        roi = frame[y0:y0 + height, x0:x0 + width]  # 获取手势框图

        blur = cv2.bilateralFilter(roi, 9, 75, 75)

        kernel = np.ones((3, 3), np.uint8)  # 设置卷积核
        erosion = cv2.erode(blur, kernel)  # 腐蚀操作
        dilation = cv2.dilate(erosion, kernel)  # 膨胀操作

        # 肤色检测
        skin = skinMask(dilation)

        # 轮廓检测
        ret = fourierDesciptor(skin)

        cv2.imshow('frame', frame)
        cv2.imshow('ret', ret)

        key = cv2.waitKey(1) & 0xFF  # 按键判断并进行一定的调整
        # 按'q'键退出录像， 按's'发送开始指令
        # 1 2 3 4 5 数据采集
        i = 1
        # 1的ascii码为49
        if key >= ord('1') and key <= ord('5'):
            index = key - 49
            while (os.path.exists('./data/' + labels[index] + '_' + str(i) + '.jpg')):
                i += 1
            cv2.imwrite('./data/' + labels[index] + '_' + str(i) + '.jpg', ret)
            if os.path.exists('./data/' + labels[index] + '_' + str(i) + '.jpg'):
                print(labels[index] + '_' + str(i) + 'is already sucessfully saved.')
        elif key == ord('s'):
            y0 += 5
        elif key == ord('a'):
            x0 -= 5
        elif key == ord('d'):
            x0 += 5
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break
