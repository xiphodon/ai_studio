import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_img():
    """
    读取图片
    BGR
    :return:
    """
    img = cv2.imread('./data/lena.jpg')
    print(img.shape)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_img_gray():
    """
    读取图片为灰度
    :return:
    """
    img = cv2.imread('./data/lena.jpg', flags=cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    cv2.imshow('gray_img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_video():
    """
    读取视频/摄像头
    :return:
    """
    # 视频路径为获取资源，id序号则为摄像头id，0为默认第一个摄像头id
    # cap = cv2.VideoCapture(r'./resources/video.mp4')
    cap = cv2.VideoCapture(0)
    # cap propId 0-18
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)  # 3为宽
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)  # 4为高
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 5)  # 10为亮度

    while True:
        success, img = cap.read()
        if not success:
            break

        cv2.imshow('video', img)
        # 图片截取
        cut_video_img = img[:300, 200:600]
        cv2.imshow('cut_video_img', cut_video_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 有按键则返回按键ASCII码，无按键则返回-1
            # 取按键返回ASCII码二进制后8位，为按键'q'退出循环
            break

    cap.release()
    cv2.destroyAllWindows()


def color_channel():
    """
    颜色通道
    :return:
    """
    img = cv2.imread('./data/lena.jpg')
    print('img shape:', img.shape)

    # split 按颜色通道拆分
    b, g, r = cv2.split(img)
    # 也可按照通道取
    # b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # 通道合并
    # img = cv2.merge(mv=(b, g, r))

    print('img b channel shape:', b.shape)

    # 只保留g通道
    img[:, :, 0] = 0  # b通道值置位0
    img[:, :, 2] = 0  # r通道值置位0
    cv2.imshow('img g', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def make_border():
    """
    制作边界
    :return:
    """
    img = cv2.imread('./data/lena.jpg')
    top, bottom, left, right = 20, 40, 60, 80
    # replicate 重复最边像素
    border_replicate_img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
    # 沿边镜像，最边像素也镜像
    border_reflect_img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)
    # 沿边镜像，最边像素不镜像，相当于101中的0
    border_reflect101_img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
    # 平铺重复
    border_wrap_img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_WRAP)
    # 常量0
    border_constant_img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)

    plt.subplot(231)
    plt.imshow(img, 'gray')
    plt.title('original')

    plt.subplot(232)
    plt.imshow(border_replicate_img, 'gray')
    plt.title('border_replicate_img')

    plt.subplot(233)
    plt.imshow(border_reflect_img, 'gray')
    plt.title('border_reflect_img')

    plt.subplot(234)
    plt.imshow(border_reflect101_img, 'gray')
    plt.title('border_reflect101_img')

    plt.subplot(235)
    plt.imshow(border_wrap_img, 'gray')
    plt.title('border_wrap_img')

    plt.subplot(236)
    plt.imshow(border_constant_img, 'gray')
    plt.title('border_constant_img')

    plt.show()


def img_px_value():
    """
    图片像素数值相关
    :return:
    """
    # 切片，使用 img[img_slice] 相当于 img[:5, :10, 0]
    img_slice = (slice(0, 5), slice(0, 10), 0)

    img = cv2.imread('./data/lena.jpg')
    print('img: ', img[img_slice])

    # 每个像素+10
    img_2 = img + 10
    print('img + 10: ', img_2[img_slice])

    # 每个像素+200  (图片“+”会自动对256取模)
    img_3 = img + 200
    print('img + 200: ', img_3[img_slice])

    # 两个尺寸一致的图片对应像素相加  (图片“+”会自动对256取模)
    img_4 = img + img_2
    print('img + img_2: ', img_4[img_slice])

    # 使用cv2.add 将两个尺寸一致的图片对应像素值相加，cv2.add方法超过255数值均为255，相当于每个像素点value执行 min(value, 255)
    img_5 = cv2.add(img, img_2)
    print('cv2.add(img, img_2): ', img_5[img_slice])


def img_mix():
    """
    图片混合
    :return:
    """
    img_dog = cv2.imread('./data/dog.jpg')
    img_cat = cv2.imread('./data/cat.jpg')
    # 将img_dog尺寸置为img_cat大小，宽w、高h
    img_dog_resize = cv2.resize(img_dog, dsize=(img_cat.shape[1], img_cat.shape[0]))

    # cv2.imshow('img_dog', img_dog)
    # cv2.imshow('img_cat', img_cat)
    # cv2.imshow('img_dog_resize', img_dog_resize)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print('img_cat shape: ', img_cat.shape)
    print('img_dog shape: ', img_dog.shape)
    print('img_dog_resize shape: ', img_dog_resize.shape)

    # 按权重将两张图叠加在一起   dst = src1*alpha + src2*beta + gamma;
    img_res = cv2.addWeighted(src1=img_cat, alpha=0.4, src2=img_dog_resize, beta=0.6, gamma=0)
    cv2.imshow('img_mix', img_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_scale():
    """
    图片比例缩放
    :return:
    """
    # 除了根据具体尺寸resize，也可以按照原图比例缩放
    img = cv2.imread('./data/lena.jpg')
    # 等比例放大
    img_2_2 = cv2.resize(src=img, dsize=(0, 0), fx=2, fy=2)
    # 非等比例放大
    img_3_2 = cv2.resize(src=img, dsize=(0, 0), fx=3, fy=2)
    cv2.imshow('img', img)
    cv2.imshow('img x*2 y*2', img_2_2)
    cv2.imshow('img x*3 y*2', img_3_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_convert_color():
    """
    图片转换颜色
    :return:
    """
    img = cv2.imread('./data/lena.jpg')
    # bgr 转 灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bgr 转 hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow('img_gray', img_gray)
    cv2.imshow('img_hsv', img_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_threshold():
    """
    图片阈值处理
    一般传入灰度图，做二值化操作
    :return:
    """
    img_gray = cv2.imread('./data/lena.jpg', flags=cv2.IMREAD_GRAYSCALE)

    ret, img_bin = cv2.threshold(src=img_gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
    ret, img_bin_inv = cv2.threshold(src=img_gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV)
    ret, img_trunc = cv2.threshold(src=img_gray, thresh=127, maxval=255, type=cv2.THRESH_TRUNC)
    ret, img_2zero = cv2.threshold(src=img_gray, thresh=127, maxval=255, type=cv2.THRESH_TOZERO)
    ret, img_2zero_inv = cv2.threshold(src=img_gray, thresh=127, maxval=255, type=cv2.THRESH_TOZERO_INV)

    title_list = ['img_gray', 'img_bin', 'img_bin_inv', 'img_trunc', 'img_2zero', 'img_2zero_inv']
    img_list = [img_gray, img_bin, img_bin_inv, img_trunc, img_2zero, img_2zero_inv]

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img_list[i], 'gray')
        plt.title(title_list[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def img_filter_blur():
    """
    图片过滤和模糊
    :return:
    """
    img_noise = cv2.imread('./data/lena_noise.png')
    # plt绘制需要rgb图像
    img_noise = cv2.cvtColor(img_noise, code=cv2.COLOR_BGR2RGB)

    # 均值滤波
    img_blur = cv2.blur(src=img_noise, ksize=(3, 3))
    # 高斯模糊
    img_gaussian_blur = cv2.GaussianBlur(src=img_noise, ksize=(3, 3), sigmaX=1)
    # box滤波(均值滤波)
    img_box = cv2.boxFilter(src=img_noise, ddepth=-1, ksize=(3, 3), normalize=False)
    # box滤波(均值滤波)标准化
    img_box_normalize = cv2.boxFilter(src=img_noise, ddepth=-1, ksize=(3, 3), normalize=True)
    # 中值滤波
    img_median = cv2.medianBlur(src=img_noise, ksize=5)

    title_list = ['img_noise', 'img_blur', 'img_gaussian_blur', 'img_box', 'img_box_normalize', 'img_median']
    img_list = [img_noise, img_blur, img_gaussian_blur, img_box, img_box_normalize, img_median]

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.title(title_list[i])
        plt.imshow(img_list[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def img_erode_dilate():
    """
    图片腐蚀膨胀
    :return:
    """
    img = cv2.imread('./data/binary.png', flags=cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(src=img, dsize=(0, 0), fx=2, fy=2)
    ret, img = cv2.threshold(src=img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    # 腐蚀
    img_erode = cv2.erode(src=img, kernel=kernel, iterations=1)
    # 膨胀
    img_dilate = cv2.dilate(src=img, kernel=kernel, iterations=1)

    # 开运算，先腐蚀，后膨胀（消去暗区域中的亮区域）
    img_opening = cv2.morphologyEx(src=img, op=cv2.MORPH_OPEN, kernel=kernel)
    # 闭运算，先膨胀，后腐蚀（消去亮区域中的暗区域）
    img_closing = cv2.morphologyEx(src=img, op=cv2.MORPH_CLOSE, kernel=kernel)

    # 礼帽与黑帽
    # 礼帽运算，变换前图 - 开运算后图，即查看消除掉的亮区域
    img_top_hat = cv2.morphologyEx(src=img, op=cv2.MORPH_TOPHAT, kernel=kernel)
    # 黑帽运算，闭运算后图 - 变换前图，即查看消除掉的暗区域
    img_black_hat = cv2.morphologyEx(src=img, op=cv2.MORPH_BLACKHAT, kernel=kernel)

    img_merge_1 = np.hstack((img, img_erode, img_dilate))
    img_merge_2 = np.hstack((img, img_opening, img_closing))
    img_merge_3 = np.hstack((img, img_top_hat, img_black_hat))

    img_merge = np.vstack((img_merge_1, img_merge_2, img_merge_3))
    cv2.imshow('img_merge', img_merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_gradient():
    """
    图像梯度
    :return:
    """
    img = cv2.imread('./data/lena.jpg', cv2.IMREAD_GRAYSCALE)

    # 1、sobel算子
    #
    #   x方向上梯度计算矩阵（3*3为例），y方向上梯度计算矩阵（3*3为例）
    #       -1  0  +1                   -1  -2  -1
    #       -2  0  +2                    0   0   0
    #       -1  0  +1                   +1  +2  +1
    #
    # sobel x方向计算梯度
    sobel_x = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    # sobel y方向计算梯度
    sobel_y = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    # sobel 同时计算xy方向梯度
    sobel_xy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
    # 可视化梯度大小（正负表示梯度方向，可视化无法直接显示负值，仅可以可视化大小）
    sobel_x_abs = cv2.convertScaleAbs(src=sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(src=sobel_y)
    sobel_xy_abs = cv2.convertScaleAbs(src=sobel_xy)

    # 叠加x、y方向上梯度大小
    # (这种单独计算x与y方向上的梯度大小后再叠加，因不考虑正负方向，直观效果比用sobel算子直接计算x与y方向上梯度后再abs的效果好一点点)
    sobel_xy_w = cv2.addWeighted(src1=sobel_x_abs, alpha=0.5, src2=sobel_y_abs, beta=0.5, gamma=0)

    #####################

    # 2、scharr算子
    #
    #   x方向上梯度计算矩阵（3*3），y方向上梯度计算矩阵（3*3）
    #       -3   0   +3                  -3   -10  -3
    #       -10  0   +10                  0    0    0
    #       -3   0   +3                  +3   +10  +3
    #
    # scharr x方向计算梯度
    scharr_x = cv2.Scharr(src=img, ddepth=cv2.CV_64F, dx=1, dy=0)
    # scharr y方向计算梯度
    scharr_y = cv2.Scharr(src=img, ddepth=cv2.CV_64F, dx=0, dy=1)
    # 可视化梯度大小（正负表示梯度方向，可视化无法直接显示负值，仅可以可视化大小）
    scharr_x_abs = cv2.convertScaleAbs(src=scharr_x)
    scharr_y_abs = cv2.convertScaleAbs(src=scharr_y)

    # 叠加x、y方向上梯度大小
    scharr_xy_w = cv2.addWeighted(src1=scharr_x_abs, alpha=0.5, src2=scharr_y_abs, beta=0.5, gamma=0)

    ###########################

    # 3、 laplacian算子
    #
    #   (矩阵已3*3为例)
    #           0    1    0
    #           1   -4    1
    #           0    1    0
    #
    laplacian = cv2.Laplacian(src=img, ddepth=cv2.CV_64F, ksize=3)
    laplacian_abs = cv2.convertScaleAbs(src=laplacian)

    img_merge_1 = np.hstack((img, sobel_x_abs, sobel_y_abs, sobel_xy_abs, sobel_xy_w))
    img_merge_2 = np.hstack((img, scharr_x_abs, scharr_y_abs, scharr_xy_w, laplacian_abs))
    img_merge = np.vstack((img_merge_1, img_merge_2))

    cv2.imshow('img_merge', img_merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _extend_line(img, row, col, threshold_low):
    """
    从确定的边界点往外延伸
    :param img: 梯度值矩阵
    :param row: 当前确定边界点的行
    :param col: 当前确定边界点的列
    :param threshold_low: 低阈值
    :return:
    """
    if img[row, col] != 0:
        return
    img_rows, img_cols = img.shape
    img[row, col] = 255
    for i in range(-1, 2, 1):
        new_row = row + i
        if new_row < 0 or new_row >= img_rows:
            continue
        for j in range(-1, 2, 1):
            new_col = col + j
            if new_col < 0 or new_col >= img_cols:
                continue
            if img[new_row, new_col] >= threshold_low:
                _extend_line(img=img, row=new_row, col=new_col, threshold_low=threshold_low)


def img_canny():
    """
    图片边缘检测
    :return:
    """
    # 边缘检测处理流程
    # 1、使用高斯滤波器，平滑图像，滤除噪音
    # 2、计算每个像素点的梯度（大小 + 方向）
    # 3、应用非极大值抑制，消除杂散响应（通过每个点的梯度方向看梯度大小是否为极大值，极大值则保留，非极大值则置0）
    # 4、应用双阈值检测来确定真实和潜在的边缘（大于高阈值置255，小于低阈值置0，二者之间通过8连通区域判断，连接高阈值点接受）

    img = cv2.imread('./data/lena.jpg', flags=cv2.IMREAD_GRAYSCALE)
    img_shape = img.shape
    # 1、使用高斯滤波器，平滑图像，滤除噪音
    img_blur = cv2.GaussianBlur(src=img, ksize=(3, 3), sigmaX=0.8)
    # 2、计算每个像素点的梯度（大小 + 方向）
    img_sobel_x = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    img_sobel_y = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    # 计算梯度方向（tan(dy / dx) = 梯度方向）
    img_gradient_tan_sita = img_sobel_y / (img_sobel_x + 1e-10)
    img_gradient_sita = np.arctan(img_gradient_tan_sita)
    # 计算梯度大小（一维距离，二维距离均可）
    # img_gradient_value = np.abs(img_sobel_x) + np.abs(img_sobel_y)
    img_gradient_value = np.sqrt(np.square(img_sobel_x) + np.square(img_sobel_y))

    # 方向编码为：
    # 7  8  9
    # 4  5  6
    # 1  2  3
    img_sita_area = np.zeros_like(img_gradient_sita)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            sita = img_gradient_sita[i, j]
            if -np.pi / 8.0 < sita <= np.pi / 8.0:
                img_sita_area[i, j] = 456
            elif np.pi / 8.0 < sita <= 3 * np.pi / 8.0:
                img_sita_area[i, j] = 159
            elif sita > 3 * np.pi / 8.0 or sita <= -3 * np.pi / 8.0:
                img_sita_area[i, j] = 258
            elif -3 * np.pi / 8.0 < sita <= -np.pi / 8.0:
                img_sita_area[i, j] = 357
            else:
                print(f'error location:{i}, {j}, value: {sita}')

    # 3、应用非极大值抑制，消除杂散响应（通过每个点的梯度方向看梯度大小是否为极大值，极大值则保留，非极大值则置0）
    img_nms = np.zeros_like(img_sita_area)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if i == 0 or j == 0 or i == img_shape[0] - 1 or j == img_shape[1] - 1:
                continue
            cur_area_value = img_sita_area[i, j]
            img_gradient_i_j = img_gradient_value[i, j]
            img_gradient_i_j_n1 = img_gradient_value[i, j - 1]
            img_gradient_i_j_p1 = img_gradient_value[i, j + 1]
            img_gradient_i_n1_j = img_gradient_value[i + 1, j]
            img_gradient_i_p1_j = img_gradient_value[i - 1, j]
            img_gradient_i_p1_j_n1 = img_gradient_value[i + 1, j - 1]
            img_gradient_i_n1_j_p1 = img_gradient_value[i - 1, j + 1]
            img_gradient_i_n1_j_n1 = img_gradient_value[i - 1, j - 1]
            img_gradient_i_p1_j_p1 = img_gradient_value[i + 1, j + 1]

            if cur_area_value == 456:
                if img_gradient_i_j >= img_gradient_i_j_n1 and img_gradient_i_j >= img_gradient_i_j_p1:
                    img_nms[i, j] = img_gradient_i_j
            elif cur_area_value == 159:
                if img_gradient_i_j >= img_gradient_i_p1_j_n1 and img_gradient_i_j >= img_gradient_i_n1_j_p1:
                    img_nms[i, j] = img_gradient_i_j
            elif cur_area_value == 258:
                if img_gradient_i_j >= img_gradient_i_n1_j and img_gradient_i_j >= img_gradient_i_p1_j:
                    img_nms[i, j] = img_gradient_i_j
            elif cur_area_value == 357:
                if img_gradient_i_j >= img_gradient_i_n1_j_n1 and img_gradient_i_j >= img_gradient_i_p1_j_p1:
                    img_nms[i, j] = img_gradient_i_j
            else:
                print(f'nms error, location: {i}, {j}, are_value: {cur_area_value}')

    # 4、应用双阈值检测来确定真实和潜在的边缘（大于高阈值置255，小于低阈值置0，二者之间通过8连通区域判断，连接高阈值点接受）
    threshold_low = 20
    threshold_high = 60
    img_double_threshold = np.zeros_like(img_nms)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if i == 0 or j == 0 or i == img_shape[0] - 1 or j == img_shape[1] - 1:
                continue
            cur_nms_value = img_nms[i, j]
            if cur_nms_value >= threshold_high:
                _extend_line(img=img_double_threshold, row=i, col=j, threshold_low=threshold_low)
            elif cur_nms_value < threshold_low:
                img_double_threshold[i, j] = 0

    img_cv2_canny = cv2.Canny(image=img, threshold1=40, threshold2=80)

    cv2.imshow('img_blur', img_blur)
    cv2.imshow('img_sobel_x', img_sobel_x)
    cv2.imshow('img_sobel_y', img_sobel_y)
    cv2.imshow('img_nms', img_nms)
    cv2.imshow('img_double_threshold', img_double_threshold)
    cv2.imshow('img_cv2_canny', img_cv2_canny)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_pyr():
    """
    图像金字塔
    :return:
    """
    img = cv2.imread('./data/lena.jpg')
    # 高斯金字塔
    # 向上采样（放大）偶数行列填充0，与高斯内核卷积
    img_pyr_up = cv2.pyrUp(src=img)
    # 向下采样（缩小）与高斯内核卷积，偶数行列剔除
    img_pyr_down = cv2.pyrDown(src=img)

    # 拉普拉斯金字塔
    img_pyr_laplacian = img - cv2.resize(src=cv2.pyrUp(cv2.pyrDown(img)), dsize=(263, 263))

    cv2.imshow('img', img)
    cv2.imshow('img_pyr_up', img_pyr_up)
    cv2.imshow('img_pyr_down', img_pyr_down)
    cv2.imshow('img_pyr_laplacian', img_pyr_laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_find_contours():
    """
    图片搜索轮廓
    :return:
    """
    img = cv2.imread('./data/shapes.png')
    img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(src=img_gray, ksize=(5, 5), sigmaX=1)
    # ret, img_binary = cv2.threshold(src=img_gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
    img_binary_canny = cv2.Canny(image=img_blur, threshold1=10, threshold2=80)

    # 寻找轮廓，返回轮廓和层级
    contours, hierarchy = cv2.findContours(image=img_binary_canny, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    img_copy = img.copy()
    # 绘制轮廓
    cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2)

    img_copy_2 = img.copy()
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        if area > 200:
            # 筛选出面积大于200像素的轮廓
            # 计算曲线长度，曲线为闭合曲线
            arc_length = cv2.arcLength(curve=contour, closed=True)
            # 多边形近似曲线(折线与曲线偏离的最大距离可设置为周长的1%-4%)
            approx = cv2.approxPolyDP(curve=contour, epsilon=0.03 * arc_length, closed=True)
            cv2.drawContours(image=img_copy_2, contours=[approx], contourIdx=-1, color=(0, 0, 255), thickness=2)

            # 获取轮廓外接矩形边界
            x, y, w, h = cv2.boundingRect(contour)
            # 绘制外接矩形
            cv2.rectangle(img=img_copy_2, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)

            # 获取轮廓外接圆边界
            (x, y), r = cv2.minEnclosingCircle(contour)
            cv2.circle(img=img_copy_2, center=(int(x), int(y)), radius=int(r), color=(0, 255, 0), thickness=2)

    cv2.imshow('img_copy', img_copy)
    cv2.imshow('img_copy_2', img_copy_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_fft():
    """
    图片快速傅里叶变换
    高通滤波(变化大的区域为高频，高通滤波会更关注边界区域)
    低通滤波(变化小的区域为低频，低通滤波会忽视边界，模糊图像)
    :return:
    """
    img = cv2.imread('./data/lena.jpg', flags=cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    img_f32 = np.float32(img)
    # 傅里叶变换获得频域矩阵  shape为(w, h, 2)， 最后一个维度0表示实部，1表示虚部
    dft = cv2.dft(src=img_f32, flags=cv2.DFT_COMPLEX_OUTPUT)
    # 将频域矩阵低频区从左上角移动至矩阵中间
    dft_shift = np.fft.fftshift(dft)

    # 将频域图像可视化
    # 根据复数实部和虚部计算出距离，用距离来可视化，x参数为实部， y参数为虚部，magnitude为，(x^2 + y^2)^(1/2)
    m = cv2.magnitude(x=dft_shift[:, :, 0], y=dft_shift[:, :, 1])
    m2 = np.sqrt(np.square(dft_shift[:, :, 0]) + np.square(dft_shift[:, :, 1]))
    # 计算m 与 m2的距离，可以看出，两矩阵大小近似，m和m2计算方法类似
    d_m_m2 = np.sum(a=np.sum(a=np.abs(m - m2), axis=1), axis=0)
    print(d_m_m2)

    # 标准化距离，方便可视化
    m_log = 20 * np.log(m)
    m_normal = cv2.normalize(src=m_log, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # print(m_normal)

    # img图像中心
    img_center_r, img_center_c = img.shape[0] // 2, img.shape[1] // 2

    mask_size = 60
    # 制作一个低通滤波蒙板(作用于低频中间化后的频域矩阵)
    low_pass_mask = np.zeros_like(a=dft_shift, dtype=np.uint8)
    low_pass_mask[img_center_r - mask_size // 2: img_center_r + mask_size // 2,
    img_center_c - mask_size // 2: img_center_c + mask_size // 2] = 1

    # 低通蒙板作用于低频中间化后的频域矩阵
    dft_shift_low_pass = dft_shift * low_pass_mask
    # 将中间化得频域矩阵移回原处(shift逆变换)
    dft_i_shift_low_pass = np.fft.ifftshift(dft_shift_low_pass)
    # 傅里叶逆变换
    dft_i_low_pass = cv2.idft(src=dft_i_shift_low_pass)

    # 可视化低通逆变换后的图像
    img_low_pass = cv2.magnitude(x=dft_i_low_pass[:, :, 0], y=dft_i_low_pass[:, :, 1])
    # 标准化像素值
    img_low_pass_normal = cv2.normalize(src=img_low_pass, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # 制作一个高通蒙板
    high_pass_mask = np.ones_like(dft_shift)
    high_pass_mask[img_center_r - mask_size // 2: img_center_r + mask_size // 2,
    img_center_c - mask_size // 2: img_center_c + mask_size // 2] = 0

    # 高通蒙板作用于低频中间化后的频域矩阵
    dft_shift_high_pass = dft_shift * high_pass_mask
    # 将中间化的频域矩阵移回原处（shift逆变换）
    dft_i_shift_high_pass = np.fft.ifftshift(dft_shift_high_pass)
    # 傅里叶逆变换
    dft_i_high_pass = cv2.idft(src=dft_i_shift_high_pass)

    # 可视化高通逆变换后的图像
    img_high_pass = cv2.magnitude(x=dft_i_high_pass[:, :, 0], y=dft_i_high_pass[:, :, 1])
    img_high_pass_normal = cv2.normalize(src=img_high_pass, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # print(img_high_pass.shape)

    cv2.imshow('img', img)
    # cv2.imshow('m_log', m_log)
    cv2.imshow('m_normal', m_normal)
    cv2.imshow('img_low_pass_normal', img_low_pass_normal)
    cv2.imshow('img_high_pass_normal', img_high_pass_normal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_histogram():
    """
    图片直方图绘制
    直方图均衡化
    :return:
    """
    img = cv2.imread('./data/lena.jpg')
    img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    print(img_gray.shape)
    # 灰度图直方图
    hist = cv2.calcHist(images=[img_gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    plt.hist(x=img_gray.ravel(), bins=256)
    plt.plot(hist, color=(0, 0, 0))
    print(hist.shape)

    # bgr图像直方图
    color = ('b', 'g', 'r')
    for i, c in enumerate(color):
        hist_item = cv2.calcHist(images=[img], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
        plt.plot(hist_item, color=c)
        plt.xlim([0, 256])

    plt.show()

    # 使用mask蒙板，并绘制直方图
    mask = np.zeros_like(a=img_gray, dtype=np.uint8)
    mask[100: 200, 100: 200] = 255

    # 图片覆盖蒙板
    img_masked = cv2.bitwise_and(src1=img_gray, src2=img_gray, mask=mask)

    hist_full = cv2.calcHist(images=[img_gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    hist_mask = cv2.calcHist(images=[img_gray], channels=[0], mask=mask, histSize=[256], ranges=[0, 256])

    plt.subplot(2, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(img_masked, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.plot(hist_full)
    plt.plot(hist_mask)
    plt.show()


def img_hist_equalize():
    """
    直方图均衡化
    :return:
    """
    img_gray = cv2.imread('./data/lena.jpg', flags=cv2.IMREAD_GRAYSCALE)
    # print(img_gray)
    # print(img_gray.ravel())

    # 直方图均衡化
    equ_hist = cv2.equalizeHist(src=img_gray)
    # 自适应分窗口局部均衡化(clipLimit颜色对比度阈值， tileGridSize直方图均衡化窗口大小)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    res_clahe = clahe.apply(img_gray)

    plt.subplot(2, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.subplot(2, 3, 2)
    plt.imshow(equ_hist, cmap='gray')
    plt.subplot(2, 3, 3)
    plt.imshow(res_clahe, cmap='gray')
    plt.subplot(2, 3, 4)
    plt.hist(x=img_gray.ravel(), bins=256)
    plt.subplot(2, 3, 5)
    plt.hist(x=equ_hist.ravel(), bins=256)
    plt.subplot(2, 3, 6)
    plt.hist(x=res_clahe.ravel(), bins=256)
    plt.show()


def img_match_template():
    """
    图像匹配模板
    :return:
    """
    # 图像尺寸（W*H），模板尺寸（w*h），从图像原点开始匹配距离，匹配结果尺寸为（W-w+1 * H-h+1）
    img_gray = cv2.imread('./data/lena.jpg', flags=cv2.IMREAD_GRAYSCALE)
    template = cv2.imread('./data/face.png', flags=cv2.IMREAD_GRAYSCALE)
    template = cv2.resize(src=template, dsize=(0, 0), fx=0.2, fy=0.2)

    img_gray_w, img_gray_h = img_gray.shape
    template_w, template_h = template.shape

    # - TM_SQDIFF：计算平方不同，计算出来的值越小，越相关
    # - TM_SQDIFF_NORMED：计算归一化平方不同，计算出来的值越接近0，越相关
    # - TM_CCORR：计算相关性，计算出来的值越大，越相关
    # - TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，越相关
    # - TM_CCOEFF：计算相关系数，计算出来的值越大，越相关
    # - TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，越相关

    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for i, m in enumerate(methods, start=1):
        img_item = img_gray.copy()

        method = eval(m)
        res = cv2.matchTemplate(image=img_item, templ=template, method=method)
        # 通过卷积，按照小范围聚集的极值点来判定匹配位置
        kernel = np.ones(shape=(7, 7), dtype=np.float32)
        res = cv2.filter2D(src=res, ddepth=-1, kernel=kernel)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src=res)

        # 通过不同匹配方法，找到匹配最佳的位置
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            x, y = min_loc
        else:
            x, y = max_loc

        pt1 = (x, y)
        pt2 = (x + template_w, y + template_h)

        # print(img_item.shape)
        # print(pt1, pt2)

        img_item = cv2.rectangle(img=img_item, pt1=pt1, pt2=pt2, color=255, thickness=2)
        # print(img_item)

        # 按照匹配度，超过阈值输出
        threshold = 0.75
        res_normal = cv2.normalize(src=res, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            loc = np.where(res_normal <= 1 - threshold)
        else:
            loc = np.where(res_normal >= threshold)

        # 将超出阈值的点，算出平均位置，相当于算出匹配度较高的点聚集中心，使用最终位置为匹配位置
        loc_x_arr, loc_y_arr = loc
        loc_x_mean = int(np.mean(loc_x_arr))
        loc_y_mean = int(np.mean(loc_y_arr))
        print(loc_x_mean, loc_y_mean)
        t_pt1 = (loc_x_mean, loc_y_mean)
        t_pt2 = (loc_x_mean + template_w, loc_y_mean + template_h)
        cv2.rectangle(img=img_item, pt1=t_pt1, pt2=t_pt2, color=0, thickness=2)

        plt.subplot(2, len(methods), i)
        plt.title(label=m)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.imshow(res, cmap='gray')
        plt.subplot(2, len(methods), len(methods) + i)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.imshow(img_item, cmap='gray')
    plt.show()






if __name__ == '__main__':
    # read_img()
    # read_img_gray()
    # read_video()
    # color_channel()
    # make_border()
    # img_px_value()
    # img_mix()
    # img_scale()
    # img_convert_color()
    # img_threshold()
    # img_filter_blur()
    # img_erode_dilate()
    # img_gradient()
    # img_canny()
    # img_pyr()
    # img_find_contours()
    # img_fft()
    # img_histogram()
    # img_hist_equalize()
    img_match_template()
