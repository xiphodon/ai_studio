import cv2
import matplotlib.pyplot as plt


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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)    # 3为宽
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
    img[:, :, 0] = 0    # b通道值置位0
    img[:, :, 2] = 0    # r通道值置位0
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



if __name__ == '__main__':
    # read_img()
    # read_img_gray()
    # read_video()
    # color_channel()
    # make_border()
    # img_px_value()
    # img_mix()
    img_scale()
