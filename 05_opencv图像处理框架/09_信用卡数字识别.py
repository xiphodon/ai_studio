import cv2
import numpy as np
import matplotlib.pyplot as plt


class CreditCardOCR(object):
    """
    信用卡数字识别
    """

    def __init__(self, img_path, template_path):
        self.img_path = img_path
        self.template_path = template_path

        self.img = None
        self.template = None
        self.template_binary = None
        self.digits_img_mapping_dict = dict()

    def load_images(self, is_show=True):
        """
        加载图片
        :return:
        """
        self.img = cv2.imread(self.img_path)
        self.template = cv2.imread(self.template_path)

        if is_show:
            self.cv2_imshow('img', self.img)
            self.cv2_imshow('template', self.template)

    def images_process(self, is_show=True):
        """
        信用卡图片处理
        :return:
        """
        self.img = self.resize_img(img=self.img, width=300)
        img_gray = cv2.cvtColor(src=self.img, code=cv2.COLOR_RGB2GRAY)
        # 初始化卷积核(指定卷积核形状和尺寸) cv2.MORPH_RECT 为矩形，cv2.MORPH_CROSS 为十字形，cv2.MORPH_ELLIPSE 为椭圆形
        kernel_rect = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(13, 3))
        kernel_rect_small = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
        kernel_square = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
        # 礼帽运算，获取高亮区
        img_tophat = cv2.morphologyEx(src=img_gray, op=cv2.MORPH_TOPHAT, kernel=kernel_rect)

        # sobel算子计算横向梯度大小并标准化，强化明暗边界，为了后面将数字横向连在一起
        img_grad_x = cv2.Sobel(src=img_tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
        img_grad_x_abs = np.abs(img_grad_x)
        img_grad_x_normal = cv2.normalize(src=img_grad_x_abs, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_8U)

        # 使用闭运算，将数字连在一起
        img_grad_x_close = cv2.morphologyEx(src=img_grad_x_normal, op=cv2.MORPH_CLOSE, kernel=kernel_rect)

        # # 可视化灰度图，判断二值化阈值
        # img_grad_x_close_hist = cv2.calcHist(
        #     images=[img_grad_x_close], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        # plt.plot(img_grad_x_close_hist, color=(0, 0, 0))
        # plt.hist(x=img_grad_x_close.ravel(), bins=256)
        # plt.xlim([0, 256])
        # plt.show()

        # 自动寻找二值化阈值，使用cv2.THRESH_OTSU，需要将阈值设置为0。（非常适用于灰度直方图具有双峰情况）
        threshold_auto, img_binary = cv2.threshold(
            src=img_grad_x_close, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 闭运算，使用方核，将连接的数字更像“凝聚”
        img_binary_close = cv2.morphologyEx(src=img_binary, op=cv2.MORPH_CLOSE, kernel=kernel_square)

        # 寻找轮廓
        box_contours, hierarchy = cv2.findContours(image=img_binary_close, mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_SIMPLE)
        # 筛选轮廓
        loc_list = list()
        for contour in box_contours:
            x, y, w, h = cv2.boundingRect(contour)
            w_h_ratio = w / h
            if 2.5 < w_h_ratio < 4:
                if 40 < w < 55 and 10 < h < 20:
                    loc_list.append((x, y, w, h))

        # 筛选出来的轮廓排序
        loc_list = sorted(loc_list, key=lambda x_y_w_h: x_y_w_h[0])
        # print(loc_list)

        card_box_img_list = list()
        card_box_img_binary_list = list()
        all_box_digit_list = list()
        for i, (box_x, box_y, box_w, box_h) in enumerate(loc_list):
            # 每个大方框中的数字
            box_digit_list = list()
            delta_px = 2
            box_img = img_gray[box_y - delta_px: box_y + box_h + delta_px, box_x - delta_px: box_x + box_w + delta_px]
            # print(box_img.shape)
            card_box_img_list.append(box_img)

            # 每四个数字的box进行二值化，如果使用 cv2.THRESH_OTSU 阈值thresh需要设置到0
            thresh, box_img_binary = cv2.threshold(src=box_img, thresh=0, maxval=255,
                                                   type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # print(thresh)
            # box_img_binary = cv2.dilate(src=box_img_binary, kernel=(9, 9), iterations=2)
            box_img_binary = cv2.morphologyEx(src=box_img_binary, op=cv2.MORPH_CLOSE, kernel=kernel_rect_small)
            card_box_img_binary_list.append(box_img_binary)

            # 获取单个数字轮廓
            box_digits_contours, hierarchy = cv2.findContours(image=box_img_binary, mode=cv2.RETR_EXTERNAL,
                                                              method=cv2.CHAIN_APPROX_SIMPLE)
            # 获得单个数字最小外接四边形，并按数字顺序排序
            box_digits_rect_list = [cv2.boundingRect(contour) for contour in box_digits_contours]
            box_digits_rect_list = sorted(box_digits_rect_list, key=lambda x_y_w_h: x_y_w_h[0])
            # 将单个数字图像切出来
            for box_digits_rect in box_digits_rect_list:
                delta_px_2 = 2
                x, y, w, h = box_digits_rect
                digit_img = box_img_binary[y - delta_px_2: y + h + delta_px_2, x - delta_px_2: x + w + delta_px_2]
                # self.cv2_imshow('-', digit_img)
                digit_img = cv2.resize(src=digit_img, dsize=(57, 88))

                # 用模板切好的数字模板来匹配信用卡切出来的数字
                card_digit_img_match_score_list = list()
                for d, digit_template_img in self.digits_img_mapping_dict.items():
                    match_res = cv2.matchTemplate(image=digit_img, templ=digit_template_img,
                                                  method=cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src=match_res)
                    card_digit_img_match_score_list.append(max_val)
                # 与所有模板的匹配值中，找到匹配度最高的数字
                best_match_digit = np.argmax(card_digit_img_match_score_list)

                box_digit_list.append(str(best_match_digit))
            all_box_digit_list.append({
                'box_digit_list': box_digit_list,
                'box_loc': (box_x, box_y, box_w, box_h)
            })

        if is_show:
            self.cv2_imshow('img_resize', self.img)
            self.cv2_imshow('img_gray', img_gray)
            self.cv2_imshow('img_tophat', img_tophat)
            self.cv2_imshow('img_grad_x_normal', img_grad_x_normal)
            self.cv2_imshow('img_grad_x_close', img_grad_x_close)
            self.cv2_imshow('img_binary', img_binary)
            self.cv2_imshow('img_binary_close', img_binary_close)

            # 绘制轮廓
            img_2 = self.img.copy()
            cv2.drawContours(image=img_2, contours=box_contours, contourIdx=-1, color=(0, 0, 255), thickness=2)
            box_contours_rect_list = [cv2.boundingRect(contour) for contour in box_contours]
            for x, y, w, h in box_contours_rect_list:
                cv2.rectangle(img=img_2, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            for x, y, w, h in loc_list:
                cv2.rectangle(img=img_2, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=1)
            self.cv2_imshow('img_contours', img_2)

            card_box_img_list = [
                cv2.resize(
                    src=item_img, dsize=tuple(reversed(card_box_img_list[0].shape))
                ) for item_img in card_box_img_list
            ]
            card_box_img_binary_list = [
                cv2.resize(
                    src=item_img, dsize=tuple(reversed(card_box_img_binary_list[0].shape))
                ) for item_img in card_box_img_binary_list
            ]
            # print(card_box_img_list)
            card_box_img_merge = np.vstack(card_box_img_list)
            card_box_img_binary_merge = np.vstack(card_box_img_binary_list)
            card_box_img_and_binary_merge = np.hstack((card_box_img_merge, card_box_img_binary_merge))
            self.cv2_imshow('box_img_and_binary', card_box_img_and_binary_merge)
            # for box_img, box_img_binary in zip(card_box_img_list, card_box_img_binary_list):
            #     self.cv2_imshow('box_img', box_img)
            #     self.cv2_imshow('box_img_binary', box_img_binary)

            # 绘制检测识别结果
            img_3 = self.img.copy()
            delta_px = 5
            for box_item in all_box_digit_list:
                box_digit = ''.join(box_item['box_digit_list'])
                box_x, box_y, box_w, box_h = box_item['box_loc']
                cv2.rectangle(img=img_3, pt1=(box_x - delta_px, box_y - delta_px),
                              pt2=(box_x + box_w + delta_px, box_y + box_h + delta_px),
                              color=(0, 255, 0), thickness=2)
                cv2.putText(img=img_3, text=box_digit, org=(box_x, box_y - 2 * delta_px),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 255, 0), thickness=2)
            self.cv2_imshow('detect_res', img_3)

    def split_template_digits(self, is_show=True):
        """
        分割模板上的数字，并对应出图片
        :param is_show:
        :return:
        """
        template_gray = cv2.cvtColor(src=self.template, code=cv2.COLOR_RGB2GRAY)
        ret, self.template_binary = cv2.threshold(src=template_gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(
            image=self.template_binary,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        # 单个数字最小矩阵框，并排序
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        bounding_boxes = sorted(bounding_boxes, key=lambda x_y_w_h: x_y_w_h[0])

        delta_px = 10
        # 将数字与对应数字模板图片映射保存字典中
        for i, box in enumerate(bounding_boxes):
            x, y, w, h = box
            # 截取对应数字图片
            digit_img = self.template_binary[y - delta_px: y + h + delta_px, x - delta_px: x + w + delta_px]
            # self.cv2_imshow('digit_img', digit_img)
            digit_img = cv2.resize(src=digit_img, dsize=(57, 88))
            self.digits_img_mapping_dict[i] = digit_img

        if is_show:
            self.cv2_imshow('template_binary', self.template_binary)
            # 绘制轮廓
            template_2 = self.template.copy()
            cv2.drawContours(image=template_2, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2)
            # 绘制矩阵框
            for x, y, w, h in bounding_boxes:
                cv2.rectangle(img=template_2, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
            self.cv2_imshow('template_contours', template_2)

    def detect(self):
        """
        开始ocr识别探测
        :return:
        """
        # 加载信用卡图片和数字模板图片
        self.load_images()
        # 分割模板数字，并生成数字与数字模板映射字典
        self.split_template_digits()
        # 信用卡图片处理
        self.images_process()

    @staticmethod
    def resize_img(img, width=None, height=None, inter=cv2.INTER_AREA):
        """
        按照原图宽高比重置尺寸，width和height只需传一个
        cv2.INTER_AREA 是缩放较好的插值法
        :param img:
        :param width:
        :param height:
        :param inter:
        :return:
        """
        h, w = img.shape[:2]
        if width is None and height is None:
            return img
        if width is None:
            width = int(height / h * w)
        else:
            height = int(width / w * h)
        return cv2.resize(src=img, dsize=(width, height), interpolation=inter)

    @staticmethod
    def cv2_imshow(img_name, img):
        """
        图像展示
        :param img_name:
        :param img:
        :return:
        """
        cv2.imshow(img_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cc_ocr = CreditCardOCR(img_path=r'./data/credit_card_01.png', template_path=r'./data/ocr_reference.png')
    cc_ocr.detect()
