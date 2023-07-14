import random

import cv2
import numpy as np


def harris():
    """
    角点检测

    通过窗口在水平和竖直方向上细微移动，检测变化量。
    进而得到变化量矩阵M，特征值λ1，λ2
    角点响应R=λ1*λ2 - k(λ1 + λ2)^2
    k一般取值为[0.04, 0.06]
    R>0: 角，λ1和λ2都较大
    R<0: 边，λ1 >> λ2 或 λ2 >> λ1
    R≈0：面，λ1和λ2都较小
    :return:
    """
    img = cv2.imread('./data/shapes.png')
    img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(src=img_gray, blockSize=2, ksize=3, k=0.04)
    # 定义角阈值，边阈值大小
    corner_threshold = 0.05 * dst.max()
    edge_threshold = 0.05 * dst.min()
    # 将边角处颜色分别替换为红色和蓝色
    img[dst > corner_threshold] = [0, 0, 255]
    img[dst < edge_threshold] = [255, 0, 0]

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sift():
    """
    图像尺度空间变换
    Scale Invariant Feature Transform (sift)
    :return:
    """
    img = cv2.imread('./data/lena.jpg')
    img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    sift_obj = cv2.SIFT_create()
    kp = sift_obj.detect(img_gray, None)

    img_kp = cv2.drawKeypoints(image=img_gray, keypoints=kp, outImage=None)

    cv2.imshow('draw_key_points', img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def feature_match():
    """
    特征匹配
    :return:
    """
    img_box = cv2.imread(filename='./data/box.png', flags=cv2.IMREAD_GRAYSCALE)
    img_boxes = cv2.imread(filename='./data/box_in_scene.png', flags=cv2.IMREAD_GRAYSCALE)

    # 检测两图片关键点
    sift_obj = cv2.SIFT_create(nfeatures=500)
    kp_1, des_1 = sift_obj.detectAndCompute(image=img_box, mask=None)
    kp_2, des_2 = sift_obj.detectAndCompute(image=img_boxes, mask=None)

    bf = cv2.BFMatcher()
    # 1、暴力匹配
    matches = bf.match(queryDescriptors=des_1, trainDescriptors=des_2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:20]  # 选取匹配最佳的一些

    img_matches = cv2.drawMatches(img1=img_box, keypoints1=kp_1, img2=img_boxes, keypoints2=kp_2, matches1to2=matches,
                                  outImg=None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('img_matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2、knn 匹配(返回匹配对，每对k个匹配)
    matches = bf.knnMatch(queryDescriptors=des_1, trainDescriptors=des_2, k=2)
    good_matches = list()
    for item_group in matches:
        if len(item_group) < 2:
            # 最近邻不足k个
            continue
        # 每个匹配对的前k个匹配
        m1, m2 = item_group[:2]
        # m1 < a * m2
        # 距离越近，点对匹配度越高；如果匹配度最高的点对距离远小于第二匹配度点对，则第一个点对（匹配对）的可靠度较高
        if m1.distance < 0.5 * m2.distance:
            good_matches.append(m1)

    if len(good_matches) < 2:
        print('未检测到匹配目标及关键点')

    img_knn_matches = cv2.drawMatches(img1=img_box, keypoints1=kp_1, img2=img_boxes, keypoints2=kp_2,
                                      matches1to2=good_matches, outImg=None,
                                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('img_knn_matches', img_knn_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_stitcher():
    """
    图片拼接
    :return:
    """
    img_a = cv2.imread('./data/img_left.png')
    img_b = cv2.imread('./data/img_right.png')
    img_a_row, img_a_col, img_a_channel = img_a.shape
    img_b_row, img_b_col, img_b_channel = img_b.shape
    # 转为灰度图
    img_a_gray = cv2.cvtColor(src=img_a, code=cv2.COLOR_BGR2GRAY)
    img_b_gray = cv2.cvtColor(src=img_b, code=cv2.COLOR_BGR2GRAY)
    # 检测出关键点和其描述
    sift_obj = cv2.SIFT_create()
    kp_a, des_a = sift_obj.detectAndCompute(image=img_a_gray, mask=None)
    kp_b, des_b = sift_obj.detectAndCompute(image=img_b_gray, mask=None)

    kp_a_pts = np.float32([kp.pt for kp in kp_a])
    kp_b_pts = np.float32([kp.pt for kp in kp_b])

    # 暴力匹配器
    bf_matcher = cv2.BFMatcher()
    # knn 匹配算法
    matches = bf_matcher.knnMatch(queryDescriptors=des_b, trainDescriptors=des_a, k=2)

    good_matches = list()
    good_kp_a_list = list()
    good_kp_b_list = list()
    for item_group in matches:
        if len(item_group) < 2:
            # 最近邻不足k个
            continue
        # 每个匹配对的前k个匹配
        m1, m2 = item_group[:2]
        # m1 < λ * m2
        # 距离越近，点对匹配度越高；如果匹配度最高的点对距离远小于第二匹配度点对，则第一个点对（匹配对）的可靠度较高
        if m1.distance < 0.7 * m2.distance:
            good_matches.append(m1)
            good_kp_a_list.append(kp_a_pts[m1.trainIdx])
            good_kp_b_list.append(kp_b_pts[m1.queryIdx])

    if len(good_matches) < 4:
        print('未检测到匹配目标及关键点')
        return

    good_kp_a_np = np.float32(good_kp_a_list)
    good_kp_b_np = np.float32(good_kp_b_list)
    # 计算视角变换矩阵 (h_matrix 为3*3视角变换矩阵)
    # 计算多个二维点对之间的最优单映射变换矩阵H
    h_matrix, mask = cv2.findHomography(srcPoints=good_kp_b_np, dstPoints=good_kp_a_np,
                                        method=cv2.RANSAC, ransacReprojThreshold=4.0)

    # 透视
    # (先透视拉伸右图，在将左图直接放入位置)
    img_result = cv2.warpPerspective(src=img_b, M=h_matrix, dsize=(img_b_col + img_a_col + 180, img_b_row))
    # img_result = cv2.warpPerspective(src=img_b, M=h_matrix, dsize=(img_b_col, img_b_row))
    img_result[0: img_a_row, 0: img_a_col] = img_a

    # 绘制匹配点
    img_a_b = np.hstack((img_a, img_b))
    for gm, m in zip(good_matches, mask):
        if m == 1:
            pt_1 = (int(kp_a_pts[gm.trainIdx][0]), int(kp_a_pts[gm.trainIdx][1]))
            pt_2 = (int(kp_b_pts[gm.queryIdx][0]) + img_a_col, int(kp_b_pts[gm.queryIdx][1]))
            cv2.line(img=img_a_b, pt1=pt_1, pt2=pt_2,
                     color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                     thickness=1)

    cv2.imshow('img_result', img_result)
    cv2.imshow('img_a_b', img_a_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_stitcher_2():
    """
    图片拼接
    cv2内置方法
    :return:
    """
    # 放入待拼接的图，按照从左到右的顺序
    images = [cv2.imread('./data/img_left.png'), cv2.imread('./data/img_right.png')]

    stitcher = cv2.Stitcher().create()
    status, panorama = stitcher.stitch(images)
    if status == cv2.STITCHER_OK:
        cv2.imshow('panorama', panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('无法拼接为全景图')


if __name__ == '__main__':
    # harris()
    # sift()
    # feature_match()
    # img_stitcher()
    img_stitcher_2()
