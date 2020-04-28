import math
from skimage import color
from PIL import Image
import os
import cv2 as cv
import numpy as np


def handle_uploaded_file(f):
    if f:
        with open('./images_uploaded/1.jpg', 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)


"""读取图片(仅用于调试)

Returns:
    img: 图像矩阵
    img_width,img_height:图像长宽
"""


def read_img(path):
    img = cv.imread(path)
    img_width = img.shape[1]
    img_height = img.shape[0]
    return img, img_width, img_height


"""插值缩小

Returns:
    img_grey: 灰度矩阵
    zoom_width, zoom_height: 缩小后长宽
    zoom: 缩小倍数
"""


def insertion_zoom(img, width, height):
    zoom = max(width, height)/720
    if zoom < 1:
        zoom = 1
    zoom_width = width/zoom
    zoom_height = height/zoom
    # 灰度化
    img_grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # 缩放(避免图像分辨率过大时直线检测过于苛刻)
    if zoom > 1:
        img_grey = cv.resize(img_grey, (int(zoom_width), int(
            zoom_height)), interpolation=cv.INTER_CUBIC)
    return img_grey, zoom_width, zoom_height, zoom


"""自适应canny算法

Returns:
    img_canny: 边缘图
"""


def adaptive_canny(img):
    # 先对灰度图进行预处理
    # 高斯滤波
    img_gsb = cv.GaussianBlur(img, (7, 7), 0)

    # 闭运算
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    img_close = cv.morphologyEx(img_gsb, cv.MORPH_CLOSE, element)
    # 自适应canny算法
    ret3, th3 = cv.threshold(
        img_close, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    img_canny = cv.Canny(img_close, 0, ret3/2)
    return img_canny


"""概率霍夫直线检测

Returns:
    img_line: 霍夫变换直线图
"""


def hough_transform(origin, img_canny, zoom_width, zoom_height, zoom):

    # 霍夫直线检测
    # 霍夫变换+划线
    lines = cv.HoughLinesP(img_canny, 1, np.pi/180, int(min(zoom_width/4, zoom_height/4)),
                           minLineLength=135, maxLineGap=1080)
    lines = lines[:, 0, :]
    img_line = origin.copy()
    for x1, y1, x2, y2 in lines:
        x1 = int(x1*zoom)
        y1 = int(y1*zoom)
        x2 = int(x2*zoom)
        y2 = int(y2*zoom)
        cv.line(img_line, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return lines


"""角点检测

Returns:
    vtx: 角点集合
"""


def skew_correction(lines, img_width, img_height, zoom):
    # 求信息区域四角顶点
    vtx = np.float32([[img_width/2, img_height/2], [img_width/2, img_height/2],
                      [img_width/2, img_height/2], [img_width/2, img_height/2]])
    for x1, y1, x2, y2 in lines:
        rad1 = np.arctan2(y2-y1, x2-x1)
        for x3, y3, x4, y4 in lines:
            rad2 = np.arctan2(y4-y3, x4-x3)
            if abs(rad2-rad1) > np.pi/3 and abs(rad2-rad1) < np.pi*2/3 or abs(rad2-rad1) > 4*np.pi/3 and abs(rad2-rad1) < np.pi*5/3:  # 排除相近直线 阈值pi/3
                k = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                x0 = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))*zoom/k
                y0 = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))*zoom/k
                if x0 >= 0 and x0 <= img_width and y0 >= 0 and y0 <= img_height:
                    if x0+y0 < vtx[0, 0]+vtx[0, 1]:
                        vtx[0] = [x0, y0]
                    if x0-y0 > vtx[1, 0]-vtx[1, 1]:
                        vtx[1] = [x0, y0]
                    if x0-y0 < vtx[2, 0]-vtx[2, 1]:
                        vtx[2] = [x0, y0]
                    if x0+y0 > vtx[3][0]+vtx[3][1]:
                        vtx[3] = [x0, y0]
    return vtx


"""透视变换

Returns:
    info: 信息区域图
"""


def perspective_transform(img, vtx):
    # 透视变换
    height = max(np.sqrt(np.square(vtx[0][0]-vtx[2][0])+np.square(vtx[0][1]-vtx[2][1])),
                 np.sqrt(np.square(vtx[1][0]-vtx[3][0])+np.square(vtx[1][1]-vtx[3][1])))
    width = max(np.sqrt(np.square(vtx[0][0]-vtx[1][0])+np.square(vtx[0][1]-vtx[1][1])),
                np.sqrt(np.square(vtx[2][0]-vtx[3][0])+np.square(vtx[2][1]-vtx[3][1])))
    vtx1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    mtx = cv.getPerspectiveTransform(vtx, vtx1)
    info = cv.warpPerspective(img, mtx, (width, height))
    return info


"""光照不均匀补偿

Returns:
    rgb: 光照补偿图
"""


def lighting_uneven(info, info_height, info_width):
    # 将rgb图像转换为hsv图像
    hsv = color.rgb2hsv(info)
    # 分离hsv通道
    h, s, v = cv.split(hsv)

    size = 32
    info_row = int(np.ceil(info_height/size)+1)
    info_col = int(np.ceil(info_width/size)+1)
    info_mean = np.empty((info_row, info_col))
    info_std = np.empty((info_row, info_col))
    info_bgb = np.empty((info_row, info_col))

    for i in range(info_row):
        rowmin = (i-1)*size
        rowmax = (i+1)*size
        if rowmin < 0:
            rowmin = 0
        if rowmax > info_height:
            rowmax = info_height
        for j in range(info_col):
            colmin = (j-1)*size
            colmax = (j+1)*size
            if colmin < 0:
                colmin = 0
            if colmax > info_width:
                colmax = info_width
            block = v[rowmin:rowmax, colmin:colmax]
            info_mean[i, j] = np.mean(block)
            info_std[i, j] = np.std(block)
            info_bgb[i, j] = info_mean[i, j]+info_std[i, j]/2
    info_tg = np.mean(info_bgb)+np.std(info_bgb)
    for i in range(info_row):
        for j in range(info_col):
            info_bgb[i, j] = info_tg/info_bgb[i, j]
    info_bgb = cv.resize(info_bgb, (info_width, info_height),
                         interpolation=cv.INTER_CUBIC)
    v = info_bgb*v
    hsv1 = cv.merge([h, s, v])
    rgb = color.hsv2rgb(hsv1)
    rgb = rgb / rgb.max()
    rgb = 255 * rgb
    rgb = rgb.astype(np.uint8)
    return rgb


"""去除图像边缘

Returns:
    img: 去除边缘后的图
"""


def remove_edge(img, info_height, info_width):
    codn = []
    for i in range(info_height):
        codn.append([i, 0])
        codn.append([i, info_width-1])
    for j in range(info_width):
        codn.append([0, j])
        codn.append([info_height-1, j])
    while len(codn) > 0:
        arr = codn.pop()
        if img[arr[0], arr[1], 0] != 255 or img[arr[0], arr[1], 1] != 255 or img[arr[0], arr[1], 2] != 255:
            img[arr[0], arr[1]] = [255, 255, 255]
            codn.append([max(0, arr[0]-1), arr[1]])
            codn.append([min(info_height-1, arr[0]+1), arr[1]])
            codn.append([arr[0], max(0, arr[1]-1)])
            codn.append([arr[0], min(info_width-1, arr[1]+1)])
    return img


"""自适应阈值

Returns:
    out: 结果图
"""


def countAdaptiveThresh(I, winSize, ratio=0.15):
    # 均值平滑
    I_mean = cv.boxFilter(I, cv.CV_32FC1, winSize)
    # 原图矩阵和平滑结果作差
    out = I - (1.0-ratio)*I_mean
    out[out >= 0] = 255
    out[out < 0] = 0
    out = out.astype(np.uint8)
    return out


def img_process_opencv(bytes):
    nparr = np.fromstring(bytes, np.uint8)
    img_np = cv.imdecode(nparr, cv.IMREAD_COLOR)
    img = img_np
    img_width = img.shape[1]
    img_height = img.shape[0]
    # img, img_width, img_height = read_img(file)
    img_grey, zoom_width, zoom_height, zoom = insertion_zoom(
        img, img_width, img_height)
    img_canny = adaptive_canny(img_grey)
    lines = hough_transform(img, img_canny, zoom_width, zoom_height, zoom)
    vtx = skew_correction(lines, img_width, img_height, zoom)
    info = perspective_transform(img, vtx)
    rgb = lighting_uneven(info, info.shape[0], info.shape[1])
    res = countAdaptiveThresh(rgb, (31, 31), 0.07)
    info_height = info.shape[0]
    info_width = info.shape[1]
    _res = remove_edge(res, info_height, info_width)
    for i in range(info_height):
        for j in range(info_width):
            if _res[i, j, 0] == 255:
                rgb[i, j] = _res[i, j]
    _hash = abs(hash(rgb.tobytes()))
    path = "./images_handled/"+str(_hash)+".jpg"
    cv.imwrite(path, rgb)
    return path, _hash
