import cv2 as cv
import numpy as np


def handle_uploaded_file(f):
    if f:
        with open('./images_uploaded/1.jpg', 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
# 窗口设置与居中展示
# cv.namedWindow("*",0)
# cv.resizeWindow("*",int(zoom_width),int(zoom_height))
# cv.moveWindow("*",int(960-zoom_width/2),int(540-zoom_height/2))
# cv.imshow("*",*)


def img_process_opencv():
    img = cv.imread('./images_uploaded/1.jpg')
    img_width = img.shape[1]
    img_height = img.shape[0]
    # 缩放倍率
    zoom = max(img_width, img_height)/720
    if zoom < 1:
        zoom = 1
    zoom_width = img_width/zoom
    zoom_height = img_height/zoom
    cv.namedWindow("img", 0)
    cv.resizeWindow("img", int(zoom_width), int(zoom_height))
    cv.moveWindow("img", int(960-zoom_width/2), int(540-zoom_height/2))
    cv.imshow("img", img)

    # 灰度化
    img_grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # 缩放(避免图像分辨率过大时直线检测过于苛刻)
    if zoom > 1:
        img_grey = cv.resize(img_grey, (int(zoom_width), int(
            zoom_height)), interpolation=cv.INTER_CUBIC)

    # 高斯滤波
    c = 7
    img_gsb = cv.GaussianBlur(img_grey, (c, c), 0)

    # 闭运算
    d = 15
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (d, d))
    img_close = cv.morphologyEx(img_gsb, cv.MORPH_CLOSE, element)

    # 边缘提取
    img_canny = cv.Canny(img_close, 0, 40)
    cv.namedWindow("img_canny", 0)
    cv.resizeWindow("img_canny", int(zoom_width), int(zoom_height))
    cv.moveWindow("img_canny", int(960-zoom_width/2), int(540-zoom_height/2))
    cv.imshow("img_canny", img_canny)

    # # 霍夫变换+划线
    lines = cv.HoughLinesP(img_canny, 1, np.pi/180, int(min(zoom_width/4, zoom_height/4)),
                           minLineLength=min(zoom_height, zoom_width)/8, maxLineGap=max(zoom_width, zoom_height))

    lines = lines[:, 0, :]
    img_line = img.copy()
    for x1, y1, x2, y2 in lines:
        x1 = int(x1*zoom)
        y1 = int(y1*zoom)
        x2 = int(x2*zoom)
        y2 = int(y2*zoom)
        cv.line(img_line, (x1, y1), (x2, y2), (0, 255, 0), 1)
    # cv.namedWindow("img_line", 0)
    # cv.resizeWindow("img_line", int(zoom_width), int(zoom_height))
    # cv.moveWindow("img_line", int(960-zoom_width/2), int(540-zoom_height/2))
    # cv.imshow("img_line", img_line)

    # # 求信息区域四角顶点
    # vtx = np.float32([[img_width/2, img_height/2], [img_width/2, img_height/2],
    #                   [img_width/2, img_height/2], [img_width/2, img_height/2]])
    # for x1, y1, x2, y2 in lines:
    #     rad1 = np.arctan2(y2-y1, x2-x1)
    #     for x3, y3, x4, y4 in lines:
    #         rad2 = np.arctan2(y4-y3, x4-x3)
    #         if abs(rad2-rad1) > np.pi/3 and abs(rad2-rad1) < np.pi*2/3 or abs(rad2-rad1) > 4*np.pi/3 and abs(rad2-rad1) < np.pi*5/3:  # 排除相近直线 阈值pi/3
    #             k = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    #             x0 = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))*zoom/k
    #             y0 = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))*zoom/k
    #             if x0 >= 0 and x0 <= img_width and y0 >= 0 and y0 <= img_height:
    #                 if x0+y0 < vtx[0][0]+vtx[0][1]:
    #                     vtx[0] = [x0, y0]
    #                 if x0-y0 > vtx[1][0]-vtx[1][1]:
    #                     vtx[1] = [x0, y0]
    #                 if x0-y0 < vtx[2][0]-vtx[2][1]:
    #                     vtx[2] = [x0, y0]
    #                 if x0+y0 > vtx[3][0]+vtx[3][1]:
    #                     vtx[3] = [x0, y0]

    # # 透视变换
    # height = max(np.sqrt(np.square(vtx[0][0]-vtx[2][0])+np.square(vtx[0][1]-vtx[2][1])),
    #              np.sqrt(np.square(vtx[1][0]-vtx[3][0])+np.square(vtx[1][1]-vtx[3][1])))
    # width = max(np.sqrt(np.square(vtx[0][0]-vtx[1][0])+np.square(vtx[0][1]-vtx[1][1])),
    #             np.sqrt(np.square(vtx[2][0]-vtx[3][0])+np.square(vtx[2][1]-vtx[3][1])))
    # vtx1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # mtx = cv.getPerspectiveTransform(vtx, vtx1)
    # info = cv.warpPerspective(img, mtx, (width, height))
    with open('./images_processed/1.jpg', 'wb+') as destination:
        cv.imwrite('./images_processed/1.jpg', img_canny)
        return destination.read()
# cv.namedWindow("info", 0)
# cv.resizeWindow("info", int(width), int(height))
# cv.moveWindow("info", int(960-width/2), int(540-height/2))

# cv.imshow("info", info)
# cv.waitKey()
# cv.destroyAllWindows()
