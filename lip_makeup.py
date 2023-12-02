from pylab import *
from skimage import color
import dlib
import cv2
import os
import numpy as np
from scipy.interpolate import CubicSpline

data_path = "./imgs"



def write_file(lips):
    img, _ = os.path.splitext(img_name)
     # 指定要寫入的 txt 檔案名稱
    file_name = f"{img}_point.txt"

    # 開啟 txt 檔案以寫入模式
    with open(f'./landmark/{file_name}', "w") as file:
        for point in lips:
            x, y = point.x, point.y
            # 將每個座標點寫入文件，並以空格或其他分隔符分隔
            file.write(f"{x} {y}\n")  # 使用換行符分隔每個座標點

    print(f"座標已寫入 {file_name}")

def detect():
    # 初始化 dlib 的臉部檢測器和特徵點檢測器
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")

    # 讀取靜態圖像
    image = cv2.imread(f"{img_path}")

    # 將圖像轉換為灰度圖像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用 dlib 的臉部檢測器檢測人臉位置
    faces = face_detector(gray)

    for face in faces:
        # 使用 dlib 的特徵點檢測器檢測 68 個臉部特徵點的位置
        landmarks = shape_predictor(gray, face)

        # 通常，嘴唇的外部輪廓點位於特徵點索引 48 到 67 之間
        lips = landmarks.parts()[48:68]

    write_file(lips)

def inter(landmark_y, landmark_x):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 對嘴唇關键點進行插值
    lip_x = landmark_x
    lip_y = landmark_y
    cubic_interp_x = CubicSpline(np.arange(len(lip_x)), lip_x)
    cubic_interp_y = CubicSpline(np.arange(len(lip_y)), lip_y)

    # 生成插值的點
    interp_points_x = cubic_interp_x(np.linspace(0, len(lip_x)-1, 5000))
    interp_points_y = cubic_interp_y(np.linspace(0, len(lip_y)-1, 5000))

    # 將差值轉為整數
    interp_points = np.column_stack((interp_points_x, interp_points_y)).astype(int)

    # 創建 mask
    mask = np.zeros_like(gray)

    # 在嘴唇區域内部设定為白色（255）
    cv2.fillPoly(mask, [interp_points], color = (255))

    # 獲取 mask 中值為白色的坐标
    coordinates = np.column_stack(np.where(mask == 255))
    x = coordinates[:, 1]
    y = coordinates[:, 0]
    return x, y


def color_switch(co_y, co_x):
    original_im = imread(img_path)
    val = color.rgb2lab((original_im[co_y, co_x] / 255.).reshape(len(co_x), 1, 3)).reshape(len(co_x), 3)
    L, A, B = mean(val[:, 0]), mean(val[:, 1]), mean(val[:, 2])
    L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
    ll, aa, bb = L1 - L, A1 - A, B1 - B
    val[:, 0] += ll
    val[:, 1] += aa
    val[:, 2] += bb
    new_im = original_im.copy()
    new_im[co_y, co_x] = color.lab2rgb(val.reshape(len(co_x), 1, 3)).reshape(len(co_x), 3) * 255
    gca().set_aspect('equal', adjustable='box')
    combine_img = np.concatenate((original_im, new_im), axis=1)
    imshow(combine_img)
    show()
    img, _ = os.path.splitext(img_name)
    imsave(f'{img}_output.jpg', new_im)
    print(f'已寫入{img}_output')



def main():
    img, _ = os.path.splitext(img_name)
    detect()
    landmark = np.loadtxt(f'./landmark/{img}_point.txt')
    y, x = landmark[:, 0], landmark[:, 1]
    co_y, co_x = inter(y, x)
    color_switch(co_y, co_x)


if __name__ == '__main__':
    # 要換照片改下面的變數就好
    # 照片要放在imgs裡面
    img_name = 'star7.jpg'
    img_path = f'./imgs/{img_name}'
    # 改變口紅顏色
    r, g, b = (207., 40., 57.)  # lipstick color
    main()
