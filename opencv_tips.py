OpenCV转换成PIL.Image格式：


import cv2  
from PIL import Image  
import numpy  
  
img = cv2.imread("plane.jpg")  
cv2.imshow("OpenCV",img)  
image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
image.show()  
cv2.waitKey()  


PIL.Image转换成OpenCV格式：

import cv2  
from PIL import Image  
import numpy  
  
image = Image.open("plane.jpg")  
image.show()  
img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)  
cv2.imshow("OpenCV",img)  
cv2.waitKey()  



# 使用ostu自动检测图像Binarization的阈值
image1 = cv2.imread("HITACHI-part+250.png")
(ori_h,ori_w,c) = image1.shape
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
ret1,th1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #因为此处自动选择阈值，所以参数设置为0，255就行
  
  #将大图分割成小图  
    img_name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            # var = np.var(crop_img / 255)
            # if var > 0.008:
            #     print(img_name, index_str, var)
            cv2.imwrite(
                os.path.join(save_folder, img_name.replace('.png', '_s{:03d}.png'.format(index))),
                crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    
    return 'Processing {:s} ...'.format(img_name)
  
  # 将BGR转为RGB
 # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]] 
  
  
  #rgb2ycbcr
  def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
  
  
#使用mask来在原图上画出轮廓区域
import cv2
import numpy as np

bg = cv2.imread('./img_0213_i.png', cv2.IMREAD_GRAYSCALE)
backtorgb = cv2.cvtColor(bg,cv2.COLOR_GRAY2RGB)
# cv2.imwrite('./out.png', backtorgb)
mask = cv2.imread('./img_0213_p.png', cv2.IMREAD_GRAYSCALE)
mask = (mask >= 200)
# print(mask)
roi = backtorgb
roi = roi[mask]
# print(roi.shape)
c = '0,0,255'
COLORS = np.array(c.split(",")).astype("int")
COLORS = np.array(COLORS, dtype="uint8")
# color = (0,255,0)
blended = ((0.1 * COLORS) + (0.9 * roi)).astype("uint8")
print(blended)
backtorgb[mask] = blended
cv2.imwrite('./out.png', backtorgb)
