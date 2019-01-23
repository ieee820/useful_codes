#coding:utf-8
from PIL import Image, ImageDraw, ImageFont
import random
import os
import numpy as np
import cv2
from imgaug import augmenters as iaa

def add_salt_pepper(img, fill=(255, 255, 255), num_ratio=None):
    """
    Args:
        img: PIL.Image type
        fill: noise color, tuple of length 3, default: Write color
        num: noise spot number, int
    Return:
        img: PIL.Image type
    """
    if num_ratio is None:
        num_ratio = np.random.uniform(0.06, 0.15)
    num = int(img.size[0] * img.size[1] * num_ratio)
    pix = img.load()
    for k in range(num):
        x = int(np.random.uniform(0, img.size[0]))
        y = int(np.random.uniform(0, img.size[1]))
        r, g, b = pix[x, y]
        img.putpixel([x, y], fill)
    return img


def dither(num, thresh=127):
    derr = np.zeros(num.shape, dtype=int)

    div = 8
    for y in range(num.shape[0]):
        for x in range(num.shape[1]):
            newval = derr[y, x] + num[y, x]
            if newval >= thresh:
                errval = newval - 255
                num[y, x] = 1.
            else:
                errval = newval
                num[y, x] = 0.
            if x + 1 < num.shape[1]:
                derr[y, x + 1] += errval / div
                if x + 2 < num.shape[1]:
                    derr[y, x + 2] += errval / div
            if y + 1 < num.shape[0]:
                derr[y + 1, x - 1] += errval / div
                derr[y + 1, x] += errval / div
                if y + 2 < num.shape[0]:
                    derr[y + 2, x] += errval / div
                if x + 1 < num.shape[1]:
                    derr[y + 1, x + 1] += errval / div
    return num[::-1, :] * 255


def hyperdither(img):
    bottom = False
    if bottom:
        m = np.array(img)[::-1, :]
        m2 = dither(m, thresh=127)
        out = Image.fromarray(m2[:, :])
    else:
        m = np.array(img)[:, :]
        m2 = dither(m, thresh=127)
        out = Image.fromarray(m2[::-1, :])
    return out


index = 0
fonts = 0
#/data/yjj/OCR-Picture-Generators-master/fonts_Chinese
fontDir = "./fonts_Chinese"
font_dir = os.walk(fontDir)

for root, dirs, files in font_dir:
    for f in files:
        fonts = fonts + 1

#print(fonts)

def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())

    return res

image_label = readfile('./text_demo.txt')
print(image_label)

for line in image_label:
    rootDir = './Piaoju-background'
    list_dir = os.walk(rootDir)

    for root, d, fs in list_dir:
        for file in fs:
            image_path = os.path.join(root, file)
            # print(image_path)
            im = Image.open(image_path)
            width, height = im.size
            # print(im.size)
            # print(width)
            # print(height)
            # print(im.mode)
            draw = ImageDraw.Draw(im)
            length = len(line)
            chineseLength = length / 3
            remainder = length % 3

            if (remainder > 0):
                length = chineseLength + 1
            else:
                length = chineseLength

            fontsize = random.randint(20, 40)

            remainderY = height - fontsize - 15
            if (remainderY <= 0):
                fontsize = height - 13
                remainderY = 3

            remainderX = width - length * (fontsize + 15)
            if (remainderX <= 0):
                length = int(width / (fontsize + 15)) - 1
                if (length <= 0):
                    print('img length skip: ', image_path, 'str: ', line)
                    continue
                remainderX = width - length * (fontsize + 15)
                if (remainderX <= 5):
                    remainderX = 6

            ll = length;

            # random select font
            font = random.randint(1, fonts)
            font_select = 1
            fontDir = "./fonts_Chinese"
            font_dir = os.walk(fontDir)
            path = ""

            for roo, dirs, files in font_dir:
                for f in files:
                    if (font_select == font):
                        path = os.path.join(roo, f)
                        # print(path)
                        break
                    font_select = font_select + 1

            ttfont = ImageFont.truetype(path, fontsize)

            try:
                radX = random.randint(0, 100)
            except:
                print('radX over range')
                continue
            try:
                radY = random.randint(0, 20)
            except:
                print('radY over range')
                continue
            color_value = random.randint(5, 100)
            draw.text((4 + radX, 2 + radY), line, fill=(color_value, color_value, color_value), font=ttfont)
            t_width, t_height = draw.textsize(line, font=ttfont)

            rX = 8 + radX + t_width
            rY = 12 + radY + t_height

            box = (radX, radY, rX, rY)
            region = im.crop(box)
            if (radX + rX) >= width:
                print("over flow width, skip: ")
                continue
            index = index + 1
            ucode = ord(line[0])
            filename = './temp_line_img/' + str(ucode) + '_' + str(index) + '.jpg'
            select = random.randint(0, 20)
            print('select: ', select)
            # 加入白色的椒盐噪声
            if select < 3:
                add_salt_pepper(region)
                print('white salt peper: ', filename)

            # 加入黑色的椒盐噪声
            if select > 7 and select < 10:
                add_salt_pepper(region, (0, 0, 0))
                print('black salt pepper: ', filename)
            # 是否添加高斯噪声或扫描件噪声
            if select >= 3 and select <= 6:
                # 添加高斯背景噪声
                background_gauss = np.ones((region.size[1], region.size[0])) * 255
                cv2.randn(background_gauss, 235, 10)
                background_gauss = Image.fromarray(background_gauss).convert('L')
                region = region.convert('L')
                mask = region.point(lambda x: 0 if x == 255 or x == 0 else 255, '1')
                # mx, my = mask.size
                background_gauss.paste(region, (0, 0), mask=mask)
                region = background_gauss
                print('gauss noise: ', filename)

            # 添加扫描件抖动噪声
            if select >= 10 and select < 12:
                if fontsize > 25:
                    region = region.convert('L')
                    region = hyperdither(region)
                    print('scan noise: ', filename)

            if select >= 12:
                numpy_region = np.asarray(region)
                blurer = iaa.GaussianBlur(1.5)
                numpy_region = blurer.augment_image(numpy_region)
                region = Image.fromarray(numpy_region)

            region.save(filename)
            # make label.txt file for every img
            des_label_file_path = filename.replace('jpg', 'txt')
            with open(des_label_file_path, 'w') as text_file:
                text_file.write(line)

