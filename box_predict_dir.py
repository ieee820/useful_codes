# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import text_detect
from apphelper.image import rotate_cut_img, solve
import ntpath
from PIL import Image
from glob import glob
import re


config=dict(MAX_HORIZONTAL_GAP=100,##字符之间的最大间隔，用于文本行的合并
                                    MIN_V_OVERLAPS=0.7,
                                    MIN_SIZE_SIM=0.7,
                                    TEXT_PROPOSALS_MIN_SCORE=0.1,
                                    TEXT_PROPOSALS_NMS_THRESH=0.3,
                                    TEXT_LINE_NMS_THRESH = 0.99,##文本行之间测iou值
                                    MIN_RATIO=1.0,
                                    LINE_MIN_SCORE=0.2,
                                    TEXT_PROPOSALS_WIDTH=0,
                                    MIN_NUM_PROPOSALS=0,
                )


def romvChinese(context):
    # context = context.decode("utf-8") # convert context from str to unicode
    filtrate = re.compile(u'[\u4E00-\u9FA5]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
    # context = context.encode("utf-8") # convert unicode back to str
    return context


imglistPNG = glob('/data/share_data/junling_img/input/*.png')
imglistJPG = glob('/data/share_data/junling_img/input/*.jpg')
imglist = imglistPNG + imglistJPG
subdir_interval = 1000
line_outputPath = '/data/share_data/junling_img/output'
img_count = 0
# make sub dir
# subdir = os.path.join(line_outputPath, str(img_count))
# if not os.path.isdir(subdir):
#     os.makedirs(subdir)
subdir = os.path.join(line_outputPath, str(img_count))
if not os.path.isdir(subdir):
    os.makedirs(subdir)
    print(subdir)

for imgPath in imglist:
    # print('img_count: ', img_count)


    img = Image.open(imgPath).convert("RGB")

    config['img'] = img
    text_recs = text_detect(**config)
    # print(text_recs)
    boxes = sorted(text_recs,key=lambda x:sum([x[1],x[3],x[5],x[7]]))
    i = 0
    filename = ntpath.basename(imgPath)
    ori_filename = romvChinese(filename)

    for index, box in enumerate(boxes):
        filename = ori_filename
        degree, w, h, cx, cy = solve(box)
        partImg, newW, newH = rotate_cut_img(img, degree, box, w, h, leftAdjust=True, rightAdjust=True, alph=0.2)
        if partImg.size[1] < 32:
            scale = partImg.size[1] * 1.0 / 32
            w = partImg.size[0] / scale
            w = int(w)
            partImg = partImg.resize((w,32), Image.BILINEAR)
        filename = filename[:-4] + '_' + str(i) + '_.jpg'
        partImgPath = os.path.join(subdir, filename)
        partImg.save(partImgPath)
        i += 1
        img_count += 1
        # print('img_count: ', img_count)
        if img_count % subdir_interval == 0:
            subdir = os.path.join(line_outputPath, str(img_count))
            if not os.path.isdir(subdir):
                os.makedirs(subdir)
                print(subdir)


