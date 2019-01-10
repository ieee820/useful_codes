#!/usr/bin/python
# encoding: utf-8
import shutil
import os
from crnn.keys import alphabetChinese

def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic

def checkChar(label, alphdict):
    for x in label:
        if x not in alphdict:
            return ''

    return label


image_label = readfile('./360W_lable_zh/360_train_randn.txt')
src_dir = '/data/share_data/360W'
des_dir = '/data/share_data/crnn_train/gen_from_360w/train_60w'
counter = 0
for img_name, img_l in image_label.items():
    if counter == 600000:
        print('10000 data gen complete, exit loop .')
        break
    label_checked = checkChar(img_l[0], alphabetChinese)
    if label_checked == '':
       print(img_name, 'label has char not in alphabetChinese: ', img_l[0])
       continue  #when illegal char , don't copy the img to des dir

    #copy img to des dir , dir name == prefix
    output_dir = os.path.join(des_dir, img_name[:4])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    src_file_path = os.path.join(src_dir, img_name)
    des_file_path = os.path.join(output_dir, img_name)
    shutil.copy(src_file_path, des_file_path)
    #make label.txt file for every img
    des_label_file_path = des_file_path.replace('jpg', 'txt')
    with open(des_label_file_path, 'w') as text_file:
        text_file.write(img_l[0])

    counter += 1
