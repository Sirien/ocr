# !/usr/bin/env python3

import os
import random
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import pickle

_rect = [280, 10, 450, 140]

# front registed
_images_dir = "./front"

class DataGenerater(object):
    def __init__(self, font_path, pool_num=100, src_dir=_images_dir, char_size=(20, 20)):
        fns = os.listdir(_images_dir)
        fns_selected = random.sample(fns, pool_num)
        imgs = [cv2.imread(os.path.join(_images_dir, fn), cv2.CV_8UC1) for fn in fns_selected]
        x0,y0, x1,y1 = _rect
        self.src_pool = [img[y0:y1,x0:x1] for img in imgs]
        self.w, self.h = char_size
        self.font = ImageFont.truetype(font_path, int(self.w * 0.95),)

    def gen_bg(self, num=10, bg_size=(20, 20)):
        bgs = []
        self.bg_w,self.bg_h = bg_size
        for i in range(num):
            img = random.choice(self.src_pool)
            assert img.shape > bg_size, "bg shape error"

            _y = random.randint(0, img.shape[0]-bg_size[1])
            _x = random.randint(0, img.shape[1]-bg_size[0])
            img_crop = img[_y:_y+bg_size[1], _x:_x+bg_size[0]]

            img_crop = cv2.resize(img_crop, (self.w, self.h))

            bgs.append(img_crop)
        return bgs

    def add_text(self, img, char, pos=None, rotate=None):
        if not pos:
            pos = (self.w*0.15, self.h*0.15)
        img_p = Image.fromarray(img)
        draw = ImageDraw.Draw(img_p)
        draw.text(pos, char, 0, font=self.font)
        if rotate:
            img_p = img_p.rotate(angle=rotate)
        img = np.array(img_p)
        return img

    def gen_text_mask(self, char, pos=None, bias=255):
        white = np.zeros((self.h, self.w), dtype=np.uint8)+bias
        img = self.add_text(white, char, pos)
        p = random.random()
        if p < 0.2:
            kernel = 20 + np.int(50*p)
        elif p < 0.4:
            kernel = 30 + np.int((p-0.4)*50)
        elif p < 0.6:
            kernel = 40 + np.int((p-0.6)*50)
        elif p < 0.8:
            kernel = 50 + np.int((p-0.8)*50)
        elif p < 0.9:
            kernel = 60 + np.int((p-0.9)*100)
        else:
            kernel = 70 + np.int((p-0.95)*200)
        if kernel%2 == 0:
            kernel += 1
        # kernel += 10
        img = cv2.GaussianBlur(img,(kernel, kernel),0)
        return img

    def to_image(self, bg, char):
        ratio = (random.uniform(-0.05, 0.05),random.uniform(-0.05, 0.05))
        pos = (self.w*ratio[0], self.h*ratio[1])

        img = self.gen_text_mask(char, bias=255, pos=pos)
        img = img/255.0
        img = np.power(img, 0.6)
        bg = bg/255.0
        bg = img*bg*255.0
        bg = bg.astype(np.uint8)

        bg = cv2.resize(bg,(self.bg_w,self.bg_h),interpolation=cv2.INTER_LINEAR)
        return bg


def get_label_dict():
    f=open('./ocr/chinese_labels','r')
    label_dict = pickle.load(f)
    f.close()
    return label_dict

def main():
    font_path = "./ocr/chinese_fonts/fangzheng_heiti.TTF"
    DG = DataGenerater(char_size=(100,100), font_path=font_path)
    char = u"\u5929"
