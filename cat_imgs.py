import cv2
import os
import numpy as np

imgs_dir = "./dataset/train/00000/"
out_dir = "./dataset/merges/"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

img_fns = os.listdir(imgs_dir)

print("loading images...")
imgs = [cv2.imread(os.path.join(imgs_dir, fn), cv2.CV_8UC1) for fn in img_fns]

print("%s images to be merged..." % (len(imgs)))

W, H = 20, 20

h, w = imgs[0].shape

to_be_merged = []
IMG = np.zeros((H*h, W*w), dtype=np.uint8)

count = 0

for img in imgs:
    to_be_merged.append(img)
    if len(to_be_merged) == W*H:
        for idx,im in enumerate(to_be_merged):
            i,j = idx//W, idx%W
            IMG[h*i:h*(i+1), w*j:w*(j+1)] = im
        IMG_fn = str(count).zfill(4) + ".png"
        cv2.imwrite(os.path.join(out_dir, IMG_fn), IMG)
        IMG = np.zeros((H*h, W*w), dtype=np.uint8)
        to_be_merged = []
        count += 1
        print(count, len(imgs) // (W*H))

if to_be_merged:
    for idx,im in enumerate(to_be_merged):
        i,j = idx//W, idx%W
        IMG[h*i:h*(i+1), w*j:w*(j+1)] = im
    IMG_fn = str(count).zfill(4) + ".png"
    cv2.imwrite(os.path.join(out_dir, IMG_fn), IMG)
    to_be_merged = []
    count += 1


