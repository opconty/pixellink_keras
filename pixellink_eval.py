#-*- coding:utf-8 -*-
#'''
# Created on 18-10-15
#
# @Author: Greg Gao(laygin)
#'''
import os
import numpy as np
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import imutils
from pixellink_model import create_pixellink_model
from pixellink_utils import *

r_mean = 123.
g_mean = 117.
b_mean = 104.
rgb_mean = [r_mean, g_mean, b_mean]


img_path = r'./samples/img_1099.jpg'
save_weights = r'./weights/pixellink.h5'
model = create_pixellink_model(acf='relu')
model.load_weights(save_weights)

image = cv2.imread(img_path)
image, *r = resize_image(image)
image_ori = image.copy()
image = image[...,::-1] - rgb_mean
image = np.expand_dims(image, axis=0)

pixel_pos_scores, link_pos_scores = model.predict(image)
pixel_pos_scores = softmax(pixel_pos_scores, axis=-1)
link_pos_scores_reshaped = link_pos_scores.reshape(link_pos_scores.shape[:-1]+(8, 2))
link_pos_scores = softmax(link_pos_scores_reshaped, axis=-1)

masks = decode_batch(pixel_pos_scores, link_pos_scores, pixel_conf_threshold=0.75, link_conf_threshold=0.9)

bboxes = mask_to_bboxes(masks[0], image_ori.shape)


image_c = image_ori.copy()
for box in bboxes:
    points = np.reshape(box, [4, 2])
    cv2.line(image_c,tuple(points[0]),tuple(points[1]),(0,0,255),2)
    cv2.line(image_c,tuple(points[0]),tuple(points[3]),(0,0,255),2)
    cv2.line(image_c,tuple(points[1]),tuple(points[2]),(0,0,255),2)
    cv2.line(image_c,tuple(points[2]),tuple(points[3]),(0,0,255),2)


cv2.imshow('image', image_c)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('img_1099_res.jpg', image_c)
