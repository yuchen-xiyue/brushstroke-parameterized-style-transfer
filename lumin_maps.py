
from PIL import Image
import os
from os import path
import argparse
import numpy as np
CWD = os.getcwd()
IMGS_PATH = path.join(CWD, 'images')
MAPS_PATH = path.join(IMGS_PATH, 'maps')
CONT_PATH = path.join(IMGS_PATH, 'content')
STYL_PATH = path.join(IMGS_PATH, 'style')

def rgb2yiq(X): 
    '''X: array of image, HxWxC'''
    RGB2YIQ = np.array(
    [[0.2990,  0.5870,  0.1140], 
     [0.5959, -0.2746, -0.3213], 
     [0.2115, -0.5227,  0.3112]])
    return np.einsum('ij,hwj->hwi', RGB2YIQ, X)

def main(file_path, hard_mode=False):
    img = Image.open(file_path)
    img_array = np.array(img).astype(float)/255.0
    img_array_lumin = rgb2yiq(img_array)[..., 0]
    results = []
    if not hard_mode: 
        img_array_lumin_reverse = 1 - img_array_lumin

        img_lumin = Image.fromarray((img_array_lumin * 255.).astype('uint8'))
        img_lumin_reverse = Image.fromarray((img_array_lumin_reverse * 255.).astype('uint8'))
        results = [img_lumin, img_lumin_reverse]
    else: 
        img_array_white, img_array_gray, img_array_black = np.zeros_like(img_array_lumin), np.zeros_like(img_array_lumin), np.zeros_like(img_array_lumin)
        img_array_white[img_array_lumin >= .7] = 1
        img_array_gray[np.logical_and(img_array_lumin<.7, img_array_lumin >= .3)] = 1
        img_array_black[img_array_lumin <.3] = 1

        img_array_white = Image.fromarray((img_array_white * 255.).astype('uint8'))
        img_array_gray = Image.fromarray((img_array_gray * 255.).astype('uint8'))
        img_array_black = Image.fromarray((img_array_black * 255.).astype('uint8'))
        results = [img_array_white, img_array_gray, img_array_black]

    filename = file_path.split('.')[0].split('/')[-1]
    savepath = path.join(MAPS_PATH, filename)

    if not os.path.exists(savepath): 
        os.mkdir(savepath)

    for i, result in enumerate(results):
        result.save(path.join(savepath, str(i)+'.jpg'))

def parse_arguments(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path')
    parser.add_argument('-m', '--hard_mode', action='store_true')
    parser.set_defaults(guided=False)
    args = parser.parse_args()
    return [args.file_path, args.hard_mode]

if __name__ == '__main__': 
    args = parse_arguments()
    main(*args)