from PIL import Image
import model
import spatial_control_model
import os
from os import path
import argparse
import glob
# import sys
# sys.path.append('/usr/local/lib/python3.5/dist-packages')
CWD = os.getcwd()
IMGS_PATH = path.join(CWD, 'images')
MAPS_PATH = path.join(IMGS_PATH, 'maps')
CONT_PATH = path.join(IMGS_PATH, 'content')
STYL_PATH = path.join(IMGS_PATH, 'style')

cont_imgs = ['olive_trees_greece.jpg', 'tree.jpg']#'elefant.jpg', 'cabin.jpg']##'church.jpg']#'golden_gate.jpg'] 
styl_imgs = ['van_gogh_trees.jpg', 'street.jpg']#'scream.jpg' , 'van_gogh_trees.jpg']##'van_gogh_starry_night.jpg']

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--guided', action='store_true')
parser.set_defaults(guided=False)
args = parser.parse_args()

if args.guided:
	for cont_name, styl_name in zip(cont_imgs, styl_imgs):
		cont_img = Image.open(path.join(CONT_PATH, cont_name))
		cont_name = cont_name.split('.')[0]
		cont_map_path = path.join(MAPS_PATH, cont_name)
		cont_maps = [Image.open(filepath) for filepath in glob.glob(path.join(cont_map_path, '[0-9]*.*'))]

		styl_img = Image.open(path.join(STYL_PATH, styl_name))
		styl_name = styl_name.split('.')[0]
		styl_map_path = path.join(MAPS_PATH, styl_name)
		styl_maps = [Image.open(filepath) for filepath in glob.glob(path.join(styl_map_path, '[0-9]*.*'))]

		stylized_img = spatial_control_model.stylize(cont_img,
													 styl_img,
													 cont_maps,
													 styl_maps,
													 num_strokes=5000,
													 num_steps=100,
													 content_weight=1.0,
													 style_weight=3.0,
													 num_steps_pixel=1000)
		stylized_img.save(path.join(IMGS_PATH, cont_name+'_stylized_by_'+styl_name+'_guided.jpg'))

else:
	for cont_name, styl_name in zip(cont_imgs, styl_imgs):

		cont_img = Image.open(path.join(CONT_PATH, cont_name))
		styl_img = Image.open(path.join(STYL_PATH, styl_name))
		cont_name = cont_name.split('.')[0]
		styl_name = styl_name.split('.')[0]

		stylized_img = model.stylize(cont_img,
								styl_img,
								num_strokes=5000,
								num_steps=100,
								content_weight=1.0,
								style_weight=3.0,
								num_steps_pixel=1000)
		stylized_img.save(path.join(IMGS_PATH, cont_name+'_stylized_by_'+styl_name+'.jpg'))