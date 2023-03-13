from PIL import Image
import model
import sys
sys.path.append('/usr/local/lib/python3.5/dist-packages')
cont_imgs = ['elefant.jpg', 'cabin.jpg']#'tree.jpg']#'church.jpg']#'golden_gate.jpg'] 
styl_imgs = ['scream.jpg' , 'van_gogh_trees.jpg']#'street.jpg']#'van_gogh_starry_night.jpg']

for i in range(len(cont_imgs)): 

	content_img = Image.open('images/content/'+cont_imgs[i])
	style_img   = Image.open('images/style/'+styl_imgs[i])

	stylized_img = model.stylize(content_img,
							style_img,
							num_strokes=5000,
							num_steps=100,
							content_weight=1.0,
							style_weight=3.0,
							num_steps_pixel=1000)
	cont_name = cont_imgs[i].split('.')[0]
	styl_name = styl_imgs[i].split('.')[0]
	stylized_img.save('images/'+cont_name+'_stylized_by_'+styl_name+'.jpg')