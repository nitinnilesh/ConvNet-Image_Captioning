import os
from PIL import Image

def resize_images(image_dir, output_dir, image_size):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	images = os.listdir(image_dir)
	num_images = len(images)
	for i, image in enumerate(images):
		with open(os.path.join(image_dir, image), 'r+b') as f:
			with Image.open(f) as img:
				img = img.resize(image_size, Image.ANTIALIAS)
				img.save(os.path.join(output_dir, image), img.format)
		if i % 100 == 0:
			print ("[%d/%d] Resized the images and saved into '%s'."%(i, num_images, output_dir))


image_dir = '/media/pi/Study/Monsoon_2017_courses/CSE471_SMAI/SMAI_Project/Flickr8k_Dataset/Flicker8k_Dataset'
output_dir = '/media/pi/Study/Monsoon_2017_courses/CSE471_SMAI/SMAI_Project/Flickr8k_Dataset/resizeFlickr_Dataset'
image_size = [256, 256]
resize_images(image_dir, output_dir, image_size)
