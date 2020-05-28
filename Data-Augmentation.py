from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import sys
import os

file_paths = []

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
input_group = ap.add_mutually_exclusive_group(required=True)
input_group.add_argument("-i", "--image",
	help="path to the input image")
input_group.add_argument("-f","--folder",
	help="path to floder of input images")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store augmentation examples")
ap.add_argument("-t", "--total", type=int, default=100,
	help="# of training samples to generate")
args = vars(ap.parse_args())

# define imput file path
if args["image"]:
	file_paths.append(args["image"])
elif args["folder"]:
	for filepath in os.listdir(args["folder"]):
		if filepath.endswith(".png") or filepath.endswith(".jpg") or filepath.endswith(".jpeg"):
			file_paths.append(os.path.join(args["folder"],filepath))
else:
	print('[ERROR] missing image option args')
	sys.exit()

print("[INFO] total {} image files will be processed".format(len(file_paths)))

for filepath in file_paths:
	# load the input image, convert it to a NumPy array, and then
	# reshape it to have an extra dimension
	print("[INFO] processing file : " + filepath)
	print("[INFO] loading image...")
	image = load_img(filepath)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# construct the image generator for data augmentation then
	# initialize the total number of images generated
	aug = ImageDataGenerator(
		rotation_range=30,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")
	total = 0

	# construct the actual Python generator
	print("[INFO] generating images...")
	imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
		save_prefix="image", save_format="jpg")
	# loop over examples from our image data augmentation generator
	for image in imageGen:
		# increment our counter
		total += 1
		# if we have reached the specified number of examples, break
		# from the loop
		if total == args["total"]:
			break