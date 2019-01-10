import cv2
from six.moves import urllib
import numpy as np
import os

def save_images(imagenet_url, out_dir):
	image_urls = urllib.request.urlopen(imagenet_url).read()
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	picture_num = 0
	for i in image_urls.split('\n'):
		if ("flickr" not in i):
			continue
		try:
			urllib.request.urlretrieve(i, out_dir+'/'+str(i.split("/")[-1])+'.jpg')
		except:
			continue
		img = cv2.imread(out_dir+'/'+str(i.split("/")[-1])+'.jpg')
		resized_image = cv2.resize(img, (299, 299))
		cv2.imwrite(out_dir+'/'+str(picture_num)+'.jpg', resized_image)
		picture_num += 1


if __name__ == '__main__':
	save_images("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03126707", "crane")
	save_images("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00007846", "people_no_hardhat")
	save_images("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03492922", "people_with_hardhat")

	






