import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_video():
	path = './output/img'
	filelist = os.listdir(path)
	video = cv2.VideoWriter('demo.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15 ,(640, 480))
	for i in tqdm(range(6, len(filelist) + 6)):
		img_p = os.path.join(path, 'frame%04d.png'%(i))
		img1 = cv2.imread(img_p)
		video.write(img1)
	video.release()

if __name__ == '__main__':
	generate_video()