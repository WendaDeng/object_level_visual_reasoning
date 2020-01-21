import glob
import os
import time
import cv2


def main():
	for video_dir in os.listdir('videos'):
		img_list = glob.glob(os.path.join('videos', video_dir, '*/*/*.jpg'))

		for i in img_list:
			img = cv2.imread(i)
			new_img = cv2.resize(img, (256, 256))
			img_name = i.replace('videos', 'videos_256x256_30')
			img_dir = os.path.dirname(img_name)
			if not os.path.exists(img_dir):
				os.makedirs(img_dir, mode=0o666)
			cv2.imwrite(img_name, new_img)



if __name__ == '__main__':
	start = time.time()
	main()
	print('The total runtime is ', time.time() - start)
