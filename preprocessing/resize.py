import glob
import os
import time
import cv2


def main():
	video_dir = '/mnt/shared_40t/dwd/epic-kitchens/dataset/EPIC_KITCHENS_2018/interim/rgb_train_segments'
	for vdir in os.listdir(video_dir):
		img_list = glob.glob(os.path.join(video_dir, vdir, '*/*/*.jpg'))

		for i in img_list:
			img = cv2.imread(i)
			new_img = cv2.resize(img, (256, 256))
			names = i.split('/')
			img_name = os.path.join('/mnt/shared_40t/dwd/epic-kitchens/orn/data/epic/videos_256x256_30', '/'.join(names[-4:]))
			img_dir = os.path.dirname(img_name)
			if not os.path.exists(img_dir):
				os.makedirs(img_dir)
			print('Write to ', img_name)
			cv2.imwrite(img_name, new_img)


if __name__ == '__main__':
	start = time.time()
	main()
	print('The total runtime is ', time.time() - start)
