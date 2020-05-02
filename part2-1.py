#!/l/python3.5.2/bin/python3

import cv2
import numpy as np

def warp():

	img = cv2.imread("../part2-images/book2.jpg", cv2.IMREAD_GRAYSCALE)

	warped_img = np.zeros(img.shape)

	trans_matrix = np.array([[0.907, 0.258, -182], [-0.153, 1.44, 58], [-0.000306, 0.000731, 1]])
	inverse_matrix = np.linalg.inv(trans_matrix)

	for x in range(warped_img.shape[0]):
		for y in range(warped_img.shape[1]):
			warped_coordinate = np.array([y, x, 1])
			old_coordinate = inverse_matrix @ warped_coordinate
			old_coordinate_w = old_coordinate[2]
			old_coordinate_x = old_coordinate[1] / old_coordinate_w
			old_coordinate_y = old_coordinate[0] / old_coordinate_w
			a = int(old_coordinate_x)+1-old_coordinate_x
			b = int(old_coordinate_y)+1-old_coordinate_y
			if (old_coordinate_y) >= warped_img.shape[1] - 1 or (old_coordinate_y) < 0:
				warped_img[x][y] = 0
				continue
			if (old_coordinate_x) >= warped_img.shape[0] - 1 or (old_coordinate_x) < 0:
				warped_img[x][y] = 0
				continue

			left_top = img[int(old_coordinate_x)][int(old_coordinate_y)]
			right_top = img[int(old_coordinate_x)][int(old_coordinate_y)+1]
			left_bottom = img[int(old_coordinate_x)+1][int(old_coordinate_y)]
			right_bottom = img[int(old_coordinate_x)+1][int(old_coordinate_y)+1]
			warped_img[x][y] = (1-b)*(1-a)*left_top + (1-b)*a*left_bottom + b*(1-a)*right_top + b*a*right_bottom

	cv2.imwrite("../part2-images/book2_warped.jpg", warped_img)

if __name__ == '__main__':
	warp()