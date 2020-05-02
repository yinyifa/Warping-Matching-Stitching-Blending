#!/l/python3.5.2/bin/python3

import cv2
import sys
import numpy as np

def warp(trans_matrix, img2_path, img_output_path):

	img = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

	warped_img = np.zeros(img.shape)

	#trans_matrix = np.array([[0.907, 0.258, -182], [-0.153, 1.44, 58], [-0.000306, 0.000731, 1]])
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

	cv2.imwrite(img_output_path, warped_img)

'''
	img1_x1 = 318
	img1_y1 = 256
	img2_x1 = 141
	img2_y1 = 131
	img1_x2 = 534
	img1_y2 = 372
	img2_x2 = 480
	img2_y2 = 159
	img1_x3 = 316
	img1_y3 = 670
	img2_x3 = 493
	img2_y3 = 630
	img1_x4 = 73
	img1_y4 = 473
	img2_x4 = 64
	img2_y4 = 601
'''

if __name__ == '__main__':


	# input
	part = sys.argv[1]
	n = int(sys.argv[2])
	img1_path = sys.argv[3]
	img2_path = sys.argv[4]
	img_output_path = sys.argv[5]

	img1_x1, img1_y1 = int(sys.argv[6].split(',')[0]), int(sys.argv[6].split(',')[1])
	img2_x1, img2_y1 = int(sys.argv[7].split(',')[0]), int(sys.argv[7].split(',')[1])
	img1_x2, img1_y2 = int(sys.argv[8].split(',')[0]), int(sys.argv[8].split(',')[1])
	img2_x2, img2_y2 = int(sys.argv[9].split(',')[0]), int(sys.argv[9].split(',')[1])
	img1_x3, img1_y3 = int(sys.argv[10].split(',')[0]), int(sys.argv[10].split(',')[1])
	img2_x3, img2_y3 = int(sys.argv[11].split(',')[0]), int(sys.argv[11].split(',')[1])
	img1_x4, img1_y4 = int(sys.argv[12].split(',')[0]), int(sys.argv[12].split(',')[1])
	img2_x4, img2_y4 = int(sys.argv[13].split(',')[0]), int(sys.argv[13].split(',')[1])

	# end input


	
	if n == 1:
		tx = img1_x1 - img2_x1
		ty = img1_y1 - img2_y1
		trans_matrix = np.array([[1, 0, tx],[0, 1, ty], [0, 0, 1]])

	elif n == 2:
		a = np.array([[img2_x1, img2_y1],[img2_x2, img2_y2]])
		b = np.array([img1_x1, img1_x2])
		x = np.linalg.solve(a,b)

		a = np.array([[img2_x1, img2_y1],[img2_x2, img2_y2]])
		b = np.array([img1_y1, img1_y2])
		y = np.linalg.solve(a,b)

		x = np.append(x, [0], axis=0)
		y = np.append(y, [0], axis=0)



		trans_matrix = np.array([x, y, [0, 0, 1]])

		print(trans_matrix)
		

	elif n == 3:
		a = np.array([[img2_x1, img2_y1, 1],[img2_x2, img2_y2, 1], [img2_x3, img2_y3, 1]])
		b = np.array([img1_x1, img1_x2, img1_x3])
		x = np.linalg.solve(a,b)

		a = np.array([[img2_x1, img2_y1, 1],[img2_x2, img2_y2, 1], [img2_x3, img2_y3, 1]])
		b = np.array([img1_y1, img1_y2, img1_y3])
		y = np.linalg.solve(a,b)

		trans_matrix = np.array([x, y, [0,0,1]])

		
	elif n == 4:
		
		X = np.array([[img2_x1, img2_y1, 1, 0, 0, 0, -img2_x1*img1_x1, -img2_y1*img1_x1],
					 [0, 0, 0, img2_x1, img2_y1, 1, -img2_x1*img1_y1, -img2_y1*img1_y1],
					 [img2_x2, img2_y2, 1, 0, 0, 0, -img2_x2*img1_x2, -img2_y2*img1_x2],
					 [0, 0, 0, img2_x2, img2_y2, 1, -img2_x2*img1_y2, -img2_y2*img1_y2],
					 [img2_x3, img2_y3, 1, 0, 0, 0, -img2_x3*img1_x3, -img2_y3*img1_x3],
					 [0, 0, 0, img2_x3, img2_y3, 1, -img2_x3*img1_y3, -img2_y3*img1_y3],
					 [img2_x4, img2_y4, 1, 0, 0, 0, -img2_x4*img1_x4, -img2_y4*img1_x4],
					 [0, 0, 0, img2_x4, img2_y4, 1, -img2_x4*img1_y4, -img2_y4*img1_y4]
					 ])
		Y = np.array([img1_x1, img1_y1, img1_x2, img1_y2, img1_x3, img1_y3, img1_x4, img1_y4])

		Z = np.linalg.inv(X) @ Y

		Z = np.append(Z, [1], axis=0)

		trans_matrix = Z.reshape((3,3))

	warp(trans_matrix, img2_path, img_output_path)

		





		
	














