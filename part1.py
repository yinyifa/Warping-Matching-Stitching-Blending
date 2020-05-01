#!/usr/bin/python

# Authors : Xizi Wang/xiziwang, Yifan Ying/yinyifa, Chang Liu/liu472

import cv2
import numpy as np
import sys
import time
import os
from datetime import datetime
import json


# pic1 and pic2 would be index of data
# This function returns the feature distance between two PICTURES
def feature_dist(data, pic1, pic2):
    # Distance = ||f1 - f2 || / || f1 - f2’ ||
    # where f2 is best SSD match to f1 in J,
    # f2’ is 2nd best SSD match to f1 in J
    Distance = []

    # make descriptors for pic1 and pic2
    pic1_descriptors = data[pic1][1]
    pic2_descriptors = data[pic2][1]

    # loop through each key-point in pic1 and record best matches and their Distance
    for i in range(len(pic1_descriptors)):
        # to record dist between this point in pic1 and every points in pic2
        temp_dist = []
        for j in range(len(pic2_descriptors)):
            # for each key_point, calculate the distance between descriptors
            dist = cv2.norm(pic1_descriptors[i], pic2_descriptors[j], cv2.NORM_HAMMING)
            temp_dist.append(dist)

        sorted_temp = sorted(temp_dist)
        # get first and second
        [first, second] = sorted_temp[:2]
        normalized_dist = first / second
        Distance.append(normalized_dist)

    # count using thresh hold
    count = 0
    for each in Distance:
        count += 1 if each <= 0.8 else 0

    return count


# returns key-points and descriptors given file name
def get_keyp_and_des(picture_name):
    img = cv2.imread(picture_name, cv2.IMREAD_GRAYSCALE)
    # you can increase nfeatures to adjust how many features to detect
    orb = cv2.ORB_create(nfeatures=1000)
    # detect features
    (a, b) = orb.detectAndCompute(img, None)
    return a, b


# return clustered index of given list
def k_cluster(picture_list, k, output_filename):
    # print(picture_list)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # print("Start Time =", current_time)
    # data is list of tuples that stores key points and descriptor
    # order is corresponding to the picture list
    data = []
    for each in picture_list:
        key_point, descriptor = get_keyp_and_des(each)
        if descriptor is None:
            print("This got a NoneType descriptor (possible file DNE) :", each)
        data.append([key_point, descriptor])

    # create a data matrix to store all feature distance between pictures
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # print("Finished importing data =", current_time)
    data_matrix = []
    for i in range(len(picture_list)):
        data_matrix.append([])
    for i in range(len(picture_list)):
        for j in range(len(picture_list)):
            data_matrix[i].append(feature_dist(data, i, j))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # print("Finished feature distance calculation =", current_time)

    # print(data)
    # start with first k key-points as centers
    centers = []
    while len(centers) < k:
        x = np.random.randint(0, len(picture_list))
        if x not in centers:
            centers.append(x)

    # print("pre-made centers :", centers)
    # the result
    res_cluster = []

    # not converged yet
    converged = False
    iteration = 0
    while not converged:
        # cluster is a list of index referring to picture list
        #  associate each data point to the closest medoid
        cur_centers = centers
        cur_cluster = [[] for x in cur_centers]
        for i in range(len(data)):
            largest = - np.inf
            choice = len(cur_centers)
            for j in range(len(cur_centers)):
                counts = data_matrix[i][cur_centers[j]] + data_matrix[cur_centers[j]][i]
                if counts > largest:
                    largest = counts
                    choice = j
            cur_cluster[choice].append(i)
        #  for each medoid 𝑗 and each data point 𝑖 associated with 𝑗,
        #  swap 𝑗 and 𝑖 and compute the total cost of the configuration
        #  (which is, the total similarity of 𝑖 to all the data points associated to 𝑗)
        #  Select the medoid 𝑗 with the highest cost of the configuration.
        pending_swaps = []
        pending_swaps_cost = []
        for j in range(len(cur_centers)):
            swap_cost = 0
            swap_this_point = len(data)
            for i in range(len(data)):
                # compute the total cost of the configuration
                total_cost = 0
                for point_in_cur_cluster in cur_cluster[j]:
                    total_cost += data_matrix[i][point_in_cur_cluster] + data_matrix[point_in_cur_cluster][i]
                if total_cost > swap_cost:
                    swap_cost = total_cost
                    swap_this_point = i
            pending_swaps_cost.append(swap_cost)
            # here its [center index, data index]
            pending_swaps.append([j, swap_this_point])

        old_medoid, new_medoid = [x for _, x in sorted(zip(pending_swaps_cost, pending_swaps))][0]
        cur_centers[old_medoid] = new_medoid
        # print("past and new centers")
        # print(centers, cur_centers)
        if set(cur_centers) == set(centers):
            converged = True
            res_cluster = cur_cluster
        centers = cur_centers

        iteration += 1
        # print("Finished Iteration %d" % iteration)
        #
        # print("Current Clusters : ")
        for each in cur_cluster:
            temp = []
            for every in each:
                temp.append(picture_list[every][len("part1-images/"):len("part1-images/") + 6])
        #     print(temp)
        #     print("")
        # print("")
        # print("")
    # cluster of index
    output = []
    # print("End result : ")
    for each in res_cluster:
        temp = []
        temp_output = ""
        for every in each:
            # print(every, picture_list, picture_list[every])
            temp_output += picture_list[every] + " "
            temp.append(picture_list[every])
        output.append(temp_output)
        # print(temp)
        # print("")
    # print(output)
    with open(output_filename, "w") as output_file:
        for each_line in output:
            output_file.write(each_line)
            output_file.write("\n")
        # json.dump(output, output_file)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # print("Finished all =", current_time)
    return res_cluster


if __name__ == '__main__':
    # test = np.array([[[1,2,3]],[[3,2,3]],[[4,2,3]],[[2,2,3]]])
    # print([x for _, x in sorted(zip([y[0,0] for y in test], test))])
    arguments = sys.argv
    # which_part = arguments[1]
    k = arguments[1]
    picture_list = arguments[2:-1]
    output_filename = arguments[-1]
    # print(picture_list, k, output_filename)
    k_cluster(picture_list, int(k), output_filename)
    # python part1.py 2 bigben_6.jpg bigben_8.jpg eiffel_18.jpg eiffel_19.jpg part1_output.txt


    # test = [[2,4,1],
    #         [5,0,9]]
    # np_test = np.array(test)
    # temp_init_centers = np.argpartition([5,0,9], 2)
    # print(temp_init_centers)
    # print(np.unravel_index(np.argmin(np_test, axis=None), np_test.shape))

    # pic1 = "part1-images/bigben_2.jpg"
    # pic2 = "part1-images/bigben_3.jpg"
    # pic3 = "part1-images/bigben_12.jpg"
    # pic4 = "part1-images/bigben_6.jpg"
    # pic5 = "part1-images/bigben_13.jpg"
    #
    # pic10 = "part1-images/eiffel_18.jpg"
    # pic11 = "part1-images/eiffel_22.jpg"
    # pic12 = "part1-images/eiffel_5.jpg"
    # pic13 = "part1-images/eiffel_6.jpg"
    # pic14 = "part1-images/eiffel_15.jpg"
    #
    # start_time = time.time()
    # k_cluster([pic1, pic2, pic3, pic4, pic5, pic10, pic11, pic12, pic13, pic14], 2, "part1_output.txt")
    # end_time = time.time()
    # print("Time taken :", round((end_time - start_time)//60), ":", round((end_time - start_time) % 60))

    # pic1 = "part1-images/bigben_2.jpg"
    # pic2 = "part1-images/bigben_3.jpg"
    # pic3 = "part1-images/bigben_6.jpg"
    # pic4 = "part1-images/bigben_7.jpg"
    #
    # pic7 = "part1-images/colosseum_3.jpg"
    # pic8 = "part1-images/colosseum_4.jpg"
    # pic9 = "part1-images/colosseum_6.jpg"
    # pic10 = "part1-images/colosseum_8.jpg"
    #
    # pic11 = "part1-images/eiffel_22.jpg"
    # pic12 = "part1-images/eiffel_5.jpg"
    # pic13 = "part1-images/eiffel_6.jpg"
    # pic14 = "part1-images/eiffel_15.jpg"
    #
    # pic15 = "part1-images/sanmarco_1.jpg"
    # pic16 = "part1-images/sanmarco_3.jpg"
    # pic17 = "part1-images/sanmarco_4.jpg"
    # pic18 = "part1-images/sanmarco_5.jpg"
    #
    # start_time = time.time()
    # k_cluster([pic1, pic2, pic3, pic4, pic7, pic8, pic9, pic10,
    #            pic11, pic12, pic13, pic14, pic15, pic16, pic17, pic18], 4)
    # end_time = time.time()
    # length = end_time - start_time
    # print("Time taken :", round(length // 60), ":", round(length % 60))

    ###########################################################################################################
    # path = "part1-images/"
    #
    # files = []
    # # r=root, d=directories, f = files
    # for r, d, f in os.walk(path):
    #     for file in f:
    #         if len(files) >= 20:
    #             break
    #         files.append(os.path.join(r, file))
    # # print(files)
    # start_time = time.time()
    # k_cluster(files, 10)
    # end_time = time.time()
    # length = end_time - start_time
    # print("Time taken :", round(length // 60), ":", round(length % 60))

    # img1 = cv2.imread(pic1, cv2.IMREAD_GRAYSCALE)
    # # you can increase nfeatures to adjust how many features to detect
    # orb = cv2.ORB_create(nfeatures=1000)
    # # detect features
    # tup1 = orb.detectAndCompute(img1, None)
    #
    # img2 = cv2.imread(pic2, cv2.IMREAD_GRAYSCALE)
    # # you can increase nfeatures to adjust how many features to detect
    # orb = cv2.ORB_create(nfeatures=1000)
    # # detect features
    # tup2 = orb.detectAndCompute(img2, None)
    #
    # # calculate feature distance for each point
    # dist = feature_dist(tup1, tup2)
    # print(dist)
    # # put a little X on each feature
    # for i in range(0, len(keypoints)):
    #     print("Keypoint #%d: x=%d, y=%d, descriptor=%s, distance between "
    #           "this descriptor and descriptor #0 is %d" %
    #           (i, keypoints[i].pt[0], keypoints[i].pt[1], np.array2string(descriptors[i]),
    #            cv2.norm(descriptors[0], descriptors[i], cv2.NORM_HAMMING)))
    #     for j in range(-5, 5):
    #         img[int(keypoints[i].pt[1]) + j, int(keypoints[i].pt[0]) + j] = 0
    #         img[int(keypoints[i].pt[1]) - j, int(keypoints[i].pt[0]) + j] = 255
    #
    # cv2.imwrite("lincoln-orb.jpg", img)
