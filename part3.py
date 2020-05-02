#!/l/python3.5.2/bin/python3
import cv2
import numpy as np
import getopt
import sys
import random
import time


# ransac part learn from HomographyEstimation
# https://github.com/hughesj919/HomographyEstimation/tree/1a29d5f673852e5ac21fc4ab5c0b12164b1a2423
##
def readImage(filename):
    img = cv2.imread(filename, 0)
    if img is None:
        print('Invalid image:' + filename)
        return None
    else:
        print('Image successfully read...')
        return img

#
# def findFeatures(img):
#     print("Finding Features...")
#     orb = cv2.ORB_create()
#     keypoints, descriptors = orb.detectAndCompute(img, None)
#     return keypoints, descriptors


def Homography(cl):
    # loop through correspondences and create assemble matrix
    aList = []
    for corr in cl:
        p1 = np.mat([corr.item(0), corr.item(1), 1])
        p2 = np.mat([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.mat(aList)

    # svd composition
    u, s, v = np.linalg.svd(matrixA)

    # reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    # normalize and now we have h
    h = (1 / h.item(8)) * h
    return h


def Distance(c, h):
    p1 = np.transpose(np.mat([c[0].item(0), c[0].item(1), 1]))
    e = np.dot(h, p1)
    e = (1 / e.item(2)) * e

    p2 = np.transpose(np.mat([c[0].item(2), c[0].item(3), 1]))
    error = p2 - e
    return np.linalg.norm(error)


# def matchFeatures(desc1, desc2):
#     print("Matching Features...")
#     matcher = cv2.BFMatcher(cv2.NORM_L2, True)
#     matches = matcher.match(desc1, desc2)
#     print("Features Matched...")
#     return matches


def ransac(cl, thresh):
    maxInliers = []
    finalH = None
    j = 0
    # for i in range(10000):
    while len(maxInliers) <= 30:
        j += 1
        print(str(j)+"th iteration")
        for i in range(1000):
            print("ransac inner loop time:", i, ";", "best number of inliers:", len(maxInliers))
            # find 4 random points to calculate a homography
            corr1 = cl[random.randrange(0, len(cl))]
            corr2 = cl[random.randrange(0, len(cl))]
            randomFour = np.vstack((corr1, corr2))
            corr3 = cl[random.randrange(0, len(cl))]
            randomFour = np.vstack((randomFour, corr3))
            corr4 = cl[random.randrange(0, len(cl))]
            randomFour = np.vstack((randomFour, corr4))

            # call the homography function on those points
            h = Homography(randomFour)
            inliers = []

            for i in range(len(cl)):
                d = Distance(cl[i], h)
                if d < 5:
                    inliers.append(cl[i])

            if len(inliers) > len(maxInliers):
                maxInliers = inliers
                finalH = h
            if len(maxInliers) > (len(cl) * thresh):
                break
    return finalH, maxInliers


def warp(finalH, img2_path, img_output_path, offset, offset2, newXOffset, newYOffset):
    img = cv2.imread(img2_path)
    warped_img = np.zeros((img.shape[0] + offset + offset2, img.shape[1] + offset + offset2, 3))
    translation = np.array([[0, 1, -newYOffset], [1, 0, -newXOffset], [0, 0, 1]])
    inverse_matrix = np.asarray(finalH)
    for x in range(warped_img.shape[0]):
        for y in range(warped_img.shape[1]):
            if y < warped_img.shape[0] and x < warped_img.shape[1]:
                warped_coordinate = np.array([y, x, 1])
                # old_coordinate = inverse_matrix @ warped_coordinate
                old_coordinate1 = translation @ warped_coordinate
                old_coordinate = inverse_matrix @ old_coordinate1
                # old_coordinate = inverse_H @ old_coordinate2
                old_coordinate_w = old_coordinate[2]
                old_coordinate_x = old_coordinate[1] / old_coordinate_w
                old_coordinate_y = old_coordinate[0] / old_coordinate_w
                a = int(old_coordinate_x) + 1 - old_coordinate_x
                b = int(old_coordinate_y) + 1 - old_coordinate_y
                if old_coordinate_y >= img.shape[1] - 1 or old_coordinate_y < 0:
                    warped_img[y][x] = [0, 0, 0]
                    continue
                if old_coordinate_x >= img.shape[0] - 1 or old_coordinate_x < 0:
                    warped_img[y][x] = [0, 0, 0]
                    continue

                left_top = img[int(old_coordinate_x)][int(old_coordinate_y)]
                right_top = img[int(old_coordinate_x)][int(old_coordinate_y) + 1]
                left_bottom = img[int(old_coordinate_x) + 1][int(old_coordinate_y)]
                right_bottom = img[int(old_coordinate_x) + 1][int(old_coordinate_y) + 1]
                warped_img[y][x] = (1 - b) * (1 - a) * left_top + (1 - b) * a * left_bottom + b * (
                            1 - a) * right_top + b * a * right_bottom
    cv2.imwrite(img_output_path, warped_img)


def main(p1, p2, of):
    start_time = time.time()
    args, img_name = getopt.getopt(sys.argv[1:], '', ['threshold='])
    args = dict(args)

    estimation_thresh = args.get('--threshold')
    if estimation_thresh is None:
        estimation_thresh = 0.80

    # img1name = str(img_name[0])
    # img2name = str(img_name[1])
    # img_output_path = str(img_name[2])
    # print("Image 1 Name: " + img1name)
    # print("Image 2 Name: " + img2name)
    # img1 = readImage(img_name[0])
    # img2 = readImage(img_name[1])
    img1name = p1
    img2name = p2
    img_output_path = of
    print("Image 1 Name: " + img1name)
    print("Image 2 Name: " + img2name)
    img1 = readImage(p1)
    img2 = readImage(p2)

    cl = []
    if img1 is not None and img2 is not None:
        print("Finding Features...")
        orb = cv2.ORB_create()
        kp1, desc1 = orb.detectAndCompute(img1, None)
        kp2, desc2 = orb.detectAndCompute(img2, None)
        # kp1, desc1 = findFeatures(img1)
        # kp2, desc2 = findFeatures(img2)
        keypoints = [kp1, kp2]
        print("Matching Features...")
        matcher = cv2.BFMatcher(cv2.NORM_L2, True)
        matches = matcher.match(desc1, desc2)
        print("Features Matched...")
        # matches = matchFeatures(desc1, desc2)
        for match in matches:
            (x1, y1) = keypoints[0][match.queryIdx].pt
            (x2, y2) = keypoints[1][match.trainIdx].pt
            cl.append([x1, y1, x2, y2])

        corrs = np.mat(cl)

        # run ransac algorithm
        finalH, inliers = ransac(corrs, estimation_thresh)
        print("Final inliers count: ", len(inliers))

        ## (y,x,1)-> (shape[1],shape[0],1)
        p = np.asarray([[0, img1.shape[1], img1.shape[1], 0], [0, 0, img1.shape[0], img1.shape[0]], [1, 1, 1, 1]])
        inverse_matrix = np.linalg.inv(np.asarray(finalH))
        p2 = inverse_matrix @ p

        xl = p2[1] / p2[2]
        yl = p2[0] / p2[2]
        # offset to adjust transformation
        xmove = 0
        ymove = 0
        xoffset = 0
        yoffset = 0
        newXOffset = 0
        newYOffset = 0
        # left right corner
        if min(xl) < 0:
            newXOffset = int(abs(min(xl)))
            xmove += int(abs(min(xl)))
            if max(xl) > img1.shape[0]:
                xoffset += int(max(xl) - img1.shape[0] + xmove)
        else:
            if max(xl) > img1.shape[0]:
                xoffset += int(max(xl) - img1.shape[0])
        if min(yl) < 0:
            newYOffset = int(abs(min(yl)))
            ymove += int(abs(min(yl)))
            if max(yl) > img1.shape[1]:
                yoffset += int(max(yl) - img1.shape[1] + ymove)
        else:
            if max(yl) > img1.shape[1]:
                yoffset += int(max(yl) - img1.shape[1])
        offset = max(xoffset, yoffset)
        offset2 = max(img1.shape[0], img1.shape[1])
        warp(finalH, img2name, img_output_path, offset, offset2, newXOffset, newYOffset)
        two = cv2.imread(img_output_path)
        one = cv2.imread(img1name)
        img_new1 = np.zeros((img1.shape[0] + offset + offset2, img1.shape[1] + offset + offset2, 3))
        img_new1[newXOffset:newXOffset + one.shape[0], newYOffset:newYOffset + one.shape[1]] = one
        img_new1 = img_new1.astype("uint8")
        img = cv2.addWeighted(img_new1, 0.5, two, 0.5, 0)
        cv2.imwrite("result" + img_output_path, img)

        end_time = time.time()
        print("Time taken :", round((end_time - start_time) // 60), ":", round((end_time - start_time) % 60))


if __name__ == "__main__":
    arguments = sys.argv
    p1, p2, of = arguments[1:]
    # p1, p2, of = ["eiffel_18.jpg", "eiffel_19.jpg", "part3_demo.jpg"]
    main(p1, p2, of)
