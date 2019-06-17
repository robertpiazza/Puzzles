# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:27:37 2019

@author: 593787
"""

#Needed libraries
#Standard data manipulation and display libraries
import numpy as np #Workhorse of the data manipulations
import pandas as pd #Will be needed for some
from matplotlib import pyplot as plt
import os

#Various utilities
import math #calcuting distances between points
from itertools import compress #single use for matching bools
import re # for renaming the solution images so we can retain original name and save a hi-res version of the solution

import cv2 as cv #This needs to be 3.4.1!! Based on stackoverflow, I used "conda install -c menpo opencv" to install 3.4.1

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AffinityPropagation

os.chdir('C:\\Users\\593787\\Documents\\GitHub\\Puzzles\\Puzzles')

#%% Best Image match:

canvas = 'gap.jpg'
piece = 'opencv_frame_0.png'
#best_image('gap.jpg', 'IMG_0277_inside.jpg')
def best_image(canvas, piece, Hessian_distance = 400, threshold_distance = 50, num_points =2):
    scene_color = cv.imread(canvas)
    piece_color = cv.imread(piece)
    horizantal_pieces = 27 #number of actual puzzle pieces across- found by completing the edges of the puzzle
    vertical_pieces = 38  #same as above but for height, I believe these are fairly standard for this size puzzle

    h_width = scene_color.shape[1]/horizantal_pieces
    v_width = scene_color.shape[0]/vertical_pieces

    #Draw horizantal and vertical lines for better placement
    for i in range(1, horizantal_pieces):
        cv.line(scene_color, (int(i*h_width), 0), (int(i*h_width), scene_color.shape[0]), (255,255,255), 2)
    for i in range(1, vertical_pieces):
        cv.line(scene_color, (0, int(i*v_width)), (scene_color.shape[1], int(i*v_width)), (255,255,255), 2)
    #label cells
    for h in range(1, horizantal_pieces+1):
        for v in range(1, vertical_pieces+1):
            cv.putText(scene_color, f'{h}, {v} ', \
            (int((h-.9)*h_width), int((v-.1)*v_width)), cv.FONT_HERSHEY_PLAIN, .75, (255,255,255))
    #scene_color will be re-used when drawing the matches but is ignored for actual matching
    #For matching, we'll use grayscale

    img_object = cv.imread(piece, cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread(canvas, cv.IMREAD_GRAYSCALE)
    if img_object is None or img_scene is None:
        print('Could not open or find the images!')
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = Hessian_distance
    surf = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)

    #surf.setUpright(True)
    #surf.setExtended(True) # These were attempt to say the pieces were upright but didn't affect performance

    keypoints_obj, descriptors_obj = surf.detectAndCompute(img_object, None)
    keypoints_scene, descriptors_scene = surf.detectAndCompute(img_scene, None)
    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    print(f"In the object and scene, found {len(keypoints_obj)} and {len(keypoints_scene)} keypoints respectively")
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    #SURF uses k-nearest-neighbors for matching the descriptors
    knn_matches = flann.knnMatch(descriptors_obj,descriptors_scene,k=2)

    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.90
    good_begin = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_begin.append(m)

    list_keypoints = [keypoints_scene[mat.trainIdx].pt for mat in good_begin]

    xy_points = np.array([[x, y] for x,y in list_keypoints])

    threshold_distance = threshold_distance
    num_points = num_points

    #create a dataframe for indexing all the good points so far and we'll perform clustering there
    #we'll compare each good_match point to every other and determine the distance to each other point
    named_pts = pd.DataFrame()
    for index, pt in enumerate(xy_points):
        named_pts['i'+str(index)] = pd.DataFrame(xy_points, columns = ['x', 'y']).apply(lambda row: math.hypot(row.x-pt[0], row.y-pt[1]), axis = 1)

    #We have a grid of distances, apply a filter of points who meet the criteria
    #apply the filters of threshold_distance and number of points
    good_bool = list((named_pts < threshold_distance).sum()>num_points)
    #apply the boolean logic to the original list of matches to get the filtered and eliminate the weird indexes
    good_matches = list(compress(good_begin, good_bool))
    print(f"Got {((named_pts < threshold_distance).sum()>num_points).sum()} good matching points")
    if ((named_pts < threshold_distance).sum()>num_points).sum() < 2:
        return
    final_keypoints = (pd.DataFrame([keypoints_scene[mat.trainIdx].pt for mat in good_matches]).drop_duplicates()).values.tolist()

    #we don't know how many clusters we're going to use so I'm using the silhouette score
    #to choose the number of clusters with the highest score.

    #Probably could have used affinity clustering on this one as well.
    silhouette = 0 #initial score to beat
    n_clusters = 2 #minimum number of clusters

    #max number of clusters is the number of keypoints minus 2
    for clusters in range(2,len(final_keypoints)-2):
        km = KMeans(n_clusters=clusters)
        alldistances = km.fit_transform(final_keypoints)
        if silhouette_score(final_keypoints, km.labels_) > silhouette:
            n_clusters = clusters
            silhouette = silhouette_score(final_keypoints, km.labels_)
    km = KMeans(n_clusters=n_clusters)
    alldistances = km.fit_transform(final_keypoints)
    (pd.DataFrame(final_keypoints).drop_duplicates()).values.tolist()

    #-- Draw matches
    img_matches = np.empty((max(piece_color.shape[0], scene_color.shape[0]), piece_color.shape[1]+scene_color.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(piece_color, keypoints_obj, scene_color, keypoints_scene, good_matches, img_matches, matchColor = (128, 0, 0), flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #draw rectangles of the desired threshold distance around each cluster center
    for center in km.cluster_centers_:
        cv.rectangle(img_matches, (int(center[0])-threshold_distance+img_object.shape[1], int(center[1])-threshold_distance), (int(center[0])+threshold_distance+img_object.shape[1], int(center[1])+threshold_distance), (0, 0,255), 5)

    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    homography = True
    try:
        H, homography_array =  cv.findHomography(obj, scene, cv.RANSAC)
    except:
        H, homography_array =  cv.findHomography(obj, scene, 0)
    if type(H) == type(None):
        H, _ =  cv.findHomography(obj, scene, 0)
    if type(H) == type(None):
        homography = False
        print(f"Hoomography translation invalid")
    if homography:
    #-- Get the corners from the image_1 ( the object to be "detected" )
        obj_corners = np.empty((4,1,2), dtype=np.float32)
        obj_corners[0,0,0] = 0
        obj_corners[0,0,1] = 0
        obj_corners[1,0,0] = img_object.shape[1]
        obj_corners[1,0,1] = 0
        obj_corners[2,0,0] = img_object.shape[1]
        obj_corners[2,0,1] = img_object.shape[0]
        obj_corners[3,0,0] = 0
        obj_corners[3,0,1] = img_object.shape[0]
        scene_corners = cv.perspectiveTransform(obj_corners, H)

    #Old code for drawing a green rectangle around the most probable location
    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
    #cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
    #    (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    #cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
    #    (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    #cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
    #    (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    #cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
    #    (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)

    #Use that scene corners instead for listing the centroid of the translated points and put the grid location on the left side
        solution_center_x = int((scene_corners[:,0,0].mean()/img_scene.shape[1])*horizantal_pieces*h_width + h_width/2)
        cols_right = int((scene_corners[:,0,0].mean()/img_scene.shape[1])*horizantal_pieces)+1
        solution_center_y = int((scene_corners[:,0,1].mean()/img_scene.shape[0])*vertical_pieces*v_width + v_width/2)
        rows_down = int((scene_corners[:,0,1].mean()/img_scene.shape[0])*vertical_pieces)+1
        cv.putText(img_matches, \
        f'C:{cols_right}', \
        (20, img_object.shape[0]+100), cv.FONT_HERSHEY_COMPLEX, 4, (255,255,255))
        cv.putText(img_matches, \
        f'R:{rows_down}', \
        (20, img_object.shape[0]+200), cv.FONT_HERSHEY_COMPLEX, 4, (255,255,255))

    #-- Show detected matches
    cv.imwrite(re.sub(r'.(jpg|png)', '_solution.jpg', piece), img_matches)
    print(f"Saved {re.sub(r'.(jpg|png)', '_solution.jpg', piece)}")
    plt.figure(figsize = (20,20))
    plt.imshow(img_matches[...,::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return

#%% Camera Capture

cam = cv.VideoCapture(1)

cv.namedWindow("Preview")

img_counter = 0
webcam_res = (640, 480)
x = int(webcam_res[0]/4)
w = int(webcam_res[0]/2)
y = int(webcam_res[1]/4)
h = int(webcam_res[1]/2)
while True:
    ret, frame = cam.read()
    cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
    cv.imshow("Preview", frame)
    if not ret:
        break
    k = cv.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame2.png"#.format(img_counter)
        crop_img = frame[y:y+h, x:x+w]
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        try:
            best_image('gap.jpg', 'opencv_frame2.png')
        except ValueError:
            "Value Error"
cam.release()

cv.destroyAllWindows()
