#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 00:03:50 2019

@author: clingsz
"""
import cv2
import numpy as np
import os
import time
from shutil import copyfile
from os.path import isfile, join

HTML_REFRESH_RATE = 10

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8") 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def run_webcam(mirror=False, night_vision=False):
    cam = cv2.VideoCapture(1)

    session_total = 0
    
    queue = []
    diff = []
    mean_img = 0
    window = 60
    size = 0
    fps = 10
    gamma = 1
    
    tstr = time.strftime('%Y_%m_%d_%H_%M_%S')
    session = 'session_' + tstr
    path = './session/{}/'.format(session)

    video_path = path + 'video.avi'
    os.makedirs(path)    
    img_path = path + 'images/'
    os.makedirs(img_path)
    
    img_size = (1280, 960)
#    file_name = 'diff.txt'        
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, img_size)
    while True:
        ret_val, img = cam.read()
        session_total += 1        
        if mirror: 
            img = cv2.flip(img, 1)
            
        cv2.imshow('Cam', img)        
        tstr = time.strftime('%Y_%m_%d_%H_%M_%S')        
        # record difference information
        queue.append(img)
        d = np.mean(abs(img - mean_img))
        diff.append(d)
        if size < window:
            mean_img = (mean_img * size + img) * 1.0 / (size + 1)
            size += 1
        elif size == window:
            front_img = queue[0]
            queue = queue[1:size] + [img]
            mean_img = mean_img - front_img * 1.0/size + img * 1.0/size

        dstr = int(d*100)
        # write to img file
        file_name = img_path + 'img_{}_{}.jpg'.format(tstr,dstr)
        cv2.imwrite(file_name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        # write to video
        out.write(img)
        # update html:
        if (session_total % HTML_REFRESH_RATE == 0):
            update_html(file_name)
                
        key = cv2.waitKey(1)
        if key == 27: 
                break  # esc to quit
        elif key == 113:
                break # "q" to quit
        elif key == 91: # '[' to decrease gamma
            gamma = max(0.5,gamma - 0.5)
        elif key == 93: # ']' to increase gamma
            gamma = gamma + 0.5
        time.sleep(1)        
    out.release()
    cv2.destroyAllWindows()
    print('Session {} finished, summarizing...'.format({session}))
    summarize(session)
    print('All done for Session {}'.format(session))
    
def update_html(filename):    
    # load the file
    copyfile(filename, './web/img.jpg')
    raw = """
    <!DOCTYPE html>
    <html>
    <body>    
    <h2>Current view</h2>
    <p>Display filename: {}</p>
    <img src="img.jpg" alt="Not exist" width="1280" height="960">
    </body>
    </html>
    """.format(
        filename
    )
    with open("./web/index.html", "w") as outf:
        outf.write(raw)

def summarize(session, fps=1, threshold=450):
    # Analyze the percentile and the threshold relationship
    pathIn = './session/{}/images/'.format(session)
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    print('Session {} has {} frames.'.format(session, len(files)))
    files = sorted(files)
    dif = [int(f.split('.')[0].split('_')[-1]) for f in files]
    for prc in [98,95,90,85,80]:
        print('Percentile {} threshold {:.0f} with {:.0f} frames.'.format(
                prc, 
                np.percentile(dif, prc), 
            len(files)*(100-prc)/100))

    # Use the selected threshold to generate video file
    n = np.sum(np.asarray(dif)>=threshold)
    print('Using threshold of {} obtain {} frames.'.format(threshold, n))
    video_path = './session/{}/video_over_{}.avi'.format(session, threshold)
    img_size = (1280, 960)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, img_size)
    print('Generating videos from the following files:')
    for i in range(len(files)):
        if dif[i] >= threshold:
            img = cv2.imread(pathIn + files[i])
            print(files[i],dif[i])
            out.write(img)
    out.release()
    print('Video {} finished.'.format(video_path))

run_webcam()

