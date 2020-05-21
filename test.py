###coded by sno....

# ### Load libs
import os
import cv2
import time
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from  model import autoencoder


                                
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


#Test any image
def IsImageHasAnomaly(autoencoder,filePath,threshold):  
    #im = cv2.resize(cv2.imread(filePath), (420, 420))
    im = cv2.resize(filePath, (420, 420))
    im = im * 1./255
    datas = np.zeros((1,  420, 420, 3))
    image = im.reshape((1,) + im.shape)
    # print(image.shape)  
    predicted_image = autoencoder.predict(image)
    _mse = mse(predicted_image, image) 
    # print('_mse: {}'.format(_mse))
    # if _mse > threshold:
    #     print('image is anomaly')
    return _mse 



# autoencoder.load_weights('autoencoder-vgg.h5')

# # set threshold manually
# threshold=4 #0.14

def main():
    autoencoder.load_weights('autoencoder-vgg.h5')

    videopath = 'video.avi' # 
    outpath = './output.avi'

    vid = cv2.VideoCapture(videopath)

    cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Stream1',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Stream2',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Stream3',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stream', (800,600))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret,frame = vid.read()
    vw = frame.shape[1]
    vh = frame.shape[0]
    # print ("Video size", vw,vh)
    outvideo = cv2.VideoWriter(videopath.replace("avi",'det1.avi'),fourcc,30,(1520,880))

    frames = 0
    starttime = time.time()
    while(True):
        ret, frame = vid.read()
        if not ret:
            break
        frames += 1
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # set threshold manually
        threshold=7 #0.14
        frame = frame[100:980,200:1720]
        # M = frame.shape[0]//2
        # N = frame.shape[1]//2
        # tiles = [frame[x:x+M,y:y+N] for x in range(0,frame.shape[0],M) for y in range(0,frame.shape[1],N)]
        
        # for i in range(len(tiles)):
        MSE =IsImageHasAnomaly(autoencoder, frame, threshold)
        print(MSE)
        if MSE > threshold:# or MSE < 3:
            cv2.putText(frame, "Anomaly" , (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,0,0), 5)
        cv2.imshow('Stream', frame)
        print(frame.shape)
            # cv2.imshow('Stream1', tiles[1])
            # cv2.imshow('Stream2', tiles[2])
            # cv2.imshow('Stream3', tiles[3])
        outvideo.write(frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    totaltime = time.time()-starttime
    print(frames, "frames", totaltime/frames, "s/frame")
    cv2.destroyAllWindows()
    outvideo.release()

if __name__ == "__main__":
    main()
    pass
