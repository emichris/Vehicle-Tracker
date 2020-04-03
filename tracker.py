from imutils.video import VideoStream 
import argparse, datetime, imutils 
import time, cv2 , numpy as np

# construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser() 
ap.add_argument("-v", "--video", help="path to the video file") 
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size") 
args = vars(ap.parse_args()) 

# Get First Frame and use as background model 
vs = cv2.VideoCapture(args["video"]) 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
firstFrame = None; DESIRED_WIDTH = 750; 

while True:
    frame = vs.read() 
    frame = frame if args.get("video", None) is None else frame[1] 
    
    if (firstFrame is None) and (frame is None):  #unable to read video
        print("ERROR: Unable to read video")
        break; 

    if frame is None: #End of video
        break;   
    
    frame = imutils.resize(frame, width=DESIRED_WIDTH) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale 
    gray = cv2.GaussianBlur(gray, (21, 21), 0) #Blur to minimize noise # compute the absolute difference between the current frame and # first frame 
    
    #Assign first frame
    if firstFrame is None:
        firstFrame = gray; 
        continue; 
        
    frameDelta = cv2.absdiff(firstFrame, gray) 
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1] 
    # dilate the thresholded image to fill in holes, then find contours # on thresholded image
    
    thresh = cv2.dilate(thresh, None, iterations=2) 
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cnts = imutils.grab_contours(cnts) # loop over the contours 
    for index, c in enumerate(cnts): # if the contour is too small, ignore it 
        if cv2.contourArea(c) < args["min_area"]: 
            continue 
        # compute the bounding box for the contour, draw it on the frame, 
        # and update the text 
        (x, y, w, h) = cv2.boundingRect(c) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) 
        cv2.putText(frame, "{}".format(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), index) 

# perform connected component labelling, using propagated
    cv2.imshow("Traffic Video", frame)  #show video
    if (cv2.waitKey(1) & 0xFF) == ord("q"):  #exit if user presses 'q'
        break 
    #firstFrame = gray 
    
# exit video player and close all windows
vs.release() 
cv2.destroyAllWindows()