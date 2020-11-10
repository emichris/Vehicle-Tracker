# import the necessary packages
from imutils.video import VideoStream
import argparse
import numpy as np, cv2
import imutils, time


# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=3):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker(); maxobjectID = 0;

###############################



from imutils.video import VideoStream
import argparse
import imutils, numpy as np
import time, cv2

args = {
    'video': "stable_Baltimore & Charles - AM (1)_TrimEnd.mp4", #"../overpass.mp4",
    'tracker':'kcf'
}

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

ROI_CAPTURED = False; refPt = []
video_dir = "Guilford & Madison - AM.avi" #
video_dir = "stable_Baltimore & Charles - AM (1)_TrimEnd.mp4"
#video_dir = "Guilford & LexingtonFayette - AM.avi"



backSub = cv2.createBackgroundSubtractorMOG2()
trackers = cv2.MultiTracker_create()    # Create Multi-Tracker Object
vs = cv2.VideoCapture(video_dir)        # Load video
print("[INFO] video path loaded..")


''' PROCESS VIDEO USING MY BACKGROUND SUBTRACTION METHOD'''
WIDTH = 500; N = 10; cur_ind = N-1
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = vs.get(cv2.CAP_PROP_FPS)
#size = (frame.shape[1], frame.shape[0])
#out = cv2.VideoWriter("video_output/video1.avi", fourcc, fps, size)
#out = cv2.VideoWriter('video_output/1.avi', fourcc, fps, (1000, 562))

# VideoStream or VideoCapture object
_, frame_o = vs.read()

# check to see if we have reached the end of the stream
if frame_o is None:
    exit()
    
# resize the frame (so we can process it faster)
frame = imutils.resize(frame_o, width=WIDTH)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
#gray = cv2.blur(gray,(5,5))
images = [gray]
cur_images = [gray]

number_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)

for i in range(N-1):
    # VideoStream or VideoCapture object
    _, frame_o = vs.read()
    
    # check to see if we have reached the end of the stream
    if frame_o is None:
        break
    # resize the frame (so we can process it faster)
    frame = imutils.resize(frame_o, width=WIDTH)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #gray = cv2.blur(gray,(5,5))
    images = np.append(images, [gray], axis=0)
    cur_images = np.append(cur_images, [gray], axis=0)
    
#k1=gray.copy(); k2=k1.copy(); k3 = k2.copy();      
while True:    # loop over frames from the video stream
    _, frame_o = vs.read()
    
    # check to see if we have reached the end of the stream
    if frame_o is None:
        break
        
    # resize the frame (so we can process it faster)
    frame = imutils.resize(frame_o, width=WIDTH)

    # Perform background subtraction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale        
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.blur(gray,(5, 5))
    avg_frame = np.mean(images, axis=0) 
    diff = cv2.convertScaleAbs(gray - avg_frame)
    images[cur_ind] = gray.copy();
    #k3 = k2.copy(); k2=k1.copy(); k1=gray.copy(); images[cur_ind] = k3.copy();
    cur_ind = cur_ind + 1 if cur_ind < N-1 else 0
    cv2.imshow("gray", gray); #cv2.imshow('avg', avg_frame)
    
    #diff = cv2.GaussianBlur(diff, (5, 5), 0) #Blur to minimize
    #cv2.imshow("diff", diff)

    '''
    alpha, thresh = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)
    cv2.imshow("Threshold - 0", thresh);

    thresh = cv2.erode(thresh, np.ones((3,3), np.uint8), iterations = 2)
    cv2.imshow("Threshold - 1", thresh);

    thresh = cv2.dilate(thresh, np.ones((5,5), np.uint8), iterations = 3)
    cv2.imshow("Threshold - 2", thresh);

    thresh = cv2.erode(thresh, np.ones((3,3), np.uint8), iterations = 2)
    cv2.imshow("Threshold - 3", thresh);

    thresh = cv2.dilate(thresh, np.ones((3,3), np.uint8), iterations = 1)
    cv2.imshow("Threshold - 4", thresh);

    ''' # Works fine
    alpha, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    cv2.imshow("Threshold - 0", thresh);

    thresh = cv2.erode(thresh, np.ones((5,3), np.uint8), iterations = 1)
    cv2.imshow("Threshold - 1", thresh);

    thresh = cv2.dilate(thresh, np.ones((11,5), np.uint8), iterations = 2)
    cv2.imshow("Threshold - 2", thresh);
    
    '''
    from matplotlib import pyplot as plt
    img = diff.copy()
    #img = cv2.medianBlur(img,5)
    ret,th1 = cv2.threshold(img,25,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,15,15)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,15,15)
    titles = ['Original Image', 'Global Thresholding (v = 25)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show(); thresh = th1.copy()

    '''
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cnts = imutils.grab_contours(cnts) # loop over the contours 
    
    rects = []

    # loop over the detections
    for index, c in enumerate(cnts):
        # if the contour is too small, ignore it
        # compute the (x, y)-coordinates of the bounding box for
        # the object, then update the bounding box rectangles list
        (x, y, w, h) = cv2.boundingRect(c)
        if True: #h > 20 and cv2.contourArea(c) > 150:
            rects.append([x, y, x + w, y + h])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        '''
        objects = ct.update(rects) # send detections to centroid tracker
        k=0; text = "New detections:"
        
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            if objectID+1 > maxobjectID:
                maxobjectID = objectID+1
                
            text = text + " {}".format(objectID); k = k+1
        cv2.rectangle(frame, (0, 0), (200, 30), (255, 255, 255), -1) #(100, 100, 255)
        #cv2.putText(frame, text, (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Total detected: {}".format(maxobjectID), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 150), 2)
        '''
        
    # show the output frame
    cv2.imshow("Frame", frame)
    #out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
vs.release(); #out.release()
cv2.destroyAllWindows()
