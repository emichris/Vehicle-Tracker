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

state = [0, 1] # track states


print("PRESS 's' to select anchor points.")

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    _, frame_o = vs.read()
    
    # check to see if we have reached the end of the stream
    if frame_o is None:
        break
        
    # resize the frame (so we can process it faster)
    frame = imutils.resize(frame_o, width=1000)
    
    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    (success, boxes) = trackers.update(frame)
    # loop over the bounding boxes and draw then on the frame
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        vs = cv2.VideoCapture(video_dir)        # Load video
        _, frame_o = vs.read()

        # resize the frame (so we can process it faster)
        frame = imutils.resize(frame_o, width=1000)

        for i in state:
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            box = cv2.selectROI("Frame", frame, fromCenter=False,
                showCrosshair=True)
            # create a new object tracker for the bounding box and add it
            # to our multi-object tracker
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            trackers.add(tracker, frame, box)

            strip = [];
            (success, boxes) = trackers.update(frame)
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                strip.append((x+w//2, y+h//2))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #break out of loop
        break
        
    # press `r` to reset 
    elif key == ord("r"):
        trackers = cv2.MultiTracker_create() # reset multi-object tracker

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        exit()

cv2.destroyAllWindows()
print("\nGOING THROUGH THE VIDEO USING THE ANCHOR TAGS\n")
print("PRESS 'l' to draw strip.")


#Get the y-value from a line using two points on the line
def getyfrom(y, point1, point2):
    (x1, y1) = point1
    (x2, y2) = point2
    m = (y2-y1) / (x2-x1)

    return int(y1 + m*(x-x1) )


#Add method for cropping and rotating roi
def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.line(clone, refPt[-2], refPt[-1], (0, 255, 0), 2)
        cv2.imshow("Draw R-O-I", clone)
        #cv2.imshow("mask", mask)

# Get regions of Interest
strip = [];     (success, boxes) = trackers.update(frame)
for box in boxes:
    (x, y, w, h) = [int(v) for v in box]
    strip.append((x+w//2, y+h//2))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

((x1, y1), (x2, y2)) = (strip[0], strip[1])
strip.append((x2, y2+20)); strip.append((x1, y1+20))
cv2.line(frame, strip[0], strip[1], (15,15,15), 1)
cv2.line(frame, strip[1], strip[2], (15,15,15), 1)
cv2.line(frame, strip[2], strip[3], (15,15,15), 1)
cv2.line(frame, strip[3], strip[0], (15,15,15), 1)

# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("Draw R-O-I")
cv2.setMouseCallback("Draw R-O-I", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("Draw R-O-I", clone)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        refPt = [];  cropping = False
        clone = frame.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break
   
#refPt = [(484, 423), (591, 508), (591, 508), (594, 496), (594, 496), (495, 418), (495, 418), (486, 418)]

# if there are two reference points, then crop the region of interest
if len(refPt) >= 3:
    roi_corners = np.array([refPt], dtype=np.int32)
    (x1, y1) = strip[0]
    free_draw_dist = [(x1-x2, y1-y2) for (x2,y2) in refPt]
    cv2.destroyWindow("Draw R-O-I"); cv2.destroyWindow("Masked Image");  
    #free_draw_dist = [(x1-x2, y1-y2) for ((x1,y1), (x2, y2)) in zip(strip, refPt)]

    # APPLY ROI ROTATION AND CROP
    rect = cv2.minAreaRect(roi_corners) #print("rect: {}".format(rect))
    box = cv2.boxPoints(rect);   box = np.int0(box)

    # img_crop will the cropped rectangle, img_rot is the rotated image
    img_crop, img_rot = crop_rect(frame, rect)
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    images = [gray]; cur_images = [gray];

roi = [];


''' PROCESS VIDEO USING MY BACKGROUND SUBTRACTION METHOD'''
WIDTH = 1000; N = 10; cur_ind = N-1
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#width  = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
##height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = vs.get(cv2.CAP_PROP_FPS)
size = (frame.shape[1], frame.shape[0])
out = cv2.VideoWriter("video_output/video1.avi", fourcc, fps, size)
#out = cv2.VideoWriter('video_output/1.avi', fourcc, fps, (1000, 562))

for i in range(N-1):
    # VideoStream or VideoCapture object
    _, frame_o = vs.read()
    
    # check to see if we have reached the end of the stream
    if frame_o is None:
        break
        
    # resize the frame (so we can process it faster)
    frame = imutils.resize(frame_o, width=WIDTH)

    # grab the updated bounding box coordinates 
    (success, boxes) = trackers.update(frame)
    # loop over the anchor bounding boxes and draw them on the frame
    strip = []
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        strip.append((x+w//2, y+h//2))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

    ((x1, y1), (x2, y2)) = (strip[0], strip[1])
    strip.append((x2, y2+20)); strip.append((x1, y1+20))

    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) >= 3:
        (x1, y1) = strip[0]
        refPt = [(x1-x2, y1-y2) for (x2,y2) in free_draw_dist]
        roi_corners = np.array([refPt], dtype=np.int32)
        
        # APPLY ROI ROTATION AND CROP
        rect = cv2.minAreaRect(roi_corners) #print("rect: {}".format(rect))
        box = cv2.boxPoints(rect);   box = np.int0(box)

        # img_crop will the cropped rectangle, img_rot is the rotated image
        img_crop, img_rot = crop_rect(frame, rect)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.blur(gray,(3,3))
        images = np.append(images, [gray], axis=0)
        cur_images = np.append(cur_images, [gray], axis=0)
        
while True:    # loop over frames from the video stream
    _, frame_o = vs.read()
    
    # check to see if we have reached the end of the stream
    if frame_o is None:
        break
        
    # resize the frame (so we can process it faster)
    frame = imutils.resize(frame_o, width=1000)
    
    # grab the updated bounding box coordinates 
    (success, boxes) = trackers.update(frame)
    # loop over the anchor bounding boxes and draw them on the frame
    strip = []
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        strip.append((x+w//2, y+h//2))
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

    ((x1, y1), (x2, y2)) = (strip[0], strip[1])
    strip.append((x2, y2+20)); strip.append((x1, y1+20))

    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) >= 3:
        (x1, y1) = strip[0]
        refPt = [(x1-x2, y1-y2) for (x2,y2) in free_draw_dist]
        roi_corners = np.array([refPt], dtype=np.int32)
        
        # APPLY ROI ROTATION AND CROP
        rect = cv2.minAreaRect(roi_corners) #print("rect: {}".format(rect))
        box = cv2.boxPoints(rect);   box = np.int0(box)

        # img_crop will the cropped rectangle, img_rot is the rotated image
        img_crop, img_rot = crop_rect(frame, rect)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        
        # Perform background subtraction
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY) #convert to grayscale        
        #gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.blur(gray,(3,3))
        avg_frame = np.mean(images, axis=0)
        diff = cv2.convertScaleAbs(gray - avg_frame)
        images[cur_ind] = gray;
        cur_ind = cur_ind + 1 if cur_ind < N-1 else 0
        
        #diff = cv2.GaussianBlur(diff, (5, 5), 0) #Blur to minimize
        
        alpha, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, np.ones((5,3), np.uint8), iterations = 1)
        thresh = cv2.dilate(thresh, np.ones((7,5), np.uint8), iterations = 2)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        cnts = imutils.grab_contours(cnts) # loop over the contours 
        
        rects = []

        # loop over the detections
        for index, c in enumerate(cnts):
            # if the contour is too small, ignore it
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            (x, y, w, h) = cv2.boundingRect(c)
            if h > 20 and cv2.contourArea(c) > 150:
                rects.append([x, y, x + w, y + h])
                cv2.rectangle(img_crop, (x, y), (x + w, y + h), (0, 0, 255), 2) 

        #cv2.imshow("thresh", thresh); cv2.imshow("diff", diff)
        #cv2.imshow("back-sub video", img_crop)
        white = np.zeros(img_crop.shape, dtype=np.uint8); white[: :]=255
        concat_img = cv2.hconcat([cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB), white,
                                  cv2.cvtColor(diff,cv2.COLOR_GRAY2RGB), white,
                                  cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB), white,
                                  img_crop])
        cv2.imshow("diff  thresh  img_crop", concat_img)
        
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

    #draw anchors
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # show the output frame
    cv2.imshow("Frame", frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
vs.release(); out.release()
cv2.destroyAllWindows()
