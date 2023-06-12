import cv2
from object_detection import ObjectDetection
import math

od = ObjectDetection()

cap = cv2.VideoCapture("los_angeles.mp4")

center_prev_pt = []

tracking_id = {}
track_id = 0

count = 0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

result = cv2.VideoWriter('filename.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         25, size)

while True:
    
    ret, frame = cap.read()
    count = count+1

    if not ret:
        break

    center_pt = []
    
    (class_ids, scores, bboxes) = od.detect(frame)


    for box in bboxes:
        (x, y, w, h) = box
        cx = int((x+x+w)/2)
        cy = int((y+y+h)/2)
        center_pt.append((cx,cy))
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)
        #cv2.circle(frame, (cx,cy), 5, (0,255,0),-1)
        

    # Only in beginning
    if count<=2:
        for pt2 in center_pt:
            for pt1 in center_prev_pt:
                distance = math.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])

                if distance < 10:
                    tracking_id[track_id] = pt2
                    track_id = track_id + 1

    
    else:
        tracking_id_copy = tracking_id.copy()
        current_pt_copy = center_pt.copy()
        #print(current_pt_copy)
        for object_id, pt1 in tracking_id_copy.items():
            object_exists = False
            for pt2 in current_pt_copy:
                distance = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])

                if distance<10:
                    tracking_id[object_id] = pt2
                    object_exists = True
                    if pt2 in center_pt:
                        center_pt.remove(pt2)
                        continue

            if not object_exists:
                tracking_id.pop(object_id)
            
        for pt in center_pt:
            tracking_id[track_id] = pt
            track_id += 1

        print(tracking_id)
        for object_id, pt in tracking_id.items():
            cv2.circle(frame, pt, 5, (0,255,0),-1)
            cv2.putText(frame, str(object_id), (pt[0],pt[1]-7), 1,2, (0,0,255), 2)


    center_prev_pt = center_pt.copy()

    cv2.imshow("Frame", frame)

    result.write(frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

result.release()


