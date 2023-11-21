import cv2
import json
from gaze_tracking import GazeTracking
from deepface import DeepFace

gaze = GazeTracking()
video_capture = cv2.VideoCapture("vid1.flv")
num_looking_center = 0
num_looking_left = 0
num_looking_right = 0
num_faces_detected = 0
lst, grp, a, em = [], [], 0 ,[]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

prev_gaze_direction = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    face_analysis = DeepFace.analyze(frame,actions=["emotion"], enforce_detection=False)
    print(face_analysis)
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    
    print(len(faces))
    for (x, y, w, h) in faces:
        num_faces_detected += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if len(faces) <= 1:   # __
        lst.append(0)
    else:
        lst.append(1)     # ___    
    if len(faces) ==1:
        face_analysis = DeepFace.analyze(frame,actions=["emotion"], enforce_detection=False)
        em.append(face_analysis[0]["emotion"])

    gaze.refresh(frame)
    new_frame = gaze.annotated_frame()

    text = ""

    if gaze.is_center():
        text = "Looking center"
        if prev_gaze_direction != 'center':
            num_looking_center += 1
    elif gaze.is_left():
        text = "Looking left"
        if prev_gaze_direction != 'left':
            num_looking_left += 1
    elif gaze.is_right():
        text = "Looking right"
        if prev_gaze_direction != 'right':
            num_looking_right += 1

    prev_gaze_direction = text.split()[1] if text else None

    text += f"\nFaces detected: {num_faces_detected}"
    cv2.putText(new_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)

    cv2.imshow("Demo", new_frame)

    if cv2.waitKey(1) == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
lst.append(0)
for i in range(len(lst)):
    if lst[i] == lst[i-1] and lst[i] == 1:
        a += 1
    if lst[i] != lst[i-1]:
        if a > 3:
            grp.append(1)
        else:
            grp.append(0)
        a = 0

print(grp)
print(grp.count(1))
output = {
    "num_looking_center": num_looking_center,
    "num_looking_left": num_looking_left,
    "num_looking_right": num_looking_right,
    "num_faces_detected": grp.count(1)
}

with open("output.json", "w") as f:
    json.dump(output, f)
