import cv2
import json
from gaze_tracking import GazeTracking

gaze = GazeTracking()
video_capture = cv2.VideoCapture("vid1.flv")

num_looking_center = 0
num_looking_left = 0
num_looking_right = 0

while True:
  ret, frame = video_capture.read()
  if not ret:
    break

  gaze.refresh(frame)

  new_frame = gaze.annotated_frame()
  text = ""

  if gaze.is_center():
    text = "Looking center"
    num_looking_center += 1
  elif gaze.is_left():
    text = "Looking left"
    num_looking_left += 1
  elif gaze.is_right():
    text = "Looking right"
    num_looking_right += 1

  cv2.putText(new_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
  cv2.imshow("Demo", new_frame)

  if cv2.waitKey(1) == 27:
    break

video_capture.release()
cv2.destroyAllWindows()

output = {
    "num_looking_center": num_looking_center,
    "num_looking_left": num_looking_left,
    "num_looking_right": num_looking_right
}

with open("output1.json", "w") as f:
    json.dump(output, f)