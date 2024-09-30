#pip install cmake
#pip install face-recognition
#pip install opencv-python
#pip install numpy
#pip install --upgrade pip setuptools wheel  ,if face-recognition package doesn't get installed

import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Load Known Faces
absar_image = face_recognition.load_image_file("faces/absar.jpg")
absar_encoding = face_recognition.face_encodings(absar_image)[0]

shadab_image = face_recognition.load_image_file("faces/shadab.jpg")
shadab_encoding = face_recognition.face_encodings(shadab_image)[0]

known_face_encoding = [absar_encoding, shadab_encoding]
known_face_names = ["Mohd Absar Alam", "Shadab Alam"]

# List of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open a CSV file for attendance
f = open(f"{current_date}.csv", "w", newline="")
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Time"])

while True:
    # Capture frame-by-frame
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Add the text if a person is present
            if name in students:
                students.remove(name)
                current_time = datetime.now().strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

                # Display attendance on the screen
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_corner_of_text = (10, 100)
                font_scale = 1.5
                font_color = (255, 0, 0)
                thickness = 3
                line_type = 2
                cv2.putText(frame, f"{name} Present", bottom_left_corner_of_text, font, font_scale, font_color,
                            thickness, line_type)

    # Display the resulting frame
    cv2.imshow("Attendance", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
f.close()
