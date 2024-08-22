import face_recognition  # type: ignore
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize video capture
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Cannot open video capture device")
    exit()

# Load and encode images
try:
    ishu_image = face_recognition.load_image_file("photos/ishu.jpg")
    ishu_encodings = face_recognition.face_encodings(ishu_image)
    if ishu_encodings:
        ishu_encodings = ishu_encodings[0]
    else:
        print("No faces found in ishu.jpg")
        exit()

    mayank_image = face_recognition.load_image_file("photos/mayankAg.jpg")
    mayank_encodings = face_recognition.face_encodings(mayank_image)
    if mayank_encodings:
        mayank_encodings = mayank_encodings[0]
    else:
        print("No faces found in mayankAg.jpg")
        exit()

except FileNotFoundError:
    print("Image files not found")
    exit()

known_face_encodings = [
    ishu_encodings,
    mayank_encodings,
]

known_face_names = [
    "ishu",
    "mayank",
]

students = known_face_names.copy()

# Initialize lists
face_locations = []
face_encodings = []
face_names = []

# Get current date for the CSV file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open the CSV file
try:
    with open(current_date + '.csv', 'w+', newline='') as f:
        lnwriter = csv.writer(f)

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            if not ret:
                print("Cannot read video frame")
                break

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1]) # type: ignore

            # Find all the face locations in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)

            # Ensure face_locations is not empty before encoding
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            else:
                face_encodings = []

            face_names = []
            for face_encoding in face_encodings:
                # Check if the face matches known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = ""

                if matches:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                face_names.append(name)

                # Record the attendance
                if name in students:
                    students.remove(name)
                    now = datetime.now()  # Update time inside the loop
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])
                    f.flush()  # Flush the buffer
                    print(f"Attendance recorded for {name} at {current_time}")

            # Display the resulting frame
            cv2.imshow('Attendance System', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f"An error occurred: {e}")