"""
Day 2 — DSM Project: Labeled Data Logging
Goal: Log EAR/MAR to a CSV, tagged with a state label you control live via
keypress. This becomes your first small labeled dataset for the LSTM model.

Controls while running:
    1 -> label frames as "alert"
    2 -> label frames as "drowsy"
    3 -> label frames as "distracted"
    0 -> pause labeling (label = "none", frames still shown but not logged)
    q -> quit and save CSV

Suggested recording plan (30-60 sec each):
    - Press 1, sit normally, blink naturally               -> alert
    - Press 2, mime drowsiness: slow blinks, head nodding   -> drowsy
    - Press 3, mime distraction: look away, check a "phone" -> distracted

Setup:
    (venv already active from Day 1)

Run:
    python day2_data_logger.py
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from datetime import datetime

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 39, 181, 0, 17]

LABEL_MAP = {
    ord('1'): "alert",
    ord('2'): "drowsy",
    ord('3'): "distracted",
    ord('0'): "none",
}

LABEL_COLORS = {
    "alert": (0, 255, 0),
    "drowsy": (0, 165, 255),
    "distracted": (0, 0, 255),
    "none": (200, 200, 200),
}


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    vertical_1 = euclidean(pts[1], pts[5])
    vertical_2 = euclidean(pts[2], pts[4])
    horizontal = euclidean(pts[0], pts[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices]
    vertical = euclidean(pts[2], pts[3])
    horizontal = euclidean(pts[0], pts[1])
    return vertical / (horizontal + 1e-6)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"dsm_data_{timestamp_str}.csv"
    csv_file = open(csv_filename, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame_time", "ear", "mar", "label"])
    csv_file.flush()

    current_label = "none"
    frame_count = 0
    logged_count = 0

    print(f"Logging to: {csv_filename}")
    print("Press 1=alert  2=drowsy  3=distracted  0=pause  q=quit")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles
                    .get_default_face_mesh_tesselation_style(),
                )

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(landmarks, MOUTH, w, h)

                if current_label != "none":
                    writer.writerow([time.time(), f"{avg_ear:.4f}",
                                      f"{mar:.4f}", current_label])
                    csv_file.flush()
                    logged_count += 1

                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.3f}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            color = LABEL_COLORS[current_label]
            cv2.putText(frame, f"LABEL: {current_label.upper()}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Logged rows: {logged_count}", (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "1=alert 2=drowsy 3=distracted 0=pause q=quit",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

            cv2.imshow("DSM - Day 2: Data Logger", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key in LABEL_MAP:
                current_label = LABEL_MAP[key]
                print(f"-> Switched label to: {current_label}")

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\nDone. Saved {logged_count} labeled rows to {csv_filename}")


if __name__ == "__main__":
    main()
