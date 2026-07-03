"""
Day 1 — DSM Project Starter
Goal: Confirm MediaPipe Face Mesh runs live on your webcam, and start pulling
out the two signals the whole project is built on: EAR (eye aspect ratio)
and MAR (mouth aspect ratio).

Setup:
    pip install mediapipe opencv-python numpy

Run:
    python day1_face_landmarks.py
    (press 'q' to quit)
"""

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Landmark indices for eyes and mouth (MediaPipe's 468-point face mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 39, 181, 0, 17]  # left, right, top-outer, top-inner, ...


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
        print("Could not open webcam. Check camera permissions/index.")
        return

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

                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.3f}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Rough thresholds — you will tune these after watching your
                # own EAR values for a normal blink vs eyes-closing-slowly.
                if avg_ear < 0.20:
                    cv2.putText(frame, "EYES CLOSING", (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("DSM - Day 1: Landmarks + EAR/MAR", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
