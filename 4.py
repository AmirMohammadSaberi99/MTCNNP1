# webcam_mtcnn_realtime.py

"""
Real‐time face detection (and landmarks) on webcam using MTCNN.
"""

import cv2
from mtcnn import MTCNN

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    # Initialize MTCNN detector
    detector = MTCNN()

    print("Starting webcam face detection. Press 'q' or ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed, exiting")
            break

        # MTCNN expects RGB images
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces + landmarks
        detections = detector.detect_faces(rgb)

        # Draw results
        for face in detections:
            x, y, w, h = face['box']
            conf = face['confidence']
            keypoints = face['keypoints']

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Draw confidence
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            # Draw landmarks
            for point in keypoints.values():
                cv2.circle(frame, point, 3, (0, 0, 255), -1)

        # Show result
        cv2.imshow("MTCNN Real-Time Face Detection", frame)

        # Exit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
