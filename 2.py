# face_landmarks_mtcnn.py

"""
Detect faces and facial landmarks using MTCNN, draw boxes & landmark points.
"""

import cv2
from mtcnn import MTCNN

def detect_and_draw_landmarks(image_path: str, output_path: str = None):
    # Load image in BGR (OpenCV default)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Convert to RGB for MTCNN
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Initialize MTCNN detector
    detector = MTCNN()

    # Detect faces + landmarks
    detections = detector.detect_faces(img_rgb)
    print(f"Detected {len(detections)} face(s)")

    for i, face in enumerate(detections, start=1):
        # Bounding box
        x, y, w, h = face['box']
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Confidence
        conf = face['confidence']
        label = f"{conf:.2f}"
        cv2.putText(img_bgr, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Landmarks
        keypoints = face['keypoints']
        # draw each landmark as a small circle
        for name, (lx, ly) in keypoints.items():
            color = (0, 0, 255)  # red for landmarks
            cv2.circle(img_bgr, (lx, ly), 3, color, -1)
            # optional: label each point
            cv2.putText(img_bgr, name, (lx + 5, ly - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Show result
    cv2.imshow("MTCNN Face & Landmarks", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save if requested
    if output_path:
        cv2.imwrite(output_path, img_bgr)
        print(f"Saved annotated image to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect faces + landmarks in an image with MTCNN"
    )
    parser.add_argument(
        "input_image",
        help="Path to input image (e.g. photo.jpg)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Optional path to save the annotated image"
    )
    args = parser.parse_args()

    detect_and_draw_landmarks(args.input_image, args.output)
