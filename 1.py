# face_detection_mtcnn.py

"""
Detect faces in a static image using MTCNN and draw bounding boxes.
"""

import cv2
from mtcnn import MTCNN

def detect_and_draw(image_path: str, output_path: str = None):
    # Load image with OpenCV (BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to RGB for MTCNN
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Initialize MTCNN face detector
    detector = MTCNN()

    # Detect faces
    detections = detector.detect_faces(img_rgb)
    print(f"Found {len(detections)} face(s)")

    # Draw bounding boxes and confidence
    for i, face in enumerate(detections, start=1):
        x, y, width, height = face['box']
        conf = face['confidence']

        # Draw rectangle on the original BGR image
        cv2.rectangle(
            img_bgr,
            (x, y),
            (x + width, y + height),
            (0, 255, 0), 2
        )
        # Label with confidence
        label = f"{conf:.2f}"
        cv2.putText(
            img_bgr,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Show result
    cv2.imshow("MTCNN Face Detection", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save to disk if output_path provided
    if output_path:
        cv2.imwrite(output_path, img_bgr)
        print(f"Annotated image saved to: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect faces in an image using MTCNN"
    )
    parser.add_argument(
        "input_image",
        help="Path to the input image file (e.g. photo.jpg)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="(Optional) Path to save the annotated image"
    )
    args = parser.parse_args()

    detect_and_draw(args.input_image, args.output)
