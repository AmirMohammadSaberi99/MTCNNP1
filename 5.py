# compare_face_detectors.py

"""
Compare face detection on a single image using:
 - OpenCV Haar Cascade
 - Dlib HOG detector
 - MTCNN
Draws results side-by-side in separate windows.
"""

import cv2
import dlib
from mtcnn import MTCNN
import argparse
import sys

def detect_haar(gray, face_cascade):
    rects = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # convert to (x,y,w,h) list
    return rects

def detect_dlib(gray, dlib_detector):
    dets = dlib_detector(gray, 1)
    boxes = [(d.left(), d.top(), d.width(), d.height()) for d in dets]
    return boxes

def detect_mtcnn(rgb, mtcnn_detector):
    dets = mtcnn_detector.detect_faces(rgb)
    boxes = []
    for det in dets:
        x, y, w, h = det['box']
        boxes.append((x, y, w, h))
    return boxes

def draw_boxes(img, boxes, color, label):
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            img, label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2
        )

def main():
    parser = argparse.ArgumentParser(
        description="Compare Haar Cascade, Dlib, and MTCNN face detectors"
    )
    parser.add_argument(
        "image", help="Path to input image"
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not read image {args.image}", file=sys.stderr)
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1) OpenCV Haar Cascade
    haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_xml)
    boxes_haar = detect_haar(gray, face_cascade)

    # 2) Dlib HOG detector
    dlib_detector = dlib.get_frontal_face_detector()
    boxes_dlib = detect_dlib(gray, dlib_detector)

    # 3) MTCNN
    mtcnn_detector = MTCNN()
    boxes_mtcnn = detect_mtcnn(rgb, mtcnn_detector)

    # Print counts
    print(f"[Haar Cascade] detected {len(boxes_haar)} faces")
    print(f"[Dlib HOG]     detected {len(boxes_dlib)} faces")
    print(f"[MTCNN]        detected {len(boxes_mtcnn)} faces")

    # Prepare visuals
    img_haar  = img.copy()
    img_dlib  = img.copy()
    img_mtcnn = img.copy()

    draw_boxes(img_haar,  boxes_haar,  (0, 255, 0), "Haar")
    draw_boxes(img_dlib,  boxes_dlib,  (255, 0, 0), "Dlib")
    draw_boxes(img_mtcnn, boxes_mtcnn, (0, 0, 255), "MTCNN")

    # Show results
    cv2.imshow("Haar Cascade", img_haar)
    cv2.imshow("Dlib HOG",     img_dlib)
    cv2.imshow("MTCNN",        img_mtcnn)

    print("Press any key to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
