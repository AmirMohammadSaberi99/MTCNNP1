# face_align_mtcnn.py

"""
Detect faces with MTCNN, align them by the eye line, crop to fixed size,
and save each aligned face.
"""

import os
import cv2
import numpy as np
from mtcnn import MTCNN


def align_and_crop_face(
    img: np.ndarray,
    left_eye: tuple,
    right_eye: tuple,
    output_size: tuple = (256, 256),
    eye_pos: tuple = (0.35, 0.35)
) -> np.ndarray:
    """
    Aligns the face in img such that the line between left_eye and right_eye
    is horizontal, then crops a region of size output_size with eyes at
    relative position eye_pos.
    
    Args:
        img:       Original BGR image.
        left_eye:  (x, y) coords of left eye.
        right_eye: (x, y) coords of right eye.
        output_size: (width, height) of the aligned crop.
        eye_pos:     Relative output position of the left eye (fractions).
    
    Returns:
        Aligned & cropped BGR face image of size output_size.
    """
    # compute center between eyes
    lx, ly = left_eye
    rx, ry = right_eye
    eyes_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)

    # angle between the eye line and horizontal
    dx = rx - lx
    dy = ry - ly
    angle = np.degrees(np.arctan2(dy, dx))

    # desired positions
    ow, oh = output_size
    ex, ey = eye_pos
    # desired x-coordinate of right eye in output
    desired_rx = 1.0 - ex
    # current inter-eye distance
    dist = np.sqrt(dx * dx + dy * dy)
    # desired inter-eye distance in output
    desired_dist = (desired_rx - ex) * ow
    scale = desired_dist / dist

    # get rotation+scale matrix around eyes_center
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    # adjust translation so that eyes_center maps to (ow*0.5, oh*ey)
    tx = ow * 0.5 - eyes_center[0]
    ty = oh * ey  - eyes_center[1]
    M[0,2] += tx
    M[1,2] += ty

    # warp and crop
    aligned = cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_CUBIC)
    return aligned


def process_image(
    image_path: str,
    output_dir: str,
    output_size=(256,256),
    eye_pos=(0.35,0.35)
):
    """
    Detects faces in image_path, aligns & crops them, and saves to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # MTCNN expects RGB
    detector = MTCNN()
    detections = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not detections:
        print("No faces found.")
        return

    for i, face in enumerate(detections, start=1):
        box = face['box']             # [x, y, w, h]
        keypoints = face['keypoints'] # dict with 'left_eye','right_eye', etc.

        left_eye  = keypoints['left_eye']
        right_eye = keypoints['right_eye']

        aligned_face = align_and_crop_face(
            img, left_eye, right_eye,
            output_size=output_size,
            eye_pos=eye_pos
        )

        out_path = os.path.join(
            output_dir,
            f"face_{i:02d}.png"
        )
        cv2.imwrite(out_path, aligned_face)
        print(f"Saved aligned face #{i} to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect & align faces using MTCNN landmarks."
    )
    parser.add_argument(
        "image", help="Path to input image (e.g. photo.jpg)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="aligned_faces",
        help="Directory to save aligned face crops"
    )
    parser.add_argument(
        "--width", type=int, default=256,
        help="Output face crop width (default: 256)"
    )
    parser.add_argument(
        "--height", type=int, default=256,
        help="Output face crop height (default: 256)"
    )
    parser.add_argument(
        "--eye_x", type=float, default=0.35,
        help="Relative x-position of left eye in output (0–1)"
    )
    parser.add_argument(
        "--eye_y", type=float, default=0.35,
        help="Relative y-position of eyes in output (0–1)"
    )
    args = parser.parse_args()

    process_image(
        args.image,
        args.output_dir,
        output_size=(args.width, args.height),
        eye_pos=(args.eye_x, args.eye_y)
    )
