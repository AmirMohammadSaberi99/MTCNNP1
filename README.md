# Face Detection & Alignment Toolkit

This repository provides a set of tools for face detection, landmark extraction, alignment, and comparative benchmarking using MTCNN, OpenCV’s Haar cascades, and Dlib.

## Repository Structure

```
├── 1.py    # 1. Static Face Detection with MTCNN
├── 2.py    # 2. Facial Landmark Extraction with MTCNN
├── 3.py    # 3. Face Alignment & Cropping with MTCNN Landmarks
├── 4.py    # 4. Real-Time MTCNN Face Detection on Webcam
├── 5.py    # 5. Comparative Benchmark: MTCNN vs. Haar Cascade vs. Dlib HOG Detectors
└── README.md                    # Documentation
```

---

## Prerequisites

Install the necessary Python packages:

```bash
pip install opencv-python mtcnn dlib tensorflow
```

* **opencv-python**: Image I/O and drawing utilities.
* **mtcnn**: Face detection & landmark extraction.
* **dlib**: HOG-based face detector.
* **tensorflow**: Backend for the MTCNN package.

> Note: On some systems, installing `dlib` may require additional build tools (CMake, Boost). See [Dlib installation instructions](http://dlib.net/).

---

## Scripts & Usage

### 1. Static Face Detection with MTCNN

**File:** `detect_faces_mtcnn.py`
Detects faces in a single image and draws green bounding boxes with confidence scores.

```bash
python detect_faces_mtcnn.py path/to/image.jpg [-o output.jpg]
```

* `-o output.jpg`: (optional) save annotated image.

### 2. Facial Landmark Extraction

**File:** `face_landmarks_mtcnn.py`
Detects faces and draws five key landmarks (eyes, nose, mouth corners).

```bash
python face_landmarks_mtcnn.py path/to/image.jpg [-o output.jpg]
```

* Landmark points shown in red, labeled by feature name.

### 3. Face Alignment & Cropping

**File:** `face_align_mtcnn.py`
Aligns each detected face so eyes are horizontal, crops to fixed size (default 256×256), and saves aligned crops.

```bash
python face_align_mtcnn.py path/to/image.jpg \
    --output_dir aligned_faces \
    [--width 256 --height 256] \
    [--eye_x 0.35 --eye_y 0.35]
```

* `--output_dir`: directory for aligned face images.
* `--eye_x`, `--eye_y`: relative eye position in the output crop.

### 4. Real-Time Face Detection (Webcam)

**File:** `webcam_mtcnn_realtime.py`
Captures video from your webcam and performs MTCNN face detection + landmarks live.

```bash
python webcam_mtcnn_realtime.py
```

* Press **q** or **ESC** to quit.

### 5. Comparative Benchmark: MTCNN vs. Haar vs. Dlib

**File:** `compare_face_detectors.py`
Runs three detectors on a single image, prints counts, and displays three windows with color-coded boxes:

* **Green**: Haar Cascade
* **Blue**: Dlib HOG
* **Red**: MTCNN

```bash
python compare_face_detectors.py path/to/image.jpg
```

Press any key to close the windows.

---

## License

This project is licensed under the MIT License. Feel free to use and modify.

---

## Acknowledgements

* MTCNN implementation: [mtcnn Python package](https://pypi.org/project/mtcnn/)
* Face detectors: OpenCV Haar cascades, Dlib HOG detector
