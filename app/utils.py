import os
import cv2
import numpy as np
import tensorflow as tf
from typing import List
from matplotlib import pyplot as plt
import dlib

# Initialize the face detector and shape predictor
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("./detector/shape_predictor_68_face_landmarks.dat")

vocab = [x for x in " অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎংঃঁািীুূেৈোৌৃ"]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


# Load Video Function with Additional Debug Statements
def load_video(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    target_height, target_width = 54, 90  # Target dimensions

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = hog_face_detector(frame)
        if len(faces) == 0:
            # If no faces are detected, append an empty frame
            empty_frame = np.zeros((target_height, target_width, 1), dtype=np.uint8)
            frames.append(empty_frame)
            continue

        x67, y67 = 100, 100
        horizontal_padding, vertical_padding = 45, 27

        for face in faces:
            face_landmarks = dlib_facelandmark(frame, face)
            x67 = face_landmarks.part(67).x
            y67 = face_landmarks.part(67).y

        # Define lip window with padding
        y_start = max(y67 - vertical_padding, 0)
        y_end = min(y67 + vertical_padding, frame.shape[0])
        x_start = max(x67 - horizontal_padding, 0)
        x_end = min(x67 + horizontal_padding, frame.shape[1])

        lip_window = frame[y_start:y_end, x_start:x_end]

        # Pad the lip window if it's smaller than the target size
        if lip_window.shape[0] != target_height or lip_window.shape[1] != target_width:
            padded_lip_window = np.zeros((target_height, target_width), dtype=np.uint8)
            padded_lip_window[:lip_window.shape[0], :lip_window.shape[1]] = lip_window
            lip_window = padded_lip_window

        # Resize to the required dimensions (54x90) and add channel dimension
        lip_window = cv2.resize(lip_window, (target_width, target_height))
        lip_window = np.expand_dims(lip_window, axis=-1)

        frames.append(lip_window)

    cap.release()

    if len(frames) == 0:
        return np.zeros((0, target_height, target_width, 1))

    frames = np.array(frames)
    mean = np.mean(frames)
    std = np.std(frames)

    normalized_frames = (frames - mean) / std

    return normalized_frames

def load_alignments(path: str) -> List[int]:
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return char_to_num([" "])[:-1]  # Return an empty list if the file does not exist or is empty

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        tokens = []
        for line in lines:
            line = line.split()
            for word in line:
                for char in word:
                    tokens.extend(char)
                tokens.extend(' ')
        return char_to_num(tokens)[:-1]
    except Exception as e:
        print(f"Error loading alignments from {path}: {e}")
        return char_to_num([" "])[:-1]  # Return an empty list in case of any error


# def load_alignments(path: str) -> List[int]:
#     with open(path, 'r',encoding='utf-8') as f:
#         lines = f.readlines()
#     tokens = []
#     for line in lines:
#         line = line.split()
#         for word in line:
#             for char in word:
#                 tokens.extend(char)
#             tokens.extend(' ')
#     return char_to_num(tokens)[:-1]

def load_data(path: tf.Tensor): 
    
    
    path = path.numpy().decode('utf-8')
    file_name = os.path.splitext(os.path.basename(path))[0]
    parent_dir = os.path.dirname(os.path.dirname(path))
    subdirectory = os.path.basename(os.path.dirname(path))
   
    # print(parent_dir)
    # Adjust the paths according to your dataset structure
    video_path = os.path.join(parent_dir, 'clips', file_name + '.mp4')
    alignment_path = os.path.join(parent_dir, 'texts', file_name + '.txt')

    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    try:
        alignments = load_alignments(alignment_path)
    except FileNotFoundError:
        print(f"Alignment file not found: {alignment_path}")
        alignments = char_to_num([" "])[:-1]
    return frames, alignments

def mappable_function(path: str) -> List[tf.Tensor]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result
