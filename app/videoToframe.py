import pickle
import numpy as np
import os

def collect_and_pad_single_video(video, label, max_frame_length=115, frame_shape=(54, 90, 1), max_label_length=55):
    # Convert video and label to numpy arrays
    video = np.array(video)
    label = np.array(label)

    # Pad frames
    padded_frames = np.zeros((max_frame_length, *frame_shape), dtype=np.float32)
    frame_length = video.shape[0]
    padded_frames[:frame_length] = video

    # Pad labels
    padded_labels = np.zeros((max_label_length,), dtype=np.int64)
    label_length = label.shape[0]
    padded_labels[:label_length] = label

    return padded_frames, padded_labels
