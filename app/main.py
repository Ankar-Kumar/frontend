import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
from modelutil import load_model
from utils import load_data, num_to_char, load_video
from videoToframe import collect_and_pad_single_video  # Adjust import based on your directory structure

# Set the layout to the Streamlit app as wide
st.set_page_config(page_title="LipReading App", page_icon="🧊", layout="wide", initial_sidebar_state="auto")

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Lip-Reading')
    st.info('This application is originally developed by the deep learning model.')

st.title('LipReading App')

base_dir = 'test'

subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

selected_subdir = st.selectbox('Choose subdirectory from directory test', subdirs)

# Generate the clips directory path based on the selected subdirectory
clips_dir = os.path.join(base_dir, selected_subdir, 'clips')

options = os.listdir(clips_dir)
options.insert(0,'Use Webcam')  # Add option to use webcam
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

def capture_webcam_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return None
    st.info("Capturing video from webcam. Please remain steady for 2 seconds.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    
    out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 320))
    frames = []
    for i in range(60):  # Capture 2 seconds of video at 30 FPS
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture video frame.")
            break
        out.write(frame)
        frames.append(frame)
    
    cap.release()
    out.release()
    st.success("Video captured successfully.")
    return 'test.mp4'

if selected_video == 'Use Webcam':
    if st.button('Start Webcam'):
        file_path = capture_webcam_video()
        if file_path:
            selected_video = file_path

# If webcam video is captured or a pre-existing video is selected
if selected_video and selected_video != 'Use Webcam':
    file_path = 'test.mp4' if selected_video == 'test.mp4' else os.path.join(clips_dir, selected_video)
    
    video = open(file_path, 'rb')
    video_bytes = video.read()
    with col1:
        st.info('Video which is inputted for the testing')
        st.video(video_bytes)

    with col2:
        if selected_video == 'test.mp4':
            video = load_video(file_path)
            annotations = np.array([-1])
        else:
            video, annotations = load_data(tf.convert_to_tensor(file_path))
            st.info('Actual sentence which is spoken')
            st.text(tf.strings.reduce_join(num_to_char(annotations)).numpy().decode('utf-8'))
        
        st.info('Predicted sentences by our model')
        
        model = load_model()
        
        video_padded, annotations_padded = collect_and_pad_single_video(video, annotations)
        video_padded = np.expand_dims(video_padded, axis=0)
        
        yhat = model.predict(video_padded)
        decoder = tf.keras.backend.ctc_decode(yhat, [115], greedy=True)[0][0].numpy()
        
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
        st.info('This is the output of the deep learning model as tokens')
        st.text(np.array(annotations))

