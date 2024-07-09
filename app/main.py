import streamlit as st
import os
import cv2  # Import OpenCV for webcam video capturing
import tempfile
import shutil  # Import shutil to copy files
import time
from subprocess import call
import numpy as np
import tensorflow as tf
from modelutil import load_model
from utils import load_data, num_to_char
from videoToframe import collect_and_pad_single_video  # Adjust import based on your directory structure

# Set the layout to the Streamlit app as wide
st.set_page_config(page_title="LipReading App", page_icon="🧊", layout="wide", initial_sidebar_state="auto")

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Lip-Reading')
    st.info('This application is originally developed by the deep learning model.')

st.title('LipReading App')

# Function to capture live video for 2 seconds
def capture_video():
    cap = cv2.VideoCapture(0)  # Open the default webcam

    if not cap.isOpened():
        st.error('Error: Could not open webcam.')
        return None
    
    frames = []
    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Use default fps if unable to get
    while (time.time() - start_time) < 3:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    
    # Save the video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()
    
    return temp_video.name

# Function to generate a unique filename
def generate_unique_filename(base_dir, prefix="clip", extension=".mp4"):
    existing_files = os.listdir(base_dir)
    existing_numbers = [int(f[len(prefix):-len(extension)]) for f in existing_files if f.startswith(prefix) and f.endswith(extension)]
    next_number = max(existing_numbers, default=0) + 1
    return f"{prefix}{next_number}{extension}"

# Function to process video using ffmpeg
def process_video(file_path):
    base_dir = os.path.dirname(file_path)
    unique_output_name = generate_unique_filename(base_dir, prefix="clip", extension=".mp4")
    output_file = os.path.join(base_dir, unique_output_name)
    ffmpeg_command = f"ffmpeg -i {file_path} -vf scale=320:320 {output_file}"
    call(ffmpeg_command, shell=True)
    os.remove(file_path)  # Remove the original file
    shutil.move(output_file, file_path)  # Move the processed file to the original file location
    return file_path

# Directory to save captured videos
save_dir = './test/captured_videos/clips/'

# Ensure the directory exists or create it
os.makedirs(save_dir, exist_ok=True)

# Capture live video for 2 seconds
if st.button('Capture Live Video for 3 seconds'):
    video_path = capture_video()

    if video_path:
        # Generate a unique filename and save the captured video
        unique_filename = generate_unique_filename(save_dir)
        saved_video_path = os.path.join(save_dir, unique_filename)
        shutil.copy(video_path, saved_video_path)  # Copy the file instead of moving
        os.remove(video_path)  # Remove the temporary file
        
        # Process the video using ffmpeg
        processed_video_path = process_video(saved_video_path)
        
        st.success(f'Video captured, processed, and saved as {processed_video_path}')
        
    else:
        st.error('Failed to capture video.')

base_dir = 'test'

subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

selected_subdir = st.selectbox('Choose subdirectory from directory test', subdirs)

# Generate the clips directory path based on the selected subdirectory
clips_dir = os.path.join(base_dir, selected_subdir, 'clips')

options = os.listdir(clips_dir)
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    # Rendering the video
    with col1:
        st.info('Video which is inputted for the testing')
        file_path = os.path.join(clips_dir, selected_video)
        
        # Process the video using ffmpeg
        # file_path = process_video(file_path)
        
        video = open(file_path, 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
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
else:
    st.warning('No videos found in the selected subdirectory.')
