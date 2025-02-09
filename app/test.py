import streamlit as st
import os
import imageio
import numpy as np
import tensorflow as tf
from modelutil import load_model
from utils import load_data, num_to_char
from videoToframe import collect_and_pad_single_video
import base64

# Set the layout to the Streamlit app
st.set_page_config(page_title="LipReading App", page_icon="🧊", layout="wide", initial_sidebar_state="auto")

# Custom CSS for styling
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
}
.sidebar .sidebar-content {
    background-color: #f0f0f0;
}
.video-container {
    display: flex;
    justify-content: center;  /* Center the container */
    margin: 0 auto;  /* Center the container */
    max-width: 800px;  /* Set a maximum width for the container */
    padding-left: 20px;  /* Add padding on the left side */
}

</style>
""", unsafe_allow_html=True)

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
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    # Rendering the video
    with col1:
        st.info('Video which is inputted for the testing')
        file_path = os.path.join(clips_dir, selected_video)
        video = open(file_path, 'rb')
        video_bytes = video.read()

        # Base64 encode the video
        video_base64 = base64.b64encode(video_bytes).decode()

        # Set explicit width and height for the video
        video_width = 320  # Desired width
        video_height = 240  # Desired height

        # Use HTML and CSS to render the video
        st.markdown(
            f'''
            <div class="video-container">
                 <video controls>
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            ''',
            unsafe_allow_html=True
        )

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
