    
# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('SpeachReading')
    st.info('This application is originally developed from the deep learning model.')

st.title('LipReading App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join( 'data','clips'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('data','clips', selected_video)
        # os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open(file_path, 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        st.info('Actual sentence which is spoken')
        st.text(tf.strings.reduce_join(num_to_char(annotations)).numpy().decode('utf-8'))
        st.info('This is the output of the deep learning model as tokens')
        print(1)
        model = load_model()
        print(video.shape)
        num_frames = video.shape[0]

        target_num_frames = 115
        padding = [(0, max(target_num_frames - num_frames, 0))] + [(0, 0)] * (video.ndim - 1)
        video_padded = np.pad(video, padding, mode='edge')

        # Expand dimensions to match the expected input shape of the model
        video_padded = np.expand_dims(video_padded, axis=0)
        print(video_padded.shape)
        yhat = model.predict(video_padded)
        decoder = tf.keras.backend.ctc_decode(yhat, [115], greedy=True)[0][0].numpy()
        
        # Convert prediction to text
        st.info('Predicted sentences by our model')        
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)





  # text_file_path = os.path.join('test', 'p15', 'texts', f"{os.path.splitext(selected_video)[0]}.txt")
        # with open(text_file_path, 'r', encoding='utf-8') as file:
        #     actual_text = file.read().strip()

        # st.info('Actual sentence which is spoken')
        # st.text(actual_text)