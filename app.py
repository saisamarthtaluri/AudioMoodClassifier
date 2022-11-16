import streamlit as st
import os
from predict import pred_feature_index
from predict import predictor
from playsound import playsound
import time

st.title('Music Mood Classification')
up=st.file_uploader(label="Choose an Audio File",type=['.mp3'])


if up is not None:
    print(up)
    st.write(up.name)
    if st.button("play"):
        playsound(os.path.abspath(up.name))
      
    f = pred_feature_index(up.name)
    k = predictor(f)
    st.write('Output class is: ')
    st.write(k)

