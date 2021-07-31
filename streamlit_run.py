import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
from PIL import Image
from u2net_run import u2net_run
from u2net_post import u2net_post
import os

#### Refs
# https://towardsdatascience.com/how-to-run-and-share-your-deep-learning-web-app-on-colab-a13f9d2cbc4e


#### Initialize session

if 'ipath' not in st.session_state:
    st.session_state['ipath'] = None


#### Run

st.title('Background removal by U-2-Netp')
st.write('By litemoment.com')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    latest_iteration = st.empty()
    bar = st.progress(0)
    latest_iteration.text('Loading uploaded image')
    image = Image.open(uploaded_file)
    #dataframe = pd.read_csv(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Name: {} Type: {} Size: {}".format(uploaded_file.name, uploaded_file.type, uploaded_file.size))
    st.write("")
    bar.progress(10)

    now = int(time.time() * 1000)
    cwd = os.getcwd()
    uploaded_dir = os.path.join(cwd, 'uploaded')
    if not os.path.exists(uploaded_dir):
        os.mkdir(uploaded_dir)
    uploaded_path = os.path.join(uploaded_dir, 'u{}.jpg'.format(now))
    image.save(uploaded_path)

    img_name = uploaded_path.split(os.sep)[-1]
    d_dir = uploaded_path[0:-len(img_name)]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    output_file = d_dir+imidx+'.png'

    latest_iteration.text('Segmenting uploaded image')
    u2net_run(uploaded_path, output_file)
    bar.progress(85)
    latest_iteration.text('Post processing')
    result = u2net_post(uploaded_path, output_file)
    st.image(result, caption='Result Image', use_column_width=True)
    st.write("")
    bar.progress(100)
    latest_iteration.text('Done.')
