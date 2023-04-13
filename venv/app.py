import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from io import StringIO
from main import sinogram_creation, save_dicom, save_image_at_iteration_2, save_image_at_iteration_sinogram
import pydicom
from PIL import Image


uploaded_file = None
image_iterations = None
n_iteration = None


def change_inputs_is_active():
    if st.session_state["disabled"]:
        st.session_state["disabled"] = False
    else:
        st.session_state["disabled"] = True


def show_iter_image():
    save_image_at_iteration_2(st.session_state["iters_photos"][int(n_iteration)], int(n_iteration))
    save_image_at_iteration_sinogram(st.session_state["iter_sinograms"][int(n_iteration)], int(n_iteration))

    st.session_state["img_iter_sinogram"] = col1.image(f"stages/iteration_sinogram_{int(n_iteration)}.png", caption="n-th iteration sinogram")
    st.session_state["img_iter"] = col1.image(f"stages/iteration_{int(n_iteration)}.png", caption="n-th iteration result")


def perform_ct_scan():
    if uploaded_file is not None or st.session_state['performed_ct']:
        st.session_state['performed_ct'] = True
        bytes_data = uploaded_file.getvalue()

        with open('init_image.png', 'wb') as file:
            file.write(bytes_data)
        image = Image.open('init_image.png')
        holder.image(image, caption='Initial image', width=200)
        image_path = os.path.abspath('init_image.png')
        image_read = cv2.imread(image_path, 0)
        width, height = np.shape(image_read)
        with st.spinner('Generating results'):
            result_image, image_iterations, iter_sinograms = sinogram_creation(image_read, width, height, angle, emiters, span, additive_or_substractive=False)

        col1, col2, col3 = holder.columns([1, 2, 1])

        with col2:
            col2.image('result.png', caption="Result Images")
            st.session_state["iters_number"] = len(image_iterations)
            st.session_state["iters_photos"] = image_iterations
            st.session_state["iter_sinograms"] = iter_sinograms

        if not st.session_state["disabled"]:
            save_dicom(result_image, str(name_input), str(description_input))


def load_dicom_image():
    if uploaded_file2 is not None:

        bytes_data = uploaded_file2.getvalue()
        with open('dummy_dcm.dcm', 'wb') as file:
            file.write(bytes_data)

        ds = pydicom.dcmread("dummy_dcm.dcm")
        img = Image.fromarray(ds.pixel_array)

        # save the image to a file
        img.save('loaded_dicom.png')
        st.image('loaded_dicom.png', caption="Loaded DICOM", width=800)


if "disabled" not in st.session_state:
    st.session_state["disabled"] = True
if "performed_ct" not in st.session_state:
    st.session_state["performed_ct"] = False

if not st.session_state.get("performed_ct", False):
    st.set_page_config(layout="wide")
    st.title('CT scan simulator')
    holder = st.empty()
    container = holder.container()
    # with holder.container():
    col1, col3, col4, col2 = container.columns([1, 1, 1, 1])

    col1.checkbox(label="Generate DICOM Image", on_change=change_inputs_is_active)
    name_input = col1.text_input("Patient Name", key='text_name', disabled=st.session_state["disabled"])
    description_input = col1.text_input("Description", key='text_description', disabled=st.session_state["disabled"])
    uploaded_file = col1.file_uploader("Choose a file")
    uploaded_file2 = col2.file_uploader("Choose a DICOM")
    btnResult = col1.button('Run', on_click=perform_ct_scan)
    btnResult2 = col2.button('Load DICOM', on_click=load_dicom_image)
    angle = col3.number_input(label="∆α", step=1, min_value=1, value=4)
    emiters = col3.number_input(label="n", step=1, min_value=0, value=180)
    span = col3.number_input(label="l", step=1, min_value=0, value=180)

if "holder" not in st.session_state:
    st.session_state["holder"] = None

if st.session_state['performed_ct']:
    if "img_iter" not in st.session_state:
        st.session_state["img_iter"] = None
    if st.session_state["holder"] is None:
        holder = st.empty()
        st.session_state["holder"] = holder
    else:
        holder = st.session_state["holder"]
    container = holder.container()
    # with holder.container():

    col1, col3, col4, col2 = container.columns([1, 1, 1, 1])
    st.image('result.png', caption="Result Images")
    n_iteration = col1.slider('Show iteration', min_value=1, max_value=st.session_state["iters_number"] - 1)
    col1.button("Generate iteration", on_click=show_iter_image)
    iter_img = st.session_state["img_iter"]
    if not st.session_state["disabled"]:
        with open('result.dcm', 'rb') as f:
            container.download_button('Download Image', f, file_name='result.dcm')
