import streamlit as st

st.set_page_config(layout = 'wide')

st.markdown(
    """
    <div style="text-align: center;">
        <h2>Electrode Contact Fingerprinting</h2>
    </div>
    
<div style="text-align: center;">
    <h3>Isaac Levy</h3>
    <p>DS7400 - Deep Learning, Fall 2026</p>
</div>
    """,
    unsafe_allow_html=True
    
)

st.markdown(
    """
    <div style="text-align: center;">
        <h2>Home Page</h2>
        <p>To navigate this site, please click on the desired page in the left sidebar.</p>
        <p>
            This project explores fingerprinting (i.e., identifying unique signatures) of contacts from 
            stereo-electroencephalography (SEEG) electrodes. These electrodes were implanted in the brains of patients
            with drug resistant epilepsy and interictal, i.e., resting-state time periods, are used here.
            The goal is to use deep learning in various ways to identify specific electrode contacts.
            The data come from the <a href="https://doi.org/10.18112/openneuro.ds004100.v1.1.3">HUP iEEG Epilepsy Dataset</a>.
        </p>
    </div>
    
    """,
    unsafe_allow_html=True
)