# SEEG Contact Fingerprinting Dashboard
Isaac Levy, DS7400-Deep Learning, Fall 2026

## Overview
This project explores fingerprinting (i.e., identifying unique signatures) of contacts from stereo-electroencephalography (SEEG) electrodes. These electrodes were implanted in the brains of patients with drug resistant epilepsy and interictal, i.e., resting-state time periods, are used here. The goal is to use deep learning to identify specific electrode contacts. The data come from the [HUP iEEG Epilepsy Dataset](https://doi.org/10.18112/openneuro.ds004100.v1.1.3) (see the data README for a full citation).

This project was built using `streamlit` and relies on `mne`, `plotly`, and `pandas` for data processing and visualization. 

Currenly implemented features:
1. 3 example patients, each with one ~5 minute interictal recording (only the first 30 seconds are used in the viewer for now).
2. An interactive data viewer where the user can select which patient and which electrode(s) time series to view. Users can zoom and adjust the viewing window.
3. The ability to add and view annotations. Annotations are created using a form, where the user can select which contact/channel the annotation applies to (one or all), the time window, and the type of annotation (artifact, spike, noisy, other), as well as write an optional descriptive note about the annotation. After an annotation is created, it is saved in a CSV file and rendered on the viewer as an orange box. Annotations can also be removed by the user. A handful of example annotations for one subject are pre-loaded.

## Running the application locally

### Clone the repository
```bash
git clone git@github.com:i-levy/seeg-dashboard.git
```
### Create the environment and open the app
```bash
cd seeg-dashboard
conda env create -f environment.yml
conda activate dashboard
streamlit run Home_Page.py
```