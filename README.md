# SEEG Contact Fingerprinting Dashboard
Isaac Levy, DS7400-Deep Learning, Fall 2026

## Overview
This project explores fingerprinting (i.e., identifying unique signatures) of contacts from stereo-electroencephalography (SEEG) electrodes. These electrodes were implanted in the brains of patients with drug resistant epilepsy and interictal, i.e., resting-state time periods, are used here. The goal is to use deep learning to identify specific electrode contacts. The data come from the [HUP iEEG Epilepsy Dataset](https://doi.org/10.18112/openneuro.ds004100.v1.1.3) (see `/example_data/README` for a full citation).

This project was built using `streamlit` and relies on `mne`, `plotly`, and `pandas` for data processing and visualization. It requires `pytorch` for the deep learning component.

Currenly implemented features:
1. 3 example patients, each with one ~5 minute interictal recording (only the first 30 seconds are used in the viewer for now).
2. An interactive data viewer where the user can select which patient and which electrode channel(s) to view. Users can zoom and adjust the viewing window and select multiple channels at once.
3. The ability to add and view annotations. Annotations are created using a form, where the user can select which contact/channel the annotation applies to (one or all), the time window, and the type of annotation (artifact, spike, noisy, other), as well as write an optional descriptive note about the annotation. After an annotation is created, it is saved in a CSV file and rendered on the viewer as an orange box. Annotations can also be removed by the user. A handful of example annotations for one subject are pre-loaded.
4. A comparison between a simple MLP with 2 hidden layers versus a more complex CNN for electrode contact fingerprinting (classification). Training/validation loss curves for both models are shown, as well as accuracy on the test set and on an entirely separate recording. Separate models were trained for each patient, since the number of electrodes and locations are entirely different. Chance accuracy ranges from ~1-3% - ~2.7%, depending on the number of electrode contacts.

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

## Other notes
* The reference was not documented, and there is no evidence for common average referencing (CAR) or bipolar reference, so I am applying CAR as preprocessing step. Additionally, I am applying a low-pass filter at 1 Hz, and a notch filter at 60 Hz (and harmonics) for power line noise.
* The basic setup of the simple CNN I used comes from a [GitHub repository](https://github.com/hsd1503/resnet1d/blob/master/cnn1d.py) by Shenda Hong, with some modifications, such as decreaseing the original dropout percentage from 0.5 to 0.1.