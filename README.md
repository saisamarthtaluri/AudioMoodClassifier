# DAProject2022

We often want to listen to music that best fits our
current emotion. A grasp of emotions in songs might be a great
help for us to effectively discover music. Automated music
mood recognition constitutes an active task in the field of MIR
(Music Information Retrieval). In our project, we aim to
provide an effective mechanism to classify music into various
human emotions based on audio and the metadata. We will be
using Mel-frequency cepstral coefficients (MFCCs) extracted
from Mel spectrograms as attributes for training our CNN
model. In order to extract the MFCC, we will be using Discrete
Cosine Transform (DCT) as well as Fast Fourier Transform
(FFT). Currently classification of music based on mood is
manually done by selecting songs that belong to a particular
mood and naming the playlist according to the mood, such as
“relaxing”. Here we investigate the possibility of assigning such
information automatically, without user interaction.

# Instructions to run

Run the files in the order
1. features_ext.py
2. model_training.py
3. predict.py in another terminal
4. Run the command streamlit run app.py with predict.py running in the background.
