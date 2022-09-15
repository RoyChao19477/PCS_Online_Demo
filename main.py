import time
import streamlit as st
import numpy as np
import pandas as pd
import scipy

import torch
import torchaudio
import io

from pesq import pesq

import matplotlib.pyplot as plt
import librosa
import librosa.display

from PIL import Image


# ------- init setting -------
st.set_page_config(
    page_title="Perceptual Contrast Stretching on Target Feature for Speech Enhancements : Post-processing PCS",
    #page_icon="random",
    page_icon=Image.open('figs/icon_CITI.jpeg'),
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.title('Perceptual Contrast Stretching on Target Feature for Speech Enhancements')
st.subheader("Post-processing PCS")
# ------- end -------

# select box:
topic = st.selectbox("Select a PP-PCS version",
        (
            '(1) Original PP-PCS', 
            '(2) PP-PCS400',
        )
    )

# Stable Variances:
if 'state_1' not in st.session_state:
    st.session_state['state_1'] = 0
if 'state_2' not in st.session_state:
    st.session_state['state_2'] = 0
if 'state_3' not in st.session_state:
    st.session_state['state_3'] = 0
if 'state_4' not in st.session_state:
    st.session_state['state_4'] = 0
if 'state_5' not in st.session_state:
    st.session_state['state_5'] = 0

# ----- Warning Part -----
st.write("PCS Github: [https://github.com/RoyChao19477/PCS](https://github.com/RoyChao19477/PCS)")
st.write("Author: RongChao@2022")
st.caption("If you find the code useful in your research, please cite:")
cite = '''@article{chao2022perceptual,
  title={Perceptual Contrast Stretching on Target Feature for Speech Enhancement},
  author={Chao, Rong and Yu, Cheng and Fu, Szu-Wei and Lu, Xugang and Tsao, Yu},
  journal={Proc. of INTERSPEECH},
  year={2022}
}
'''
st.code(cite)
# ----- end -----

# used function
def no_fn():
    pass

def Sp_and_phase(signal, PCS, n_fft, hop_length):
    signal_length = signal.shape[0]
    y_pad = librosa.util.fix_length(signal, signal_length + n_fft // 2)
    y_pad = torch.tensor(y_pad)

    F = torch.stft(y_pad, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hamming_window(n_fft), return_complex=True)
    Lp = PCS * torch.transpose(torch.log1p(torch.abs(F)), 1, 0)
    phase = torch.angle(F)

    NLp = torch.transpose(Lp, 1, 0)

    return NLp, phase, signal_length


def SP_to_wav(mag, phase, signal_length, n_fft, hop_length):
    mag = torch.expm1(mag)
    Rec = torch.multiply(mag, torch.exp(1j*phase))
    result = torch.istft(Rec,
                           n_fft=n_fft,
                           hop_length=hop_length,
                           win_length=n_fft,
                           window=torch.hamming_window(n_fft), length=signal_length)
    return result

def draw(name, title):
  y, sr = librosa.load(io.BytesIO(name))
  x = librosa.stft( y )
  fig, ax = plt.subplots()
  img = librosa.display.specshow(
    librosa.amplitude_to_db(x,ref=np.max),
    y_axis='log', x_axis='time', ax=ax)
  ax.set_title(f'Power spectrogram: {title}')
  st.pyplot(fig)

# ------- HW1 - 1 -------
if topic == '(1) Original PP-PCS':
    st.header("PP-PCS")

    n_fft = 512
    hop_length = 256

    PCS = torch.ones(257)      # Perceptual Contrast Stretching
    PCS[0:3] = 1
    PCS[3:6] = 1.070175439
    PCS[6:9] = 1.182456140
    PCS[9:12] = 1.287719298
    PCS[12:138] = 1.4       # Pre Set
    PCS[138:166] = 1.322807018
    PCS[166:200] = 1.238596491
    PCS[200:241] = 1.161403509
    PCS[241:256] = 1.077192982

elif topic == '(2) PP-PCS400':
    st.header("PP-PCS400")
    
    n_fft = 400
    hop_length = 100

    PCS = torch.ones(201)
    PCS[0:3] = 1
    PCS[3:5] = 1.070175439
    PCS[5:8] = 1.182456140
    PCS[8:10] = 1.287719298
    PCS[10:110] = 1.4       # Pre Set
    PCS[110:130] = 1.322807018
    PCS[130:160] = 1.238596491
    PCS[160:190] = 1.161403509
    PCS[190:202] = 1.077192982

# ------- Upload Clean Wav -------
st.subheader("You can upload the audio manually here or use the sample audio.")
clean_wav, enh_wav = None, None
    
clean_wav_f = st.file_uploader("Upload Clean Audio", type=['.wav', '.mp3'], accept_multiple_files=False)
if clean_wav_f is not None:
    clean_wav = clean_wav_f.read()

enh_wav_f = st.file_uploader("Upload Noisy or Enhanced Audio", type=['.wav', '.mp3'], accept_multiple_files=False)
if enh_wav_f is not None:
    enh_wav = enh_wav_f.read()

colA, colB = st.columns(2)
with colA:
    if st.button("Use sample audio A"):
        st.write("A sample from VoiceBank-DEMAND dataset was adopted.")
        clean_wav_f = open('audio/p232_005_clean.wav', 'rb')
        clean_wav = clean_wav_f.read()
        enh_wav_f = open('audio/p232_005_noisy.wav', 'rb')
        enh_wav = enh_wav_f.read()

with colB:
    if st.button("Use sample audio B"):
        st.write("A sample from VoiceBank-DEMAND dataset was adopted.")
        clean_wav_f = open('audio/p232_005_clean.wav', 'rb')
        clean_wav = clean_wav_f.read()
        enh_wav_f = open('audio/p232_005_enh.wav', 'rb')
        enh_wav = enh_wav_f.read()

colC, colD = st.columns(2)
with colC:
    if st.button("Use sample audio C"):
        st.write("A sample from VoiceBank-DEMAND dataset was adopted.")
        clean_wav_f = open('audio/p232_007_clean.wav', 'rb')
        clean_wav = clean_wav_f.read()
        enh_wav_f = open('audio/p232_007_noisy.wav', 'rb')
        enh_wav = enh_wav_f.read()

with colD:
    if st.button("Use sample audio D"):
        st.write("A sample from VoiceBank-DEMAND dataset was adopted.")
        clean_wav_f = open('audio/p232_007_clean.wav', 'rb')
        clean_wav = clean_wav_f.read()
        enh_wav_f = open('audio/p232_007_enh.wav', 'rb')
        enh_wav = enh_wav_f.read()

if clean_wav is not None:
    st.write("Clean audio:")
    clean, sr = torchaudio.load(io.BytesIO(clean_wav))
    st.audio(clean_wav, format='audio/wav')
    draw(clean_wav, 'clean')


col1, col2 = st.columns(2)
with col1:
    if enh_wav is not None:
        st.write("Enhanced/Noisy audio:")
        noisy, sr = torchaudio.load(io.BytesIO(enh_wav))
        noisy = noisy/noisy.abs().max()
        st.info(f"PESQ score: {pesq(sr, clean.squeeze(0).numpy(), noisy.squeeze(0).numpy(), 'wb')}")
        st.audio(enh_wav, format='audio/wav')
        draw(enh_wav, 'noisy/enhanced')

with col2:
    if enh_wav is not None:
        noisy, sr = torchaudio.load(io.BytesIO(enh_wav))
        noisy_LP, Nphase, signal_length = Sp_and_phase(noisy.squeeze(), PCS, n_fft, hop_length)
        enhanced_wav = SP_to_wav(noisy_LP, Nphase, signal_length, n_fft, hop_length)
        enhanced_wav = enhanced_wav/enhanced_wav.abs().max()

        st.write("Enhanced/Noisy + PP-PCS/PP-PCS400 audio:")
        st.info(f"PESQ score: {pesq(sr, clean.squeeze(0).numpy(), enhanced_wav.squeeze(0).numpy(), 'wb')}")

        torchaudio.save(
            'cache/cache1.wav',
                torch.unsqueeze(enhanced_wav, 0),
            sr,
        )

        enhanced_wav_f = open('cache/cache1.wav', 'rb')
        enhanced_wav = enhanced_wav_f.read()

        st.audio(enhanced_wav, format='audio/wav')
        draw(enhanced_wav, 'noisy/enhanced')

