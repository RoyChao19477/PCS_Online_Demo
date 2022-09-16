import streamlit as st
import numpy as np
import scipy
import pandas as pd

import torch
import torchaudio
import io

from pesq import pesq
import soundfile as sf
from pystoi import stoi

import matplotlib.pyplot as plt
import librosa
import librosa.display

from PIL import Image


# ------- init setting -------
st.set_page_config(
    page_title="Perceptual Contrast Stretching on Target Feature for Speech Enhancements : Post-processing PCS",
    #page_icon="random",
    #page_icon=Image.open('figs/icon_CITI.jpeg'),
    page_icon=Image.open('figs/icon_CITI.png'),
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
            '(3) Original PP-PCS (with tunable parameters)'
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

elif topic == '(3) Original PP-PCS (with tunable parameters)':
    st.header("PP-PCS tunable parameters")

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

    BIF = torch.ones(9)
    BIF[0] = 0.0000
    BIF[1] = 0.010
    BIF[2] = 0.026
    BIF[3] = 0.041
    BIF[4] = 0.057
    BIF[5] = 0.046
    BIF[6] = 0.034
    BIF[7] = 0.023
    BIF[8] = 0.011

    colG1, colG2 = st.columns(2)
    with colG1:
        g_min = st.slider(
                'Minimun of gamma:',
                0.0, 5.0, value=1.0, step=0.01)
    with colG2:
        g_max = st.slider(
                'Maximum of gamma:',
                g_min, 5.0, value=1.4, step=0.01)
    if g_min >= g_max:
        st.error('Error: Minimum should smaller than maximum', icon="🚨")
    else:
        PCS = torch.ones(257)      # Perceptual Contrast Stretching
        PCS[0:3]        = (g_max - g_min) / 0.057 * BIF[0] + g_min
        PCS[3:6]        = (g_max - g_min) / 0.057 * BIF[1] + g_min
        PCS[6:9]        = (g_max - g_min) / 0.057 * BIF[2] + g_min
        PCS[9:12]       = (g_max - g_min) / 0.057 * BIF[3] + g_min
        PCS[12:138]     = (g_max - g_min) / 0.057 * BIF[4] + g_min
        PCS[138:166]    = (g_max - g_min) / 0.057 * BIF[5] + g_min
        PCS[166:200]    = (g_max - g_min) / 0.057 * BIF[6] + g_min
        PCS[200:241]    = (g_max - g_min) / 0.057 * BIF[7] + g_min
        PCS[241:256]    = (g_max - g_min) / 0.057 * BIF[8] + g_min

    f_x = torch.ones(8000)
    f_x[0:100] = PCS[0]
    f_x[100:200] = PCS[3]
    f_x[200:300] = PCS[6]
    f_x[300:400] = PCS[9]
    f_x[400:4400] = PCS[12]
    f_x[4400:5300] = PCS[138]
    f_x[5300:6400] = PCS[166]
    f_x[6400:7700] = PCS[200]
    f_x[7700:8000] = PCS[241]

    f_o = torch.ones(8000)
    f_o[0:100] = 1
    f_o[100:200] = 1.070175439
    f_o[200:300] = 1.182456140
    f_o[300:400] = 1.287719298
    f_o[400:4400] = 1.4
    f_o[4400:5300] = 1.322807018
    f_o[5300:6400] = 1.238596491
    f_o[6400:7700] = 1.161403509
    f_o[7700:8000] = 1.077192982


    chart_data = pd.DataFrame(
            torch.stack([f_x, f_o]).permute(1,0).numpy(),
            columns=['Your Gamma-PCS', 'Original Gamma-PCS'])
    st.area_chart(chart_data)
    


# ------- Upload Clean Wav -------
st.subheader("You can upload the audio manually here or use the sample audio.")
clean_wav, enh_wav = None, None
    
colU1, colU2 = st.columns(2)
with colU1:
    clean_wav_f = st.file_uploader("Upload Clean Audio", type=['.wav', '.mp3'], accept_multiple_files=False)
    if clean_wav_f is not None:
        clean_wav = clean_wav_f.read()

with colU2:
    enh_wav_f = st.file_uploader("Upload Noisy or Enhanced Audio", type=['.wav', '.mp3'], accept_multiple_files=False)
    if enh_wav_f is not None:
        enh_wav = enh_wav_f.read()

colA, colB, colC, colD = st.columns(4)
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
        len_min = clean.size(1) if clean.size(1) < noisy.size(1) else noisy.size(1)
        st.info(f"PESQ score: {pesq(sr, clean.squeeze(0).numpy(), noisy.squeeze(0).numpy(), 'wb')}")
        st.info(f"STOI score: {stoi(clean.squeeze(0)[:len_min].numpy(), noisy.squeeze(0)[:len_min].numpy(), sr, extended=False)}")
        st.audio(enh_wav, format='audio/wav')
        draw(enh_wav, 'noisy/enhanced')

with col2:
    if enh_wav is not None:
        noisy, sr = torchaudio.load(io.BytesIO(enh_wav))
        noisy_LP, Nphase, signal_length = Sp_and_phase(noisy.squeeze(), PCS, n_fft, hop_length)
        enhanced_wav = SP_to_wav(noisy_LP, Nphase, signal_length, n_fft, hop_length)
        enhanced_wav = enhanced_wav/enhanced_wav.abs().max()

        len_min = clean.size(1) if clean.size(1) < enhanced_wav.size(0) else enhanced_wav.size(0)
        st.write("Enhanced/Noisy + PP-PCS/PP-PCS400 audio:")
        st.info(f"PESQ score: {pesq(sr, clean.squeeze(0).numpy(), enhanced_wav.squeeze(0).numpy(), 'wb')}")
        st.info(f"STOI score: {stoi(clean.squeeze(0)[:len_min].numpy(), enhanced_wav.squeeze(0)[:len_min].numpy(), sr, extended=False)}")

        torchaudio.save(
            'cache/cache1.wav',
                torch.unsqueeze(enhanced_wav, 0),
            sr,
        )

        enhanced_wav_f = open('cache/cache1.wav', 'rb')
        enhanced_wav = enhanced_wav_f.read()

        st.audio(enhanced_wav, format='audio/wav')
        draw(enhanced_wav, 'noisy/enhanced')

