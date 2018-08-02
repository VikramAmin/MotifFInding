import librosa
from librosa.display import specshow
import matplotlib as plt
import mir_eval

y = librosa.core.load('./audio/katy_perry.wav', sr=44100)
#y_harmonic, y_percussive = librosa.effects.hpss(y[0])

# We'll use a CQT-based chromagram here.  An STFT-based implementation also exists in chroma_cqt()
# We'll use the harmonic component to avoid pollution from transients
C = librosa.feature.chroma_cqt(y=y[0], sr=y[1])

# Make a new figure
#plt.figure(figsize=(12,4))

# Display the chromagram: the energy in each chromatic pitch class as a function of time
# To make sure that the colors span the full range of chroma values, set vmin and vmax

specshow(C, sr=y[1], x_axis='time', y_axis='chroma', vmin=0, vmax=1)

