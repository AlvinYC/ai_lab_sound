import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

fs = 44100 # Hz
f1 = 512 # Hz
samples = fs * 5 # n_of_samples within 5 seceonds
x1 = np.arange(samples)
y1 = np.array([np.sin(2*np.pi*f1 * (i/fs)) for i in x1])

f2 = 2 * f1
sample = fs * 5
x2 = x1 #np.arange(sample) # the same as x1
y2 = np.array([np.sin(2*np.pi*f2 * (i/fs)) for i in x2])

y = y1 + y2

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(36,16))
ax1.plot(x1[:(fs//100)], y1[:(fs//100)]); # show the wave within 100 ms
ax1.plot(x2[:(fs//100)], y2[:(fs//100)]); # show the wave within 100 ms
ax1.grid(True)

ax2.plot(x2[:(fs//100)], y[:(fs//100)]); # show the wave within 100 ms
ax2.grid(True)

# From sound courses, frame duration between 20ms and 40ms was suggested
# From other examples,
# - n_fft: # samples of a frame = 2048 samples@22050Hz, 4096 samples@44100Hz = 92.879818594104ms = frame duration
# - hop_length = n_fft * 3 // 4, that means overlap n_fft / 4 = 512 samples@22050Hz, 1024 samples@44100Hz = 23.219954648526 = overlap duration
sr = fs
n_fft = 4096
hop_length = n_fft * 3 // 4
linear_freq = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
ax3.set_title('Power Spectrogram', fontsize=8)
librosa.display.specshow(librosa.amplitude_to_db(linear_freq, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log');
plt.colorbar(format='%+2.0f dB')

plt.tight_layout(pad=8, w_pad=1, h_pad=1)
plt.show()