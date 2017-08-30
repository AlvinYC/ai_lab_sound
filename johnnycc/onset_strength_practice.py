import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from mirex2016_dataset import verify_number_of_frames

# y, sr = librosa.load('dataset/beat_test_song.mp3', sr=44100, offset=72.5, duration=11.5) # default mono=True
y, sr = librosa.load('dataset/mirex_beat_tracking_2016/train/train1.wav', sr=44100, mono=True)

# From sound courses, frame duration between 20ms and 40ms was suggested
# From other examples,
# - n_fft: # samples of a frame = 2048 samples@22050Hz, 4096 samples@44100Hz = 92.879818594104ms = frame duration
# - hop_length = n_fft * 3 // 4, that means overlap n_fft / 4 = 512 samples@22050Hz, 1024 samples@44100Hz = 23.219954648526 = overlap duration
n_fft = 4096
hop_length = n_fft * 3 // 4
print(hop_length)

linear_freq = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
print('linear_freq.shape=', end='')
print(linear_freq.shape)
plt.figure(figsize=[16,4])
# amplitude_to_db(linear_freq) = power_to_db(linear_freq**2)
librosa.display.specshow(librosa.amplitude_to_db(linear_freq, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log');
#librosa.display.specshow(np.abs(linear_ft), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log');
plt.title('Power Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

melfb = librosa.filters.mel(sr=sr, n_fft=n_fft) # mel-frequency filter bank
plt.figure(figsize=[16,4])
librosa.display.specshow(melfb, x_axis='linear')
plt.ylabel('Mel filter')
plt.title('Mel filter bank')
plt.colorbar()
plt.tight_layout()
plt.show()

linear_ps = np.abs(linear_freq)**2 # Linear power spectram
mel_s = librosa.feature.melspectrogram(S=linear_ps)
print('mel_s.shape=', end='')
print(mel_s.shape)
plt.figure(figsize=[16, 4])
librosa.display.specshow(librosa.power_to_db(mel_s, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

S = librosa.power_to_db(mel_s, ref=np.max)
onset_env = librosa.onset.onset_strength(S=S, hop_length=hop_length, n_fft=n_fft)
print('onset_env.shape1=', end='')
print(onset_env.shape)
frames = range(len(onset_env))
t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

plt.figure(figsize=[16,6])
plt.plot(t, onset_env)
plt.xlim(0, t.max())
plt.ylim(0)
plt.xlabel('Time(sec)')
plt.title('Onset Strength Envelope 1');
plt.show()

onset_env = librosa.onset.onset_strength(y, sr=sr, hop_length=hop_length, n_fft=n_fft)
print('onset_env.shape2=', end='')
print(onset_env.shape)
frames = range(len(onset_env))
t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

plt.figure(figsize=[16,6])
plt.plot(t, onset_env)
plt.xlim(0, t.max())
plt.ylim(0)
plt.xlabel('Time(sec)')
plt.title('Onset Strength Envelope 2');
plt.show()
