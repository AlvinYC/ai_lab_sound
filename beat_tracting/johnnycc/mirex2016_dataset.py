import librosa
import numpy as np
import os
import sys
import argparse
import pathlib

# Verify the number of frames
# - The number of frames (n_of_frames)
#  if 0 < hop_length <= n_fft,
#  (n_of_frames - 1) * hop_length + n_fft = sr * the total time in second
#  n_of_frames * hop_length - hop_length + n_fft = sr * the total time in second
# 
#  - Formula : n_of_frames = (sr * the total time in second + hop_length - n_fft) / hop_length
def verify_number_of_frames(n_of_frames, y, sr, n_fft, hop_length):
    __total_time = len(y) / sr
    __n_of_frames = int(np.ceil((sr * __total_time + hop_length - n_fft) / hop_length))
    if __n_of_frames != n_of_frames:
        print("%d != %d, total_time=%d" % (n_of_frames, __n_of_frames, __total_time))
        raise ValueError

def save_trainXY(traindata_path, trainX, trainY):
    np.save(traindata_path + 'trainX.npy', trainX, allow_pickle=True, fix_imports=False)
    np.save(traindata_path + 'trainY.npy', trainY, allow_pickle=True, fix_imports=False)
    
def load_trainXY(traindata_path):
    trainX_path = traindata_path + 'trainX.npy'
    trainY_path = traindata_path + 'trainY.npy'
    n_of_frames = 0
    n_of_freq_bins = 0
    trainX = np.ndarray([])
    trainY = np.ndarray([])
    if os.path.isfile(trainX_path) and os.access(trainX_path, os.R_OK) and \
       os.path.isfile(trainY_path) and os.access(trainY_path, os.R_OK):
        trainX = np.load(trainX_path)
        trainY = np.load(trainY_path)
        n_of_frames, n_of_freq_bins = trainX.shape
        n_of_frames = int((n_of_frames / 40) / 20) # total 20 songs, each song was labeled by 40 listeners
        print('Load trainX.npy and trainY.npy (%d, %d)' % (n_of_frames, n_of_freq_bins))
    else:
        print('Generate trainX.npy and trainY.npy')
        for i in range(1, 21):
            train_x = np.load(traindata_path + 'train%d_x.npy' % (i))
            train_y = np.load(traindata_path + 'train%d_y.npy' % (i))
            #print('train_x.shape: '+ str(train_x.shape))
            #print('train_y.shape: '+ str(train_y.shape))

            # train_x
            #   - dimension = [n_of_frames, n_of_freq_bins] = [431, 2049] or [431, 128]
            #   - train1_x[i][j] is the "amplitude" of jth_freq_bin@ith_frame
            # train_y, each train audio was labled beats by 40 listeners
            #   - dimension = [n_of_listeners, n_of_frames] = [40, 431]
            #   - train_y[i,j] indicates if jth_frame@ith_listener was labled having a beat
            #   - jth_frame@ith_listener was labled having a beat call "beat frame"
            n_of_frames, n_of_freq_bins = train_x.shape
            n_of_listeners, __n_of_frames = train_y.shape

            if n_of_frames != __n_of_frames:
                print('[ValueError] train_x.shape = ' + str(train_x.shape))
                print('[ValueError] train_y.shape = ' + str(train_y.shape))
                raise ValueError

            #trainX = np.tile(trainX, 40).reshape(BATCH_SIZE * 40, INPUT_SIZE)
            train_x = np.tile(train_x, (n_of_listeners, 1))
            train_y.resize((train_y.size, 1), refcheck=False)
            if trainX.ndim == 0 and trainY.ndim == 0:
                trainX = train_x
                trainY = train_y
            else:
                trainX = np.concatenate((trainX, train_x))
                trainY = np.concatenate((trainY, train_y))
        #print('trainX.shape: '+ str(trainX.shape))
        #print('trainY.shape: '+ str(trainY.shape))
    return trainX, trainY, n_of_frames, n_of_freq_bins

def encode_training_data(n_of_songs, sr, n_fft, hop_length, traindata_path, trainxy_path):
    # #### MIREX Audio Formats
    # - CD-quality (PCM, 16-bit, 44100 Hz)
    # - single channel (mono)
    # - file length between 2 and 36 seconds (total time: 14 minutes)
    # 
    # From experience,
    # - n_fft: # samples of a frame = 2048 samples@22050Hz, 4096 samples@44100Hz = 92.879818594104ms = frame duration
    # - hop_length = n_fft * 3 // 4, that means overlap n_fft / 4 = 512 samples@22050Hz, 1024 samples@44100Hz = 23.219954648526 = overlap duration

    frame_duration = n_fft / sr
    print("frame_duration = %f ms" % (frame_duration))

    # Default 40 listeners for each audio
    # Default 431 frames for each 30-minute audio
    # frames_beats = [  [0, 0, 0, 1, ...., 1, 0], => row is listener
    #                   [0, 0, 0, 1, ...., 1, 0], => column is frame
    #                   ......                  , => the value 0/1 represents which frame with beat
    #                   [0, 0, 0, 1, ...., 1, 0] ]
    frames_beats = np.ndarray([], dtype=int)

    trainxy_dir = ''
    for i in range(1, (n_of_songs+1)):
        train_fprefix = traindata_path + '/train%d' % (i)
        print(train_fprefix)

        audio_raw_data, sr = librosa.load(train_fprefix +'.wav', sr=sr, mono=True) # default mono=True
        linear_freq = librosa.stft(y=audio_raw_data, n_fft=n_fft, hop_length=hop_length)
        n_of_freq_bins, n_of_frames = linear_freq.shape
        #verify_number_of_frames(n_of_frames, y=audio_raw_data, sr=sr, n_fft=n_fft, hop_length=hop_length)

        beat_sequences = np.genfromtxt(train_fprefix + '.txt', dtype='str', delimiter='\r')
        # beat_sequences.shape = (40,) # each element is one beat sequence in seconds
        n_of_listeners = beat_sequences.shape[0]

        frames_beats.resize((n_of_listeners, n_of_frames), refcheck=False)
        frames_beats.fill(0)
        for ith_listener in range(n_of_listeners):
            beat_seq = beat_sequences[ith_listener] # the ith listener's beat sequence

            # step1. split into beats list. each element is a beat in seconds.
            beats_sec = [float(b) for b in beat_seq.split('\t')] # beats_sec = [0.625, 1.235, 1.740, ...]
            
            # step2. use beats_sec to indicate which frame has a beat.
            frames_idx = range(n_of_frames) # frames_idx = [0, 1, 2, ..., n_of_frames-1]
            frames_time = librosa.frames_to_time(frames_idx, sr=sr, hop_length=hop_length) # frames_time = [0, t1, t2, ...]
            start_fi = 0
            for beat_sec in beats_sec:
                for fi in frames_idx[start_fi:]:
                    ti = frames_time[fi]
                    if beat_sec < ti:
                        frames_beats[ith_listener][fi - 1] = 1
                        start_fi = fi
                        break # found, to do next beat
                else:
                    if beat_sec < (frames_time[-1] + frame_duration): # check if the beat is within the last frame
                        frames_beats[ith_listener][-1] = 1
                    else:
                        # not found, there must be someting wrong here
                        raise ValueError

            # then next listener's beat_sequence

        # ex: 'x431x128_y40x431/train1'
        n_of_frames = 862 if n_of_frames > 432 else 431
        trainxy_dir = trainxy_path + '/x%dx%d_y%dx%d/' % (n_of_frames, n_of_freq_bins, n_of_listeners, n_of_frames)
        pathlib.Path(trainxy_dir).mkdir(parents=True, exist_ok=True) 

        trainxy_fprefix = trainxy_dir + 'train%d' % (i)

        # [Option1] Linear frequency
        # linear_freq.shape = (n_of_freq_bins, n_of_frames) = (2049, 431)
        #linear_freq = librosa.amplitude_to_db(linear_freq, ref=np.max)

        # [OPtion2] Mel-frequency
        # mel_freq.shape = (n_of_freq_bins, n_of_frames) = (128, 431)
        linear_ps = np.abs(linear_freq)**2 # Linear power spectram
        mel_freq = librosa.feature.melspectrogram(S=linear_ps)
        mel_freq = librosa.power_to_db(mel_freq, ref=np.max)

        # train1_x.npy, train2_x.npy ...
        # [Option1] (n_of_frames, n_of_freq_bins)
        #train_x = linear_freq.T # (431, 2049)
        train_x = mel_freq.T[:n_of_frames, :] # (431, 128) only train16_x.shape = (432, 128), skip the last frame

       # [Option2] train_x.shape = (n_of_listeners, n_of_freq_bins, n_of_frames) = (40, 431, 2049)
        #train_x = [linear_freq.T] * n_of_listeners #=> shape(40,431,2049)

        # [Option3] train_x.shape = (n_of_listeners*n_of_frames, n_of_freq_bins) = (40*431, 2049)
        #train_x = np.tile(linear_freq.T, (n_of_listeners, 1)) #=> shape(17240, 2049)

        np.save(trainxy_fprefix + '_x.npy', train_x, allow_pickle=True, fix_imports=False)
        #np.savetxt(train_fprefix + '_x.txt', train_x, fmt='%f', delimiter=',', newline='\n')

        # train_y.npy
        # frames_beats.shape = (n_of_listeners, n_of_frames) = (40, 431)
        # Only train16_y.shape = (40, 432) and the last frames don't contain any beats.
        train_y = frames_beats[:, :n_of_frames]

        # [Option2] train_y.shape = (n_of_frames, n_of_listeners) = (431, 40)
        #train_y = frames_beats.T

        # [Option3] train_y.shape = (n_of_frames*n_of_listeners, 1) NOTICE: not n_of_listeners*n_of_frames. size is the same. content is different.
        #train_y = frames_beats
        #train_y.resize((train_y.size, 1), refcheck=False)

        np.save(trainxy_fprefix + '_y.npy', train_y, allow_pickle=True, fix_imports=False)
        #np.savetxt(train_fprefix + '_y.txt', train_y, fmt='%d', delimiter=',', newline='\n')

        # then next training file
    return trainxy_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nsg', '--nsongs', type=int, default=20, help='the number of audio songs')
    parser.add_argument('-sr', '--samplerate', type=int, default=44100, help='sampling rate')
    parser.add_argument('-spf', '--samplesframe', type=int, default=2048, help='the number of samples per frame')
    parser.add_argument('-hpr', '--hoprate', type=float, default=0.75, help='the rate of hopping samples per frame')
    parser.add_argument('-in', '--inputpath', type=str, default='dataset/mirex_beat_tracking_2016/train/', help='the path of training files')
    parser.add_argument('-out', '--outputpath', type=str, default='', help='the path of output data')
    args = parser.parse_args()

    n_of_songs = args.nsongs
    sr = args.samplerate
    n_fft = args.samplesframe
    hop_length = int(args.hoprate * n_fft)
    input_path = args.inputpath
    output_path = input_path if args.outputpath == '' else args.outputpath

    trainXY_path = encode_training_data(n_of_songs, sr, n_fft, hop_length, input_path, output_path) # generate train1_x, train1_y, ..., train20_x ,train20_y
    trainX, trainY, n_of_frames, n_of_freq_bins = load_trainXY(trainXY_path)
    save_trainXY(trainXY_path, trainX, trainY)
