# Evaluate performance of audio beat tracking

## ToDo Tasks:

- [x] Learn the usage of librosa STFT - [stft_practice.py](stft_practice.py)
- [x] Check onset strength process - [onset_strength_practice.py](onset_strength_practice.py)
- [x] Realize MIREX 2016 training data - [train_data_quicklook.py](train_data_quicklook.py)
- [x] Preprocess training data - [mirex2016_dataset.py](mirex2016_dataset.py)
- [x] Single LSTM Layer - [lstm_beat_tracking.py](lstm_beat_tracking.py)
- [ ] Design Model with multiple LSTM unit
- [ ] Taking Mel-freq bin as input_x
- [ ] Bi-directon RNN
- [ ] Try different loss functions
- [ ] Demo

## Result
- Data
    * Total 20 songs, each 30 sec
    * Sample rate = 44100 Hz
    * 1 frame = 4096 samples = 92.88ms
    * Total 431 frames (because overlap 1/4 frame)
    * Each audio correspond to 40 beat results (40 listeners)
    * Beat frame : Blank frame = 11.27 : 88.73

- Linear frequency
    * Single LSTM layer
    * Loss function : binary_crossentropy
    * Optimizer : RMSprop
    * LSTM activation : tanh
    * The number of features = 2049
    * Loss, Accuracy = 0.377582, 0.867927 on only train1 data

## Public Dataset

- [MIREX Database](http://www.music-ir.org/mirex/wiki/2016:Audio_Beat_Tracking)
    * Beat locations have been annotated in each excerpt by 40 different listeners (39 listeners for a few excerpts) 
    * The length of each excerpt is 30 seconds
    * This dataset contains 217 excerpts around 40s each, of which 19 are "easy" and the remaining 198 are "hard"
    * The harder excerpts were drawn from the following musical styles: Romantic music, ﬁlm soundtracks, blues, chanson and solo guitar
    * This dataset has been designed for radically new techniques which can contend with challenging beat tracking situations like: quiet accompaniment, expressive timing, changes in time signature, slow tempo, poor sound quality etc


### Model
```
```

## Reference
- Paper: [Improved Musical Onset Detection With Convolutional Neural Networks](http://ieeexplore.ieee.org/document/6854953/)
