import matplotlib.pyplot as plt
import numpy as np

def get_train_data(filepath):
    train = np.genfromtxt(filepath, dtype='str', delimiter='\r')
    #print(train.shape)

    listeners = []
    beat_times = []
    for i in range(len(train)):
        beats = [float(x) for x in train[i].split('\t')]
        beat_times.extend(beats)
        listeners.extend([i]*len(beats))
    print("%s : %d listeners, %d beats" % (filepath, len(train), len(beat_times)))
    return beat_times, listeners

train_file = 'dataset/mirex_beat_tracking_2016/train/train1.txt'
beat_times, listeners = get_train_data(train_file)
plt.figure(figsize=[36,12])
plt.scatter(beat_times, listeners)
plt.xlabel('Time(sec)')
plt.ylabel('Listener')
plt.title(train_file)
plt.show()

from mpl_toolkits.axes_grid1 import Grid
fig = plt.figure(figsize=[36,12])
grid = Grid(fig, rect=111, nrows_ncols=(10, 2), axes_pad=0.25, label_mode='L')
for i, ax in zip(range(20), grid):
    train_file = 'dataset/mirex_beat_tracking_2016/train/train%d.txt' % (i+1)
    beat_times, listeners = get_train_data(train_file)
    ax.scatter(beat_times, listeners, s=1, marker='.')
    ax.locator_params(nbins=1)
    ax.set_xlabel('Time(sec)', fontsize=8)
    ax.set_ylabel('Listener', fontsize=8)
    ax.set_title(train_file, fontsize=8)
    #ax.title.set_visible(False)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
