# # Creates and re-saves processed (z-scored PSDs and phase diffss) UrbanSound8K files for quicker loading

import numpy as np
import pandas as pd
import librosa
import os
from scipy.stats import zscore
from phaseshift import phasediff
from math import pi

# parameters to choose (exp refers to log2 of param)
# maybe make this into function
windexp = 10 # any larger will fail on <90 msec files
stepexp = 8
precision = "float16"

####

info_path = "data/UrbanSound8K.csv"
data_dir = "data"
FS = 22050 # expected samp freq from librosa load
wind = 2**windexp
step = 2**stepexp
save_dir = "processed_wind" + str(windexp) + "_step" + str(stepexp) # where fold dirs will be saved

csv_in = pd.read_csv(info_path)

# make directory structure if none exists for these parameters
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
    n_folds = max(csv_in.loc[:,"fold"])
    for f in range(n_folds):
        fold = str(f+1)
        os.mkdir(save_dir + "/fold" + fold)

# main loop, process and save psds and phdiffs
for i in [j+4300 for j in range(len(csv_in))]:
    if i % 100 == 0:
        print("Writing files " + str(i+1) + " to " + str(i+100))

    row = csv_in.loc[i]
    fold = row.fold
    filename = row.slice_file_name
    save_prefix = save_dir + "/fold" + str(fold) + "/" + filename[0:-4]

    sound, FS1 = librosa.load(data_dir + "/fold" + str(fold) + "/" + filename)
    cut = len(sound) % step
    if cut!=0:
        sound = sound[0:-cut]
    phdiff, psds, _, _ = phasediff(sound, windsize=wind, stepsize=step, sampfreq=FS)
    # z-score normalization - prior to padding, probably good since padding shouldn't activate?
    psds = zscore(psds,axis=None)
    phdiff = phdiff-pi/2 # no need to normalize since already in range 0-pi, but want zeros to have non-zero activation
    np.save(save_prefix + "_psd.npy", psds.astype(precision))
    np.save(save_prefix + "_phdiff.npy", phdiff.astype(precision))
