import numpy as np
import math
import random
pi = math.pi

def phasediff(signal, windsize=2**10, stepsize=2**8, sampfreq=1):
    """Tim C Whalen last edited May 22 2020
    Compute phase difference (Whalen et al. 2020 JNeurophys; phase shift is time mean of
    phase difference) and non-windowed power spectral density using Welch's method. 

    Arguments:
        signal {list[float]} -- time series of which to compute phase shift
        windsize {int} -- length (in samples) of FFT window (default = 2**10)
        stepsize {int} -- length (in samples) to shift window each step (default = 2**8)
        sampfreq {float} -- sampling frequency (default = 1)

    Returns:
        phdiff {2D np.array[float]} -- phdiff[f,t] gives phase diff of fth frequency bin
                                           at (t,t-1)th time difference
        psds {2D np.array[float]} -- psds[f,t] gives power spectral density at fth frequency
                                     bin and tth time bin
        times {list[float]} -- time axis for plotting psds, in sec if sampfreq is in Hz
        freqs {list[float]} -- frequency axis for plotting psds, in Hz if sampfreq is in Hz

    to-do: make input signal array-like
    """
    timebins = len(signal)
    nsteps = math.floor((timebins-windsize)/stepsize)+1

    # times and freqs only needed for return
    times = [(windsize/2 + i*stepsize)/sampfreq for i in range(nsteps)]
    rayleigh = sampfreq/windsize
    freqs = [rayleigh*i for i in range(round(windsize/2+1))]

    fts = np.empty((len(freqs),nsteps),dtype=complex)
    psds = np.empty((len(freqs),nsteps),dtype=float)
    phs = np.empty((len(freqs),nsteps),dtype=float) # phases
    
    for s in range(nsteps):
        seg = signal[stepsize*s:stepsize*s+windsize]
        ft = np.fft.fft(seg-np.mean(seg))
        fts[:,s] = ft[0:math.ceil(len(ft)/2)+1]
        psds[:,s] = abs(fts[:,s])**2
        phs[:,s] = np.angle(fts[:,s])

    stepsize_sec = stepsize/sampfreq
    wph = [[(pi + (phs[f,s] - 2*pi*s*stepsize_sec*freqs[f]) % (2*pi))-pi 
        for s in range(nsteps)] 
        for f in range(len(freqs))] # window-corrected phase
    phdiff_unlooped = [np.diff(wph[f]) for f in range(len(freqs))] # phase diff over time as func of freq
    phdiff = np.asarray([[pi-abs(abs(phdiff_unlooped[f][s])-pi) 
        for s in range(nsteps-1)] 
        for f in range(len(freqs))])  # wrap so 2pi is like 0, pi farthest
    
    return phdiff, psds, times, freqs

def phaseshift(phdiff):
    """Tim C Whalen last edited May 22 2020
    Mean phase shift (Whalen et al. 2020, JNeurophys) from phase difference (first output 
    of phasediff)

    Arguments:
        phdiff {2D np.array[float]} -- phdiff[t,f] gives phase diff of fth frequency bin
                                           at (t,t-1)th time difference (output of phasediff)

    Returns:
        phshift [np.array[float]] -- mean phase shift of input as function of frequency (same
                                     freqs as output by phdiff)
    """
    
    return np.mean(phdiff,axis=1)