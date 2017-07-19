#This script loads a wav file and divides it into data points, then loads
#split times and assigns truth values to data points

import librosa as lb
import numpy as np
import math

prefix = "C:\Users\Joe\Downloads\sampleChop\\"
sample = raw_input("Please enter audio file name ")
name = sample[:-4]  #to be used when loading truth file and writing data to file
filename = prefix + sample

##Features ###################################################
#load file
y, sr = lb.load(filename, sr=44100)
hop_length = 256
#compute constant-Q transformation
print "Generating Data"
cqt = np.abs(lb.core.cqt(y, sr=sr, fmin=lb.note_to_hz('F2'),
                         n_bins=48, hop_length=hop_length, norm=2, real=False))

#separate harmonic and percussive elements to aid in tonnetz
y_har, y_per = lb.effects.hpss(y)

#compute tonnetz using harmonic elements only
tonnetz = lb.feature.tonnetz(y=y_har,sr=sr,
                             chroma=lb.feature.chroma_cqt(y=y_har, sr=sr,fmin=lb.note_to_hz('C2'),
                                                          n_octaves=7,hop_length=hop_length,norm=np.inf))
#no negative values, scale up to cqt values
tonnetz[tonnetz < 0] = 0
#scale tonnetz values relative to amplitudes of cqt
mean_c = np.mean(cqt)
mean_t = np.mean(tonnetz)
tonnetz = tonnetz  * (mean_c/mean_t)

#if tonnetz and cqt aren't the same shape
difference = cqt.shape[1] - tonnetz.shape[1]
if (difference < 0):
    tonnetz = tonnetz[:,:-1]
elif (difference > 0):
    cqt = cqt[:,:-1]


#combine cqt and tonnetz
song_data = np.concatenate((cqt, tonnetz), axis=0)
data_pts = []

for i in range(30, (song_data.shape[1])-30):
    data_pt = song_data[:,(i-30):(i+31):1]
    data_pts.append(data_pt)

#normalize features
mean = np.mean(song_data)
std = np.std(song_data)
for i in range(0, len(data_pts)):
    data_pts[i] = data_pts[i] - mean
    data_pts[i] = data_pts[i] / std


## Target values ####################################
#get text file that contains truth values
filename = prefix + name + '.txt'
truth_vals = open(filename, 'r')

#get split times
truth_times = truth_vals.read().split(',')
truth_times = [float(x) for x in truth_times]

#get split frames
truth_frames = lb.core.time_to_frames(truth_times, sr=sr, hop_length=hop_length)
truth_frames = [(x - 30) for x in truth_frames]

j=0     #truth value array pointer
truth_vals = []
positives = len(truth_frames)*3     #3 consecutive positive frames per row
num_data_pts = len(data_pts)

##labeling data
for i in range(0,num_data_pts):
    if((truth_frames[j] == i)or(truth_frames[j] == (i+1))or(truth_frames[j] == (i-1))):
        truth_vals.append(int(1))
        if((j < ((positives/3) - 1))and(truth_frames[j] == (i-1))):
            j = j + 1
    else:
        truth_vals.append(int(0))


##  upsampling  ###########################################
truth_frames = lb.core.time_to_frames(truth_times, sr=sr, hop_length=hop_length)
for i in  range(-3,4):
    if (i != 0):
        print "Upsampling with pitch changed by " + str(i) + " half-steps"
        y_shift = lb.effects.pitch_shift(y, sr, n_steps = i)
        #compute constant-Q transformation
        cqt = np.abs(lb.core.cqt(y_shift, sr=sr, fmin=lb.note_to_hz('F2'),
                                 n_bins=48, hop_length=hop_length, norm=2,real=False))
        #separate harmonic and percussive elements to aid in tonnetz
        y_har, y_per = lb.effects.hpss(y_shift)
        #compute tonnetz using harmonic elements only
        tonnetz = lb.feature.tonnetz(y=y_har,sr=sr,
            chroma=lb.feature.chroma_cqt(y=y_har, sr=sr,fmin=lb.note_to_hz('C2'),
                n_octaves=7,hop_length=hop_length,norm=np.inf))
        #no negative values
        tonnetz[tonnetz < 0] = 0
        #scale tonnetz values relative to amplitudes of cqt
        mean_c = np.mean(cqt)
        mean_t = np.mean(tonnetz)
        tonnetz = tonnetz  * (mean_c/mean_t)
        #if tonnetz and cqt aren't the same shape
        difference = cqt.shape[1] - tonnetz.shape[1]
        if (difference < 0):
            tonnetz = tonnetz[:,:-1]
        elif (difference > 0):
            cqt = cqt[:,:-1]
        #combine cqt and tonnetz
        song_data = np.concatenate((cqt, tonnetz), axis=0)
        mean = np.mean(song_data)
        std = np.std(song_data)
        for x in truth_frames:
            for j in range(-1,2):
                data_pt = song_data[:,((x+j)-30):((x+j)+31):1]
                data_pt = data_pt - mean
                data_pt = data_pt / std
                data_pts.append(data_pt)
                truth_vals.append(int(1))
                positives += 1


print positives
print len(data_pts)
print float(positives)/float(len(data_pts))


## write data to .txt files #########################################
with file(prefix + name + 'features.txt', 'w') as outfile:
    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in data_pts:
        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        np.savetxt(outfile, data_slice, fmt='%-7.2f')

        # Writing out a break to indicate different slices...
        outfile.write('# New slice\n')

with file(prefix + name + 'targets.txt', 'w') as outfile:
    np.savetxt(outfile, truth_vals)


import plotly as py
import plotly.graph_objs as go
import pandas as pd
data = [go.Surface(z=pd.DataFrame(data_pts[len(data_pts) - 5]).as_matrix())]
layout = go.Layout(autosize = True)
fig = go.Figure(data=data,layout=layout)
py.offline.plot(fig)



