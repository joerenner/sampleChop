import librosa as lb
import numpy as np
from sklearn.externals import joblib
from scipy.spatial.distance import euclidean

# generator used to classify frames
# input: cqt transform
# yields tuples (index, example data point)
def getData(cqt):
    for i in range(30,(cqt.shape[1])-30):
        pt = cqt[:,(i-30):(i+31):1]
        pt = np.array(map(sum, zip(*pt)))
        if not (np.isnan(pt).any()):
            pt = (pt - pt.mean(axis=0)) / pt.std(axis=0)
            yield [i, pt]

# function to clean up times: aggregate chops that are close to eachother, return highest probability chops
# input: list of positive times, probability of these times, number of final chops to send
# output: list of cleaned tuples (time, probability)
def clean_times(times, prob, num):
    new_times = []
    new_prob = []
    worker = []
    workerpb = []
    prev = 0
    for i in range(0, len(times)):
        if times[i] - prev > 0.01:
            if worker:
                new_times.append(np.mean(worker))
                new_prob.append(np.mean(workerpb))
                worker = []
                workerpb = []
            prev = times[i]
        elif new_times:
            if times[i] - new_times[len(new_times)-1] > 0.25:
                worker.append(times[i])
                workerpb.append(prob[i])
                prev = times[i]
        else:
            worker.append(times[i])
            workerpb.append(prob[i])
            prev = times[i]
    if worker:
        new_times.append(np.mean(worker))
        new_prob.append(np.mean(workerpb))
    if len(new_times) < num:
        return zip(new_times, new_prob)
    else:
        new_times_probs = sorted(zip(new_times,new_prob), key=lambda pair: pair[1], reverse=True)
        return new_times_probs[0:num-1]

# greedy algorithm. Selects samples that maximize probability and diversity function
# input: cqt (used for diversity function), sample times and probabilities, number of samples wanted
# output: final set of chop frames
def max_avg_diversity(cqt, frames_prob, n, lamb):
    if len(frames_prob) <= n:
        return [i[0] for i in frames_prob]
    if lamb < 0.0 or lamb > 1.0:
        lamb = 0.9
    H, P = lb.decompose.hpss(cqt)
    distances = make_dist_dict(H, frames_prob)
    prob = [i[1] for i in frames_prob]
    max_i = prob.index(max(prob))
    F = [frames_prob[max_i][0]]
    del frames_prob[max_i]
    while len(F) < n - 1:     # n - 1 frame divisions equals n total audio segments
        max_i = 0
        max_val = 0.0
        for i in range(0, len(frames_prob)):
            score = ((1 - lamb) * frames_prob[i][1]) + (lamb * diversity(distances, frames_prob[i][0], F))
            if score > max_val:
                max_val = score
                max_i = i
        F.append(frames_prob[max_i][0])
        del frames_prob[max_i]
    return F

# function to create distance dictionary: the euclidean distance between the harmonic amplitudes of two frames
# dictionary of dictionaries: if i < j, the distance from i to j is stored dist[i][j]
# input: Harmonic cqt, frames and probabilites
def make_dist_dict(H, frames_prob):
    dist = {}
    max_dist = 0        # used for scaling distances between 0 and 1
    for i in xrange(len(frames_prob)-1):
        dist[str(frames_prob[i][0])] = {}
        for j in range(i+1, len(frames_prob)):
            distance = euclidean(H[:,frames_prob[i][0]],H[:,frames_prob[j][0]])
            dist[str(frames_prob[i][0])][str(frames_prob[j][0])] = distance
            if distance > max_dist:
                max_dist = distance
    # scaling distances down to same and probability scaling
    for start in dist:
        for end in dist[start]:
            dist[start][end] = float(dist[start][end]) / max_dist

    return dist

# diversity function: measures average distance from candidate frame to all frames already picked
# input: distance dict of dicts, candidate frame, and set of frames already selected
# output: average distance from candidate frame to all frames already picked
def diversity(dist, cand_frame, F):
    total_distance = 0.0
    for sample in F:
        if cand_frame > sample:
            total_distance += dist[str(sample)][str(cand_frame)]
        else:
            total_distance += dist[str(cand_frame)][str(sample)]
    return total_distance / float(len(F))



class Sample(object):

    # input: path to wav file, offset (where to start chopping in seconds), how long to mine chops (in seconds)
    # whether to only consider harmonic aspects of wav (will get rid of drums while chopping resulting chops will
    # be without drums or other percussive elements), the lowest frequency (note in MIDI form) to
    # consider while chopping
    def __init__(self, file_name, offset=0.0, duration=None, harmonic=False, start_note='F2'):
        self.file_name = file_name
        self.sr = 44100
        self.hp_len = 256
        y, sample_rate = lb.load(self.file_name, sr=self.sr, offset=offset, duration=duration)
        if harmonic:
            self.y = lb.effects.harmonic(y)
        else:
            self.y = y
        self.cqt = np.abs(lb.core.cqt(y, sr=sample_rate, fmin=lb.note_to_hz(start_note),
                             n_bins=48, hop_length=self.hp_len, norm=2))

    # function that uses neural network to classify frames
    # input: cqt data
    # output: chop times and probabilities
    def classify(self, num = 15, lamb = 0.9):
        clf = joblib.load("..\sampleChop\\nn32_16_8_4.pkl")
        datagen = getData(self.cqt)
        times = []
        prob = []
        for pt in datagen:
            pred = clf.predict_proba([pt[1]])
            if pred[0][1] > .7:
                times.append(lb.core.frames_to_time([pt[0]], sr=self.sr, hop_length=self.hp_len)[0])
                prob.append(pred[0][1])
        time_probs = clean_times(times, prob, num * 2)
        # convert times to frame numbers
        frames = lb.core.time_to_frames([i[0] for i in time_probs], sr=self.sr, hop_length=self.hp_len)
        # arrange frames with probabilities in chronological order
        frames_prob = sorted(zip(frames, [i[1] for i in time_probs]), key=lambda pair: pair[0])
        # run greedy diversification algorithm to get chop frames
        self.final_frames = sorted(max_avg_diversity(self.cqt, frames_prob, num, lamb))

    # function that uses final sample indices of the chops to cut and rewrite audio into several chops
    # input: y, sample rate, prefix of new samples, sample indices
    def write_samples(self, samples_prefix = "sample"):
        samples = lb.core.frames_to_samples(self.final_frames, hop_length=self.hp_len)
        for i in range(0, len(samples)):
            if i > 0:
                lb.output.write_wav(samples_prefix + str(i + 1) + ".wav", self.y[samples[i - 1]:samples[i]], sr=self.sr)
            else:
                lb.output.write_wav(samples_prefix + str(i + 1) + ".wav", self.y[0:samples[i]], sr=self.sr)
        lb.output.write_wav(samples_prefix + str(len(samples) + 1) + ".wav", self.y[samples[len(samples) - 1]:],
                            sr=self.sr)

# driver/example
def main():
    samp = Sample(file_name="../finishedsamples/Los Angeles Negros - Le Huella de Mi Amor  (Instrumental) 1.wav",
                  offset=0.0, duration=None, harmonic=True, start_note='F2')
    samp.classify(15, lamb = 0.95)
    samp.write_samples()


if __name__ == '__main__':
    main()
