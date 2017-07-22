import librosa as lb
import numpy as np
from sklearn.externals import joblib
from scipy.spatial.distance import euclidean
sr = 44100
hp_len = 256

# generator used to classify frames
# input: cqt transform
def getData(cqt):
    for i in range(30,(cqt.shape[1])-30):
        pt = cqt[:,(i-30):(i+31):1]
        pt = np.array(map(sum, zip(*pt)))
        pt = (pt - pt.mean(axis=0)) / pt.std(axis=0)
        yield [i, pt]

# function that uses neural network to classify frames
# input: cqt data
# output: chop times and probabilities
def classify(cqt):
    clf = joblib.load("..\sampleChop\\nn32_16_8_4.pkl")
    datagen = getData(cqt)
    times = []
    prob = []
    for pt in datagen:
        pred = clf.predict_proba([pt[1]])
        if pred[0][1] > .7:
            times.append(lb.core.frames_to_time([pt[0]], sr = sr, hop_length = hp_len)[0])
            prob.append(pred[0][1])
    return times, prob

# function to clean up times: aggregate chops that are close to eachother
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

    new_times_probs = sorted(zip(new_times,new_prob), key=lambda pair: pair[1], reverse=True)
    return new_times_probs[0:num-1]

# greedy algorithm. Selects samples that maximize probability and diversity function
# input: cqt (used for diversity function), sample times and probabilities, and number of samples wanted
# output: final set of  chop frames
def max_avg_diversity(cqt, frames_prob, n):
    H, P = lb.decompose.hpss(cqt)
    distances = make_dist_dict(H, frames_prob)
    prob = [i[1] for i in frames_prob]
    max_i = prob.index(max(prob))
    F = [frames_prob[max_i][0]]
    del frames_prob[max_i]
    while len(F) < n - 1:
        max_i = 0
        max_val = 0.0
        for i in range(0, len(frames_prob)):
            score = frames_prob[i][1] + diversity(distances, frames_prob[i][0], F)
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
    for i in xrange(len(frames_prob)-1):
        dist[str(frames_prob[i][0])] = {}
        for j in range(i+1, len(frames_prob)):
            dist[str(frames_prob[i][0])][str(frames_prob[j][0])] = euclidean(H[:,frames_prob[i][0]],H[:,frames_prob[j][0]])
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

# function that uses final sample indices of the chops to cut and rewrite audio into several chops
# input: y, sample rate, prefix of new samples, sample indices
def write_samples(y, sample_rate, samples_prefix, samples):
    for i in range(0, len(samples)):
        if i > 0:
            lb.output.write_wav(samples_prefix + str(i+1) + ".wav", y[samples[i-1]:samples[i]], sr=sample_rate)
        else:
            lb.output.write_wav(samples_prefix + str(i+1) + ".wav", y[0:samples[i]], sr=sample_rate)
    lb.output.write_wav(samples_prefix + str(len(samples) + 1) + ".wav", y[samples[len(samples)-1]:], sr=sample_rate)
    return

def main():
    f = 'finishedsamples/02 - 21St Century - The Way We Were 1.wav'
    y, sample_rate = lb.load(f, sr=sr)
    cqt = np.abs(lb.core.cqt(y, sr=sample_rate, fmin=lb.note_to_hz('F2'),
                             n_bins=48, hop_length=hp_len, norm=2, real=False))
    times, prob = classify(cqt)
    time_probs = clean_times(times, prob, 10)
    frames = lb.core.time_to_frames([i[0] for i in time_probs], sr=sample_rate, hop_length=hp_len)
    frames_prob = sorted(zip(frames, [i[1] for i in time_probs]), key=lambda pair:pair[0])
    final_frames = sorted(max_avg_diversity(cqt, frames_prob, 5))
    write_samples(y, sample_rate, 'sample', lb.core.frames_to_samples(final_frames, hop_length=hp_len))

if __name__ == '__main__':
    main()