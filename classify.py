import librosa as lb
import numpy as np
from sklearn.externals import joblib
sr = 44100
hp_len = 256

def getData(f, sr = 44100, hp_len = 256):
    y, sr = lb.load(f, sr=sr)
    cqt = np.abs(lb.core.cqt(y, sr=sr, fmin=lb.note_to_hz('F2'),
                             n_bins=48, hop_length=hp_len, norm=2, real=False))

    for i in range(30,(cqt.shape[1])-30):
        pt = cqt[:,(i-30):(i+31):1]
        pt = np.array(map(sum, zip(*pt)))
        pt = (pt - pt.mean(axis=0)) / pt.std(axis=0)
        yield [i, pt]

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
        return new_times
    else:
        print
        new_times = [x for (y,x) in sorted(zip(new_prob,new_times), key=lambda pair: pair[0], reverse=True)]
        return sorted(new_times[0:num-1])

def classify(f, num, sr = 44100, hp_len = 256):
    clf = joblib.load("..\sampleChop\\nn32_16_8_4.pkl")
    datagen = getData(f, sr, hp_len)
    times = []
    prob = []
    for pt in datagen:
        pred = clf.predict_proba([pt[1]])
        if pred[0][1] > .7:
            times.append(lb.core.frames_to_time([pt[0]], sr = sr, hop_length = hp_len)[0])
            prob.append(pred[0][1])

    times = clean_times(times, prob, num)
    return times, lb.core.time_to_samples(times, sr = sr)

def write_samples(f, samples_prefix, samples):
    y, sample_rate = lb.load(f, sr=sr)
    for i in range(0, len(samples)):
        if i > 0:
            lb.output.write_wav(samples_prefix + str(i+1) + ".wav", y[samples[i-1]:samples[i]], sr=sample_rate)
        else:
            lb.output.write_wav(samples_prefix + str(i+1) + ".wav", y[0:samples[i]], sr=sample_rate)
    lb.output.write_wav(samples_prefix + str(len(samples) + 1) + ".wav", y[samples[len(samples)-1]:], sr=sample_rate)
    return


def main():
    f = 'C:\Users\Joe\Downloads\sampleChop\\finishedsamples\\Inez Foxx - Let Me Down Easy.wav'
    times, samples = classify(f, 5)
    write_samples(f, 'sample', samples)


if __name__ == '__main__':
    main()