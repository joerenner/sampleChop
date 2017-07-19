# sampleChop

This python project trains a neural network to divide samples for the purpose of chopping https://en.wikipedia.org/wiki/Chopping_(sampling_technique) for the aide of hip hop production. 

Process:
I framed the problem as a binary classification sample: For each frame of an audio file, it would be positive if it was a good break point for a chop, and negative otherwise.
I spent a large amount of time chopping samples by hand and recording where in a wav file I chopped.
Then, I created a script to turn the raw data (audio file, text file with times) into data features and labels.
This was done by first taking the constant q transform aka cqt (https://en.wikipedia.org/wiki/Constant-Q_transform) and generating the tonnetz data (https://en.wikipedia.org/wiki/Tonnetz , for more harmonic components), and then having each frame be accompanied by the previous thirty frames and following thirty frames (for context).
Since the classes are extremely unbalanced (a lot more negatives than positives), I had to use a couple upsampling techniques:
  - First, if a frame was positive, the previous and following frame's is considered positive as they are so close together they would both work
  - Next, I retuned every positive data point from -3 half steps to 3 half steps in order to have new (slightly different but still positive) examples

At this point, I had about 400,000 data points from about 100 audio files. Since this took a large amount of time and I hope to make a website using this data, I left the data off of github.

Next, I train a neural network on this data. After experimenting with many architectures, I decided to make the data simpler.
I first tried a convolutional neural network on the cqt and appended tonnetz data. This did not work well and was computationally expensive, so I got rid of the tonnetz data and summed the cqt data for each frame (including 30 before and 30 after) for each data point. This took the data points from (61 x 54) to (61 x 1). I then trained a deep feed forward neural network on this data, and it worked better. I experimented with a few different architectures, and found one that worked well on validation audio samples.


Files:
- get_data : script to process raw data into data points
- dataOps : script that reprocesses data from a lot of little files into a few large files (made training easier)
- sampleChop : file that loads data (in batches: over 400,000 data points!), trains and saves the network, and performs on a validation audio file
- classify : file that takes new audio file, generated features, classifies its' frames, filters the positive samples, breaks the audio file on these final positive chops, then writes the new broken up samples to wav files. 

Future work: 
- Create website for this using classify.py as backend. Possibly using Django framework
- Introduce greedy filtering algorithm on positive samples, to introduce trade-off between chop diversity and chop confidence
- Find way to use with mp3s and not just wav audio files
