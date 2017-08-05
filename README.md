# sampleChop

This python project uses a trained neural network to divide samples for the purpose of chopping https://en.wikipedia.org/wiki/Chopping_(sampling_technique) for the aide of hip hop production. 

Process:

I framed the problem as a binary classification sample: For each frame of an audio file, it would be positive if it was a good break point for a chop (chop: point in the song in which to break the original audio file), and negative otherwise.
I spent a large amount of time chopping samples by hand and recording where in a wav file I chopped.
Then, I created a script to turn the raw data (audio file, text file with times) into data features and labels.
This was done by first taking the constant q transform aka cqt (https://en.wikipedia.org/wiki/Constant-Q_transform) and generating the tonnetz data (https://en.wikipedia.org/wiki/Tonnetz , for more harmonic components), and then having each frame be accompanied by the previous thirty frames and following thirty frames (for context).
Since the classes are extremely unbalanced (a lot more negatives than positives), I had to use a couple upsampling techniques:
  - First, if a frame was positive, the previous and following frame's is considered positive as they are so close together they would both work
  - Next, I retuned every positive data point from -3 half steps to 3 half steps in order to have new (slightly different but still positive) examples

At this point, I had about 400,000 data points from about 100 audio files. Since this took a large amount of time and I hope to make a website using this data, I left the data off of github.

Next, I train a neural network on this data. After experimenting with many architectures, I decided to make the data simpler.
I first tried a convolutional neural network on the cqt and appended tonnetz data. This did not work well and was computationally expensive, so I got rid of the tonnetz data and summed the cqt data for each frame (including 30 before and 30 after) for each data point. This took the data points from (61 x 54) to (61 x 1). I then trained a deep feed forward neural network on this data, and it worked better. I experimented with a few different architectures, and found one that worked well on validation audio samples.

Finally, I added some post filtering on the results of the neural network:
- cleaned the times: any positively labeled frames that are microseconds away from eachother are aggregated into one chop
- took most likely chops: took the top-N most likely chops based on probabilities from neural network
- applied greedy diversification algorithm: see below

Greedy Diversification Algorithm: Maximal Average Diversification
Once the neural network results have been cleaned, we end of with N candidates for breaking the wav file into smaller chops.
Since many songs have repetitive parts in them (chords, progressions, notes, etc), a lot of the final chops can be very similar if only judging on the probability. This can lead to the system generating 5 chops that basically sound the same and offer little variation for the artist to play with. Thus, we want diverse chops. I achieve this by introducing a greedy algorithm based on an objective I named Maximal Average Diversification, which is a small variation on Max-Sum Diversification (https://arxiv.org/pdf/1203.6397.pdf), an objective used for adding diversification to recommendations.

Basically, what Max-Sum Diversification and Maximal Average Diversification (M.A.D) aim to achieve is a balance between the relevance of an item and the diversity of the list of items as a whole. In the domain of sampleChop, this is a balance between thee probability of a chop and how different it is from the other chops.

The way the algorithm works is it takes the cleaned frames from the output of the neural network (we will call them the Candidate set, or C) and then greedily selects frames from the set that maximize the M.A.D. objective and adds them to the final set of chops, which we will call the F set. The objective has two parts: the relevance expression and the diversity expression, with a hyperparameter lambda that controls the trade-off between the two. The relevance expression is simply the probability output from the network. After experimenting with the greedy algorithm, I discovered a lambda value of 0.999 (almost maximum diversity) works best, indicating that since only the highest probability chops are passed into the greedy algorithm as candidates, there is no need to include the relevance expression in the objective function. Thus, the final greedy algorithm simply takes the most probable chops and selects the most diverse among them, after selecting a seed chop (the most probable chop according the the network). The diversity expression is calculated as follows:
   - first, the spectrogram is converted to only its harmonic components, meaning no drums or other percussive elements
   - next, the euclidean distance is taken from the harmonic spectrogram of the candidate frame to every frame in F and averaged. This measures how different the candidate frame is to each of the frames that have already been selected into F.
One candidate from is selected at each iteration and added to F, until there are enough chops in F to return them as the final set.

Once the final set is obtained, the original audio file is split at each frame number, and the resulting chops are written to wav files, ready to program into an MPC.

Files:
- get_data : script to process raw data into data points
- dataOps : script that reprocesses data from a lot of little files into a few large files (made training easier)
- sampleChop : file that loads data (in batches: over 400,000 data points), trains and saves the network, and performs on a validation audio file
- classify : file that takes new audio file, generated features, classifies its' frames, filters the positive samples, applies diversification, breaks the audio file on these final positive chops, then writes the new broken up samples to wav files. 

Future work: 
- Create website for this using classify.py as backend. Possibly using Django framework
- Find way to use with mp3s and not just wav audio files
