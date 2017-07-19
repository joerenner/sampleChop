import sampleChop
import numpy as np

# script to combine summed data into larger files to make training easier
dataGen = sampleChop.dataGenerator()
i = 0
files = 0
filename = "C:\Users\Joe\Downloads\sampleChop\data\summed\\"

for songData in dataGen:
    if files == 0:
        features = songData[0]
        targets = songData[1]
    else:
        features = np.concatenate((features, songData[0]))
        targets = np.concatenate((targets, songData[1]))
    files += 1
    if files == 8:
        print "Saving file " + str(i) + "..."
        print features.shape
        print targets.shape
        ## write data to .txt files #########################################
        with file(filename + str(i) + 'features.txt', 'w') as outfile:
            # Iterating through a ndimensional array produces slices along
            # the last axis. This is equivalent to data[i,:,:] in this case
            for data_slice in features:
                for num in data_slice:
                    outfile.write(str(num) + " ")
                outfile.write('\n')

        with file(filename + str(i) + 'targets.txt', 'w') as outfile:
            np.savetxt(outfile, targets)

        i += 1
        files = 0
