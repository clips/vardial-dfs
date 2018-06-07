## Vardial-dfs

CLiPS submission for the Discriminating between Dutch and Flemish in Subtitles (DFS) shared task at [VarDial 2018](http://alt.qcri.org/vardial2018/).

### Authors: Tim Kreutz and Walter Daelemans
### Paper: Not yet available

### Overview

We explore different ways to combine classifiers for language variety identification. The task is to discriminate Netherlandic Dutch subtitles from Flemish Dutch subtitles. We combine word n-gram features and part-of-speech n-gram in different ways:

1. by linking the two feature vectors,
2. by training base classifiers for both and letting the classifier which has the highest probability for a label decide the label,
3. or by training a meta-classifier on the probabilities output of the individual base classifiers.

Our second submission yielded the best result of the three (F-score: 63.6%) which achieved third place out of 12 participating teams in the shared task.