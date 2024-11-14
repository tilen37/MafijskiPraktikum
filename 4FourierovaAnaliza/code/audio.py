import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
i, j = -1, 0

for file in os.listdir(r'4FourierovaAnaliza/code/audioData'):
    i += 1
    if i > 2:
        i = 0
        j += 1

    y, sr = librosa.load(os.path.join(
        r'4FourierovaAnaliza/code/audioData', file))
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, fmin=0, fmax=2500)
    img = librosa.power_to_db(S, ref=np.max)[::1]

    ax[i, j].imshow(img)
    ax[i, j].set_title(file[6:-4])
    ax[i, j].set_ylim([0, 120])

    # plt.imshow(img)
    # plt.title(file[9:-4])
    # plt.xlabel('čas')
    # plt.ylim([0, 120])
    # plt.show()

plt.show()

# Ni lepo :(
