import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
plt.style.use('customStyle.mplstyle')

# Import the audio files
sounds = []
sound_sr = []
signals = []
file_names = []

for file in os.listdir(r'5FFT/data'):
    if file[-4:] == '.wav':
        sound, sr = librosa.load(os.path.join(r'5FFT/data', file))
        sounds.append(sound)
        sound_sr.append(sr)
        file_names.append(file[:-4])
    if file[-4:] == '.txt':
        with open(os.path.join(r'5FFT/data', file), 'r') as f:
            # Decided not to use this, but only .wav files
            signal = np.array([float(line.split()[0])
                              for line in f if line != '\n'])
            signals.append(signal)

# Plot the sounds
# for i in range(len(sounds)):
#     plt.plot(np.arange(len(sounds[i]))/sound_sr[i], sounds[i])
#     plt.xlabel('Čas $t$[s]')
#     plt.ylabel('Amplituda')
#     if i == 0:
#         plt.title(r'\textit{Newton}')
#         plt.savefig('5FFT/porocilo/Newton_t.pdf')
#     elif i == 1:
#         plt.title(r'\textit{Leibnitz}')
#         plt.savefig('5FFT/porocilo/Leibnitz_t.pdf')
#     else:   
#         plt.title('Zvok ' + r'{\tt ' + file_names[i] + '}')
#     # plt.show()
#     plt.close()

# Autocoorelation of the sounds
auto_sounds = [librosa.autocorrelate(sound) for sound in sounds]

# Plot the autocorrelation of the sounds
# for i in range(len(sounds)):
#     plt.plot(np.arange(1, len(auto_sounds[i])+1)/sound_sr[i], auto_sounds[i])
#     if i == 0:
#         plt.title(r'\textit{Newton}')
#         plt.savefig('5FFT/porocilo/Newton_auto_t.pdf')
#     elif i == 1:
#         plt.title(r'\textit{Leibnitz}')
#         plt.savefig('5FFT/porocilo/Leibnitz_auto_t.pdf')
#     else:   
#         plt.title('Avtokorelacija zvoka ' + r'{\tt ' + file_names[i] + '}')
#     plt.xlabel(r'Zamik $\tau$ [s]')
#     plt.ylabel('Amplituda')
#     plt.xlim(-0.05, 1.25)
#     # plt.show()
#     plt.close()

# FFT of the sounds
fft_sounds = [np.fft.fft(sound) for sound in sounds]

# Plot the FFT of the sounds
# for i in range(len(sounds)):
#     plt.plot(np.fft.fftfreq(len(fft_sounds[i]), 1/sound_sr[i]),
#              np.abs(fft_sounds[i]))
#     plt.xlabel('Frekvenca [Hz]')
#     plt.ylabel('Amplituda')
#     plt.xlim(0, 1000)
#     if i == 0:
#         plt.title(r'\textit{Newton}')
#         plt.savefig('5FFT/porocilo/Newton_f.pdf')
#     elif i == 1:
#         plt.title(r'\textit{Leibnitz}')
#         plt.savefig('5FFT/porocilo/Leibnitz_f.pdf')
#     else:   
#         plt.title('FFT zvoka ' + r'{\tt ' + file_names[i] + '}')
#     # plt.show()
#     plt.close()

# FFT of Autocorrelations of sounds
fft_auto_sounds = [np.fft.fft(auto_sound) for auto_sound in auto_sounds]

# Plot the FFT of the autocorrelation of the sounds
for i in range(len(sounds)):
    plt.plot(np.fft.fftfreq(len(fft_auto_sounds[i]), 1/sound_sr[i]),
             np.abs(fft_auto_sounds[i]))
    plt.xlabel('Frekvenca [Hz]')
    plt.ylabel('Amplituda')
    plt.xlim(0, 650)
    # if i == 0:
    #     plt.title(r'\textit{Newton}')
    #     plt.savefig('5FFT/porocilo/Newton_auto_f.pdf')
    # elif i == 1:
    #     plt.title(r'\textit{Leibnitz}')
    #     plt.savefig('5FFT/porocilo/Leibnitz_auto_f.pdf')
    if i == 4:
        plt.title('FFT avtokoreliranega zvoka ' + r'{\tt ' + file_names[i] + '}')
        plt.savefig('5FFT/porocilo/reka1_auto_f.pdf')
    elif i == 5:
        plt.title('FFT avtokoreliranega zvoka ' + r'{\tt ' + file_names[i] + '}')
        plt.savefig('5FFT/porocilo/reka2_auto_f.pdf')
    else:  
        plt.title('FFT avtokorelacije zvoka ' + r'{\tt ' + file_names[i] + '}')
    # plt.show()
    plt.close()

######################################################################

fig, ax = plt.subplots(4, 2, figsize=(15, 15))

for j in range(2, 6):
    for i in range(2):
        ax[j-2, i].plot(np.fft.fftfreq(len(fft_auto_sounds[j]), 1/sound_sr[j]),
                        np.abs(fft_auto_sounds[j])/np.max(np.abs(fft_auto_sounds[j])), label=r'{\tt ' + file_names[j] + '}')
        ax[j-2, i].plot(np.fft.fftfreq(len(fft_auto_sounds[i]), 1/sound_sr[i]),
                        np.abs(fft_auto_sounds[i])/np.max(np.abs(fft_auto_sounds[i])), label='Sova ' + str(i+1))
        ax[j-2, i].set_xlim(0, 650)
        ax[j-2, i].legend(loc='upper left')

ax[0, 0].set_title(r'\textit{Newton}', pad=30, fontsize=25)
ax[0, 1].set_title(r'\textit{Leibnitz}', pad=30, fontsize=25)

ax[0, 0].set_ylabel(r'{\tt ' + file_names[2] + '}', labelpad=20, fontsize=25)
ax[1, 0].set_ylabel(r'{\tt ' + file_names[3] + '}', labelpad=20, fontsize=25)
ax[2, 0].set_ylabel(r'{\tt ' + file_names[4] + '}', labelpad=20, fontsize=25)
ax[3, 0].set_ylabel(r'{\tt ' + file_names[5] + '}', labelpad=20, fontsize=25)

ax[3, 0].set_xlabel('Frekvenca [Hz]', fontsize=22)
ax[3, 1].set_xlabel('Frekvenca [Hz]', fontsize=22)

fig.tight_layout()
# plt.savefig('5FFT/porocilo/soviKonec.pdf')
# plt.show()
