from scipy.signal import find_peaks
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.style.use('customStyle.mplstyle')

color_gradient = burnt_bronze = [
    '#FFFFFF', '#FFF4EA', '#FFE5D1', '#FFD3B5', '#FFC199',
    '#FFAE7D', '#FF9B61', '#E67035', '#CC5B2B', '#B34921',
    '#993817', '#662900', '#4D1F00', '#331400'
]
custom_color_gradient = LinearSegmentedColormap.from_list(
    'custom_blue', color_gradient)

# import mp3 file
audio_path = r'5FFT/.local/DavidMaček.wav'
y, sr = librosa.load(audio_path)

# nyquist frequency
nyquist = sr/2

# Cut audio
# y = y[int(sr*6.5):int(sr*22.5)]

# draw a spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max) + 80
# Color: sequential, continuous, light for low values, dark for high values
librosa.display.specshow(D, sr=sr, y_axis='log', cmap=custom_color_gradient, x_axis='time')
plt.grid(True, alpha=1, color='black', linestyle='--')
plt.colorbar(format='%+2.0f dB')
plt.xlabel(r'Čas $t$ [s]', labelpad=15)
plt.ylabel('Frekvenca [Hz]')
plt.title('Spektrogram sodomovčevega petja')
plt.tight_layout()
plt.ylim(200, 2000)
# plt.savefig('5FFT/porocilo/resitev.pdf')
plt.show()

# draw a waveform
plt.plot(np.arange(len(y))/sr, y)
plt.xlabel('Čas $t$ [s]')
plt.ylabel('Amplituda')
plt.title('Valovna oblika')
plt.tight_layout()
# plt.savefig('5FFT/porocilo/valovna_oblika.pdf')
plt.show()

# Auto-correlation
waveform = autocorr = librosa.autocorrelate(y)

# Make spectrogram from auto-correlation
hop_length = 512
n_fft = 2048

spec = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length))
spec_db = librosa.amplitude_to_db(spec, ref=np.max) + 80

# plt.figure(figsize=(30, 5))
librosa.display.specshow(spec_db, sr=sr, y_axis='log',
                         cmap=custom_color_gradient)
plt.grid(False)
plt.colorbar(format='%+2.0f dB')
plt.xlabel(r'Čas $t$ [s]', labelpad=15)
plt.ylabel('Frekvenca [Hz]')
plt.title('Spektrogram')
plt.tight_layout()
# plt.ylim(1600, 1900)
# plt.savefig('5FFT/porocilo/spektrogram_auto_f.pdf')
plt.show()

# Plot auto-correlation in time domain
plt.plot(np.arange(1, len(autocorr)+1)/sr, autocorr, label='Amplituda')
plt.title('Valovna oblika po avtokorelaciji')
plt.xlabel(r'Zamik $\tau$ [s]')
plt.ylabel('Amplituda')
plt.tight_layout()
# plt.savefig('5FFT/porocilo/avtokorelacija.pdf')
plt.show()

# Plot auto-correlation another way
plt.plot(sr/np.arange(1, len(autocorr)+1), autocorr, label='Avto korelacija')
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.axvline(nyquist, linestyle='--',
            label='Nyquistova frekvenca', color=color_cycle[1])
plt.title('Frekvence avtokorelacije')
plt.xlabel(r'Frekvenca $f$ [Hz]')
plt.ylabel('Amplituda')
plt.xlim(-nyquist*0.03, nyquist * 1.2)
plt.tight_layout()
plt.legend(loc='upper center')
# plt.savefig('boni_zanimivo.pdf')
plt.show()

# Plot auto-correlation in frequency domain
plt.plot(np.fft.fftfreq(len(autocorr), 1/sr),
         np.abs(np.fft.fft(autocorr)), label='Signal')
plt.title('FFT avtokorelacije')
plt.xlabel(r'Frekvenca $f$ [Hz]')
plt.ylabel('Amplituda')
plt.tight_layout()
# plt.xlim(1300, 4000)
# plt.xlim(100, 650)
plt.yscale('log')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(1, 4):
    continue
    plt.axvline(i*1650, linestyle='--', color=colors[1], linewidth=2)
    plt.annotate(r'$\SI{'+str(i*1650)+r'}{\hertz}$',
                 (i*1650, 1e5), textcoords='offset points', xytext=(-40, 250), ha='center', fontsize=18, color=colors[1])

    plt.axvline(i*1834, linestyle='--', color=colors[4], linewidth=2,
                label='$' + str(i) + r'\cdot \SI{'+str(1834)+r'}{\hertz}$')
    plt.annotate(r'$\SI{'+str(i*1834)+r'}{\hertz}$',
                 (i*1834, 1e5), textcoords='offset points', xytext=(40, 220), ha='center', fontsize=18, color=colors[4])
# plt.savefig('5FFT/porocilo/fft_avtokorelacija_boni2.pdf')
plt.show()
