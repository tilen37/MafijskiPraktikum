import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 14
})

# Number of files to process
num_files = len(os.listdir('4FourierovaAnaliza/code/data'))

# Generate a colormap
colors = cm.viridis(np.linspace(0, 1, num_files))

# Initialize plot
plt.figure(figsize=(10, 7))

i, j = -1, 0

for idx, file in enumerate(os.listdir('4FourierovaAnaliza/code/data')):
    print(file[6:-4])
    with open('4FourierovaAnaliza/code/data/' + file, 'r') as f:
        i += 1
        if i > 2:
            i = 0
            j += 1

        # Read the file
        data = f.read()
        num_array = [float(i) for i in data.split('\n') if i]

        y = np.fft.fft(num_array)
        y = np.abs(y)
        y = y[:len(y)//2]
        y = y / y.max()

        # Plot with gradient color
        plt.plot(y, color=colors[idx],
                 label=file[6:-4]+r'Hz', alpha=0.7)

plt.xlim([-100, 2500])
# plt.yscale('log')
plt.legend(loc='upper center', fontsize=12, shadow=True,
           fancybox=True, bbox_to_anchor=(0.5, 1.05), ncol=6, edgecolor='black')
plt.grid(alpha=0.5)
plt.xlabel('Frekvenca')
plt.ylabel('Amplituda')
plt.title('Fourierjeve transformacije različnih posnetkov', pad=35)

plt.tight_layout()
# plt.savefig(r'4FourierovaAnaliza/porocilo/figs/Bach_lin.pdf')
plt.show()
