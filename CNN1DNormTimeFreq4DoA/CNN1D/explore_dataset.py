import numpy as np
import matplotlib.pyplot as plt
from dataset import WidebandULADataset
import random

# Chargement du dataset
dataset = WidebandULADataset("dataset")
X, y = [], []
for i in range(len(dataset)):
    x_i, y_i = dataset[i]
    X.append(x_i.numpy())
    y.append(y_i.numpy()[0])
X = np.array(X)   # (N, num_antennas, num_samples)
y = np.array(y)   # (N,)

num_antennas = X.shape[1]
num_samples = X.shape[2]

# ----- STATISTIQUES GLOBALES -----
print(f"Nombre d'échantillons: {len(X)}")
print(f"Nombre d'antennes: {num_antennas}, nombre d'échantillons temporels: {num_samples}")
print(f"DoA min/max/moyenne/std: {y.min():.2f}° / {y.max():.2f}° / {y.mean():.2f}° / {y.std():.2f}°")
print(f"Amplitude moyenne (sur tous): {X.mean():.3f}, std: {X.std():.3f}")

# ----- EXEMPLE ALÉATOIRE -----
idx = random.randint(0, len(X)-1)
sample_signal = X[idx]
sample_doa = y[idx]
print(f"\nÉchantillon choisi: {idx}   DoA: {sample_doa:.2f}°")

# 1. Courbes temporelles multi-antennes
plt.figure(figsize=(14, 6))
for i in range(num_antennas):
    plt.plot(sample_signal[i], label=f'Antenne {i}')
plt.title(f"Signal temporel reçu sur chaque antenne (DoA = {sample_doa:.2f}°)")
plt.xlabel("Temps (échantillon)")
plt.ylabel("Amplitude normalisée")
plt.legend(loc="upper right", ncol=2)
plt.tight_layout()
plt.show()

# 2. Zoom sur une antenne spécifique (zoom dynamique)
for antenne_zoom in [0, num_antennas//2, num_antennas-1]:
    plt.figure(figsize=(12, 3))
    plt.plot(sample_signal[antenne_zoom])
    plt.title(f"Zoom sur l'antenne {antenne_zoom} (DoA = {sample_doa:.2f}°)")
    plt.xlabel("Temps (échantillon)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

# 3. FFT (spectre) pour plusieurs antennes
from scipy.fft import rfft, rfftfreq
fs = 10e9  # 10 GHz
freqs = rfftfreq(num_samples, d=1/fs)
plt.figure(figsize=(10, 5))
for antenne_fft in [0, num_antennas//2, num_antennas-1]:
    fft_vals = rfft(sample_signal[antenne_fft])
    plt.semilogy(freqs/1e6, np.abs(fft_vals), label=f'Antenne {antenne_fft}')
plt.title(f"Spectre fréquentiel (FFT) sur plusieurs antennes (DoA = {sample_doa:.2f}°)")
plt.xlabel("Fréquence (MHz)")
plt.ylabel("|FFT|")
plt.legend()
plt.tight_layout()
plt.show()

# 4. Heatmap brute: toute la trame d'antennes
plt.figure(figsize=(8, 4))
plt.imshow(sample_signal, aspect='auto', origin='lower', cmap='inferno')
plt.title(f"Heatmap du signal (antennes x temps), DoA = {sample_doa:.2f}°")
plt.xlabel("Temps (échantillon)")
plt.ylabel("Antenne")
plt.colorbar(label="Amplitude")
plt.tight_layout()
plt.show()

# 5. Heatmap du covariogramme d'antennes (corrélation spatiale)
R = np.dot(sample_signal, sample_signal.T)
plt.figure(figsize=(6, 5))
plt.imshow(R, cmap='viridis')
plt.title("Covariance spatiale (antennes x antennes)")
plt.xlabel("Antenne")
plt.ylabel("Antenne")
plt.colorbar(label="Cov(x, y)")
plt.tight_layout()
plt.show()

# 6. Histogramme & densité des angles DoA
plt.figure(figsize=(10, 4))
plt.hist(y, bins=60, density=True, alpha=0.6, color='steelblue', edgecolor='k', label="Histogramme")
try:
    import seaborn as sns
    sns.kdeplot(y, color="crimson", label="Densité estimée")
except ImportError:
    pass
plt.title("Distribution des angles DOA")
plt.xlabel("DoA (degrés)")
plt.ylabel("Densité de probabilité")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Corrélation énergie totale / angle
total_energy = np.sum(X**2, axis=(1,2))
plt.figure(figsize=(7,4))
plt.scatter(y, total_energy, alpha=0.4)
plt.title("Corrélation entre énergie reçue et DoA")
plt.xlabel("DoA (degrés)")
plt.ylabel("Énergie totale reçue (somme x²)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Écart-type du bruit (par antenne) pour l'échantillon
noise_std = np.std(sample_signal, axis=1)
print("Écart-type par antenne (sur cet exemple):", noise_std.round(4))
plt.figure(figsize=(7,3))
plt.bar(np.arange(num_antennas), noise_std)
plt.title("Écart-type (bruit/puissance) par antenne pour l'exemple choisi")
plt.xlabel("Antenne")
plt.ylabel("STD")
plt.tight_layout()
plt.show()

# 9. Boîte à moustaches (boxplot) de l'énergie par antenne sur tout le dataset
energy_ant = np.sum(X**2, axis=2)  # (N, num_antennas)
plt.figure(figsize=(8,5))
plt.boxplot(energy_ant, vert=True, patch_artist=True)
plt.title("Énergie par antenne (distribution sur tout le dataset)")
plt.xlabel("Antenne")
plt.ylabel("Énergie totale (x²)")
plt.tight_layout()
plt.show()

# 10. Matrice de corrélation des signaux entre antennes (pour l'exemple)
corr = np.corrcoef(sample_signal)
plt.figure(figsize=(6,5))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matrice de corrélation entre antennes (exemple)")
plt.xlabel("Antenne")
plt.ylabel("Antenne")
plt.colorbar(label="Corrélation")
plt.tight_layout()
plt.show()
