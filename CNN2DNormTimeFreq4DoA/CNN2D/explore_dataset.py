# explore_dataset.py

import os
from dataset import WidebandULASTFTDataset
import numpy as np
import matplotlib.pyplot as plt
import random

# ========== CONFIGURATION ==========
os.makedirs("figures", exist_ok=True)

def safe_filename(name):
    """Create a safe filename in the 'figures' folder."""
    return "figures/" + "".join(c if c.isalnum() or c in "_-." else "_" for c in name) + ".png"

# ========== LOAD DATASET ==========
print("Chargement du dataset...")
dataset = WidebandULASTFTDataset("dataset")
X, y = [], []
for i in range(len(dataset)):
    x_i, y_i = dataset[i]
    X.append(x_i.numpy())         # Shape: (antennas, freq, time)
    y.append(y_i.numpy()[0])      # Shape: scalar
X = np.array(X)                   # Shape: (N, antennas, freq, time)
y = np.array(y)                   # Shape: (N,)

num_antennas = X.shape[1]
num_freqs = X.shape[2]
num_time = X.shape[3]

# ========== GLOBAL STATISTICS ==========
print("\n===== Statistiques Globales =====")
print(f"Nombre d'échantillons         : {len(X)}")
print(f"Nombre d'antennes             : {num_antennas}")
print(f"Dimensions temps-fréquence    : {num_freqs} fréquences × {num_time} temps")
print(f"DoA min / max / moyenne / std : {y.min():.2f}° / {y.max():.2f}° / {y.mean():.2f}° / {y.std():.2f}°")
print(f"Amplitude moyenne globale     : {X.mean():.3f}, std: {X.std():.3f}")

# ========== RANDOM SAMPLE ==========
idx = random.randint(0, len(X) - 1)
sample_signal = X[idx]
sample_doa = y[idx]
print(f"\n--- Exemple aléatoire : index {idx}, DoA = {sample_doa:.2f}° ---")

# ========== FIGURE 1: Time-Domain Signal ==========
try:
    plt.figure(figsize=(14, 6))
    for i in range(num_antennas):
        plt.plot(sample_signal[i].mean(axis=0), label=f'Antenne {i}')
    plt.title(f"Signal temporel moyen (DoA = {sample_doa:.2f}°)")
    plt.xlabel("Temps (indice)")
    plt.ylabel("Amplitude moyenne (sur fréquences)")
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(safe_filename("Signal_Temporel_Moyen_Antennes"), dpi=300)
    plt.show()
except Exception as e:
    print(f"[Erreur] Figure 1 : {e}")

# ========== FIGURE 2: Zoom Antenna 0 ==========
try:
    plt.figure(figsize=(12, 3))
    plt.imshow(sample_signal[0], aspect="auto", origin="lower", cmap="viridis")
    plt.title(f"Zoom temps-fréquence sur l'antenne 0 (DoA = {sample_doa:.2f}°)")
    plt.xlabel("Temps (indice)")
    plt.ylabel("Fréquence (indice)")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.savefig(safe_filename("Zoom_TF_Antenne_0"), dpi=300)
    plt.show()
except Exception as e:
    print(f"[Erreur] Figure 2 : {e}")

# ========== FIGURE 3: Mean STFT Spectrum ==========
try:
    mean_spectrum = sample_signal[0].mean(axis=1)
    plt.figure(figsize=(10, 4))
    plt.semilogy(mean_spectrum)
    plt.title("Spectre STFT moyen - Antenne 0")
    plt.xlabel("Indice de fréquence")
    plt.ylabel("Amplitude moyenne")
    plt.tight_layout()
    plt.savefig(safe_filename("Spectre_STFT_Moyen_Antenne_0"), dpi=300)
    plt.show()
except Exception as e:
    print(f"[Erreur] Figure 3 : {e}")

# ========== FIGURE 4: Heatmap Signal ==========
try:
    plt.figure(figsize=(8, 4))
    plt.imshow(sample_signal.mean(axis=1), aspect='auto', origin='lower', cmap='inferno')
    plt.title(f"Heatmap moyenne sur les fréquences (DoA = {sample_doa:.2f}°)")
    plt.xlabel("Temps (indice)")
    plt.ylabel("Antenne")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.savefig(safe_filename("Heatmap_Temps_Antennes"), dpi=300)
    plt.show()
except Exception as e:
    print(f"[Erreur] Figure 4 : {e}")

# ========== FIGURE 5: Histogram of DoA ==========
try:
    plt.figure(figsize=(10, 4))
    plt.hist(y, bins=60, density=True, alpha=0.6, color='steelblue', edgecolor='black', label="Histogramme")
    try:
        import seaborn as sns
        sns.kdeplot(y, color="crimson", label="Densité estimée")
    except ImportError:
        print("[Avertissement] Seaborn non installé - courbe de densité désactivée.")
    plt.title("Distribution des angles DoA")
    plt.xlabel("DoA (degrés)")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(safe_filename("Distribution_Angles_DOA"), dpi=300)
    plt.show()
except Exception as e:
    print(f"[Erreur] Figure 5 : {e}")

# ========== FIGURE 6: Energy vs DoA ==========
try:
    total_energy = np.sum(X ** 2, axis=(1, 2, 3))  # Shape: (N,)
    plt.figure(figsize=(7, 4))
    plt.scatter(y, total_energy, alpha=0.4, color='darkgreen')
    plt.title("Corrélation entre l'énergie reçue et l'angle DoA")
    plt.xlabel("DoA (degrés)")
    plt.ylabel("Énergie totale reçue")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(safe_filename("Energie_vs_DoA"), dpi=300)
    plt.show()
except Exception as e:
    print(f"[Erreur] Figure 6 : {e}")

# ========== FIGURE 7: Per-Antenna Noise STD ==========
try:
    noise_std = np.std(sample_signal, axis=(1, 2))  # Shape: (num_antennas,)
    print("Écart-type par antenne (exemple choisi):", noise_std)
    plt.figure(figsize=(7, 3))
    plt.bar(np.arange(num_antennas), noise_std, color='orange')
    plt.title("Écart-type (bruit) par antenne")
    plt.xlabel("Antenne")
    plt.ylabel("STD")
    plt.tight_layout()
    plt.savefig(safe_filename(f"STD_Bruit_Antennes_Exemple_{idx}"), dpi=300)
    plt.show()
except Exception as e:
    print(f"[Erreur] Figure 7 : {e}")
