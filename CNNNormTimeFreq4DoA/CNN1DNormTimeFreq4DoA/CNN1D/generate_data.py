import numpy as np
import os

# PARAMETERS
c = 3e8
fc = 1e9
fs = 10e9
num_antennas = 8
d = c / (2 * fc)
num_samples = 512
snr_db_range = (-10, 30)
doa_range = (-90, 90)
num_signals = 10000

def wideband_signal(length):
    t = np.arange(length) / fs
    pulse = np.exp(-((t - t.mean()) ** 2) / (2 * (10e-9)**2))
    return pulse / np.max(np.abs(pulse))

def apply_doa(signal, doa_deg, num_antennas, d, fc):
    c = 3e8
    fs = 10e9  # assure-toi que fs est bien défini ici aussi (ou passe-le en argument)
    doa_rad = np.radians(doa_deg)
    delays = d * np.sin(doa_rad) * np.arange(num_antennas) / c
    samples_delay = (delays * fs).astype(int)
    sigs = np.zeros((num_antennas, len(signal)), dtype=np.float32)
    for i, sd in enumerate(samples_delay):
        if 0 <= sd < len(signal):
            sigs[i, sd:] = signal[:len(signal)-sd]
    return sigs


def add_noise(signal, snr_db):
    signal_power = np.mean(signal**2)
    snr = 10 ** (snr_db / 10)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def main():
    out_dir = "dataset"
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_signals):
        doa = np.random.uniform(*doa_range)
        snr = np.random.uniform(*snr_db_range)
        pulse = wideband_signal(num_samples)
        signal = apply_doa(pulse, doa, num_antennas, d, fc)
        signal_noisy = add_noise(signal, snr)
        np.save(os.path.join(out_dir, f"signal_{i:05d}.npy"), signal_noisy)
        np.save(os.path.join(out_dir, f"doa_{i:05d}.npy"), np.array([doa]))
    print("DONE: {} samples written in '{}'".format(num_signals, out_dir))

if __name__ == "__main__":
    main()
