"""audio_engine.py – DSP module for acoustic environment harness.

Manages logarithmic sine-sweep synthesis, synchronous capture via sounddevice,
and metrological extractions: T60 (Schroeder integration), comb-filter notch
detection (cepstrum), Welch PSD, RMS, and transient detection.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate, find_peaks, welch

try:
    import sounddevice as sd  # type: ignore[import]
    _SD_AVAILABLE = True
except (OSError, ImportError):
    sd = None  # type: ignore[assignment]
    _SD_AVAILABLE = False


class AudioEngine:
    """Low-level DSP engine wrapping sounddevice for acoustic measurements."""

    def __init__(self, sample_rate: int = 48000) -> None:
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Active measurement – impulse response
    # ------------------------------------------------------------------

    def measure_ir(
        self,
        f_min: float = 150.0,
        f_max: float = 14000.0,
        duration_sec: float = 2.0,
    ) -> dict:
        """Emit a windowed log-sweep, record the response, and extract acoustics.

        Parameters
        ----------
        f_min : float
            Lower sweep frequency in Hz (default 150 Hz).
        f_max : float
            Upper sweep frequency in Hz (default 14 000 Hz).
        duration_sec : float
            Sweep duration in seconds (clamped to ≥ 0.5 s).

        Returns
        -------
        dict with keys:
            t60_est_sec            – Estimated T60 via Schroeder backwards integration (s).
            comb_filter_notch_hz   – First comb-filter notch from table reflection (Hz or None).
            snr_peak_db            – Peak SNR of the measured IR (dB).
            duration_measured_sec  – Actual sweep duration used (s).
        """
        sr = self.sample_rate
        duration_sec = max(0.5, float(duration_sec))
        f_min = max(20.0, float(f_min))
        f_max = min(float(f_max), sr / 2.0 - 100.0)

        num_samples = int(sr * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)

        # --- Logarithmic sine sweep synthesis ---
        rate_ratio = f_max / f_min
        sweep = np.sin(
            2.0 * np.pi * f_min * (rate_ratio ** (t / duration_sec) - 1.0) / np.log(rate_ratio)
        )
        # Hann window + conservative amplitude to avoid transducer clipping
        sweep = sweep * 0.25 * np.hanning(num_samples)

        # --- Synchronous playback + recording (mono) ---
        if not _SD_AVAILABLE:
            return {
                "t60_est_sec": 0.08,
                "comb_filter_notch_hz": None,
                "snr_peak_db": 0.0,
                "duration_measured_sec": round(duration_sec, 2),
                "error": "sounddevice/PortAudio not available",
            }
        try:
            recording = sd.playrec(
                sweep.astype(np.float32),
                samplerate=sr,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            sig = recording.flatten()
        except sd.PortAudioError as exc:
            # Return safe defaults when no audio hardware is available
            return {
                "t60_est_sec": 0.08,
                "comb_filter_notch_hz": None,
                "snr_peak_db": 0.0,
                "duration_measured_sec": round(duration_sec, 2),
                "error": str(exc),
            }

        # --- Deconvolution by cross-correlation (approximate IR) ---
        ir = correlate(sig, sweep, mode="full")[len(sweep) - 1 :]
        max_abs = float(np.max(np.abs(ir))) + 1e-9
        ir = ir / max_abs

        # --- Schroeder backwards integration → T60 ---
        energy = np.cumsum(ir[::-1] ** 2)[::-1]
        energy_norm = energy / (energy[0] + 1e-9)
        energy_db = 10.0 * np.log10(np.clip(energy_norm, 1e-9, 1.0))

        idx_5db = np.where(energy_db <= -5.0)[0]
        idx_25db = np.where(energy_db <= -25.0)[0]

        if len(idx_5db) > 0 and len(idx_25db) > 0 and idx_25db[0] > idx_5db[0]:
            t20 = (idx_25db[0] - idx_5db[0]) / sr
            t60_est = float(round(3.0 * t20, 3))
        else:
            # Quasi-anechoic / very confined space (e.g. car cabin)
            t60_est = 0.08

        # --- Cepstrum on first 30 ms → comb-filter notch from table reflection ---
        early_window = ir[: int(0.03 * sr)]
        autocorr = correlate(early_window, early_window, mode="full")[len(early_window) - 1 :]

        min_lag = max(1, int(0.00025 * sr))  # ~0.25 ms  → ~8.5 cm path difference
        max_lag = int(0.005 * sr)             # ~5.0 ms   → ~1.7 m path difference
        search_region = autocorr[min_lag : max_lag + 1]

        comb_notch_hz: int | None = None
        if len(search_region) > 0:
            peaks, _ = find_peaks(search_region, distance=max(1, int(0.0002 * sr)))
            if len(peaks) > 0:
                tau = (int(peaks[0]) + min_lag) / sr
                comb_notch_hz = int(round(1.0 / (2.0 * tau)))

        # --- Peak SNR ---
        noise_floor_db = float(np.median(energy_db[-max(1, int(0.1 * sr)) :]))
        snr_db = float(round(float(np.max(energy_db)) - noise_floor_db, 1))

        return {
            "t60_est_sec": t60_est,
            "comb_filter_notch_hz": comb_notch_hz,
            "snr_peak_db": snr_db,
            "duration_measured_sec": round(duration_sec, 2),
        }

    # ------------------------------------------------------------------
    # Passive measurement – ambient listening
    # ------------------------------------------------------------------

    def listen_ambient(self, duration_sec: float = 5.0) -> dict:
        """Passively record the acoustic environment without emitting any sound.

        Parameters
        ----------
        duration_sec : float
            Recording duration in seconds (clamped to ≥ 0.5 s).

        Returns
        -------
        dict with keys:
            rms_db                     – Mean level in dBFS.
            dominant_stationary_freq_hz – Frequency of PSD peak (Hz).
            low_frequency_energy_ratio  – Fraction of energy below 200 Hz.
            crest_factor                – Peak-to-RMS ratio (impulsive indicator).
            transients_detected        – True when crest_factor > 4.5.
            duration_measured_sec      – Actual recording duration used (s).
        """
        sr = self.sample_rate
        duration_sec = max(0.5, float(duration_sec))
        num_samples = int(sr * duration_sec)

        if not _SD_AVAILABLE:
            return {
                "rms_db": -60.0,
                "dominant_stationary_freq_hz": 0.0,
                "low_frequency_energy_ratio": 0.0,
                "crest_factor": 1.0,
                "transients_detected": False,
                "duration_measured_sec": round(duration_sec, 2),
                "error": "sounddevice/PortAudio not available",
            }
        try:
            recording = sd.rec(num_samples, samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            sig = recording.flatten()
        except sd.PortAudioError as exc:
            return {
                "rms_db": -60.0,
                "dominant_stationary_freq_hz": 0.0,
                "low_frequency_energy_ratio": 0.0,
                "crest_factor": 1.0,
                "transients_detected": False,
                "duration_measured_sec": round(duration_sec, 2),
                "error": str(exc),
            }

        # --- Welch PSD ---
        nperseg = min(len(sig), 4096)
        freqs, psd = welch(sig, fs=sr, nperseg=nperseg)
        dominant_freq = float(freqs[int(np.argmax(psd))])

        # --- Temporal/dynamic metrics ---
        rms_val = float(np.sqrt(np.mean(sig ** 2))) + 1e-9
        rms_db = float(round(20.0 * np.log10(rms_val), 1))

        std_val = float(np.std(sig)) + 1e-9
        crest_factor = float(np.max(np.abs(sig))) / std_val
        transients_detected = bool(crest_factor > 4.5)

        # --- Low-frequency energy ratio (below 200 Hz – engine/road noise proxy) ---
        idx_low = np.where(freqs <= 200.0)[0]
        low_energy_ratio = float(round(float(np.sum(psd[idx_low])) / (float(np.sum(psd)) + 1e-9), 3))

        return {
            "rms_db": rms_db,
            "dominant_stationary_freq_hz": round(dominant_freq, 1),
            "low_frequency_energy_ratio": low_energy_ratio,
            "crest_factor": round(crest_factor, 2),
            "transients_detected": transients_detected,
            "duration_measured_sec": round(duration_sec, 2),
        }
