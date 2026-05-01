"""Synthetic neural signal generation for testing the evaluation harness.

Goal: produce signals with realistic 1/f power-law structure plus optional
band-limited oscillatory bursts, so we can construct controlled test cases
where we *know* what's in each frequency band.

These are not meant to look like real iEEG. They're meant to let us answer
"does my high-gamma NMSE metric correctly score a model that smooths out
gamma activity?" before we have real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SignalConfig:
    """Parameters for synthetic neural signal generation.

    Attributes:
        n_samples: Number of time samples per channel.
        n_channels: Number of channels.
        fs: Sampling rate in Hz.
        aperiodic_exponent: Exponent of the 1/f^alpha aperiodic component.
            iEEG empirically sits between ~1.0 and ~2.5 (Donoghue et al. 2020).
            Default 1.5 is a reasonable "average" value.
        broadband_amplitude: Amplitude scaling for the 1/f component.
        bursts: List of (center_freq_hz, amplitude, rate_per_sec) tuples
            describing band-limited oscillatory bursts to inject. Each burst
            is a Gaussian-windowed sinusoid placed at a random time.
        seed: Random seed for reproducibility.
    """

    n_samples: int
    n_channels: int = 1
    fs: float = 2048.0
    aperiodic_exponent: float = 1.5
    broadband_amplitude: float = 1.0
    bursts: list[tuple[float, float, float]] | None = None
    seed: int | None = None


def generate_pink_noise(
    n_samples: int,
    n_channels: int,
    fs: float,
    exponent: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate 1/f^exponent noise via spectral shaping.

    We construct white noise in the frequency domain, scale each frequency bin
    by 1/f^(exponent/2) (so power scales by 1/f^exponent), and inverse-FFT.
    This is the cleanest way to get exact 1/f statistics — direct time-domain
    methods (e.g., Voss-McCartney) give only approximate scaling.

    Returns:
        Signal of shape (n_channels, n_samples), zero-mean, unit-variance per
        channel, then scaled by `amplitude`.
    """
    # Use rfft frequencies; we'll construct in spectral domain then invert.
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    # Avoid divide-by-zero at DC; we'll zero out the DC bin anyway.
    scale = np.zeros_like(freqs)
    scale[1:] = freqs[1:] ** (-exponent / 2.0)

    signals = np.empty((n_channels, n_samples), dtype=np.float64)
    for c in range(n_channels):
        # Random phases, unit-magnitude white spectrum.
        phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
        white_spectrum = np.exp(1j * phases)
        # Real signal requires conjugate symmetry; rfft handles this implicitly
        # because we'll use irfft. But we need the DC bin and Nyquist bin to
        # be real-valued. Setting DC to 0 (zero-mean signal) and Nyquist to
        # real is the cleanest approach.
        white_spectrum[0] = 0.0
        if n_samples % 2 == 0:
            white_spectrum[-1] = white_spectrum[-1].real

        shaped = white_spectrum * scale
        sig = np.fft.irfft(shaped, n=n_samples)
        # Normalize to unit variance, then scale.
        sig = sig / sig.std()
        signals[c] = amplitude * sig

    return signals


def add_oscillatory_bursts(
    signal: np.ndarray,
    fs: float,
    bursts: list[tuple[float, float, float]],
    rng: np.random.Generator,
    burst_duration_sec: float = 0.15,
) -> np.ndarray:
    """Add Gaussian-windowed sinusoidal bursts to an existing signal.

    Each burst is parameterized by (center_freq, amplitude, rate_per_sec).
    Burst times are sampled as a homogeneous Poisson process per channel.

    Args:
        signal: (n_channels, n_samples) array.
        fs: Sampling rate.
        bursts: List of (freq_hz, amplitude, rate_per_sec).
        rng: Random number generator.
        burst_duration_sec: 1-sigma width of the Gaussian envelope.

    Returns:
        Signal with bursts added (modifies a copy, not in place).
    """
    out = signal.copy()
    n_channels, n_samples = signal.shape
    duration_sec = n_samples / fs

    for center_freq, amplitude, rate in bursts:
        sigma_samples = int(burst_duration_sec * fs)
        # Window length: 6 sigma covers >99.7%.
        window_len = 6 * sigma_samples
        t_window = np.arange(window_len) - window_len // 2
        envelope = np.exp(-(t_window**2) / (2 * sigma_samples**2))

        for c in range(n_channels):
            n_bursts = rng.poisson(rate * duration_sec)
            # Sample burst onset times uniformly in [0, n_samples - window_len].
            if n_samples <= window_len:
                continue
            onsets = rng.integers(0, n_samples - window_len, size=n_bursts)
            for onset in onsets:
                phase = rng.uniform(0, 2 * np.pi)
                t_local = np.arange(window_len) / fs
                burst = (
                    amplitude
                    * envelope
                    * np.sin(2 * np.pi * center_freq * t_local + phase)
                )
                out[c, onset : onset + window_len] += burst

    return out


def generate_signal(config: SignalConfig) -> np.ndarray:
    """Generate a synthetic neural signal per the given config.

    Returns:
        (n_channels, n_samples) array.
    """
    rng = np.random.default_rng(config.seed)
    signal = generate_pink_noise(
        n_samples=config.n_samples,
        n_channels=config.n_channels,
        fs=config.fs,
        exponent=config.aperiodic_exponent,
        amplitude=config.broadband_amplitude,
        rng=rng,
    )
    if config.bursts:
        signal = add_oscillatory_bursts(signal, config.fs, config.bursts, rng)
    return signal


def smooth_high_frequencies(
    signal: np.ndarray,
    fs: float,
    cutoff_hz: float,
    attenuation_db: float = 20.0,
) -> np.ndarray:
    """Simulate a model that smooths out high-frequency content.

    This is what we expect MSE-trained models to do to gamma activity. Apply a
    soft low-pass: above `cutoff_hz`, attenuate by `attenuation_db` dB. We use
    a smooth transition rather than a hard cutoff to mimic the behavior of a
    learned smoother, not a sharp filter.

    Args:
        signal: (..., n_samples) array.
        fs: Sampling rate.
        cutoff_hz: Frequency above which to attenuate.
        attenuation_db: How much to attenuate (in dB) at frequencies well
            above cutoff. 20 dB = 10x reduction in amplitude.

    Returns:
        Signal with high frequencies attenuated.
    """
    n = signal.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Smooth sigmoid transition centered on cutoff with width ~cutoff/4.
    transition_width = cutoff_hz / 4.0
    attenuation_linear = 10 ** (-attenuation_db / 20.0)
    # Sigmoid: 1 below cutoff, attenuation_linear far above.
    sigmoid = 1.0 / (1.0 + np.exp((freqs - cutoff_hz) / transition_width))
    response = attenuation_linear + (1.0 - attenuation_linear) * sigmoid

    spectrum = np.fft.rfft(signal, axis=-1)
    smoothed = np.fft.irfft(spectrum * response, n=n, axis=-1)
    return smoothed
