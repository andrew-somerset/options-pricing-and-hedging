"""Simulated price paths for the hedging experiment. Everything here is SYNTHETIC."""

import numpy as np


def price_paths(spot, drift, vol, t, n_steps, n_paths, rng,
                jump_intensity=0.0, jump_mean=0.0, jump_std=0.0):
    """Geometric Brownian motion, optionally with lognormal jumps (Merton).

    drift and vol are annual; jump_intensity is expected jumps per year, and each
    jump multiplies the price by exp(N(jump_mean, jump_std^2)). The drift is
    compensated so the expected growth rate stays `drift` with or without jumps.

    Returns an array of shape (n_paths, n_steps + 1); column 0 is `spot`.
    """
    dt = t / n_steps
    compensator = jump_intensity * (np.exp(jump_mean + 0.5 * jump_std**2) - 1)
    log_returns = ((drift - 0.5 * vol**2 - compensator) * dt
                   + vol * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps)))
    if jump_intensity > 0:
        n_jumps = rng.poisson(jump_intensity * dt, size=(n_paths, n_steps))
        log_returns += n_jumps * jump_mean + np.sqrt(n_jumps) * jump_std * rng.standard_normal((n_paths, n_steps))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = spot
    paths[:, 1:] = spot * np.exp(np.cumsum(log_returns, axis=1))
    return paths


def diffusion_vol_for_total(total_vol, jump_intensity, jump_mean, jump_std):
    """Diffusion vol that keeps total annual variance equal to total_vol^2 once jumps are added."""
    jump_variance = jump_intensity * (jump_mean**2 + jump_std**2)
    if jump_variance >= total_vol**2:
        raise ValueError("jumps alone exceed the target variance")
    return np.sqrt(total_vol**2 - jump_variance)
