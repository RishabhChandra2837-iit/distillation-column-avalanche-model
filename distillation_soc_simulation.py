import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
import json
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Parameters ───────────────────────────────────────────────────────────────
N_TRAYS      = 30          # Number of trays (connectivity nodes)
Z_CRITICAL   = 10.0        # Critical holdup threshold (flooding limit)
Z_DELTA      = 0.5         # Holdup added per perturbation step
N_STEPS      = 200_000     # Number of simulation steps
TOPPLING_FRAC = 0.4        # Fraction redistributed to neighbours on avalanche


# ── Core SOC Model ───────────────────────────────────────────────────────────

def initialize_column(n_trays, z_critical):
    """Start with random sub-critical holdup on each tray."""
    return np.random.uniform(0, z_critical * 0.6, n_trays)


def add_perturbation(z, delta, n_trays):
    """Randomly add a small liquid perturbation to one tray (feed disturbance)."""
    tray = np.random.randint(0, n_trays)
    z[tray] += delta
    return z, tray


def topple(z, n_trays, z_critical, toppling_frac):
    """
    Cascade rule:  if z[i] >= z_critical, tray i 'floods'.
    Liquid redistributes to adjacent trays (i-1 and i+1),
    simulating weeping/overflow coupling between trays.
    Returns updated z and the size of this avalanche cascade.
    """
    avalanche_size = 0
    active = True
    while active:
        active = False
        for i in range(n_trays):
            if z[i] >= z_critical:
                active = True
                avalanche_size += 1
                overflow = toppling_frac * z[i]
                z[i] -= overflow
                # Distribute to neighbours (boundary = open, liquid lost)
                if i > 0:
                    z[i - 1] += overflow / 2
                if i < n_trays - 1:
                    z[i + 1] += overflow / 2
                # At boundaries liquid exits the column
    return z, avalanche_size


def run_simulation(n_trays, z_critical, z_delta, n_steps, toppling_frac):
    """
    Main simulation loop.
    Returns avalanche size distribution and tray holdup time series.
    """
    z = initialize_column(n_trays, z_critical)
    avalanche_sizes = []
    holdup_history  = np.zeros((n_steps, n_trays))
    noise_signal    = []

    for step in range(n_steps):
        z, _ = add_perturbation(z, z_delta, n_trays)
        z, av_size = topple(z, n_trays, z_critical, toppling_frac)

        noise_signal.append(np.std(z))          # 'noise' = spread of holdup
        holdup_history[step] = z.copy()

        if av_size > 0:
            avalanche_sizes.append(av_size)

    return np.array(avalanche_sizes), holdup_history, np.array(noise_signal)


# ── Connectivity Sweep ────────────────────────────────────────────────────────

def connectivity_sweep(tray_counts, z_critical, z_delta, n_steps, toppling_frac):
    """
    Run the SOC model for different column sizes (connectivity proxy).
    Returns mean avalanche size vs connectivity.
    """
    results = {}
    for n in tray_counts:
        av, _, _ = run_simulation(n, z_critical, z_delta, n_steps, toppling_frac)
        results[n] = av
        print(f"  N={n:3d} trays | avalanches: {len(av):6d} | "
              f"mean size: {av.mean():.2f}")
    return results


# ── Analysis Helpers ──────────────────────────────────────────────────────────

def power_law_fit(sizes):
    """Fit P(s) ~ s^(-tau) via log-binned histogram."""
    if len(sizes) < 50:
        return None, None, None
    bins  = np.logspace(np.log10(max(sizes.min(), 1)),
                        np.log10(sizes.max()), 30)
    counts, edges = np.histogram(sizes, bins=bins)
    centres = np.sqrt(edges[:-1] * edges[1:])
    mask = counts > 0
    if mask.sum() < 4:
        return None, None, None
    log_s = np.log10(centres[mask])
    log_p = np.log10(counts[mask] / counts[mask].sum())
    slope, intercept, r, *_ = linregress(log_s, log_p)
    return slope, intercept, r ** 2


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all(avalanche_sizes, holdup_history, noise_signal,
             connectivity_results, n_trays, n_steps):

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Self-Organized Criticality in a Distillation Column",
                 fontsize=15, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Tray holdup heatmap (noise / fluctuations) ─────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    sample = holdup_history[::500]         # downsample for plotting
    im = ax1.imshow(sample.T, aspect='auto', origin='lower',
                    cmap='plasma',
                    extent=[0, n_steps, 0, n_trays])
    plt.colorbar(im, ax=ax1, label='Holdup (liquid volume)')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Tray number')
    ax1.set_title('Figure 1 – Tray Holdup Dynamics (Noise Landscape)')

    # ── 2. Noise signal ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(noise_signal[::200], color='steelblue', lw=0.6, alpha=0.8)
    ax2.set_xlabel('Time step (×200)')
    ax2.set_ylabel('Std dev of holdup')
    ax2.set_title('Figure 2 – System Noise Over Time')
    ax2.axhline(np.mean(noise_signal), color='red', ls='--',
                label=f'Mean = {np.mean(noise_signal):.2f}')
    ax2.legend(fontsize=8)

    # ── 3. Avalanche size distribution (power law) ────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if len(avalanche_sizes) > 0:
        bins  = np.logspace(np.log10(max(avalanche_sizes.min(), 1)),
                            np.log10(avalanche_sizes.max()), 30)
        counts, edges = np.histogram(avalanche_sizes, bins=bins)
        centres = np.sqrt(edges[:-1] * edges[1:])
        mask = counts > 0
        ax3.loglog(centres[mask], counts[mask] / counts[mask].sum(),
                   'o', color='darkorange', ms=5, label='Simulation')

        slope, intercept, r2 = power_law_fit(avalanche_sizes)
        if slope is not None:
            x_fit = centres[mask]
            y_fit = 10 ** (intercept + slope * np.log10(x_fit))
            ax3.loglog(x_fit, y_fit, 'k--',
                       label=f'Power law τ={-slope:.2f}, R²={r2:.3f}')
        ax3.set_xlabel('Avalanche size s')
        ax3.set_ylabel('P(s)')
        ax3.set_title('Figure 3 – Avalanche Size Distribution')
        ax3.legend(fontsize=8)

    # ── 4. Avalanche time series ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    # Show first 2000 avalanches
    display = avalanche_sizes[:2000] if len(avalanche_sizes) > 2000 else avalanche_sizes
    ax4.bar(range(len(display)), display, width=1.0,
            color='mediumseagreen', alpha=0.7)
    ax4.set_xlabel('Avalanche event index')
    ax4.set_ylabel('Avalanche size')
    ax4.set_title('Figure 4 – Avalanche Event Series (first 2 000 events)')

    # ── 5. Connectivity vs mean avalanche size ────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ns   = sorted(connectivity_results.keys())
    means = [connectivity_results[n].mean() if len(connectivity_results[n]) > 0
             else 0 for n in ns]
    ax5.plot(ns, means, 's-', color='crimson', ms=7)
    ax5.set_xlabel('Number of trays (connectivity)')
    ax5.set_ylabel('Mean avalanche size')
    ax5.set_title('Figure 5 – Connectivity vs Mean Avalanche Size')
    ax5.grid(True, alpha=0.3)

    plt.savefig('/mnt/user-data/outputs/soc_distillation_results.png',
                dpi=150, bbox_inches='tight')
    plt.savefig('/home/claude/soc_distillation_results.png',
                dpi=150, bbox_inches='tight')
    print("Figures saved.")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" SOC Simulation: Distillation Column")
    print("=" * 60)

    print(f"\n[1/3] Running main simulation (N={N_TRAYS} trays, "
          f"{N_STEPS:,} steps)...")
    avalanche_sizes, holdup_history, noise_signal = run_simulation(
        N_TRAYS, Z_CRITICAL, Z_DELTA, N_STEPS, TOPPLING_FRAC)

    print(f"      Total avalanche events : {len(avalanche_sizes):,}")
    print(f"      Mean avalanche size    : {avalanche_sizes.mean():.3f}")
    print(f"      Max  avalanche size    : {avalanche_sizes.max()}")

    slope, _, r2 = power_law_fit(avalanche_sizes)
    if slope is not None:
        print(f"      Power-law exponent τ   : {-slope:.3f}  (R²={r2:.4f})")

    print("\n[2/3] Connectivity sweep ...")
    tray_counts = [5, 10, 15, 20, 25, 30, 40, 50]
    connectivity_results = connectivity_sweep(
        tray_counts, Z_CRITICAL, Z_DELTA,
        n_steps=50_000,        # shorter for sweep
        toppling_frac=TOPPLING_FRAC)

    print("\n[3/3] Generating figures ...")
    plot_all(avalanche_sizes, holdup_history, noise_signal,
             connectivity_results, N_TRAYS, N_STEPS)

    # ── Save numeric results for the LaTeX report ─────────────────────────
    stats = {
        "n_trays"         : N_TRAYS,
        "n_steps"         : N_STEPS,
        "z_critical"      : Z_CRITICAL,
        "total_avalanches": int(len(avalanche_sizes)),
        "mean_av_size"    : float(avalanche_sizes.mean()),
        "max_av_size"     : int(avalanche_sizes.max()),
        "power_law_tau"   : float(-slope) if slope else None,
        "power_law_r2"    : float(r2)     if r2   else None,
        "connectivity_ns" : tray_counts,
        "connectivity_means": [float(connectivity_results[n].mean())
                               if len(connectivity_results[n]) > 0 else 0.0
                               for n in tray_counts],
    }
    with open('/home/claude/sim_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("\nSimulation complete. Stats written to sim_stats.json")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    main()
