"""
Diagnostic plot: ApRES dR/dt profiles for all 10 South Pole sites.
Left panel: raw v_smooth (cm/yr) vs range (m), 0-200 m.
Right panel: detrended v_smooth vs range, 0-200 m.
Detrending: linear fit in 200-800 m range, subtracted from all bins.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_csv(
    "/home/andrew/projects/firngrain/test/southpole/processed/"
    "apres_vertical_velocity_processed.csv"
)

sites = sorted(df["site"].unique())
cmap = plt.cm.tab10

fig, (ax_raw, ax_det) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

for i, site in enumerate(sites):
    sub = df[(df["site"] == site) & (df["coherence"] > 0.5)].copy()
    sub = sub.sort_values("range_m")

    r = sub["range_m"].values
    v = sub["v_smooth_m_yr"].values  # m/yr

    # --- Linear detrend using deep-ice fit (200-800 m range) ---
    mask_deep = (r >= 200) & (r <= 800)
    if mask_deep.sum() >= 2:
        coeffs = np.polyfit(r[mask_deep], v[mask_deep], 1)
        trend = np.polyval(coeffs, r)
    else:
        trend = np.zeros_like(v)

    v_detrended = v - trend

    # Convert to cm/yr for display
    v_cm = v * 100.0
    v_det_cm = v_detrended * 100.0

    color = cmap(i)

    # Only plot 0-200 m range
    mask_plot = r <= 200
    ax_raw.plot(v_cm[mask_plot], r[mask_plot], color=color, lw=1.0, label=site)
    ax_det.plot(v_det_cm[mask_plot], r[mask_plot], color=color, lw=1.0, label=site)

# --- Format axes ---
ax_raw.set_ylabel("Range (m)")
ax_raw.set_xlabel("dR/dt  (cm yr$^{-1}$)")
ax_raw.set_title("Raw ApRES vertical velocity")
ax_raw.invert_yaxis()
ax_raw.legend(fontsize=7, loc="lower left")
ax_raw.grid(True, alpha=0.3)

ax_det.set_xlabel("dR/dt detrended  (cm yr$^{-1}$)")
ax_det.set_title("Detrended (linear fit 200\u2013800 m removed)")
ax_det.invert_yaxis()
ax_det.legend(fontsize=7, loc="lower left")
ax_det.grid(True, alpha=0.3)

fig.suptitle("South Pole ApRES dR/dt — all 10 sites", fontsize=13, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])

outpath = (
    "/home/andrew/projects/firngrain/test/southpole/results/"
    "apres_all_sites_comparison.png"
)
fig.savefig(outpath, dpi=150)
print(f"Saved to {outpath}")
plt.close(fig)
