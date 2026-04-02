"""
B1 SHAP Analysis — Generate publication-quality SHAP plots (Fig.5).

Panels:
  A — SHAP Beeswarm (Top 20 features)
  B — SHAP Bar (Mean |SHAP|)
  C — SHAP Dependence (Top 3 features, 1×3 grid)

Output: Individual panel PNGs + PDF + combined figure.
"""

import pickle
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap

# ── Paths ──────────────────────────────────────────────────────────
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output" / "RF_RDKit"

# ── Load data ──────────────────────────────────────────────────────
print("Loading SHAP data...")
with open(OUTPUT_DIR / "shap_explanation.pkl", "rb") as f:
    data = pickle.load(f)

shap_values = data["shap_values_cls1"]      # (1829, 168)
feature_names = data["feature_names"]        # list of 168
feature_values = data["feature_values"]      # (1829, 168)
labels = data["labels"]                       # (1829,)
base_value = data["base_value"]

print(f"SHAP values: {shap_values.shape}")
print(f"Features: {len(feature_names)}")

# If base_value is array (binary clf), take class 1
if hasattr(base_value, "__len__"):
    base_val = base_value[1]
else:
    base_val = base_value

# Create a proper SHAP Explanation object for native plotting
explanation = shap.Explanation(
    values=shap_values,
    base_values=np.full(shap_values.shape[0], base_val),
    data=feature_values,
    feature_names=feature_names,
)

# Load feature importance for reference
fi_df = pd.read_csv(OUTPUT_DIR / "shap_feature_importance.csv")
top20_features = fi_df.head(20)["feature"].tolist()
top3_features = fi_df.head(3)["feature"].tolist()

# ── Plot settings ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ══════════════════════════════════════════════════════════════════
# Panel A: SHAP Beeswarm (Top 20)
# Note: beeswarm doesn't support ax= with plot_size, so we use
# plot_size=None and set figure size before calling.
# ══════════════════════════════════════════════════════════════════
print("\nGenerating Panel A: Beeswarm plot...")
fig_a = plt.figure(figsize=(7, 6))
ax_a = fig_a.add_subplot(111)
shap.plots.beeswarm(explanation, max_display=20, show=False,
                     plot_size=None, ax=ax_a)
ax_a.set_xlabel("SHAP value (impact on model output)", fontsize=10)
ax_a.set_title("")
fig_a.tight_layout()
fig_a.savefig(OUTPUT_DIR / "panel_A_beeswarm.png", dpi=600)
fig_a.savefig(OUTPUT_DIR / "panel_A_beeswarm.pdf")
print(f"  Saved panel_A_beeswarm.png/pdf")
plt.close(fig_a)


# ══════════════════════════════════════════════════════════════════
# Panel B: SHAP Bar (Mean |SHAP|, Top 20)
# ══════════════════════════════════════════════════════════════════
print("Generating Panel B: Bar plot...")
fig_b = plt.figure(figsize=(6, 6))
ax_b = fig_b.add_subplot(111)
shap.plots.bar(explanation, max_display=20, show=False, ax=ax_b)
ax_b.set_xlabel("Mean |SHAP value|", fontsize=10)
ax_b.set_title("")
fig_b.tight_layout()
fig_b.savefig(OUTPUT_DIR / "panel_B_bar.png", dpi=600)
fig_b.savefig(OUTPUT_DIR / "panel_B_bar.pdf")
print(f"  Saved panel_B_bar.png/pdf")
plt.close(fig_b)


# ══════════════════════════════════════════════════════════════════
# Panel C: SHAP Dependence Plots (Top 3 features, 1×3 grid)
# ══════════════════════════════════════════════════════════════════
print("Generating Panel C: Dependence plots...")
fig_c, axes_c = plt.subplots(1, 3, figsize=(14, 4))

for i, feat in enumerate(top3_features):
    ax = axes_c[i]

    # Use SHAP scatter with auto-detected interaction feature
    shap.plots.scatter(
        explanation[:, feat],
        color=explanation,
        ax=ax,
        show=False,
    )
    ax.set_title(feat, fontsize=10, fontweight="bold")
    if i > 0:
        ax.set_ylabel("")

fig_c.tight_layout(w_pad=3)
fig_c.savefig(OUTPUT_DIR / "panel_C_dependence.png", dpi=600)
fig_c.savefig(OUTPUT_DIR / "panel_C_dependence.pdf")
print(f"  Saved panel_C_dependence.png/pdf")
plt.close(fig_c)


# ══════════════════════════════════════════════════════════════════
# Combined Figure 5
# Layout:  [  A  |  B  ]     (top row)
#           [ C1 | C2 | C3 ] (bottom row)
# ══════════════════════════════════════════════════════════════════
print("\nGenerating combined Figure 5...")
fig = plt.figure(figsize=(14, 13))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1], hspace=0.30, wspace=0.35)

# Panel A (top-left)
ax_a2 = fig.add_subplot(gs[0, 0])
shap.plots.beeswarm(explanation, max_display=20, show=False,
                     plot_size=None, ax=ax_a2)
ax_a2.set_xlabel("SHAP value (impact on model output)", fontsize=9)
ax_a2.set_title("")

# Panel B (top-right)
ax_b2 = fig.add_subplot(gs[0, 1])
shap.plots.bar(explanation, max_display=20, show=False, ax=ax_b2)
ax_b2.set_xlabel("Mean |SHAP value|", fontsize=9)
ax_b2.set_title("")

# Panel C (bottom, spanning full width — 3 subplots)
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :], wspace=0.35)
for i, feat in enumerate(top3_features):
    ax = fig.add_subplot(gs_bottom[i])
    shap.plots.scatter(
        explanation[:, feat],
        color=explanation,
        ax=ax,
        show=False,
    )
    ax.set_title(feat, fontsize=10, fontweight="bold")
    if i > 0:
        ax.set_ylabel("")

# Panel labels
label_kw = dict(fontsize=16, fontweight="bold", fontfamily="Arial",
                verticalalignment="top", horizontalalignment="left")
fig.text(0.01, 0.99, "A", **label_kw)
fig.text(0.51, 0.99, "B", **label_kw)
fig.text(0.01, 0.42, "C", **label_kw)

fig.savefig(OUTPUT_DIR / "Fig5_SHAP_combined.png", dpi=600)
fig.savefig(OUTPUT_DIR / "Fig5_SHAP_combined.pdf")
print(f"  Saved Fig5_SHAP_combined.png/pdf")
plt.close(fig)


print("\n✅ All plots generated successfully!")
print(f"   Output directory: {OUTPUT_DIR}")
