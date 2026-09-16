# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase 3: where does the model look?
#
# Phase 2 asked how far the score can be trusted. This notebook asks whether the model gets its answers from
# the right places. There are two scales:
#
# - **Across the study:** which of the 62 three-slice windows each finding draws on, which series those come
#   from, and whether "medial" findings look at the medial side of the knee.
# - **Within a slice:** which pixels of the most important window drive the score.
#
# The executed copy of this notebook shows MRI slices from the competition data, so only this `.py` source is
# committed. Run `jupytext --to ipynb` and execute it locally to see the images.

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
import explain as ex
import plot_style as ps
import raptor_core as rc

ps.apply()
RUN = ROOT / "outputs/kaggle/gold_run"
FIG = ROOT / "outputs/figures"; FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.precision", 2)

gold = pd.read_csv(RUN / "gold_labels.csv")
Y = gold[rc.LAB].values.astype(int)
masks = np.load(RUN / "gold_masks_A.npy")
slots = json.load(open(RUN / "gold_slots.json"))["A"]
geometry = pd.read_csv(ROOT / "outputs/kaggle/gold_headers/gold_series_geometry.csv")
model, _ = rc.load_checkpoint(str(ROOT / "models/raptor_ft_coatnet_v10_full.pt"), "cpu")
logits, attn, wlogits = ex.head_outputs(model, np.load(RUN / "feats_A_imagenet.npy"))
N, K, n_find = attn.shape
own_slot, straddle = ex.window_slots()
SLOT_NAMES = [s[0] for s in ex.SLOTS]

# %% [markdown]
# ## 1. An exact decomposition
#
# The backbone scores each window on its own, and the head combines them. For finding *n*, each window *k*
# gets an attention weight $a_{kn}$ (a softmax over windows, so the weights sum to 1) and its own logit
# $w_{kn}$, and
#
# $$\text{logit}_n = \sum_k a_{kn}\, w_{kn}.$$
#
# So a study's score splits exactly into per-window contributions $a_{kn} w_{kn}$. This is unusual: most
# explanations of deep networks are approximations. The limit is what it does *not* tell us. Removing a window
# would not simply subtract its contribution, because the softmax would hand its attention to the others.
# That causal question is left to Phase 4's ablations.

# %%
err = np.abs(logits - (attn * wlogits).sum(1)).max()
print(f"max |logit - sum of window contributions| over 58 studies x 12 findings: {err:.1e}")

# %% [markdown]
# ## 2. Which series each finding draws on

# %%
share = np.stack([attn[:, own_slot == s, :].sum(1) for s in range(5)], axis=1)       # [N, 5, n]
uniform = np.array([(own_slot == s).mean() for s in range(5)])
share_tab = pd.DataFrame(share.mean(0).T * 100, index=rc.LAB, columns=SLOT_NAMES)
share_tab.loc["(even spread)"] = uniform * 100
share_tab.round(1)

# %%
ramp = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
rel = attn.mean(0).T * K                                  # [n, K]; 1 = the window's share under even spread
fig, ax = plt.subplots(figsize=(10, 4.6))
im = ax.imshow(rel, aspect="auto", cmap=ramp, vmin=0, vmax=4, interpolation="nearest")
ax.set_yticks(range(n_find), rc.LAB)
ax.set_xticks(np.arange(0, K, 5), ex.CENTRES[::5])
ax.set_xlabel("window (centre slice)")
ax.grid(False)
for s, (name, a, b) in enumerate(ex.SLOTS):
    idx = np.where(own_slot == s)[0]
    if s:
        ax.axvline(idx[0] - 0.5, color=ps.SURFACE, lw=2)
    ax.text(idx.mean(), -0.9, name, ha="center", va="bottom", fontsize=9, color=ps.INK_2)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("attention vs even spread (1 = even)", color=ps.INK_2)
cb.outline.set_visible(False)
ax.set_title("Mean attention per window, 58 studies", pad=22)
fig.savefig(FIG / "phase3_attention_heatmap.png")
plt.show()

# %% [markdown]
# **Reading it.**
#
# - **The fluid-sensitive sagittal series dominates for 11 of 12 findings**, taking 37-61% of attention
#   against 27% under an even spread. The lateral meniscus leans on it most (61%). The exception is MCL, which
#   takes slightly more from the fluid-sensitive coronal series (33% vs 31%).
# - **The model uses sequences the way a radiologist would.** Effusion, synovitis, Baker's cyst and contusion
#   are all about fluid or oedema, and give the non-fluid sagittal series only 7-9%. ACL and the three
#   osteoarthritis findings are about anatomy and cartilage, and give it 23-28%.
# - **Findings go to their best planes.** MCL and contusion take a third of their attention from the
#   fluid-sensitive coronal series (the MCL runs vertically on the medial side, best seen coronally). Baker's
#   cyst takes the most axial attention of any finding (23%), which is where a cyst behind the knee shows best.
# - **The second coronal series is nearly ignored** (1.5-6% against 13%). Phase 4 can test whether dropping it
#   costs anything.
# - **Medial meniscus and Baker's cyst each show two peaks in the fluid-sensitive sagittal block.** That is not
#   two lesions. Left and right knees run in opposite directions along the stack, and this average mixes them.
#   Section 5 lines every knee up.

# %% [markdown]
# ## 3. Does it focus when something is there?
#
# The effective number of windows, $1 / \sum_k a_{kn}^2$, is 62 when attention is spread evenly and 1 when all
# of it sits on a single window.

# %%
neff = ex.effective_windows(attn)                                 # [N, n]
focus = pd.DataFrame({"positives": [neff[Y[:, j] == 1, j].mean() for j in range(n_find)],
                      "negatives": [neff[Y[:, j] == 0, j].mean() for j in range(n_find)]}, index=rc.LAB)
focus = focus.sort_values("positives")

fig, ax = plt.subplots(figsize=(7, 4.2))
yy = np.arange(len(focus))
ax.hlines(yy, focus.positives, focus.negatives, color=ps.AXIS, lw=2, zorder=1)
ax.scatter(focus.positives, yy, s=64, color=ps.BLUE, edgecolor=ps.SURFACE, linewidth=2, zorder=3,
           label="studies with the finding")
ax.scatter(focus.negatives, yy, s=64, color=ps.ORANGE, edgecolor=ps.SURFACE, linewidth=2, zorder=2,
           label="studies without it")
ax.set_yticks(yy, focus.index)
ax.set_xlim(0, 40)
ax.set_xlabel("effective number of windows attended (of 62)")
ax.grid(axis="y", visible=False)
ax.set_title("Attention narrows when the finding is present")
ax.legend(loc="lower right", fontsize=9)
fig.savefig(FIG / "phase3_focus.png")
plt.show()

# %% [markdown]
# For every finding the model spreads its attention over fewer windows in studies that have the finding. The
# gap is large for most findings and negligible for two: medial meniscus (15 either way) and PF osteoarthritis
# (27 against 29). The narrowest focus is on lesions that sit in one place: lateral meniscus (11 windows),
# Baker's cyst (12), ACL and contusion (14). Diffuse findings stay broad even when present: PF osteoarthritis
# (27), fracture (23), MCL (22). A model that simply averaged its windows could not behave like this.

# %% [markdown]
# ## 4. Blank windows
#
# Fourteen studies have no second coronal series, so slices 44-51 are black. The windows centred on 45-50 see
# nothing but black. How much attention do they still get?

# %%
blank = masks[:, 44:52].sum(1) == 0
inner = np.isin(ex.CENTRES, np.arange(45, 51))
blank_tab = pd.DataFrame({
    "blank studies (%)": attn[blank][:, inner].sum(1).mean(0) * 100,
    "same windows, series present (%)": attn[~blank][:, inner].sum(1).mean(0) * 100,
}, index=rc.LAB)
print(f"{blank.sum()} studies with the slot blank")
blank_tab.round(1)

# %% [markdown]
# Blank windows get less attention than real ones for 11 of 12 findings (PF OA is the exception), but not
# zero: 0.5-4% per finding across the six windows. Their features are identical in every such study, so this adds a small fixed offset to those
# studies' logits. It is harmless for ranking within a finding only if the missing slot is not itself correlated
# with the label, which is a question for Phase 4.

# %% [markdown]
# ## 5. Medial versus lateral
#
# The strongest test of anatomical sense. The sagittal stack runs from one side of the knee to the other, so
# medial-compartment findings should draw on one end and lateral-compartment findings on the other.
#
# ### 5a. Without knowing which knee it is
#
# Which end is medial depends on whether it is a left or right knee and on the scanner's slice direction.
# There is a test that needs neither: within each study, compute where along the fluid-sensitive sagittal slot
# each finding's attention is centred (0 = first slice, 1 = last), then check whether medial and lateral findings
# land on opposite ends *of the same study*.

# %%
sag = np.where((own_slot == 0) & ~straddle)[0]                 # 16 windows fully inside the slot
pos = np.linspace(0, 1, len(sag))


def centre_of_mass(j):
    a = attn[:, sag, j]
    return (a / a.sum(1, keepdims=True) * pos).sum(1)


com = pd.DataFrame({f: centre_of_mass(rc.LAB.index(f)) for f in
                    ["Medial Meniscus", "Medial OA", "MCL", "Baker's", "Lateral Meniscus", "Lateral OA", "ACL"]})
opposite = ((com["Medial Meniscus"] - 0.5) * (com["Lateral Meniscus"] - 0.5) < 0).mean()
print(f"medial and lateral meniscus attention on opposite halves of the slot in {opposite:.0%} of studies")
com.corr().round(2)

# %% [markdown]
# Across studies the medial and lateral menisci correlate at -0.95: when one moves to the start of the stack, the
# other moves to the end, and they are on opposite halves in every single study. The findings split cleanly into
# two groups: medial meniscus, medial OA, MCL and **Baker's cyst** (+0.98 with the medial meniscus, and a Baker's
# cyst does form on the posteromedial side, between the medial gastrocnemius and semimembranosus) against
# lateral meniscus and lateral OA. ACL leans lateral (+0.87), consistent with its origin on the inner wall of the
# lateral femoral condyle.

# %% [markdown]
# ### 5b. With the knee identified
#
# To draw a real medial-to-lateral axis we need the knee's side and the slice direction, from a second small
# Kaggle job that read the DICOM headers.
#
# - **Slice direction.** Slices are sorted by position along the slice normal (from `ImageOrientationPatient`).
#   DICOM's patient x-axis points to the patient's left, so the sign of the normal's x-component says which way
#   the index runs.
# - **Knee side**, in order of trust: the DICOM `Laterality` tag (27 studies), then the series description (1),
#   then the knee's position in the scanner. A right knee sits on the patient's right, at negative x. Where the tag
#   exists, position agrees with it in 26 of 27 studies.

# %%
rows = []
for i, sid in enumerate(gold.StudyInstanceUID):
    ser = geometry[geometry.StudyInstanceUID == sid].set_index("SeriesInstanceUID")
    sag_series = ser.loc[slots[sid][0]["series"]]
    tag = ex.knee_side(*ser.Laterality.tolist())
    text = ex.knee_side(*ser.SeriesDescription.tolist())
    by_position = "R" if sag_series.ipp_mean_x < 0 else "L"
    side, source = (tag, "tag") if tag else ((text, "series text") if text else (by_position, "position"))
    rows.append({"side": side, "source": source, "position_says": by_position,
                 "medial_high": ex.medial_at_high_index(sag_series.normal_x, side)})
knees = pd.DataFrame(rows)
knees["model_medial_high"] = com["Medial Meniscus"] > com["Lateral Meniscus"]
knees["agree"] = knees.medial_high == knees.model_medial_high
print("knees:", knees.side.value_counts().to_dict())
print(knees.groupby("source").agree.agg(["sum", "count"]))
knees[~knees.agree]

# %% [markdown]
# **Headers and model agree in 57 of 58 studies.** The medial end worked out from DICOM geometry is where the
# model puts its medial-meniscus attention. That includes all 30 studies whose side came only from scanner
# position, so the model effectively confirms that rule. The single disagreement is a study tagged as a left knee
# whose scanner position says right, and the model sides with the position. A mislabelled tag is at least as
# likely as a model error.
#
# With every study oriented, the attention profiles can be averaged on a common medial-to-lateral axis:

# %%
profiles = {}
for f in ["Medial Meniscus", "ACL", "Lateral Meniscus"]:
    j = rc.LAB.index(f)
    a = attn[:, sag, j] / attn[:, sag, j].sum(1, keepdims=True)
    a = np.where(knees.medial_high.values[:, None], a[:, ::-1], a)     # index 0 = medial edge
    profiles[f] = a.mean(0)

fig, ax = plt.subplots(figsize=(7, 3.6))
for (f, prof), colour in zip(profiles.items(), [ps.BLUE, ps.ORANGE, ps.AQUA]):
    ax.plot(pos, prof * 100, color=colour, label=f)
    k = int(np.argmax(prof))
    ax.text(pos[k], prof[k] * 100 + 0.6, f, ha="center", fontsize=9, color=ps.INK_2)
ax.axhline(100 / len(sag), color=ps.MUTED, lw=1)
ax.text(1.0, 100 / len(sag) + 0.3, "even spread", ha="right", fontsize=8.5, color=ps.MUTED)
ax.set_xticks([0, 0.5, 1], ["medial edge", "centre", "lateral edge"])
ax.set_ylabel("% of the slot's attention")
ax.set_ylim(0, None)
ax.set_title("Fluid-sensitive sagittal slices, all 58 knees aligned")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=3, fontsize=9)
fig.savefig(FIG / "phase3_medial_lateral.png")
plt.show()

# %% [markdown]
# The medial meniscus peaks at the medial edge, the lateral meniscus at the lateral edge, and the ACL in the
# centre of the knee, which is where the cruciates cross in the intercondylar notch. Nobody told the model any of
# this: its only supervision was twelve per-study numbers read out of free-text reports in nine languages.

# %% [markdown]
# ## 6. Within a slice
#
# For four confident true positives, take the window that contributes most to the finding's score and ask which
# pixels matter. Three methods:
#
# - **Grad-CAM, last stage.** Channel activations weighted by their gradient. CoAtNet's last two stages are
#   transformer blocks, where every position can mix with every other, so this 12 x 12 map need not say *where*
#   in the image the evidence came from.
# - **Grad-CAM, last convolutional stage** (48 x 48). Positions still mean image locations here, but the
#   gradient has to pass back through the transformer stages.
# - **Occlusion.** Grey out a 48 px patch, re-score, record how much the window's logit drops, and repeat across
#   the slice (225 forward passes). It measures directly what matters, but cannot see anything larger than the
#   patch.

# %%
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_dev = model.to(device)
vols = np.load(RUN / "gold_vols_A.npy", mmap_mode="r")
probs = 1 / (1 + np.exp(-logits))
ORANGE_RGB = np.array([0xEB, 0x68, 0x34]) / 255


def overlay(ax, img, heat, title):
    ax.imshow(img, cmap="gray")
    heat = np.clip(heat / (np.percentile(heat, 99.5) + 1e-8), 0, 1)
    rgba = np.zeros(img.shape + (4,)); rgba[..., :3] = ORANGE_RGB; rgba[..., 3] = 0.65 * heat ** 1.5
    ax.imshow(rgba)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, loc="left")


cases = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Baker's"]
fig, axes = plt.subplots(3, 4, figsize=(12, 9.8))
case_rows = []
for col, f in enumerate(cases):
    j = rc.LAB.index(f)
    i = int(np.argmax(np.where(Y[:, j] == 1, probs[:, j], -1)))
    contrib = attn[i, :, j] * wlogits[i, :, j]
    k = int(np.argmax(contrib)); c = int(ex.CENTRES[k])
    window = rc.make_windows(np.asarray(vols[i]), [c], res=384)[0]
    cam_last, _ = ex.grad_cam(model_dev, window, j, device, stage=-1)
    cam_conv, _ = ex.grad_cam(model_dev, window, j, device, stage=1)
    occ, base = ex.occlusion(model_dev, window, j, device)
    img = vols[i][c]
    overlay(axes[0, col], img, cam_last, f"{f}  (p = {probs[i, j]:.2f})\nGrad-CAM, last stage")
    overlay(axes[1, col], img, cam_conv, "Grad-CAM, last conv stage")
    overlay(axes[2, col], img, np.clip(occ, 0, None), f"Occlusion: largest drop {occ.max():.2f} of {base:.2f}")
    case_rows.append({"finding": f, "p": probs[i, j], "window centre": c, "series": SLOT_NAMES[own_slot[k]],
                      "attention on it (%)": attn[i, k, j] * 100,
                      "share of logit (%)": contrib[k] / logits[i, j] * 100})
fig.savefig(FIG / "phase3_saliency.png")
plt.show()
pd.DataFrame(case_rows).set_index("finding")

# %% [markdown]
# **Reading it.**
#
# - **Occlusion lands on the right anatomy every time**: the intercondylar notch for the ACL, the meniscus at the
#   joint line for both menisci, and the neck of the Baker's cyst. The lateral meniscus is the most concentrated.
#   Greying one patch removes about two-thirds of that window's logit.
# - **One window rarely decides a study.** The top window carries 21% of the ACL study logit and only 9-11% for
#   the other three. The evidence is spread over neighbouring slices, which is what the attention pooling is for.
# - **For the Baker's cyst occlusion barely registers** (a drop of 0.3 out of 5). The cyst is bigger than the
#   patch, so hiding any one part leaves plenty visible. This is where last-stage Grad-CAM, coarse as it is,
#   gives the better picture: it covers the cyst itself.
# - **Last-conv-stage Grad-CAM is mostly noise**, with bright spots along image borders. A finer map is not
#   automatically a more faithful one.
#
# The general lesson: saliency methods disagree, and each fails in a predictable way. Before trusting a heat map,
# check it against a method that intervenes on the input (occlusion here) and against anatomy you already know.

# %% [markdown]
# ## 7. Summary
#
# | Question | Answer |
# |---|---|
# | Is the per-window explanation exact? | Yes: logit = sum of attention times window logit, to 2e-6 |
# | Which series matter? | Fluid-sensitive sagittal for 11 of 12 findings (37-61%); second coronal nearly ignored |
# | Does it pick sequences sensibly? | Fluid findings avoid non-fluid series (7-9%); ACL and OA use them (23-28%) |
# | Does it focus on lesions? | Yes: fewer windows attended when a finding is present, for all 12 findings |
# | Does it know medial from lateral? | Medial and lateral meniscus on opposite ends in 58/58 studies; matches DICOM-derived anatomy in 57/58 |
# | Where within the slice? | Occlusion finds the notch, the menisci and the cyst neck; Grad-CAM is unreliable on this architecture |
#
# **Caveat.** Attention says where the model *looks*, not what it *needs*. A window can hold a lot of attention
# and still be redundant with its neighbours. Phase 4 removes series, slices and windows to measure what the
# score actually depends on.
