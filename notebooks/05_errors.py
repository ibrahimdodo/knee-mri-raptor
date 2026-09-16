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
# # Phase 5: where and why the model is wrong
#
# AUC counts wrongly ordered pairs: a positive study scored below a negative one. The reproduced model (62 windows)
# gets 726 of the 8,288 pairs wrong. This notebook asks where those errors live, which studies are hard, what the
# worst errors look like when read against the radiology report, and whether the probabilities mean what they say.
#
# No images are shown. The adjudication table paraphrases report content in a line per case.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kruskal, mannwhitneyu, spearmanr

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
import ablation as ab
import evaluation as ev
import explain as ex
import plot_style as ps
import raptor_core as rc
import reports as rp

ps.apply()
RUN = ROOT / "outputs/kaggle/gold_run"
FIG = ROOT / "outputs/figures"; FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.precision", 3)

gold = pd.read_csv(RUN / "gold_labels.csv")
Y = gold[rc.LAB].values.astype(int)
model, _ = rc.load_checkpoint(str(ROOT / "models/raptor_ft_coatnet_v10_full.pt"), "cpu")
logits, attn, wlogits = ex.head_outputs(model, np.load(RUN / "feats_A_imagenet.npy"))
probs = ab.sigmoid(logits)
N, F = Y.shape

# %% [markdown]
# ## 1. How concentrated are the errors?
#
# Every wrong pair involves one positive and one negative study. Charge each study-finding cell with the wrong
# pairs it takes part in: a positive scored below many negatives, or a negative scored above many positives.

# %%
wrong = np.zeros((N, F))
for j in range(F):
    pos = Y[:, j] == 1
    miss = 1 - ev.psi(probs[pos, j], probs[~pos, j])          # [positives, negatives]
    wrong[pos, j], wrong[~pos, j] = miss.sum(1), miss.sum(0)
total = wrong.sum() / 2
share = np.sort(wrong.ravel() / 2)[::-1].cumsum() / total
cells_for = {q: int(np.searchsorted(share, q)) + 1 for q in (0.25, 0.5, 0.75)}
print(f"{total:.0f} wrong pairs over {N * F} study-finding cells")
for q, n in cells_for.items():
    print(f"  {q:.0%} of them come from {n} cells ({n / (N * F):.1%})")

fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(1, N * F + 1) / (N * F) * 100
ax.plot(x, share * 100, color=ps.BLUE)
ax.plot([0, 100], [0, 100], color=ps.MUTED, lw=1)
ax.text(62, 55, "if every cell erred equally", color=ps.MUTED, fontsize=8.5, rotation=31)
for q, n in cells_for.items():
    ax.scatter(n / (N * F) * 100, q * 100, s=48, color=ps.BLUE, edgecolor=ps.SURFACE, linewidth=2, zorder=3)
    ax.text(n / (N * F) * 100 + 2, q * 100 - 5, f"{q:.0%} of errors in {n} cells", fontsize=9, color=ps.INK_2)
ax.set_xlabel("% of study-finding cells, worst first")
ax.set_ylabel("% of wrong pairs")
ax.set_xlim(0, 100); ax.set_ylim(0, 102)
ax.set_title("A few cells account for most of the errors")
fig.savefig(FIG / "phase5_concentration.png")
plt.show()

# %% [markdown]
# A quarter of all wrong pairs come from 17 of the 696 cells, and half from 51. AUC errors are not a thin spread
# of near-misses: they are dominated by a few confident mistakes, which makes reading those cases worthwhile.

# %% [markdown]
# ## 2. Which studies are hard?
#
# Give each study a difficulty score: the average, over its 12 findings, of the fraction of opposite-label studies it
# is misordered with. Then compare against what we know about each study: the report language (a stand-in for the
# hospital), scanner vendor and field strength, whether a series slot was empty, whether any series was magnified by
# the 140 mm crop, and how many findings the knee has.

# %%
import json

pairs_per_cell = np.where(Y == 1, (Y == 0).sum(0), (Y == 1).sum(0))
difficulty = (wrong / pairs_per_cell).mean(1)
masks = np.load(RUN / "gold_masks_A.npy")
slots = json.load(open(RUN / "gold_slots.json"))["A"]
meta = pd.read_csv(RUN / "gold_series_meta.csv").set_index("SeriesInstanceUID")
rows = []
for i, sid in enumerate(gold.StudyInstanceUID):
    used = [s for s in slots[sid] if s["series"]]
    fov = min(min(meta.loc[s["series"], ["Rows", "Columns"]]) * s["pixel_spacing"] for s in used)
    first = meta[meta.StudyInstanceUID == sid].iloc[0]
    rows.append({"difficulty": difficulty[i], "language": rp.LANG_NAMES[rp.detect_language(gold.Report[i])],
                 "vendor": str(first.Manufacturer).split()[0].upper(), "tesla": first.MagneticFieldStrength,
                 "empty slot": masks[i].sum() < 64, "magnified series": fov < 140, "positive findings": Y[i].sum()})
studies = pd.DataFrame(rows)
studies.groupby("language").difficulty.agg(["count", "mean"]).sort_values("count", ascending=False)

# %%
tests = {
    "language (groups of 3+)": kruskal(*[g.difficulty for _, g in studies.groupby("language") if len(g) >= 3]).pvalue,
    "vendor (groups of 3+)": kruskal(*[g.difficulty for _, g in studies.groupby("vendor") if len(g) >= 3]).pvalue,
    "3 T vs 1.5 T": mannwhitneyu(studies[studies.tesla == 3].difficulty, studies[studies.tesla == 1.5].difficulty).pvalue,
    "empty slot vs not": mannwhitneyu(*[studies[studies["empty slot"] == v].difficulty for v in (True, False)]).pvalue,
    "magnified series vs not": mannwhitneyu(*[studies[studies["magnified series"] == v].difficulty for v in (True, False)]).pvalue,
    "number of positive findings (Spearman)": spearmanr(studies["positive findings"], studies.difficulty).pvalue,
}
pd.Series(tests, name="p-value").round(3).to_frame()

# %% [markdown]
# Nothing about where or how a study was acquired predicts difficulty on these 58 studies. The one borderline signal
# is the number of positive findings (Spearman +0.26, p = 0.05): multi-injury knees are harder. With groups this small
# (two German studies, three Greek) only a large effect could show, so "no evidence" is the honest reading, not "no
# effect".
#
# The reports themselves are in eight languages here and nine across the full training set (English, Spanish,
# Turkish, Croatian, Greek, German, Bulgarian, Dutch, French), so the language model that produced the training labels
# had to read all of them.

# %% [markdown]
# ## 3. Findings the model cannot tell apart
#
# If the model scored each finding from its own evidence, two findings' scores would be about as correlated as
# their labels. Compare the two correlations for all 66 pairs of findings.

# %%
iu = np.triu_indices(F, 1)
label_r, score_r = np.corrcoef(Y.T)[iu], np.corrcoef(logits.T)[iu]
corr = pd.DataFrame({"finding a": np.array(rc.LAB)[iu[0]], "finding b": np.array(rc.LAB)[iu[1]],
                     "label correlation": label_r, "score correlation": score_r})
corr["excess"] = corr["score correlation"] - corr["label correlation"]

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.scatter(corr["label correlation"], corr["score correlation"], s=40, color=ps.BLUE, edgecolor=ps.SURFACE, linewidth=1.5)
ax.plot([-0.4, 1], [-0.4, 1], color=ps.MUTED, lw=1)
ax.text(0.62, 0.52, "scores as correlated\nas labels", fontsize=8.5, color=ps.MUTED)
for (_, r), dy in zip(corr.sort_values("excess", ascending=False).head(3).iterrows(), [-3, 6, -12]):
    ax.annotate(f"{r['finding a']} / {r['finding b']}", (r["label correlation"], r["score correlation"]),
                xytext=(8, dy), textcoords="offset points", fontsize=8.5, color=ps.INK_2)
ax.set_xlabel("correlation between the two labels")
ax.set_ylabel("correlation between the two scores")
ax.set_xlim(-0.4, 1); ax.set_ylim(-0.4, 1.02)
ax.set_title("Pairs of findings the model scores alike")
fig.savefig(FIG / "phase5_entangled.png")
plt.show()
corr.sort_values("excess", ascending=False).head(6)

# %% [markdown]
# Effusion and synovitis are the extreme: their scores correlate at 0.98 across studies while their labels correlate
# at 0.40. The three osteoarthritis compartments follow (score correlations about 0.8, label correlations about 0.3).
#
# Does each score still carry its own information? Rank each finding's studies with a *sibling's* score and compare.

# %%
ix = rc.LAB.index
oa = [ix("Medial OA"), ix("Lateral OA"), ix("PF OA")]
rows = [{"finding": "Synovitis", "own score": ev.auc(Y[:, ix("Synovitis")], logits[:, ix("Synovitis")]),
         "sibling score": ev.auc(Y[:, ix("Synovitis")], logits[:, ix("Effusion")]), "sibling": "Effusion"}]
for j in oa:
    k = max((k for k in oa if k != j), key=lambda k: ev.auc(Y[:, j], logits[:, k]))
    rows.append({"finding": rc.LAB[j], "own score": ev.auc(Y[:, j], logits[:, j]),
                 "sibling score": ev.auc(Y[:, j], logits[:, k]), "sibling": rc.LAB[k]})
siblings = pd.DataFrame(rows).set_index("finding")
siblings["own minus sibling"] = siblings["own score"] - siblings["sibling score"]
effusion_pos = Y[:, ix("Effusion")] == 1
print("synovitis AUC within the %d effusion-positive studies: own score %.3f, effusion score %.3f" % (
    effusion_pos.sum(), ev.auc(Y[effusion_pos, ix("Synovitis")], logits[effusion_pos, ix("Synovitis")]),
    ev.auc(Y[effusion_pos, ix("Synovitis")], logits[effusion_pos, ix("Effusion")])))
siblings

# %% [markdown]
# - **Synovitis is effusion under another name.** Ranking studies for synovitis by the *effusion* score gives 0.800,
#   against 0.805 for the synovitis score itself. Even among effusion-positive knees the synovitis score barely
#   separates synovitis (0.68 against 0.65). This is the model's weakest finding, and the reason.
# - **The OA compartments are tangled but not merged.** Each compartment's own score still beats the best other
#   compartment by 0.07-0.10, so compartment-specific evidence exists. It is diluted by a shared "degenerative knee"
#   component.
#
# There is a clinical reason the synovitis result is unsurprising. Without intravenous contrast, synovial thickening
# and joint fluid look alike on MRI, and scoring systems for non-contrast knee MRI grade them together as
# "effusion-synovitis". Training labels read from reports would inherit whatever each radiologist chose to call it.

# %% [markdown]
# ## 4. The worst errors, read against the reports
#
# The 14 cells with the most wrong pairs, each read in its original language (English, Spanish, Bulgarian, Greek,
# Dutch) and put in one of five categories. The labels are the competition's structured labels. They may come from a
# separate expert reading rather than from these reports, so "label and report disagree" says the two sources
# conflict, not which one is right.

# %%
ADJUDICATION = {
    (47, "PF OA"): ("focal or early cartilage damage labelled OA",
                    "grade 3-4 chondromalacia of the trochlea, in an acutely injured knee (ACL tear, bone contusion)"),
    (49, "PF OA"): ("label and report disagree",
                    "patella and patellar cartilage described as unremarkable; the cartilage damage is on the medial femoral condyle"),
    (28, "PF OA"): ("focal or early cartilage damage labelled OA",
                    "thinning of medial compartment and medial patellar facet cartilage, called incipient osteoarthritis, beside ACL and MCL tears"),
    (42, "Synovitis"): ("effusion scored as synovitis",
                        "large joint effusion in a severe multi-fracture injury; synovitis not mentioned"),
    (40, "Lateral Meniscus"): ("label and report disagree",
                               "menisci described as normal; ACL rupture, MCL sprain and impaction fracture"),
    (14, "Fracture"): ("subtle finding missed",
                       "undisplaced fracture of the tibial intercondylar eminence with marrow oedema"),
    (11, "Synovitis"): ("effusion scored as synovitis",
                        "ACL, MCL and both menisci torn, effusion, bone contusions; synovitis not mentioned"),
    (47, "Synovitis"): ("label and report disagree",
                        "synovial hypertrophy indicative of synovitis, repeated in the impression"),
    (50, "Lateral OA"): ("label and report disagree",
                         "cartilage described as normal; patellar dislocation pattern (medial patellofemoral ligament tear)"),
    (10, "Synovitis"): ("mild or borderline finding",
                        "Hoffa fat-pad impingement and iliotibial band friction; no effusion described"),
    (53, "Synovitis"): ("effusion scored as synovitis",
                        "ACL tear with mild to moderate effusion; synovitis not mentioned"),
    (17, "Synovitis"): ("mild or borderline finding",
                        "mild effusion with slight diffuse synovitis"),
    (0, "PF OA"): ("focal or early cartilage damage labelled OA",
                   "a single thin focal chondral ulcer on the trochlea; patellar cartilage normal"),
    (6, "Lateral OA"): ("focal or early cartilage damage labelled OA",
                        "6 x 5 mm full-thickness chondral ulcer on the lateral femoral condyle after acute trauma"),
}
worst = sorted(((wrong[i, j], i, rc.LAB[j]) for i in range(N) for j in range(F)), reverse=True)[:14]
assert {(i, f) for _, i, f in worst} == set(ADJUDICATION), "the worst cells changed; re-read the reports"
cases = pd.DataFrame([{"study": i, "finding": f, "label": Y[i, ix(f)], "p": probs[i, ix(f)], "wrong pairs": w,
                       "language": rp.LANG_NAMES[rp.detect_language(gold.Report[i])],
                       "category": ADJUDICATION[(i, f)][0], "report says": ADJUDICATION[(i, f)][1]}
                      for w, i, f in worst])
cases

# %%
summary = cases.groupby("category").agg(cells=("finding", "size"), wrong_pairs=("wrong pairs", "sum"))
summary["share of all wrong pairs"] = summary.wrong_pairs / total
summary.sort_values("wrong_pairs", ascending=False)

# %% [markdown]
# **Reading it.**
#
# - **Only one of the 14 is a clear model miss**: an undisplaced tibial eminence fracture, a genuinely subtle finding.
# - **In four, the label and the report disagree, and the model sides with the report every time.** Two positives
#   (lateral meniscus, lateral OA) whose reports call the structure normal, one PF OA positive whose report puts the
#   cartilage damage in the medial compartment instead, and one synovitis negative whose report states synovitis.
# - **Four are focal cartilage lesions labelled as osteoarthritis**, mostly in acutely injured knees: a single chondral
#   ulcer, grade 3-4 chondromalacia of the trochlea. The model has learned OA as a diffuse degenerative pattern, and a
#   6 mm ulcer in a young trauma knee does not look like that.
# - **Three are effusion scored as synovitis**, in trauma knees with effusion and no synovitis in the report. This is
#   Section 3's entanglement seen case by case.
# - **Two are mild or borderline synovitis** that the model scored in the middle.
#
# So most of the worst errors sit on boundaries the labels themselves draw inconsistently (focal damage against OA,
# fluid against synovitis) or where label and report conflict. Before blaming the model for its 0.80 on synovitis,
# note that "synovitis" is not one consistent thing in these labels.

# %% [markdown]
# ## 5. Do the probabilities mean anything?

# %%
def reliability(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return pd.DataFrame([{"mean predicted": p[idx == b].mean(), "observed rate": y[idx == b].mean(), "n": (idx == b).sum()}
                         for b in range(bins) if (idx == b).any()])


def ece(p, y, bins=10):
    r = reliability(p, y, bins)
    return float((r.n * (r["mean predicted"] - r["observed rate"]).abs()).sum() / r.n.sum())


pd.DataFrame({"mean predicted": probs.mean(0), "prevalence": Y.mean(0)}, index=rc.LAB).T

# %% [markdown]
# Every finding's average prediction is above its prevalence, badly so for the rarer ones (MCL 0.53 against 0.16).
#
# Two training choices distort the probabilities, each in its own way:
#
# - **Positive weighting shifts the logits up.** The loss is `BCEWithLogitsLoss(pos_weight=w)` with
#   $w = (1 - \text{prevalence}) / \text{prevalence}$, clipped to 1-10. Weighting the positive term by $w$ moves the
#   loss minimum from the true odds to $w$ times the odds, so every logit comes out too high by $\log w$, most for the
#   rarest findings.
# - **Soft labels squeeze the logits towards zero.** A report that hedges becomes a target of 0.8 rather than 1, so the
#   model is taught never to be fully sure and its logits come out compressed.
#
# The matching repairs are an offset per finding, and a slope stretching all logits. Test both, fitting on 57 studies
# and applying to the one left out so the result is honest on 58.

# %%
def nll_of(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def fit(idx, shared_slope):
    """Per-finding offsets, plus one slope shared by all findings if shared_slope, fitted on studies idx."""
    def loss(params):
        slope = params[0] if shared_slope else 1.0
        return nll_of(ab.sigmoid(slope * logits[idx] + params[-F:]), Y[idx])
    x0 = np.r_[1.0, np.zeros(F)] if shared_slope else np.zeros(F)
    x = minimize(loss, x0).x
    return (x[0] if shared_slope else 1.0), x[-F:]


versions = {"as trained": probs}
for name, shared in [("offset per finding", False), ("offset per finding + shared slope", True)]:
    out = np.zeros_like(probs)
    for i in range(N):
        slope, offsets = fit(np.arange(N) != i, shared)
        out[i] = ab.sigmoid(slope * logits[i] + offsets)
    versions[name] = out

calib = pd.DataFrame({name: {"Brier": ((P - Y) ** 2).mean(), "log loss": nll_of(P, Y), "ECE": ece(P.ravel(), Y.ravel()),
                             "macro-AUC": rc.macro_auc(Y, P)[0]} for name, P in versions.items()}).T
slope_all, offsets_all = fit(np.arange(N), True)
_, shifts = fit(np.arange(N), False)
print(f"shared slope fitted on all 58 studies: {slope_all:.2f}")
print(pd.DataFrame({"offset alone": shifts, "weight it implies, exp(-offset)": np.exp(-shifts)}, index=rc.LAB).T.round(2).to_string())
calib

# %%
fig, ax = plt.subplots(figsize=(5.6, 4.8))
ax.plot([0, 1], [0, 1], color=ps.MUTED, lw=1)
for (name, P), colour in zip(versions.items(), [ps.ORANGE, ps.AQUA, ps.BLUE]):
    r = reliability(P.ravel(), Y.ravel())
    ax.plot(r["mean predicted"], r["observed rate"], color=colour, marker="o", markersize=6, label=name)
ax.set_xlabel("predicted probability")
ax.set_ylabel("observed rate of the finding")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_title("Calibration, all 12 findings pooled (leave-one-out)")
ax.legend(loc="upper left", fontsize=9)
fig.savefig(FIG / "phase5_calibration.png")
plt.show()

# %% [markdown]
# **Reading it.**
#
# - **As trained, a mid-range probability means almost nothing.** Predictions between 0.2 and 0.5 turn out positive
#   2-5% of the time. Calibration error (ECE) is 0.20.
# - **An offset per finding fixes the level but overshoots.** ECE falls to 0.08, but the curve now sits above the
#   diagonal from about 0.45 up. The offsets behave as the weighting explanation predicts: the rarest findings need the
#   largest (MCL and lateral OA -2.2, implied weights about 9), and common ones need little (effusion and PF OA -0.3).
# - **Adding one shared slope fixes the shape.** The fitted slope is 1.77: the logits were compressed, as soft labels
#   predict. With 13 parameters in all, ECE falls to 0.026, Brier score from 0.164 to 0.107 and log loss from 0.50 to 0.35,
#   and the curve stays close to the diagonal.
# - **Macro-AUC reads 0.903 after recalibration rather than 0.917.** An offset and a positive slope cannot change a
#   ranking, but each left-out study gets slightly different parameters, and on 58 studies that jitter reorders a few
#   close pairs. Parameters fitted once on separate data would leave the AUC untouched.
#
# For the competition none of this matters: the metric is AUC, and the author's inference notebook converts scores to
# ranks anyway. For anyone reading the outputs as probabilities of disease, it matters a great deal.

# %% [markdown]
# ## 6. Summary
#
# | Question | Answer |
# |---|---|
# | Are errors spread or concentrated? | Concentrated: 25% of wrong pairs from 17 of 696 study-finding cells |
# | Which studies are hard? | Nothing about site, scanner, field strength or preprocessing predicts it; multi-injury knees slightly (p = 0.05) |
# | Which findings are confused? | Synovitis scores track effusion (score correlation 0.98, labels 0.40); OA compartments partly merged |
# | What are the worst 14 errors? | 1 clear miss, 4 label/report conflicts (model agrees with the report), 4 focal cartilage lesions labelled OA, 3 effusion scored as synovitis, 2 borderline |
# | Are probabilities calibrated? | No: shifted up by positive weighting and compressed by soft labels. An offset per finding plus one shared slope cuts calibration error from 0.20 to 0.03 |
#
# Across Phases 2-5 the model looks better than its weakest numbers suggest. Its synovitis and PF OA scores are low
# partly because those labels mix different things (fluid with synovium, focal injury with degeneration), and several of
# its worst "errors" conflict with the report rather than with the image. The limits it does have are specific and
# explainable: an effusion detector standing in for synovitis, an OA detector that expects diffuse degeneration, and
# probabilities that need recalibrating before anyone reads them as risks.
