"""Re-run the backbone locally at K=24 in several precisions to see whether the last few
mismatched pairs against the checkpoint's stored AUCs are numerical rather than preprocessing."""
import json, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, "src")
import raptor_core as rc

D = "outputs/kaggle/gold_run"
K = 24
gold = pd.read_csv(f"{D}/gold_labels.csv"); Y = gold[rc.LAB].values
vols = np.load(f"{D}/gold_vols_A.npy", mmap_mode="r"); masks = np.load(f"{D}/gold_masks_A.npy")
ck = json.load(open(f"{D}/summary.json"))["checkpoint"]["aucs"]
model, _ = rc.load_checkpoint("models/raptor_ft_coatnet_v10_full.pt", "mps")

def wrong_pairs(P):
    return np.array([(P[Y[:, j] == 1, j][:, None] < P[Y[:, j] == 0, j][None, :]).sum() for j in range(12)])

ckw = np.array([Y[:, j].sum() * (len(Y) - Y[:, j].sum()) * (1 - ck[l]) for j, l in enumerate(rc.LAB)])
for name, dtype in [("fp32", None), ("bf16", torch.bfloat16), ("fp16", torch.float16)]:
    t0 = time.time(); P = np.zeros((len(Y), 12), np.float32)
    for i in range(len(Y)):
        x = rc.make_windows(np.asarray(vols[i]), rc.eval_centers(masks[i], K), res=384)[None].to("mps")
        with torch.no_grad(), torch.autocast("mps", dtype=dtype or torch.float32, enabled=dtype is not None):
            P[i] = torch.sigmoid(model(x).float())[0].cpu().numpy()
    w = wrong_pairs(P); mauc, _ = rc.macro_auc(Y, P)
    np.save(f"outputs/local/preds_A_imagenet_K24_{name}.npy", P)
    print(f"{name}: macro-AUC {mauc:.6f} (ckpt {np.mean(list(ck.values())):.6f}) | L1 pairs to ckpt "
          f"{np.abs(w - ckw).sum():.0f} | per finding {dict(zip(rc.LAB, (w - ckw).round().astype(int).tolist()))} "
          f"| {time.time() - t0:.0f}s", flush=True)
