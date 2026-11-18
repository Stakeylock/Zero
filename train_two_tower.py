"""Two-tower taste model: members and titles -> 64-d vectors, scored by dot product.

Title tower : catalog description (TF-IDF) + genres/type/maturity/year/runtime -> MLP -> 64-d
Member tower: mean of the 64-d vectors of titles the member watched (shared title tower)
              + device/country/kids mix + small member-id embedding -> MLP -> 64-d
Loss        : softmax over the whole catalog (exact, only ~300 titles), temperature tau.
Metric      : Hit Rate@10 = share of events whose true title is in the member's top 10.

Protocol (no peeking at the hidden week):
  1. TUNE   : fit on train events before 2026-11-03, score on 2026-11-03..11-09
              -> pick hyper-parameters and number of epochs.
  2. FINAL  : refit on ALL train (up to 2026-11-09) with the chosen settings,
              score ONCE on holdout 2026-11-10..11-16, against a popularity baseline.

Usage: python train_two_tower.py            (reads WANDB_API_KEY from .env or env var)
"""
import itertools
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "model"
K = 10
DIM = 64
VAL_START = pd.Timestamp("2026-11-03")   # last train week used only for tuning
CUTOFF = pd.Timestamp("2026-11-10")
SEEDS_TUNE = [0, 1, 2]
SEEDS_FINAL = [0, 1, 2, 3, 4]
MAX_EPOCHS = 60
torch.set_num_threads(4)


# ------------------------------------------------------------------ W&B ----
def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    # the key may be saved as WANDB_API or the standard WANDB_API_KEY
    if "WANDB_API" in os.environ:
        os.environ.setdefault("WANDB_API_KEY", os.environ["WANDB_API"])


load_env()
import wandb  # noqa: E402

WANDB_MODE = "online" if os.environ.get("WANDB_API_KEY") else "offline"
print(f"[wandb] mode={WANDB_MODE}")


def wb_run(name, group, config):
    return wandb.init(project="netflix-two-tower", name=name, group=group, config=config,
                      mode=WANDB_MODE, dir=str(ROOT), reinit=True)


# ----------------------------------------------------------------- data ----
cat = pd.read_csv(ROOT / "Netflix title catalog, November 2026.csv", keep_default_na=False)
train_all = pd.read_csv(ROOT / "data/clean/train.csv", parse_dates=["event_date"], keep_default_na=False)
holdout = pd.read_csv(ROOT / "data/clean/holdout.csv", parse_dates=["event_date"], keep_default_na=False)
assert train_all.event_date.max() < CUTOFF <= holdout.event_date.min()

cat["release_date"] = pd.to_datetime(cat["release_date"])
tid = {t: i for i, t in enumerate(cat.title_id)}
mid = {m: i for i, m in enumerate(sorted(set(train_all.member_id) | set(holdout.member_id)))}
N_T, N_M = len(cat), len(mid)
release = cat["release_date"].values


def title_features():
    """Content only, so brand-new titles get a vector too. Fit on catalog text, no event data."""
    tfidf = TfidfVectorizer(stop_words="english", min_df=2, max_features=1500, sublinear_tf=True)
    X_text = tfidf.fit_transform(cat["description"]).toarray()
    genres = sorted(set(cat.primary_genre) | (set(cat.secondary_genre) - {""}))
    g = np.zeros((N_T, len(genres)))
    for i, (p, s) in enumerate(zip(cat.primary_genre, cat.secondary_genre)):
        g[i, genres.index(p)] = 1.0
        if s:
            g[i, genres.index(s)] = 0.5
    typ = pd.get_dummies(cat.title_type).values.astype(float)
    mat = pd.get_dummies(cat.maturity_rating).values.astype(float)
    year = cat.release_date.dt.year.values.astype(float)
    year = (year - year.mean()) / year.std()
    rt = pd.to_numeric(cat.runtime_minutes, errors="coerce")
    rt_missing = rt.isna().values.astype(float)
    rt = ((rt.fillna(rt.median()) - rt.median()) / rt.std()).values
    X = np.hstack([X_text, g, typ, mat, year[:, None], rt[:, None], rt_missing[:, None]])
    return torch.tensor(X, dtype=torch.float32)


X_TITLE = title_features()
print(f"[data] title features: {tuple(X_TITLE.shape)}  members={N_M}  titles={N_T}")

DEVICES = ["tv", "mobile", "tablet", "web", "unknown"]
COUNTRIES = sorted(set(train_all.country) | set(holdout.country))


def member_demo(events):
    """Per-member mix of device / country / kids-profile share, from the given events only."""
    D = np.zeros((N_M, len(DEVICES) + len(COUNTRIES) + 1))
    for m, dev, ctry, prof in zip(events.member_id, events.device_type, events.country, events.profile_type):
        i = mid[m]
        D[i, DEVICES.index(dev)] += 1
        D[i, len(DEVICES) + COUNTRIES.index(ctry)] += 1
        D[i, -1] += prof == "kids"
    n = D[:, : len(DEVICES)].sum(1, keepdims=True)
    n[n == 0] = 1
    D[:, : len(DEVICES) + len(COUNTRIES)] /= np.repeat(n, len(DEVICES) + len(COUNTRIES), 1)
    D[:, -1] /= n[:, 0]
    return torch.tensor(D, dtype=torch.float32)


def history_counts(events):
    C = np.zeros((N_M, N_T), dtype=np.float32)
    for m, t in zip(events.member_id, events.title_id):
        C[mid[m], tid[t]] += 1
    return torch.tensor(C)


class Split:
    """Everything one fit needs: history events, target events, and their tensors."""

    def __init__(self, hist_events, target_events):
        self.C = history_counts(hist_events)            # member x title watch counts (history)
        self.demo = member_demo(hist_events)
        self.m = torch.tensor([mid[x] for x in target_events.member_id])
        self.t = torch.tensor([tid[x] for x in target_events.title_id])
        d = target_events.event_date.values
        self.ok = torch.tensor(release[None, :] <= d[:, None])   # title already released that day
        self.events = target_events.reset_index(drop=True)


# ---------------------------------------------------------------- model ----
class TwoTower(nn.Module):
    def __init__(self, n_feat, n_demo, use_id, hidden=128, id_dim=16, drop=0.2):
        super().__init__()
        self.title = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, DIM))
        self.use_id = use_id
        self.id_emb = nn.Embedding(N_M, id_dim) if use_id else None
        n_in = DIM + n_demo + (id_dim if use_id else 0)
        self.member = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, DIM))

    def title_vecs(self):
        return F.normalize(self.title(X_TITLE), dim=-1)

    def member_vecs(self, m, C, demo, E, t_leave_out=None):
        """History = mean title vector of what the member watched.
        While training, the target view itself is left out so it can't be copied."""
        counts = C[m].clone()
        if t_leave_out is not None:
            counts[torch.arange(len(m)), t_leave_out] -= 1
            counts.clamp_(min=0)
        n = counts.sum(1, keepdim=True)
        hist = (counts @ E) / n.clamp(min=1)
        parts = [hist, demo[m]]
        if self.use_id:
            parts.append(self.id_emb(m))
        return F.normalize(self.member(torch.cat(parts, 1)), dim=-1)


def fit_epoch(model, opt, sp, tau, bs=256):
    model.train()
    perm = torch.randperm(len(sp.m))
    for i in range(0, len(perm), bs):
        b = perm[i : i + bs]
        E = model.title_vecs()
        U = model.member_vecs(sp.m[b], sp.C, sp.demo, E, t_leave_out=sp.t[b])
        logits = (U @ E.T) / tau
        logits = logits.masked_fill(~sp.ok[b], -1e9)
        loss = F.cross_entropy(logits, sp.t[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


# -------------------------------------------------------------- scoring ----
def ranks_of_true(scores, sp, mask_seen):
    s = scores.masked_fill(~sp.ok, -1e9)
    if mask_seen:
        seen = sp.C[sp.m] > 0
        true = F.one_hot(sp.t, N_T).bool()
        s = s.masked_fill(seen & ~true, -1e9)
    s_true = s.gather(1, sp.t[:, None])
    return (s > s_true).sum(1)      # ties go in the model's / baseline's favour (conservative for us)


@torch.no_grad()
def model_scores(model, sp):
    model.eval()
    E = model.title_vecs()
    return model.member_vecs(sp.m, sp.C, sp.demo, E) @ E.T


def hit_rate(ranks):
    return (ranks < K).float().mean().item()


def popularity_scores(train_events, sp):
    pop = torch.tensor(train_events.title_id.map(tid).value_counts().reindex(range(N_T), fill_value=0).values,
                       dtype=torch.float32)
    return pop[None, :].repeat(len(sp.m), 1)


def evaluate(scores, sp):
    """Primary: every event, all released titles allowed (re-watches are valid picks).
    Secondary: only events on a title the member had NOT watched before; seen titles excluded."""
    r_all = ranks_of_true(scores, sp, mask_seen=False)
    new = (sp.C[sp.m, sp.t] == 0)
    r_new = ranks_of_true(scores, sp, mask_seen=True)
    return {"hr10": hit_rate(r_all), "hr10_new_titles": hit_rate(r_new[new]) if new.any() else float("nan"),
            "n_events": len(r_all), "n_new_events": int(new.sum()), "hits": (r_all < K).numpy()}


def train_one(cfg, seed, sp_fit, sp_eval, epochs, log_every_epoch=True):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = TwoTower(X_TITLE.shape[1], sp_fit.demo.shape[1], cfg["use_id"], drop=cfg["drop"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    curve = []
    for ep in range(1, epochs + 1):
        loss = fit_epoch(model, opt, sp_fit, cfg["tau"])
        if sp_eval is not None:
            curve.append(evaluate(model_scores(model, sp_eval), sp_eval)["hr10"])
    return model, curve


# ------------------------------------------------------------------ run ----
def main():
    OUT.mkdir(exist_ok=True)
    # ---- 1. TUNE on the last train week (holdout untouched) -----------------
    fit_ev = train_all[train_all.event_date < VAL_START]
    val_ev = train_all[train_all.event_date >= VAL_START]
    sp_fit = Split(fit_ev, fit_ev)
    sp_val = Split(fit_ev, val_ev)
    pop_val = evaluate(popularity_scores(fit_ev, sp_val), sp_val)
    print(f"[tune] fit={len(fit_ev)} val={len(val_ev)}  popularity val HR@10={pop_val['hr10']:.3f}")

    grid = list(itertools.product([1e-3, 3e-3], [1e-4, 1e-2], [True, False], [0.1, 0.2]))
    results = []
    for lr, wd, use_id, tau in grid:
        cfg = dict(lr=lr, wd=wd, use_id=use_id, tau=tau, drop=0.2)
        curves = np.array([train_one(cfg, s, sp_fit, sp_val, MAX_EPOCHS)[1] for s in SEEDS_TUNE])
        mean = curves.mean(0)
        # smooth over 3 epochs so one noisy epoch can't pick the round count
        sm = np.convolve(mean, np.ones(3) / 3, mode="valid")
        best_ep = int(sm.argmax()) + 2          # centre epoch of the best window (1-indexed)
        results.append({**cfg, "best_epoch": best_ep, "val_hr10": float(sm.max()), "curve": mean.tolist()})
        run = wb_run(f"tune lr={lr} wd={wd} id={use_id} tau={tau}", "tuning", cfg)
        for ep, v in enumerate(mean, 1):
            wandb.log({"epoch": ep, "val_hr10": v, "val_popularity_hr10": pop_val["hr10"]})
        wandb.log({"best_epoch": best_ep, "best_val_hr10_smoothed": float(sm.max())})
        run.finish()
        print(f"  lr={lr:<6} wd={wd:<6} id={use_id!s:<5} tau={tau}  best_ep={best_ep:>2}  val HR@10={sm.max():.3f}")
    best = max(results, key=lambda r: r["val_hr10"])
    cfg = {k: best[k] for k in ["lr", "wd", "use_id", "tau", "drop"]}
    epochs = best["best_epoch"]
    print(f"[tune] chosen {cfg} epochs={epochs} (val HR@10={best['val_hr10']:.3f} vs popularity {pop_val['hr10']:.3f})")

    # ---- 2. FINAL fit on all train, score holdout once ----------------------
    sp_tr = Split(train_all, train_all)
    sp_ho = Split(train_all, holdout)
    pop = evaluate(popularity_scores(train_all, sp_ho), sp_ho)
    rng_rand = evaluate(torch.rand(len(sp_ho.m), N_T, generator=torch.Generator().manual_seed(0)), sp_ho)

    run = wb_run("final", "final", {**cfg, "epochs": epochs, "dim": DIM, "seeds": SEEDS_FINAL})
    finals, models = [], []
    for s in SEEDS_FINAL:
        model, _ = train_one(cfg, s, sp_tr, None, epochs)
        ev = evaluate(model_scores(model, sp_ho), sp_ho)
        finals.append(ev); models.append(model)
        wandb.log({"seed": s, "holdout_hr10": ev["hr10"], "holdout_hr10_new_titles": ev["hr10_new_titles"]})
        print(f"  seed {s}: holdout HR@10={ev['hr10']:.3f}  new-titles-only={ev['hr10_new_titles']:.3f}")

    hr = np.array([f["hr10"] for f in finals]); hrn = np.array([f["hr10_new_titles"] for f in finals])
    # paired bootstrap over members: is the model better than popularity, beyond luck?
    members = sp_ho.m.numpy()
    model_hits = np.mean([f["hits"] for f in finals], axis=0)     # avg over seeds, per event
    pop_hits = pop["hits"].astype(float)
    uniq = np.unique(members); idx_by_m = {m: np.where(members == m)[0] for m in uniq}
    rs = np.random.default_rng(0); diffs = []
    for _ in range(2000):
        pick = np.concatenate([idx_by_m[m] for m in rs.choice(uniq, len(uniq))])
        diffs.append(model_hits[pick].mean() - pop_hits[pick].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    summary = {
        "metric": "Hit Rate@10 (share of events whose true title is in the member's top 10)",
        "cutoff_date": str(CUTOFF.date()), "train_events": len(train_all), "holdout_events": len(holdout),
        "holdout_members": int(len(uniq)), "candidate_titles": "all catalog titles released by event date",
        "model_hr10_mean": float(hr.mean()), "model_hr10_std": float(hr.std()), "model_hr10_per_seed": hr.tolist(),
        "popularity_hr10": pop["hr10"], "random_hr10": rng_rand["hr10"],
        "lift_vs_popularity": float(hr.mean() - pop["hr10"]),
        "lift_95ci_member_bootstrap": [float(lo), float(hi)],
        "new_titles_only": {"n_events": pop["n_new_events"], "model_hr10_mean": float(hrn.mean()),
                            "popularity_hr10": pop["hr10_new_titles"], "random_hr10": rng_rand["hr10_new_titles"]},
        "chosen_config": {**cfg, "epochs": epochs, "dim": DIM},
        "tuning_val_hr10_popularity": pop_val["hr10"], "tuning_val_hr10_model": best["val_hr10"],
    }
    wandb.log({k: v for k, v in summary.items() if isinstance(v, float)})
    wandb.summary.update({"holdout_hr10_mean": summary["model_hr10_mean"], "popularity_hr10": pop["hr10"]})
    run.finish()

    # ---- 3. save: weights (seed 0), embeddings, metrics --------------------
    model = models[0]
    torch.save({"state_dict": model.state_dict(), "config": cfg, "epochs": epochs, "dim": DIM,
                "member_index": mid, "title_index": tid}, OUT / "two_tower.pt")
    with torch.no_grad():
        model.eval()
        E = model.title_vecs()
        U = model.member_vecs(torch.arange(N_M), sp_ho.C, sp_ho.demo, E)
    pd.DataFrame(E.numpy(), index=cat.title_id).to_csv(OUT / "title_embeddings_64d.csv", float_format="%.5f")
    pd.DataFrame(U.numpy(), index=list(mid)).to_csv(OUT / "member_embeddings_64d.csv", float_format="%.5f")
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    (OUT / "tuning_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
