"""
Full benchmark. Produces every number in README and the dashboard.

    python -m src.evaluate --quick                  # KDD, 1 seed, fewer budgets  (~5 min CPU)
    python -m src.evaluate                          # KDD, 3 seeds                (~20-30 min CPU)
    python -m src.evaluate --dataset unsw           # UNSW-NB15, 3 seeds

Outputs (KDD in results/, UNSW-NB15 in results/unsw/)
    results.json         per-seed records, mean/std summary, XAI, metadata
    robustness_*.png     detection rate vs eps
    summary_table.md     table for the README
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from src.attack import ThreatModel, attack_malicious, pgd
from src.dataset import SPECS, load_splits
from src.explain import integrated_gradients, top_features
from src.utils import HPARAMS, evaluate, metrics_from_scores, train_model

NETS = ["mlp", "cnn"]
DEFENSES = ["none", "adv_train", "adv_mix"]
ATTACKS = ["fgsm", "pgd"]
THREATS = ["unconstrained", "constrained"]


def run(data_path, seeds, eps_list, epochs, out: Path, dataset: str = "kdd"):
    records, xai, stats_all = [], {}, []
    t0 = time.time()
    for seed in seeds:
        s = load_splits(data_path, seed=seed, dataset=dataset)
        stats_all.append(s.stats)
        Xte, yte = torch.as_tensor(s.X_test), s.y_test
        threats = {"unconstrained": ThreatModel.unconstrained(Xte.shape[1]),
                   "constrained": ThreatModel.constrained(s.prep)}
        print(f"\n=== seed {seed}: {s.stats['rows_unique']:,} unique records, "
              f"{len(yte):,} test ({int(yte.sum()):,} attacks) ===")

        nets = {}
        for m in NETS:
            for d in DEFENSES:
                t = time.time()
                nets[(m, d)] = train_model(m, s.X_train, s.y_train, s.X_val, s.y_val, seed=seed,
                                           adversarial=(d != "none"), verbose=False, epochs=epochs,
                                           adv_mode="mix" if d == "adv_mix" else "replace")
                clean = evaluate(nets[(m, d)], Xte, yte)
                records.append(dict(seed=seed, model=m, defense=d, attack="clean", threat="none",
                                    eps=0.0, **clean))
                print(f"  {m}+{d:9s} clean F1 {clean['f1']:.4f}  ({time.time()-t:.0f}s)", flush=True)

        gbt = HistGradientBoostingClassifier(random_state=seed).fit(s.X_train, s.y_train)
        records.append(dict(seed=seed, model="gbt", defense="none", attack="clean", threat="none",
                            eps=0.0, **metrics_from_scores(yte, gbt.predict_proba(s.X_test)[:, 1])))

        yte_t = torch.as_tensor(yte)
        for (m, d), net in nets.items():
            for th_name, th in threats.items():
                for atk in ATTACKS:
                    for eps in eps_list:
                        X_adv = attack_malicious(net, Xte, yte_t, atk, eps, th, **({"seed": seed} if atk == "pgd" else {}))
                        met = evaluate(net, X_adv, yte)
                        records.append(dict(seed=seed, model=m, defense=d, attack=atk, threat=th_name,
                                            eps=eps, **met))
                        # transfer the undefended nets' PGD examples to the tree model
                        if d == "none" and atk == "pgd":
                            p = gbt.predict_proba(X_adv.numpy())[:, 1]
                            records.append(dict(seed=seed, model="gbt", defense="none",
                                                attack=f"pgd_from_{m}", threat=th_name, eps=eps,
                                                **metrics_from_scores(yte, p)))
            print(f"  attacked {m}+{d}")

        if seed == seeds[0]:
            xai = explain(s, nets, threats, yte_t)
            xai["class_separation"] = class_separation(s, seed)

    summary = summarize(records)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_sec": round(time.time() - t0, 1),
        "seeds": list(seeds),
        "dataset": dataset,
        "dataset_title": SPECS[dataset].title,
        "frozen_in_constrained": SPECS[dataset].immutable,
        "data": stats_all[0],
        "eps": eps_list,
        "hyperparameters": {**HPARAMS, "epochs": epochs},
        "setting": "evasion at test time; only malicious test records are perturbed; "
                   "eps in units of each feature's standard deviation (after log1p for counts)",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform(terse=True)},
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "xai": xai, "records": records}, f, indent=1)
    print(f"\n[*] {len(records)} records -> {out/'results.json'} ({meta['runtime_sec']}s)")
    return summary, xai


def class_separation(s, seed: int, n_attack: int = 2000, n_benign: int = 40000) -> dict:
    """L-inf distance from test attacks to the nearest benign training record.

    No detector can be robust at budget eps for an attack whose nearest benign
    record is closer than eps, so this bounds what adversarial training can
    achieve on a dataset.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    ben = s.X_train[s.y_train == 0]
    att = s.X_test[s.y_test == 1]
    ben = ben[rng.choice(len(ben), min(n_benign, len(ben)), replace=False)]
    att = att[rng.choice(len(att), min(n_attack, len(att)), replace=False)]
    d = NearestNeighbors(n_neighbors=1, metric="chebyshev").fit(ben).kneighbors(att)[0][:, 0]
    return {"n_attack": len(att), "n_benign": len(ben), "median": round(float(np.median(d)), 4),
            **{f"within_{e}": round(float((d <= e).mean()), 4) for e in (0.25, 0.5, 1.0)}}


def explain(s, nets, threats, yte_t):
    """IG attributions, which features PGD moves, and evasion per attack family."""
    Xte = torch.as_tensor(s.X_test)
    names = s.prep.feature_names
    mal = torch.where(yte_t == 1)[0][:2000]
    baseline = torch.as_tensor(s.X_train[s.y_train == 0].mean(0))
    out = {}
    for m in NETS:
        attr, err = integrated_gradients(nets[(m, "none")], Xte[mal], baseline)
        out[f"ig_{m}"] = {"top": top_features(attr, names), "completeness_error_median": round(err, 4)}

    net = nets[("mlp", "none")]
    x = Xte[mal]
    x_adv = pgd(net, x, yte_t[mal], 0.5, threats["constrained"])
    delta = (x_adv - x).abs().mean(0).numpy()
    order = np.argsort(-delta)[:10]
    out["pgd_moves_mlp_constrained_eps0.5"] = [
        {"feature": names[i], "mean_abs_delta": round(float(delta[i]), 4)} for i in order]

    fam = {}
    all_mal = torch.where(yte_t == 1)[0]
    for d in DEFENSES:
        X_adv = attack_malicious(nets[("mlp", d)], Xte, yte_t, "pgd", 0.5, threats["constrained"])
        with torch.no_grad():
            pred = nets[("mlp", d)](X_adv[all_mal]).argmax(-1).numpy()
            pred0 = nets[("mlp", d)](Xte[all_mal]).argmax(-1).numpy()
        types = s.attack_types_test[all_mal.numpy()]
        for t in np.unique(types):
            sel = types == t
            if sel.sum() < 20:
                continue
            fam.setdefault(t, {"n": int(sel.sum())})
            fam[t][f"{d}_clean"] = round(float(pred0[sel].mean()), 4)
            fam[t][f"{d}_pgd"] = round(float(pred[sel].mean()), 4)
    out["detection_by_family_mlp_constrained_pgd_eps0.5"] = fam
    return out


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["defense"], r["attack"], r["threat"], r["eps"])].append(r)
    out = []
    for (m, d, a, t, e), rs in sorted(groups.items()):
        row = {"model": m, "defense": d, "attack": a, "threat": t, "eps": e, "n": len(rs)}
        for k in ("accuracy", "f1", "precision", "recall", "roc_auc"):
            v = [r[k] for r in rs if not np.isnan(r[k])]
            if v:
                row[k], row[k + "_std"] = round(float(np.mean(v)), 4), round(float(np.std(v)), 4)
        out.append(row)
    return out


def plot(summary, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(m, d, a, t):
        c = [s for s in summary if s["model"] == m and s["defense"] == d and s["attack"] == "clean"]
        pts = sorted([s for s in summary if s["model"] == m and s["defense"] == d
                      and s["attack"] == a and s["threat"] == t], key=lambda s: s["eps"])
        pts = c + pts
        return [p["eps"] for p in pts], [p["recall"] for p in pts]

    for t in THREATS:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        label = {"none": "", "adv_train": "(adv. trained)", "adv_mix": "(adv. trained, mixed)"}
        for (m, d), col, ls in [(("mlp", "none"), "#3987e5", "-"), (("cnn", "none"), "#d95926", "-"),
                                (("mlp", "adv_train"), "#3987e5", "--"), (("cnn", "adv_train"), "#d95926", "--"),
                                (("mlp", "adv_mix"), "#3987e5", "-."), (("cnn", "adv_mix"), "#d95926", "-.")]:
            x, y = series(m, d, "pgd", t)
            ax.plot(x, y, ls, marker="o", color=col, lw=2, label=f"{m.upper()} {label[d]}")
        x, y = series("gbt", "none", "pgd_from_mlp", t)
        ax.plot(x, y, ":", marker="s", color="#199e70", lw=2, label="GBT (transfer from MLP)")
        ax.set_xlabel("eps (feature std)")
        ax.set_ylabel("Detection rate on attacks")
        ax.set_title(f"PGD, {t} threat model")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"robustness_{t}.png", dpi=150)
        plt.close(fig)


def table(summary, eps=0.5) -> str:
    def cell(m, d, a, t, e):
        r = next((s for s in summary if s["model"] == m and s["defense"] == d and s["attack"] == a
                  and s["threat"] == t and abs(s["eps"] - e) < 1e-9), None)
        return f"{r['recall']:.3f} ± {r['recall_std']:.3f}" if r else "—"
    cols = [("clean", "none", 0.0, "Clean"), ("fgsm", "unconstrained", eps, f"FGSM ε={eps}"),
            ("pgd", "unconstrained", eps, f"PGD ε={eps}"), ("pgd", "constrained", eps, f"PGD ε={eps}, constrained"),
            ("pgd", "constrained", 1.0, "PGD ε=1, constrained")]
    lines = ["| Model | Defence | " + " | ".join(c[3] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for m in NETS:
        for d in DEFENSES:
            lines.append(f"| {m.upper()} | {d} | " + " | ".join(cell(m, d, a, t, e) for a, t, e, _ in cols) + " |")
    gb = [cell("gbt", "none", "clean", "none", 0.0), "—",
          cell("gbt", "none", "pgd_from_mlp", "unconstrained", eps),
          cell("gbt", "none", "pgd_from_mlp", "constrained", eps),
          cell("gbt", "none", "pgd_from_mlp", "constrained", 1.0)]
    lines.append("| GBT (transfer) | none | " + " | ".join(gb) + " |")
    return "Detection rate (recall on attack records, test split), mean ± std over seeds\n\n" + "\n".join(lines) + "\n"


def main():
    # the summary table contains ε and ±, which a Windows console code page cannot print
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Adversarial robustness benchmark for NIDS")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dataset", default="kdd", choices=sorted(SPECS))
    p.add_argument("--data", default=None, help="data file (KDD) or folder (UNSW-NB15); defaults under data/")
    p.add_argument("--out", default=None, help="defaults to results/ for KDD, results/<dataset>/ otherwise")
    a = p.parse_args()
    if a.quick:
        seeds, eps, epochs = [0], [0.1, 0.25, 0.5, 1.0], 6
    else:
        seeds, eps, epochs = [0, 1, 2], [0.1, 0.25, 0.5, 1.0, 2.0], HPARAMS["epochs"]
    out = Path(a.out or ("results" if a.dataset == "kdd" else f"results/{a.dataset}"))
    summary, _ = run(a.data, seeds, eps, epochs, out, dataset=a.dataset)
    plot(summary, out)
    t = table(summary)
    (out / "summary_table.md").write_text(t, encoding="utf-8")
    print("\n" + t)


if __name__ == "__main__":
    main()
