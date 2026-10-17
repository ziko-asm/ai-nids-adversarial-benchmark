"""
Streamlit dashboard. It only reads results/results.json produced by
`python -m src.evaluate`; no number on screen is typed in by hand.

    streamlit run src/app.py
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results" / "results.json"
NAME = {"mlp": "MLP", "cnn": "1D-CNN", "gbt": "Gradient-boosted trees"}
DEF = {"none": "No defence", "adv_train": "PGD adversarial training"}
METRIC = {"recall": "Detection rate (recall)", "f1": "F1", "accuracy": "Accuracy", "roc_auc": "ROC-AUC"}

st.set_page_config(page_title="NIDS Adversarial Robustness", page_icon="🛡️", layout="wide")


@st.cache_data
def load():
    with open(RESULTS, encoding="utf-8") as f:
        return json.load(f)


if not RESULTS.exists():
    st.title("NIDS adversarial robustness benchmark")
    st.info("No results yet. Run `python -m src.evaluate --quick` from the repository root, then reload.")
    st.stop()

R = load()
S = pd.DataFrame(R["summary"])
meta = R["meta"]


def row(model, defense, attack, threat, eps):
    q = S[(S.model == model) & (S.defense == defense) & (S.attack == attack) & (S.threat == threat)
          & ((S.eps - eps).abs() < 1e-9)]
    return None if q.empty else q.iloc[0]


def curve(model, defense, attack, threat, metric):
    c = row(model, defense, "clean", "none", 0.0)
    pts = S[(S.model == model) & (S.defense == defense) & (S.attack == attack) & (S.threat == threat)]
    pts = pts.sort_values("eps")
    xs = [0.0] + pts.eps.tolist()
    ys = ([c[metric]] if c is not None else [None]) + pts[metric].tolist()
    return pd.Series(ys, index=xs)


# ---------------------------------------------------------------- header
st.title("How easily can an adversary slip past a neural intrusion detector?")
d = meta["data"]
st.caption(f"KDD Cup 1999 (10%): {d['rows_raw']:,} records, {d['rows_unique']:,} after removing duplicates, "
           f"{d['attack_share']:.0%} attacks · {len(meta['seeds'])} seed(s) · generated {meta['generated_at']}")

mlp_clean = row("mlp", "none", "clean", "none", 0.0)
mid = 0.5 if 0.5 in meta["eps"] else meta["eps"][len(meta["eps"]) // 2]
mlp_pgd = row("mlp", "none", "pgd", "constrained", mid)
adv_pgd = row("mlp", "adv_train", "pgd", "constrained", mid)
c = st.columns(4)
c[0].metric("MLP clean detection rate", f"{mlp_clean.recall:.3f}")
c[1].metric(f"Under constrained PGD, ε={mid}", f"{mlp_pgd.recall:.3f}", f"{mlp_pgd.recall - mlp_clean.recall:+.3f}")
c[2].metric("Same attack, adversarially trained MLP", f"{adv_pgd.recall:.3f}", f"{adv_pgd.recall - mlp_pgd.recall:+.3f} vs undefended")
gbt = row("gbt", "none", "pgd_from_mlp", "constrained", mid)
if gbt is not None:
    c[3].metric("Tree model, attack transferred from MLP", f"{gbt.recall:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["Robustness curves", "Explanations", "All results", "Report"])

with tab1:
    a, b, m = st.columns(3)
    attack = a.radio("Attack", ["pgd", "fgsm"], format_func=str.upper, horizontal=True)
    threat = b.radio("Threat model", ["constrained", "unconstrained"], horizontal=True,
                     help="Constrained: protocol, service, TCP flag and victim-side flags are frozen and "
                          "features stay within the training range.")
    metric = m.radio("Metric", list(METRIC), format_func=METRIC.get, horizontal=True)
    lines = {}
    for mod in ["mlp", "cnn"]:
        for de in ["none", "adv_train"]:
            lines[f"{NAME[mod]} · {DEF[de]}"] = curve(mod, de, attack, threat, metric)
    if attack == "pgd":
        lines["Trees · transfer from MLP"] = curve("gbt", "none", "pgd_from_mlp", threat, metric)
    df = pd.DataFrame(lines)
    df.index.name = "ε (feature std)"
    st.line_chart(df, height=380)
    st.caption("Only attack records are perturbed; benign test traffic is unchanged. "
               "ε = 0 is the clean test set. Mean over seeds.")
    st.dataframe(df.round(3), use_container_width=True)

with tab2:
    x = R.get("xai", {})
    l, r = st.columns(2)
    for col, key, title in [(l, "ig_mlp", "MLP"), (r, "ig_cnn", "1D-CNN")]:
        if key in x:
            col.subheader(f"Integrated Gradients, {title}")
            col.bar_chart(pd.DataFrame(x[key]["top"]).set_index("feature")["share"], height=320)
            col.caption(f"Share of total |attribution| on 2,000 attack records. "
                        f"Median completeness error: {x[key]['completeness_error_median']:.3f}")
    k = "pgd_moves_mlp_constrained_eps0.5"
    if k in x:
        st.subheader("Features PGD changes most (MLP, constrained, ε = 0.5)")
        st.bar_chart(pd.DataFrame(x[k]).set_index("feature")["mean_abs_delta"], height=280)
    k = "detection_by_family_mlp_constrained_pgd_eps0.5"
    if k in x:
        st.subheader("Detection rate by attack family (MLP, constrained PGD, ε = 0.5)")
        fam = pd.DataFrame(x[k]).T.sort_values("n", ascending=False)
        st.dataframe(fam, use_container_width=True)

with tab3:
    show = S.copy()
    show["model"] = show.model.map(NAME).fillna(show.model)
    show["defense"] = show.defense.map(DEF)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("Download results.json", RESULTS.read_bytes(), "results.json", "application/json")

with tab4:
    st.write("A PDF summary built from the same results file.")

    def pdf_bytes() -> bytes:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=48, rightMargin=48)
        ss = getSampleStyleSheet()
        story = [Paragraph("Adversarial robustness of neural intrusion detectors", ss["Title"]),
                 Paragraph(f"KDD Cup 1999 (10%), {d['rows_unique']:,} unique records, "
                           f"{len(meta['seeds'])} seed(s). Generated {meta['generated_at']}.", ss["Normal"]),
                 Spacer(1, 12)]
        hdr = ["Model", "Defence", "Clean"] + [f"PGD ε={e}" for e in meta["eps"]]
        data = [hdr]
        for mod in ["mlp", "cnn"]:
            for de in ["none", "adv_train"]:
                cl = row(mod, de, "clean", "none", 0.0)
                data.append([NAME[mod], DEF[de], f"{cl.recall:.3f}"] +
                            [f"{row(mod, de, 'pgd', 'constrained', e).recall:.3f}" for e in meta["eps"]])
        t = Table(data)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                               ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story += [Paragraph("Detection rate under constrained PGD", ss["Heading2"]), t]
        doc.build(story)
        return buf.getvalue()

    try:
        st.download_button("Download PDF report", pdf_bytes(), "nids_robustness_report.pdf", "application/pdf")
    except ImportError:
        st.warning("Install reportlab to export the PDF report.")
