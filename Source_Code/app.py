"""
app.py — ResumeAI Pro  |  Dual-Mode ATS Analyser
=================================================
Two distinct modes:
  • 🎓 Student Mode  — upload resume + JD, see ATS score, gaps, tips
  • 👔 Recruiter Mode — paste JD, upload multiple resumes, rank candidates

Tabs (Student): Input → ATS Dashboard → Skill Match → Recommendations → Contact Info → Dataset Insights
Tabs (Recruiter): Setup → Candidate Rankings → Detailed View
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from analyser import (
    ATSScorer, RecommendationEngine, SimilarityEngine,
    SkillExtractor, parse_upload, SYNONYMS,
    ContactValidator, ContextualSkillInferencer,
)

# ── PAGE CONFIG ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeAI Pro — ATS Analyser",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root { color-scheme: light !important; }
html, body { background:#f1f5f9 !important; color:#1e293b !important; font-family:Inter,sans-serif !important; }
.main .block-container { padding:0 0 4rem; max-width:1380px; }
[data-testid="stMarkdownContainer"] * { color:#1e293b; font-family:Inter,sans-serif; }

/* ── TOP NAV ─────────────────────────────────────────────── */
.topnav {
  background:#fff;
  border-bottom:2px solid #e2e8f0;
  padding:0 32px;
  height:60px;
  display:flex;
  align-items:center;
  gap:14px;
  position:sticky;
  top:0;
  z-index:100;
}
.nav-brand {
  font-size:1.15rem;
  font-weight:900;
  color:#0f172a;
  letter-spacing:-.5px;
  display:flex;
  align-items:center;
  gap:9px;
}
.brand-icon {
  width:32px; height:32px;
  background:linear-gradient(135deg,#6366f1,#8b5cf6);
  border-radius:8px;
  display:flex;
  align-items:center;
  justify-content:center;
  color:#fff;
  font-size:.9rem;
  font-weight:900;
}
.nav-badge {
  background:#ede9fe;
  color:#5b21b6;
  font-size:.56rem;
  font-weight:700;
  padding:3px 8px;
  border-radius:5px;
  letter-spacing:1.2px;
  text-transform:uppercase;
}
.nav-right {
  flex:1;
  display:flex;
  align-items:center;
  justify-content:flex-end;
  gap:8px;
}
.nav-mode-pill {
  font-size:.72rem;
  font-weight:600;
  padding:5px 14px;
  border-radius:99px;
  border:1.5px solid;
  cursor:pointer;
  letter-spacing:.2px;
}
.mode-student { background:#eff6ff; color:#1d4ed8; border-color:#93c5fd; }
.mode-recruiter { background:#fdf4ff; color:#7c3aed; border-color:#d8b4fe; }
.mode-active-s { background:#1d4ed8 !important; color:#fff !important; border-color:#1d4ed8 !important; }
.mode-active-r { background:#7c3aed !important; color:#fff !important; border-color:#7c3aed !important; }

/* ── MODE HERO ───────────────────────────────────────────── */
.hero {
  padding:28px 32px 0;
}
.hero-title {
  font-size:1.6rem;
  font-weight:900;
  color:#0f172a;
  letter-spacing:-.6px;
  line-height:1.2;
  margin-bottom:6px;
}
.hero-sub {
  font-size:.85rem;
  color:#64748b;
  line-height:1.6;
  max-width:700px;
}
.hero-accent { color:#6366f1; }

/* ── MODE SELECTOR CARDS ─────────────────────────────────── */
.mode-grid {
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:16px;
  max-width:700px;
  margin:28px auto;
}
.mode-card {
  background:#fff;
  border:2px solid #e2e8f0;
  border-radius:16px;
  padding:28px 24px;
  text-align:center;
  cursor:pointer;
  transition:all .18s;
}
.mode-card:hover { border-color:#a5b4fc; box-shadow:0 4px 20px rgba(99,102,241,.1); }
.mode-card.selected-s { border-color:#3b82f6; background:#eff6ff; box-shadow:0 4px 20px rgba(59,130,246,.15); }
.mode-card.selected-r { border-color:#8b5cf6; background:#fdf4ff; box-shadow:0 4px 20px rgba(139,92,246,.15); }
.mc-icon { font-size:2.8rem; margin-bottom:12px; }
.mc-title { font-size:1rem; font-weight:800; color:#0f172a; margin-bottom:6px; }
.mc-desc { font-size:.8rem; color:#64748b; line-height:1.5; }

/* ── LAYOUT ──────────────────────────────────────────────── */
.pg { padding:24px 32px 0; }
.sep { border:none; border-top:1px solid #e2e8f0; margin:20px 0; }
.row2 { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.row3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; }

/* ── TYPOGRAPHY ──────────────────────────────────────────── */
.h1 { font-size:1.05rem; font-weight:700; color:#0f172a; margin:0 0 4px; letter-spacing:-.2px; }
.h2 { font-size:.92rem; font-weight:600; color:#1e293b; margin:0 0 3px; }
.sub { font-size:.77rem; color:#64748b; margin-bottom:14px; line-height:1.6; }
.lbl { font-family:'JetBrains Mono',monospace; font-size:.58rem; font-weight:600; color:#94a3b8;
       letter-spacing:1.8px; text-transform:uppercase; margin-bottom:6px; display:block; }

/* ── CARDS ───────────────────────────────────────────────── */
.card { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:20px 22px;
        margin-bottom:14px; box-shadow:0 1px 4px rgba(0,0,0,.05); color:#1e293b; }
.card-sm { background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:12px 16px;
           margin-bottom:9px; color:#1e293b; }
.card-tint { background:#f8faff; border:1px solid #e0e7ff; border-radius:10px; padding:12px 16px;
             margin-bottom:9px; color:#1e293b; }

/* ── SCORE RING ──────────────────────────────────────────── */
.score-ring {
  background:#fff;
  border:1px solid #e0e7ff;
  border-radius:18px;
  padding:28px 20px 24px;
  text-align:center;
  box-shadow:0 4px 20px rgba(99,102,241,.08);
  color:#1e293b;
}
.score-num { font-size:4.8rem; font-weight:900; line-height:1; letter-spacing:-3px; }
.score-den { font-size:.76rem; color:#94a3b8; font-weight:500; margin-top:4px; }
.score-lbl { font-family:'JetBrains Mono',monospace; font-size:.58rem; font-weight:600;
             letter-spacing:2px; text-transform:uppercase; margin-top:5px; }
.verdict-badge {
  display:inline-flex;
  align-items:center;
  gap:5px;
  font-size:.82rem;
  font-weight:800;
  padding:8px 22px;
  border-radius:99px;
  border:2px solid;
  margin-top:14px;
}

/* ── COMPONENT SCORE CARDS ───────────────────────────────── */
.cc {
  background:#fff;
  border:1px solid #e2e8f0;
  border-left:4px solid;
  border-radius:12px;
  padding:15px 18px;
  margin-bottom:11px;
  color:#1e293b;
}
.cc-row { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:7px; }
.cc-name { font-weight:700; font-size:.88rem; color:#1e293b; }
.cc-wt { font-family:'JetBrains Mono',monospace; font-size:.58rem; color:#94a3b8; margin-left:4px; }
.cc-pts { font-size:1.25rem; font-weight:900; letter-spacing:-.5px; }
.cc-of { font-size:.68rem; color:#94a3b8; }
.cc-pct { font-size:.65rem; font-weight:600; margin-top:2px; }
.why-t { font-family:'JetBrains Mono',monospace; font-size:.56rem; font-weight:700;
         letter-spacing:2px; text-transform:uppercase; color:#94a3b8; margin-top:9px; }
.why-b { font-size:.77rem; color:#475569; line-height:1.6; margin-top:3px; }
.fix-s { background:#f5f3ff; border-left:3px solid #6366f1; border-radius:0 8px 8px 0;
         padding:8px 13px; margin-top:8px; font-size:.75rem; color:#4338ca; line-height:1.55; }

/* ── BARS ────────────────────────────────────────────────── */
.btrack { background:#f1f5f9; border-radius:99px; height:6px; overflow:hidden; margin:5px 0; }
.bfill  { height:6px; border-radius:99px; }

/* ── CHIPS ───────────────────────────────────────────────── */
.chips { display:flex; flex-wrap:wrap; gap:5px; margin:7px 0; }
.sk { font-family:'JetBrains Mono',monospace; font-size:.67rem; font-weight:500;
      padding:3px 10px; border-radius:6px; display:inline-block; white-space:nowrap; }
.sk-g { background:#dcfce7; color:#15803d; border:1px solid #86efac; }
.sk-r { background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5; }
.sk-y { background:#fef9c3; color:#854d0e; border:1px solid #fde68a; }
.sk-b { background:#dbeafe; color:#1d4ed8; border:1px solid #93c5fd; }
.sk-p { background:#ede9fe; color:#5b21b6; border:1px solid #c4b5fd; }
.sk-n { background:#f1f5f9; color:#475569; border:1px solid #cbd5e1; }

/* ── STAT PILLS ──────────────────────────────────────────── */
.stats-row { display:flex; gap:10px; flex-wrap:wrap; margin:10px 0; }
.sp { background:#fff; border:1px solid #e2e8f0; border-radius:10px;
      padding:10px 14px; text-align:center; flex:1; min-width:72px; color:#1e293b; }
.sp-v { font-size:1.35rem; font-weight:900; line-height:1; letter-spacing:-.5px; }
.sp-l { font-size:.61rem; color:#64748b; margin-top:4px; font-weight:500; }

/* ── RECRUITER VERDICT ───────────────────────────────────── */
.rvb { border-radius:14px; padding:22px 24px; border:2px solid; margin-bottom:18px; }
.rvb * { color:#fff !important; }
.rv-d { font-size:1.4rem; font-weight:900; letter-spacing:-.3px; color:#fff !important; }
.rv-c { font-family:'JetBrains Mono',monospace; font-size:.6rem; font-weight:700;
        letter-spacing:1.5px; text-transform:uppercase; margin-top:3px;
        color:rgba(255,255,255,.75) !important; }
.rv-r { font-size:.82rem; margin-top:6px; line-height:1.6; color:#fff !important; }
.rv-ss { display:flex; flex-wrap:wrap; gap:5px; margin-top:10px; }
.rv-s { font-size:.63rem; padding:3px 8px; border-radius:5px; font-weight:500;
        background:rgba(255,255,255,.2); border:1px solid rgba(255,255,255,.4);
        color:#fff !important; }
.rv-n { font-size:.74rem; margin-top:10px; padding:9px 13px; border-radius:8px;
        background:rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.3);
        line-height:1.5; color:#fff !important; }

/* ── PATTERN CARDS ───────────────────────────────────────── */
.pat { border-radius:10px; padding:12px 15px; margin:6px 0; display:flex; align-items:flex-start; gap:11px; }
.pat-ok   { background:#f0fdf4; border:1px solid #86efac; color:#166534; }
.pat-warn { background:#fefce8; border:1px solid #fde68a; color:#854d0e; }
.pat-info { background:#eff6ff; border:1px solid #93c5fd; color:#1e40af; }
.pat-ic { font-size:1.1rem; flex-shrink:0; margin-top:1px; }
.pat-tt { font-weight:700; font-size:.82rem; }
.pat-ds { font-size:.74rem; margin-top:3px; line-height:1.5; opacity:.9; }

/* ── RECOMMENDATION CARDS ────────────────────────────────── */
.rc { background:#fff; border:1px solid #e2e8f0; border-left:4px solid;
      border-radius:12px; padding:15px 18px; margin-bottom:10px; color:#1e293b; }
.rc-cr { border-left-color:#ef4444; }
.rc-hi { border-left-color:#f97316; }
.rc-me { border-left-color:#eab308; }
.rc-lo { border-left-color:#3b82f6; }
.rc-h { display:flex; align-items:center; gap:7px; margin-bottom:6px; flex-wrap:wrap; }
.rc-b { font-size:.58rem; font-weight:800; letter-spacing:1px; text-transform:uppercase;
        padding:2px 8px; border-radius:5px; }
.rb-cr { background:#fee2e2; color:#b91c1c; }
.rb-hi { background:#ffedd5; color:#c2410c; }
.rb-me { background:#fef9c3; color:#854d0e; }
.rb-lo { background:#dbeafe; color:#1d4ed8; }
.rc-tt { font-weight:700; font-size:.87rem; color:#0f172a; }
.rc-ct { font-size:.62rem; color:#94a3b8; }
.rc-bd { font-size:.77rem; color:#475569; line-height:1.6; margin-top:5px; }
.rc-im { font-size:.68rem; color:#6366f1; font-weight:600; margin-top:5px; }

/* ── CONTACT CARDS ───────────────────────────────────────── */
.coc { border-radius:11px; padding:16px; color:#1e293b; }
.coc-ok   { background:#f0fdf4; border:1px solid #86efac; }
.coc-miss { background:#fef2f2; border:1px solid #fecaca; }
.coc-soft { background:#f8fafc; border:1px solid #e2e8f0; }
.coc-ic { font-size:1.4rem; margin-bottom:7px; }
.coc-lbl { font-weight:700; font-size:.85rem; color:#1e293b; }
.coc-val { font-family:'JetBrains Mono',monospace; font-size:.68rem; color:#6366f1;
           margin-top:5px; word-break:break-all; }
.coc-hint { font-size:.73rem; color:#475569; margin-top:8px; line-height:1.5;
            border-top:1px solid rgba(0,0,0,.06); padding-top:7px; }

/* ── CALLOUTS ────────────────────────────────────────────── */
.callout { border-radius:9px; padding:11px 15px; font-size:.78rem;
           margin-bottom:12px; line-height:1.55; }
.ci { background:#eff6ff; border:1px solid #93c5fd; color:#1e40af; }
.cw { background:#fefce8; border:1px solid #fde68a; color:#854d0e; }
.cs { background:#f0fdf4; border:1px solid #86efac; color:#166534; }
.ce { background:#fef2f2; border:1px solid #fecaca; color:#b91c1c; }

/* ── RECRUITER RANKING CARD ──────────────────────────────── */
.rank-card {
  background:#fff;
  border:1px solid #e2e8f0;
  border-radius:14px;
  padding:18px 20px;
  margin-bottom:12px;
  display:flex;
  align-items:center;
  gap:18px;
  color:#1e293b;
  transition:box-shadow .18s;
}
.rank-card:hover { box-shadow:0 4px 18px rgba(0,0,0,.08); }
.rank-num { font-size:1.5rem; font-weight:900; color:#94a3b8; min-width:32px; text-align:center; }
.rank-score { font-size:2.2rem; font-weight:900; letter-spacing:-1px; min-width:72px; text-align:center; }
.rank-details { flex:1; }
.rank-name { font-size:.92rem; font-weight:700; color:#0f172a; margin-bottom:4px; }
.rank-meta { font-size:.74rem; color:#64748b; margin-bottom:6px; }
.rank-dec { font-size:.72rem; font-weight:700; padding:3px 10px; border-radius:5px; }
.rank-dec-y { background:#dcfce7; color:#15803d; }
.rank-dec-m { background:#fef9c3; color:#854d0e; }
.rank-dec-n { background:#fee2e2; color:#b91c1c; }

/* ── STREAMLIT OVERRIDES ─────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background:#f1f5f9; border-radius:11px; padding:4px;
  gap:2px; border:1px solid #e2e8f0; margin:0 32px 20px;
}
.stTabs [data-baseweb="tab"] {
  background:transparent; border-radius:8px; color:#64748b;
  font-family:Inter,sans-serif; font-size:.8rem; font-weight:500;
  padding:8px 18px; border:none;
}
.stTabs [aria-selected="true"] {
  background:#fff !important; color:#0f172a !important;
  font-weight:700 !important; box-shadow:0 1px 6px rgba(0,0,0,.09) !important;
}
.stButton>button {
  background:linear-gradient(135deg,#6366f1,#4f46e5) !important;
  color:#fff !important; font-family:Inter,sans-serif !important;
  font-weight:700 !important; font-size:.83rem !important;
  border:none !important; border-radius:9px !important;
  padding:.56rem 1.8rem !important;
  box-shadow:0 2px 10px rgba(99,102,241,.28) !important;
  transition:all .18s !important;
}
.stButton>button:hover {
  box-shadow:0 4px 18px rgba(99,102,241,.42) !important;
  transform:translateY(-1px) !important;
}
.stButton>button:disabled { opacity:.5 !important; transform:none !important; }
.stTextArea textarea {
  background:#fff !important; color:#1e293b !important;
  border:1px solid #e2e8f0 !important; border-radius:9px !important;
  font-family:'JetBrains Mono',monospace !important;
  font-size:.76rem !important; line-height:1.6 !important;
}
.stTextArea textarea:focus {
  border-color:#6366f1 !important;
  box-shadow:0 0 0 3px rgba(99,102,241,.14) !important;
}
[data-testid="stFileUploader"] {
  background:#fafbff !important;
  border:2px dashed #c7d2fe !important;
  border-radius:11px !important;
}
#MainMenu, footer, [data-testid="stToolbar"] { visibility:hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# ENGINE / CACHE
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _engine():
    return ATSScorer(SimilarityEngine())

scorer = _engine()


@st.cache_data(show_spinner=False)
def _load_dataset():
    try:
        import pandas as pd
        from analyser import ALL_SKILLS_FLAT
        df = pd.read_csv("Resume.csv", encoding="utf-8", engine="python",
                         on_bad_lines="skip", escapechar="\\")
        df = df[["Resume_str", "Category"]].rename(columns={"Resume_str": "Resume"})
        df = df.dropna().copy()
        FAST = list(ALL_SKILLS_FLAT)
        df["skills"] = df["Resume"].apply(
            lambda t: [s for s in FAST if isinstance(t, str) and (" " + s + " ") in " " + t.lower() + " "]
        )
        return df, Counter(s for sl in df["skills"] for s in sl)
    except Exception:
        return None, Counter()


_df, _sc = _load_dataset()

# ── Session state defaults ───────────────────────────────────────────
for _k, _v in [
    ("resume_text", ""), ("jd_text", ""), ("result", None),
    ("app_mode", None),   # "student" | "recruiter"
    ("rec_results", []),  # list of {name, text, result}
    ("rec_jd", ""),
    ("rec_selected", 0),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
def score_color(p):
    if p >= 75: return "#22c55e"
    if p >= 55: return "#eab308"
    if p >= 35: return "#f97316"
    return "#ef4444"


def prog_bar(p, color, h=6):
    return (f'<div class="btrack"><div class="bfill" '
            f'style="width:{min(p, 100):.0f}%;background:{color};height:{h}px;"></div></div>')


def chip(lbl, k="n"):
    return f'<span class="sk sk-{k}">{lbl}</span>'


def chips(items, k="n", lim=45):
    if not items:
        return '<span style="font-size:.74rem;color:#94a3b8;font-style:italic;">None detected</span>'
    return '<div class="chips">' + "".join(chip(i, k) for i in list(items)[:lim]) + "</div>"


def stat_pill(val, label, color="#1e293b"):
    return (f'<div class="sp"><div class="sp-v" style="color:{color};">{val}</div>'
            f'<div class="sp-l">{label}</div></div>')


def no_result_placeholder(msg="Run an analysis on the <b>Input</b> tab first."):
    st.markdown(
        f'<div style="text-align:center;padding:64px 32px;background:#fff;border-radius:14px;'
        f'border:1px solid #e2e8f0;margin-top:20px;color:#64748b;font-size:.88rem;">'
        f'📋<br><br>{msg}</div>',
        unsafe_allow_html=True,
    )


def component_card(name, comp):
    pct = comp["pct"]
    col = score_color(pct)
    dtail = ""
    if "detail" in comp:
        d = comp["detail"]
        bits = []
        if "years" in d:            bits.append(f'{d["years"]}yr exp')
        if "action_verbs" in d:     bits.append(f'{d["action_verbs"]} action verbs')
        if "has_quant" in d:        bits.append("quantified ✓" if d["has_quant"] else "no metrics ✗")
        if "has_projects" in d:     bits.append(f'{d.get("project_count", 0)} project(s)')
        if "github" in d:           bits.append("GitHub ✓" if d["github"] else "no GitHub ✗")
        if "domain_relevance" in d: bits.append(d["domain_relevance"])
        if bits:
            dtail = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:8px;">' + \
                    "".join(f'<span class="sk sk-n" style="font-size:.59rem;">{b}</span>' for b in bits) + \
                    "</div>"
    return (
        f'<div class="cc" style="border-left-color:{col};">'
        f'<div class="cc-row"><div>'
        f'<span style="font-size:1rem;">{comp["icon"]}</span> '
        f'<span class="cc-name">{name}</span>'
        f'<span class="cc-wt">({comp["weight_label"]})</span></div>'
        f'<div style="text-align:right;">'
        f'<div class="cc-pts" style="color:{col};">{comp["score"]:.1f}'
        f'<span class="cc-of">/{comp["max"]}</span></div>'
        f'<div class="cc-pct" style="color:{col};">{pct:.0f}%</div></div></div>'
        f'{prog_bar(pct, col, 6)}{dtail}'
        f'<div class="why-t">WHY THIS SCORE</div>'
        f'<div class="why-b">{comp["why"]}</div>'
        f'<div class="fix-s">💡 {comp["improve"]}</div></div>'
    )


def pattern_card(pat):
    cls = {"ok": "pat-ok", "warn": "pat-warn", "info": "pat-info"}.get(pat["severity"], "pat-info")
    return (f'<div class="pat {cls}"><div class="pat-ic">{pat["icon"]}</div>'
            f'<div><div class="pat-tt">{pat["label"]}</div>'
            f'<div class="pat-ds">{pat["description"]}</div></div></div>')


def rec_card(r):
    p = r["priority"]
    cc = {"critical": "rc-cr", "high": "rc-hi", "medium": "rc-me", "low": "rc-lo"}.get(p, "")
    bc = {"critical": "rb-cr", "high": "rb-hi", "medium": "rb-me", "low": "rb-lo"}.get(p, "")
    return (
        f'<div class="rc {cc}"><div class="rc-h">'
        f'<span class="rc-b {bc}">{p.upper()}</span>'
        f'<span class="rc-tt">{r["title"]}</span>'
        f'<span class="rc-ct">· {r["category"]}</span></div>'
        f'<div class="rc-bd">{r["action"]}</div>'
        f'<div class="rc-im">⚡ {r["impact"]}</div></div>'
    )


def _corrected_years(res: dict) -> int:
    """
    Robust experience year calculator that fixes two known analyser bugs:
 
    BUG 1 — Education dates counted as experience:
      The analyser scans the whole document. A B.Tech line like
      "2022 – present" or "2022 – 2026" gets parsed as 3-4 years of work.
      Fix: re-extract years from resume text, but SKIP any line that
      contains education keywords (university, college, b.tech, degree etc).
 
    BUG 2 — Present year hardcoded as 2025:
      Current year is 2026. Add +1 when a present-marker is found in a
      genuine work-experience line.
    """
    import re as _re
 
    resume_text = (
        st.session_state.get("resume_text", "")   # student mode
        or res.get("_resume_text", "")             # recruiter mode (stored below)
    )
 
    # Education-section keywords — lines containing these are skipped
    EDU_SKIP = {
        "university","college","institute","b.tech","b.e","b.sc","m.tech","m.sc",
        "mba","phd","ph.d","bachelor","master","degree","diploma","graduation",
        "graduated","matriculation","secondary","school","cbse","icse","10th","12th",
        "cgpa","gpa","percentage","semester","ug","pg","undergraduate","postgraduate",
        "bca","bba","b.com","mca","mba","engineering college","engineering institute",
    }
 
    WORK_ROLES = {
        "engineer","developer","analyst","manager","lead","head","director",
        "architect","designer","consultant","officer","specialist","associate",
        "intern","scientist","researcher","coordinator","executive","programmer",
        "administrator","trainee","apprentice","sre","devops","mlops",
    }
 
    DATE_RE = _re.compile(
        r"(20\d\d|19\d\d)\s*[-–—to]+\s*(20\d\d|present|current|now)",
        _re.IGNORECASE,
    )
 
    if not resume_text:
        # Fallback: return raw analyser value but cap at 20
        return min(res.get("years_exp", 0), 20)
 
    nums: list[int] = []
    lines = resume_text.split("\n")
 
    # Also try to detect the education section boundaries so we can skip the whole block
    edu_section_active = False
    work_section_active = False
    WORK_HEADERS  = {"experience","work experience","employment","career","professional experience","internship"}
    EDU_HEADERS   = {"education","academic","qualification","schooling","academics"}
 
    for line in lines:
        ll = line.lower().strip()
        if not ll:
            continue
 
        # Detect section headers
        if any(h in ll for h in WORK_HEADERS) and len(ll) < 40:
            work_section_active = True
            edu_section_active  = False
            continue
        if any(h in ll for h in EDU_HEADERS) and len(ll) < 40:
            edu_section_active  = True
            work_section_active = False
            continue
        # Reset if we see a new major section (projects, skills, etc.)
        if ll in {"projects","skills","technical skills","certifications","achievements",
                  "publications","awards","extra-curricular"}:
            edu_section_active  = False
            work_section_active = False
            continue
 
        # Skip lines that are clearly education-related
        if edu_section_active:
            continue
        if any(ek in ll for ek in EDU_SKIP):
            continue
 
        # Only count dates from lines that have a role/work signal OR we're in work section
        has_role = any(rw in ll for rw in WORK_ROLES)
        if not has_role and not work_section_active:
            continue
 
        m = DATE_RE.search(line)
        if m:
            try:
                s = int(m.group(1))
                end_raw = m.group(2).lower()
                e = 2026 if end_raw in ("present", "current", "now") else int(m.group(2))
                duration = max(0, e - s)
                if duration <= 20:   # sanity: no single role > 20 years
                    nums.append(duration)
            except ValueError:
                pass
 
    # Also check explicit "X years of experience" statements anywhere
    EXP_EXPLICIT = [
        r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:professional\s*)?experience",
        r"over\s*(\d+)\s*years?\s*(?:of\s*)?experience",
        r"(\d+)\s*yrs?\s*(?:of\s*)?(?:professional\s*)?experience",
    ]
    for pat in EXP_EXPLICIT:
        for match in _re.findall(pat, resume_text.lower()):
            try:
                nums.append(int(match))
            except ValueError:
                pass
 
    if not nums:
        # No work dates found — likely fresher. Return 0 rather than leaking edu dates.
        return 0
 
    # Sum overlapping roles (candidate may have had multiple simultaneous roles)
    # but take max of any single span to avoid double-counting
    return min(max(nums), 25)

def mpl_fig(w,h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    for sp in ax.spines.values(): sp.set_color("#e2e8f0")
    ax.tick_params(colors="#374151")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════
# TOP NAV
# ══════════════════════════════════════════════════════════════════════
mode = st.session_state.get("app_mode")
mode_label = ("🎓 Student Mode" if mode == "student"
              else "👔 Recruiter Mode" if mode == "recruiter"
              else "")

st.markdown(f"""
<div class="topnav">
  <div class="nav-brand">
    <div class="brand-icon">⚡</div>
    ResumeAI Pro
  </div>
  <div class="nav-badge">ATS v3</div>
  <div style="flex:1;"></div>
  {f'<div style="font-size:.74rem;font-weight:600;color:#475569;background:#f1f5f9;padding:6px 14px;border-radius:8px;border:1px solid #e2e8f0;">{mode_label}</div>' if mode else ''}
  <div style="font-family:JetBrains Mono,monospace;font-size:.6rem;color:#94a3b8;padding-left:8px;">
    Skills 40% · Experience 25% · Projects 15% · Keywords 15% · Education 5%
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MODE SELECTION LANDING PAGE
# ══════════════════════════════════════════════════════════════════════
if mode is None:
    st.markdown("""
    <div style="text-align:center;padding:48px 32px 0;">
      <div style="font-size:2rem;font-weight:900;color:#0f172a;letter-spacing:-.8px;margin-bottom:10px;">
        Who are you today?
      </div>
      <div style="font-size:.9rem;color:#64748b;max-width:500px;margin:0 auto;line-height:1.7;">
        ResumeAI Pro serves both job seekers and hiring teams.<br>
        Select your role to get a tailored experience.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([5, 1, 5])

    with col1:
        st.markdown("""
        <div class="mode-card" style="border-color:#3b82f6;background:#eff6ff;">
          <div class="mc-icon">🎓</div>
          <div class="mc-title">Student / Job Seeker</div>
          <div class="mc-desc">
            Upload your resume and a job description.<br>
            Get your ATS score, skill gap analysis, recruiter verdict, and personalised tips to improve.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter as Student →", use_container_width=True, key="mode_student"):
            st.session_state["app_mode"] = "student"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="mode-card" style="border-color:#8b5cf6;background:#fdf4ff;">
          <div class="mc-icon">👔</div>
          <div class="mc-title">Recruiter / Hiring Manager</div>
          <div class="mc-desc">
            Paste a job description and upload multiple resumes.<br>
            Rank all candidates by ATS fit, compare skills, and get a shortlist recommendation.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter as Recruiter →", use_container_width=True, key="mode_recruiter"):
            st.session_state["app_mode"] = "recruiter"
            st.rerun()

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.stop()


# ── Switch mode button ───────────────────────────────────────────────
st.markdown('<div style="padding:10px 32px 0;display:flex;align-items:center;gap:10px;">', unsafe_allow_html=True)
if st.button("← Switch Mode", key="switch_mode"):
    st.session_state["app_mode"] = None
    st.session_state["result"] = None
    st.session_state["rec_results"] = []
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# ████████████████  STUDENT MODE  ████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════
if mode == "student":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🎓 ATS Resume Analyser</div>
      <div class="hero-sub">
        Upload your resume and the job description to get a full ATS score breakdown,
        skill gap analysis, recruiter verdict, and actionable recommendations.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📄 Input",
        "📊 ATS Dashboard",
        "🛠️ Skill Match",
        "💡 Recommendations",
        "📇 Contact Info",
        "📈 Dataset Insights",
    ])
    res = st.session_state.get("result")

    # ════════════════ TAB 1 — INPUT ══════════════════════════════════
    with t1:
        st.markdown('<div class="pg">', unsafe_allow_html=True)
        cr, cj = st.columns(2, gap="large")

        with cr:
            st.markdown('<div class="h1">📄 Your Resume</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub">Upload a PDF/DOCX or paste your resume text.</div>', unsafe_allow_html=True)
            mode_r = st.radio("", ["📁 Upload file", "✏️ Paste text"],
                              horizontal=True, label_visibility="collapsed", key="r_mode")
            if "📁" in mode_r:
                up = st.file_uploader("PDF / DOCX / TXT", type=["pdf", "docx", "txt"],
                                      label_visibility="collapsed", key="r_up")
                if up:
                    txt = parse_upload(up)
                    if txt.startswith("Parse error"):
                        st.error(txt)
                    else:
                        st.session_state["resume_text"] = txt
                        st.success(f"✅ {len(txt.split())} words extracted from **{up.name}**")
            else:
                p = st.text_area("Resume", height=360,
                                 value=st.session_state.get("resume_text", ""),
                                 placeholder="Paste your full resume here…",
                                 label_visibility="collapsed", key="r_paste")
                st.session_state["resume_text"] = p

            rv2 = st.session_state.get("resume_text", "").strip()
            if rv2:
                wc = len(rv2.split())
                if wc >= 450:
                    st.markdown(f'<div class="callout cs">✅ {wc} words — good length for ATS.</div>', unsafe_allow_html=True)
                elif wc >= 200:
                    st.markdown(f'<div class="callout ci">📝 {wc} words — consider adding more detail.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="callout cw">⚠️ Only {wc} words — resume may be too thin.</div>', unsafe_allow_html=True)

        with cj:
            st.markdown('<div class="h1">📋 Job Description</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub">Paste the full job description you are applying for.</div>', unsafe_allow_html=True)
            jd_in = st.text_area("JD", height=400,
                                 value=st.session_state.get("jd_text", ""),
                                 placeholder="Paste the full job description here…",
                                 label_visibility="collapsed", key="jd_paste")
            st.session_state["jd_text"] = jd_in
            if jd_in.strip():
                wj = len(jd_in.split())
                q = "#22c55e" if wj > 150 else "#f97316"
                st.markdown(f'<div style="font-size:.72rem;color:{q};margin-top:4px;font-weight:500;">✓ {wj} words</div>',
                            unsafe_allow_html=True)

        st.markdown('<hr class="sep">', unsafe_allow_html=True)
        R = st.session_state.get("resume_text", "").strip()
        J = st.session_state.get("jd_text", "").strip()

        bc2, mc2 = st.columns([1, 3])
        with bc2:
            go = st.button("⚡ Analyse Resume", use_container_width=True, disabled=not (R and J))
        with mc2:
            if not R:
                st.markdown('<div class="callout ci">👈 Add your resume to begin.</div>', unsafe_allow_html=True)
            elif not J:
                st.markdown('<div class="callout ci">📋 Paste a job description.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="callout cs">✅ Both inputs ready — click Analyse Resume.</div>', unsafe_allow_html=True)

        if go and R and J:
            with st.spinner("Running full ATS analysis…"):
                r2 = scorer.score(R, J)
                r2["recommendations"] = RecommendationEngine.generate(r2, J)
                st.session_state["result"] = r2
            res = r2
            st.success("✅ Analysis complete — open the **ATS Dashboard** tab.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════ TAB 2 — ATS DASHBOARD ══════════════════════════
    with t2:
        st.markdown('<div class="pg">', unsafe_allow_html=True)
        if not res:
            no_result_placeholder()
        else:
            fs = res["final_score"]
            rv3 = res["recruiter_verdict"]
            sim = res["similarity"]
            fc  = score_color(fs)
            slbl = ("ATS Ready" if fs >= 75 else "Good Match" if fs >= 55
                    else "Needs Work" if fs >= 35 else "Poor Match")
            yrs_display = _corrected_years(res)

            # ── Header row ──────────────────────────────────────────
            h1c, h2c, h3c = st.columns([1, 1.2, 2.2], gap="medium")

            with h1c:
                dec_icon = ("✅" if rv3["decision"] == "YES"
                            else "⚠️" if rv3["decision"] == "MAYBE" else "❌")
                st.markdown(
                    f'<div class="score-ring">'
                    f'<div class="score-num" style="color:{fc};">{fs:.0f}</div>'
                    f'<div class="score-den">/ 100 ATS Score</div>'
                    f'<div class="score-lbl" style="color:{fc};">{slbl}</div>'
                    f'<div class="verdict-badge" style="color:{rv3["dec_color"]};'
                    f'border-color:{rv3["dec_color"]};background:{rv3["dec_bg"]};">'
                    f'{dec_icon} {rv3["decision"]}</div></div>',
                    unsafe_allow_html=True,
                )

            with h2c:
                sim_rows = "".join(
                    f'<div style="margin-bottom:10px;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:.75rem;'
                    f'color:#1e293b;margin-bottom:3px;"><span>{lb}</span>'
                    f'<span style="font-weight:700;color:{score_color(vl)};">{vl:.0f}%</span></div>'
                    f'{prog_bar(vl, score_color(vl), 5)}</div>'
                    for lb, vl in [
                        ("Overall Similarity", sim.get("overall",0)),
                        ("TF-IDF Cosine",      sim.get("tfidf",0)),
                        ("Keyword Overlap",    sim.get("keyword",0)),
                        ("Jaccard",            sim.get("jaccard",0)),
                    ]
                )
                st.markdown(
                    f'<div class="card"><span class="lbl">SIMILARITY METRICS</span>{sim_rows}</div>',
                    unsafe_allow_html=True,
                )

            with h3c:
                rv_icon = dec_icon
                # reasoning is a list of bullets from analyser v3
                _raw_r = rv3.get("reasoning", [])
                if isinstance(_raw_r, list):
                    _rv_body = "".join(
                        f'<div style="display:flex;gap:7px;margin-bottom:4px;">'
                        f'<span style="color:rgba(255,255,255,.6);flex-shrink:0;">•</span>'
                        f'<span>{item}</span></div>'
                        for item in _raw_r
                    )
                else:
                    _rv_body = f'<div>{_raw_r}</div>'
                _next = rv3.get("next_action", rv3.get("next_step", ""))
                st.markdown(
                    f'<div class="rvb" style="color:{rv3["dec_color"]};border-color:{rv3["dec_color"]};'
                    f'background:{rv3["dec_bg"]};">'
                    f'<div class="rv-d">{rv_icon} Recruiter Verdict: {rv3["decision"]}</div>'
                    f'<div class="rv-c">{rv3["confidence"]} CONFIDENCE</div>'
                    f'<div style="margin-top:8px;">{_rv_body}</div>'
                    + (f'<div class="rv-n">📌 {_next}</div>' if _next else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ── Stat pills ───────────────────────────────────────────
            comps = res["components"]
            certs = res.get("cert_count", 0)
            secs  = res.get("sections", [])
            st.markdown(
                '<div class="stats-row">'
                + stat_pill(f'{yrs_display}yr', "Experience", score_color(min(yrs_display * 20, 100)))
                + stat_pill(str(len(sim.get("matched_skills", []))), "Skills Matched", "#22c55e")
                + stat_pill(str(len(sim.get("missing_skills", []))), "Skills Missing", "#ef4444")
                + stat_pill(str(len(secs)), "Sections Found", "#6366f1")
                + stat_pill(str(certs), "Certifications", "#eab308")
                + stat_pill(f'{sim["overall"]:.0f}%', "JD Match", score_color(sim["overall"]))
                + "</div>",
                unsafe_allow_html=True,
            )

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ── Component breakdown ──────────────────────────────────
            st.markdown('<div class="h1">Score Breakdown by Component</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub">Each component is weighted differently. Click into the ATS scoring system logic to understand what moves your score.</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2, gap="medium")
            comp_items = list(comps.items())
            for i, (name, comp) in enumerate(comp_items):
                with (col_a if i % 2 == 0 else col_b):
                    st.markdown(component_card(name, comp), unsafe_allow_html=True)

            # ── Pattern flags ────────────────────────────────────────
            patterns = res.get("patterns", [])
            if patterns:
                st.markdown('<hr class="sep">', unsafe_allow_html=True)
                st.markdown('<div class="h1">🔍 Resume Pattern Flags</div>', unsafe_allow_html=True)
                st.markdown('<div class="sub">Automatic pattern detection — positive signals and areas to watch.</div>', unsafe_allow_html=True)
                for pat in patterns:
                    st.markdown(pattern_card(pat), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════ TAB 3 — SKILL MATCH ════════════════════════════
    with t3:
        st.markdown('<div class="pg">', unsafe_allow_html=True)
        if not res:
            no_result_placeholder()
        else:
            sim = res["similarity"]
            matched  = sim.get("matched_skills", [])
            missing  = sim.get("missing_skills", [])
            partial  = sim.get("partial_matches", [])
            bonus    = sim.get("bonus_skills", [])
            jd_kws   = sim.get("jd_keywords", [])
            res_kws  = sim.get("resume_keywords", [])
            inferred = res.get("inferred_skills", [])

            # Overview summary bar
            total = len(matched) + len(missing)
            pct_match = round(len(matched) / total * 100) if total else 0
            fc_m = score_color(pct_match)
            st.markdown(
                f'<div class="card" style="border-left:4px solid {fc_m};">'
                f'<div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">'
                f'<div><div style="font-size:2.6rem;font-weight:900;color:{fc_m};letter-spacing:-1.5px;">'
                f'{pct_match}%</div><div style="font-size:.7rem;color:#64748b;font-weight:600;">SKILL COVERAGE</div></div>'
                f'<div style="flex:1;">{prog_bar(pct_match, fc_m, 9)}'
                f'<div style="font-size:.76rem;color:#475569;margin-top:6px;">'
                f'{len(matched)} matched · {len(missing)} missing · {len(partial)} partial · {len(bonus)} bonus</div>'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2, gap="medium")

            with col1:
                st.markdown('<span class="lbl">✅ MATCHED SKILLS — already in your resume</span>', unsafe_allow_html=True)
                st.markdown(chips(matched, "g"), unsafe_allow_html=True)

                st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
                st.markdown('<span class="lbl">⚡ BONUS SKILLS — not required but differentiate you</span>', unsafe_allow_html=True)
                st.markdown(chips(bonus, "b"), unsafe_allow_html=True)

                st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
                st.markdown('<span class="lbl">🔑 TOP JD KEYWORDS</span>', unsafe_allow_html=True)
                st.markdown(chips(jd_kws[:30], "p"), unsafe_allow_html=True)

            with col2:
                st.markdown('<span class="lbl">❌ MISSING SKILLS — add these to close the gap</span>', unsafe_allow_html=True)
                st.markdown(chips(missing, "r"), unsafe_allow_html=True)

                if partial:
                    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
                    st.markdown('<span class="lbl">🟡 PARTIAL MATCHES — you have the parent, add the specific</span>', unsafe_allow_html=True)
                    for resume_sk, jd_sk in partial[:10]:
                        st.markdown(
                            f'<div class="card-sm" style="display:flex;align-items:center;gap:8px;">'
                            f'{chip(resume_sk, "y")} → {chip(jd_sk, "g")}'
                            f'<span style="font-size:.7rem;color:#64748b;">add explicitly</span></div>',
                            unsafe_allow_html=True,
                        )

                st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
                st.markdown('<span class="lbl">🔑 YOUR RESUME KEYWORDS</span>', unsafe_allow_html=True)
                st.markdown(chips(res_kws[:30], "n"), unsafe_allow_html=True)

            # Inferred (hidden) skills
            if inferred:
                st.markdown('<hr class="sep">', unsafe_allow_html=True)
                st.markdown('<div class="h1">💡 Implied Skills — Add These Explicitly</div>', unsafe_allow_html=True)
                st.markdown('<div class="sub">These skills are implied by your experience but not listed. ATS systems scan for exact keywords — add them to your Skills section.</div>', unsafe_allow_html=True)
                for inf in inferred[:12]:
                    cc2 = {"high": "#22c55e", "medium": "#eab308", "low": "#94a3b8"}.get(inf["confidence"], "#94a3b8")
                    cbg = {"high": "#f0fdf4", "medium": "#fefce8", "low": "#f8fafc"}.get(inf["confidence"], "#f8fafc")
                    st.markdown(
                        f'<div class="card-sm" style="border-left:3px solid {cc2};">'
                        f'<div style="display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:4px;">'
                        f'<span style="background:{cbg};color:{cc2};border:1px solid {cc2};'
                        f'font-family:JetBrains Mono,monospace;font-size:.67rem;padding:3px 10px;'
                        f'border-radius:5px;font-weight:600;">{inf["skill"].title()}</span>'
                        f'<span style="font-size:.62rem;color:#94a3b8;font-family:JetBrains Mono,monospace;'
                        f'font-weight:600;text-transform:uppercase;letter-spacing:1px;">{inf["confidence"]} confidence</span></div>'
                        f'<div style="font-size:.71rem;color:#64748b;font-style:italic;margin-bottom:4px;">Evidence: {inf["evidence"]}</div>'
                        f'<div style="font-size:.75rem;color:#475569;line-height:1.5;">{inf["suggestion"]}</div></div>',
                        unsafe_allow_html=True,
                    )

        st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════ TAB 4 — RECOMMENDATIONS ════════════════════════
    with t4:
        st.markdown('<div class="pg">', unsafe_allow_html=True)
        if not res:
            no_result_placeholder()
        else:
            recs = res.get("recommendations", [])
            if not recs:
                st.markdown('<div class="callout cs">✅ No recommendations — your resume looks solid!</div>', unsafe_allow_html=True)
            else:
                # Count by priority
                cnt = Counter(r["priority"] for r in recs)
                st.markdown(
                    f'<div class="stats-row">'
                    + stat_pill(str(cnt.get("critical", 0)), "Critical", "#ef4444")
                    + stat_pill(str(cnt.get("high", 0)), "High", "#f97316")
                    + stat_pill(str(cnt.get("medium", 0)), "Medium", "#eab308")
                    + stat_pill(str(cnt.get("low", 0)), "Low", "#3b82f6")
                    + "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown('<hr class="sep">', unsafe_allow_html=True)
                st.markdown('<div class="h1">Personalised Action Plan</div>', unsafe_allow_html=True)
                st.markdown('<div class="sub">Prioritised from most to least impactful. Fix Critical items first — they can cause automatic rejection.</div>', unsafe_allow_html=True)
                for r in recs:
                    st.markdown(rec_card(r), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════ TAB 5 — CONTACT INFO ═══════════════════════════
    with t5:
        st.markdown('<div class="pg">', unsafe_allow_html=True)
        if not res:
            no_result_placeholder()
        else:
            ct = res.get("contact_info", {})
            if not ct:
                st.markdown('<div class="callout cw">⚠️ Contact analysis unavailable.</div>', unsafe_allow_html=True)
            else:
                # ── Patch: re-check LinkedIn/GitHub with broader patterns ──
                resume_raw = st.session_state.get("resume_text", "")
                import re as _re
                if not ct.get("linkedin", {}).get("found"):
                    li_m = _re.search(r"linkedin\.com[/\w\-\.]+", resume_raw, _re.IGNORECASE)
                    if li_m:
                        ct["linkedin"]["found"] = True
                        ct["linkedin"]["value"] = li_m.group(0)[:80]
                        ct["linkedin"]["suggestion"] = ""
                        # Recompute score
                        ct["score"] = ct.get("score", 0) + 1
                if not ct.get("github", {}).get("found"):
                    gh_m = _re.search(r"github\.com[/\w\-\.]+", resume_raw, _re.IGNORECASE)
                    if gh_m:
                        ct["github"]["found"] = True
                        ct["github"]["value"] = gh_m.group(0)[:80]
                        ct["github"]["suggestion"] = ""
                        ct["score"] = ct.get("score", 0) + 1
                # Recompute missing_critical after patch
                ct["missing_critical"] = [k for k in ["email", "phone"] if not ct[k]["found"]]

                # analyser returns key "score" (not "contact_score")
                sc2 = ct.get("score", ct.get("contact_score", 0))
                pct = round(sc2 / ct.get("max_score", 4) * 100)
                col = score_color(pct)
                lbl = ("Complete" if pct >= 80 else "Mostly complete" if pct >= 60
                       else "Incomplete" if pct >= 40 else "Critical gaps")
                max_sc = ct.get("max_score", 4)

                st.markdown(
                    f'<div class="card" style="display:flex;align-items:center;gap:22px;border-left:4px solid {col};">'
                    f'<div style="text-align:center;min-width:72px;">'
                    f'<div style="font-size:2.6rem;font-weight:900;color:{col};letter-spacing:-1px;line-height:1;">{sc2}/{max_sc}</div>'
                    f'<div style="font-size:.62rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">Contact Score</div></div>'
                    f'<div style="flex:1;">{prog_bar(pct, col, 8)}'
                    f'<div style="font-weight:700;font-size:.85rem;color:{col};margin-top:6px;">{lbl}</div>'
                    + (f'<div style="font-size:.74rem;color:#b91c1c;margin-top:3px;">⚠️ Missing: {", ".join(ct["missing_critical"])}</div>'
                       if ct.get("missing_critical") else
                       '<div style="font-size:.74rem;color:#22c55e;margin-top:3px;">✅ All required fields present.</div>')
                    + "</div></div>",
                    unsafe_allow_html=True,
                )

                st.markdown('<hr class="sep">', unsafe_allow_html=True)
                st.markdown('<span class="lbl">REQUIRED FIELDS</span>', unsafe_allow_html=True)
                rc2 = st.columns(2, gap="medium")
                for col2, fk in zip(rc2, ["email", "phone"]):
                    with col2:
                        f = ct[fk]
                        ok = f["found"]
                        cls = "coc-ok" if ok else "coc-miss"
                        st.markdown(
                            f'<div class="coc {cls}">'
                            f'<div class="coc-ic">{f["icon"]} {"✅" if ok else "❌"}</div>'
                            f'<div class="coc-lbl">{f["label"]}</div>'
                            + (f'<div class="coc-val">{f["value"]}</div>' if f["value"] else
                               '<div style="font-size:.73rem;color:#b91c1c;margin-top:4px;">Not detected</div>')
                            + (f'<div class="coc-hint">{f["suggestion"]}</div>' if f.get("suggestion") else "")
                            + "</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
                st.markdown('<span class="lbl">RECOMMENDED FIELDS</span>', unsafe_allow_html=True)
                rc3 = st.columns(3, gap="medium")
                for col2, fk in zip(rc3, ["linkedin", "github", "portfolio"]):
                    with col2:
                        f = ct[fk]
                        ok = f["found"]
                        cls = "coc-ok" if ok else "coc-soft"
                        st.markdown(
                            f'<div class="coc {cls}">'
                            f'<div class="coc-ic">{f["icon"]} {"✅" if ok else "○"}</div>'
                            f'<div class="coc-lbl">{f["label"]}</div>'
                            + (f'<div class="coc-val">{f["value"]}</div>' if f["value"] else
                               '<div style="font-size:.73rem;color:#94a3b8;margin-top:4px;">Not detected</div>')
                            + (f'<div class="coc-hint">{f["suggestion"]}</div>' if f.get("suggestion") else "")
                            + "</div>",
                            unsafe_allow_html=True,
                        )

        st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════ TAB 6 — DATASET INSIGHTS ═══════════════════════
    with t6:
        st.markdown('<div class="pg">', unsafe_allow_html=True)
        st.markdown('<div class="h1">📈 Dataset Insights</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">Aggregated analysis across Resume.csv — top skills, category distribution, skill heatmap, and skill search.</div>', unsafe_allow_html=True)

        if _df is None:
            st.markdown('<div class="callout cw">⚠️ <b>Resume.csv not found.</b> Place it in the same folder as app.py and restart. All other tabs work without it.</div>', unsafe_allow_html=True)
        else:
            top30 = _sc.most_common(30)
            if top30:
                n30, v30 = zip(*top30)
                st.markdown('<div class="h2">Top 30 Skills Across All Resumes</div>', unsafe_allow_html=True)
                fig1, ax1 = mpl_fig(13, 7)
                colors1 = plt.cm.Blues(np.linspace(0.4, 0.85, len(n30)))
                ax1.barh(list(n30)[::-1], list(v30)[::-1], color=colors1[::-1], height=0.68, edgecolor="#e2e8f0", linewidth=0.5)
                for i, (n, v) in enumerate(zip(list(n30)[::-1], list(v30)[::-1])):
                    ax1.text(v + 0.5, i, f"{v}  ({round(v / len(_df) * 100, 1)}%)", va="center", color="#374151", fontsize=9)
                ax1.set_xlim(0, max(v30) * 1.3)
                ax1.set_title(f"Top 30 Skills — {len(_df)} resumes", fontsize=13, fontweight="bold", color="#1e293b", pad=12)
                ax1.set_xlabel("Number of Resumes", color="#64748b", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig1, use_container_width=True)
                plt.close()

            st.markdown('<hr class="sep">', unsafe_allow_html=True)
            st.markdown('<div class="h2">Resume Category Distribution</div>', unsafe_allow_html=True)
            cat_counts = _df["Category"].value_counts()
            fig2, ax2 = mpl_fig(14, 4)
            bc3 = plt.cm.Purples(np.linspace(0.35, 0.8, len(cat_counts)))
            ax2.bar(cat_counts.index, cat_counts.values, color=bc3, edgecolor="#e2e8f0", linewidth=0.5)
            for i, (cat, val) in enumerate(cat_counts.items()):
                ax2.text(i, val + 0.3, str(val), ha="center", color="#374151", fontsize=8, fontweight="600")
            ax2.set_title("Resumes per Category", fontsize=12, fontweight="bold", color="#1e293b", pad=10)
            ax2.tick_params(axis="x", rotation=45, labelsize=8)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

            try:
                import pandas as pd
                import seaborn as sns
                st.markdown('<hr class="sep">', unsafe_allow_html=True)
                st.markdown('<div class="h2">Skill Presence Heatmap by Category</div>', unsafe_allow_html=True)
                st.markdown('<div class="sub">% of resumes in each category that mention each of the top 18 skills.</div>', unsafe_allow_html=True)
                top_cats  = _df["Category"].value_counts().head(12).index.tolist()
                top_sk_hm = [s for s, _ in top30[:18]]
                hm_data   = [[round(_df[_df["Category"] == cat]["skills"].apply(lambda sl: sk in sl).mean() * 100, 1)
                               for sk in top_sk_hm] for cat in top_cats]
                hm_df     = pd.DataFrame(hm_data, index=top_cats, columns=top_sk_hm)
                fig3, ax3 = plt.subplots(figsize=(16, 6))
                fig3.patch.set_facecolor("#ffffff")
                sns.heatmap(hm_df, annot=True, fmt=".0f", cmap="Blues",
                            linewidths=0.4, linecolor="#e2e8f0", ax=ax3,
                            cbar_kws={"label": "% of resumes"})
                ax3.set_title("Skill Prevalence Heatmap", fontsize=12, fontweight="bold", color="#1e293b", pad=12)
                ax3.tick_params(colors="#374151", labelsize=9)
                plt.xticks(rotation=35, ha="right")
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close()
            except Exception as e:
                st.warning(f"Heatmap unavailable: {e}")

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ── DUAL SEARCH ─────────────────────────────────────────
            ds1, ds2 = st.columns(2, gap="large")

            with ds1:
                st.markdown('<div class="h2">🔍 Job Title → Top Skills</div>', unsafe_allow_html=True)
                st.markdown('<div class="sub">Type a job category/role to see the most common skills used in those resumes.</div>', unsafe_allow_html=True)
                search_job = st.text_input("Type a job title / category",
                                           placeholder="e.g. Data Scientist, Java Developer, HR",
                                           key="ds_job")
                if search_job.strip():
                    jbl = search_job.strip().lower()
                    # Fuzzy match against category names
                    matched_cats = [c for c in _df["Category"].unique()
                                    if jbl in c.lower()]
                    if not matched_cats:
                        st.markdown(f'<div class="callout cw">No category matching "{jbl}" found. Try: {", ".join(list(_df["Category"].unique())[:6])}…</div>', unsafe_allow_html=True)
                    else:
                        for cat in matched_cats[:3]:
                            cat_df   = _df[_df["Category"] == cat]
                            sk_cnt   = Counter(s for sl in cat_df["skills"] for s in sl)
                            top_sk   = sk_cnt.most_common(20)
                            if top_sk:
                                st.markdown(f'<div class="h2" style="margin-top:12px;">📂 {cat} <span style="font-size:.72rem;color:#94a3b8;font-weight:500;">({len(cat_df)} resumes)</span></div>', unsafe_allow_html=True)
                                fig_j, ax_j = mpl_fig(9, 5)
                                names_j = [s for s, _ in top_sk]
                                vals_j  = [v for _, v in top_sk]
                                colors_j = plt.cm.Blues(np.linspace(0.35, 0.85, len(names_j)))
                                ax_j.barh(names_j[::-1], vals_j[::-1], color=colors_j[::-1],
                                          height=0.65, edgecolor="#e2e8f0", linewidth=0.4)
                                for i, (n, v) in enumerate(zip(names_j[::-1], vals_j[::-1])):
                                    pct_v = round(v / len(cat_df) * 100, 1)
                                    ax_j.text(v + 0.3, i, f"{v} ({pct_v}%)", va="center",
                                              color="#374151", fontsize=8.5)
                                ax_j.set_xlim(0, max(vals_j) * 1.35)
                                ax_j.set_title(f"Top Skills in {cat}", fontsize=11,
                                               fontweight="bold", color="#1e293b", pad=10)
                                ax_j.set_xlabel("Count", color="#64748b", fontsize=9)
                                plt.tight_layout()
                                st.pyplot(fig_j, use_container_width=True)
                                plt.close()

            with ds2:
                st.markdown('<div class="h2">🔍 Skill → Which Categories Use It?</div>', unsafe_allow_html=True)
                st.markdown('<div class="sub">Type a skill to see which job categories mention it most.</div>', unsafe_allow_html=True)
                search_sk = st.text_input("Type a skill",
                                          placeholder="e.g. python, docker, machine learning",
                                          key="ds_s")
                if search_sk.strip():
                    skl = search_sk.strip().lower()
                    cp  = {cat: round(_df[_df["Category"] == cat]["skills"].apply(
                               lambda sl: skl in sl).mean() * 100, 1)
                           for cat in _df["Category"].unique()}
                    cp  = {k: v for k, v in cp.items() if v > 0}
                    if cp:
                        scp = sorted(cp.items(), key=lambda x: x[1], reverse=True)
                        fig4, ax4 = mpl_fig(9, 5)
                        cs = [c for c, _ in scp[:12]]
                        ps = [p for _, p in scp[:12]]
                        clrs = plt.cm.Purples(np.linspace(0.35, 0.85, len(cs)))
                        ax4.barh(cs[::-1], ps[::-1], color=clrs[::-1], height=0.65, edgecolor="#e2e8f0")
                        for i, v in enumerate(ps[::-1]):
                            ax4.text(v + 0.3, i, f"{v}%", va="center", color="#374151", fontsize=9)
                        ax4.set_xlim(0, max(ps) * 1.3)
                        ax4.set_title(f'"{skl}" — categories that use it most',
                                      fontsize=11, fontweight="bold", color="#1e293b")
                        ax4.set_xlabel("% of resumes in category", color="#64748b", fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig4, use_container_width=True)
                        plt.close()
                    else:
                        st.markdown(f'<div class="callout cw">"{skl}" not found in skill database. Try a broader term like "python" or "machine learning".</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# ████████████████  RECRUITER MODE  ████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════
elif mode == "recruiter":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">👔 Recruiter Dashboard</div>
      <div class="hero-sub">
        Paste your job description, upload multiple resumes, and instantly rank all candidates
        by ATS fit. Compare skills, get shortlist recommendations, and drill into any candidate.
      </div>
    </div>
    """, unsafe_allow_html=True)

    rt1, rt2, rt3 = st.tabs([
        "📋 Setup & Upload",
        "🏆 Candidate Rankings",
        "🔍 Detailed View",
    ])

    # ════════════════ RECRUITER TAB 1 — SETUP ════════════════════════
    with rt1:
        st.markdown('<div class="pg">', unsafe_allow_html=True)

        col_jd, col_up = st.columns([1, 1], gap="large")

        with col_jd:
            st.markdown('<div class="h1">📋 Job Description</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub">Paste the full JD for this role. All resumes will be scored against this.</div>', unsafe_allow_html=True)
            jd_rec = st.text_area(
                "JD", height=420,
                value=st.session_state.get("rec_jd", ""),
                placeholder="Paste job description here…",
                label_visibility="collapsed", key="rec_jd_input",
            )
            st.session_state["rec_jd"] = jd_rec
            if jd_rec.strip():
                wj = len(jd_rec.split())
                q = "#22c55e" if wj > 150 else "#f97316"
                st.markdown(f'<div style="font-size:.72rem;color:{q};margin-top:4px;font-weight:500;">✓ {wj} words</div>', unsafe_allow_html=True)

        with col_up:
            st.markdown('<div class="h1">📁 Upload Candidate Resumes</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub">Upload up to 20 resumes (PDF, DOCX, TXT). Each file = one candidate.</div>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Resumes", type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                label_visibility="collapsed", key="rec_files",
            )
            if uploaded_files:
                st.markdown(f'<div class="callout cs">✅ {len(uploaded_files)} resume(s) uploaded.</div>', unsafe_allow_html=True)
                for uf in uploaded_files:
                    wc_est = "..."
                    st.markdown(
                        f'<div class="card-sm" style="display:flex;align-items:center;gap:10px;">'
                        f'<span style="font-size:1.1rem;">📄</span>'
                        f'<span style="font-size:.82rem;font-weight:600;color:#1e293b;">{uf.name}</span>'
                        f'<span style="font-size:.72rem;color:#94a3b8;margin-left:auto;">{uf.type}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown('<hr class="sep">', unsafe_allow_html=True)
        jd_ready  = bool(st.session_state.get("rec_jd", "").strip())
        files_ok  = bool(uploaded_files)
        bc3, mc3  = st.columns([1, 3])

        with bc3:
            run_rec = st.button("⚡ Rank All Candidates", use_container_width=True,
                                disabled=not (jd_ready and files_ok))
        with mc3:
            if not jd_ready:
                st.markdown('<div class="callout ci">📋 Paste a job description first.</div>', unsafe_allow_html=True)
            elif not files_ok:
                st.markdown('<div class="callout ci">📁 Upload at least one resume.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="callout cs">✅ Ready — {len(uploaded_files)} candidate(s) will be ranked.</div>', unsafe_allow_html=True)

        if run_rec and jd_ready and files_ok:
            JD = st.session_state["rec_jd"]
            results = []
            prog = st.progress(0, text="Analysing candidates…")
            for i, uf in enumerate(uploaded_files):
                uf.seek(0)
                txt = parse_upload(uf)
                if not txt.startswith("Parse error") and txt.strip():
                    r = scorer.score(txt, JD)
                    r["recommendations"] = RecommendationEngine.generate(r, JD)
                    results.append({
                        "name": uf.name.rsplit(".", 1)[0],
                        "filename": uf.name,
                        "text": txt,
                        "result": r,
                    })
                prog.progress((i + 1) / len(uploaded_files), text=f"Analysed {i+1}/{len(uploaded_files)}…")

            results.sort(key=lambda x: x["result"]["final_score"], reverse=True)
            st.session_state["rec_results"]  = results
            st.session_state["rec_selected"] = 0
            prog.empty()
            st.success(f"✅ {len(results)} candidate(s) ranked — open **Candidate Rankings**.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════ RECRUITER TAB 2 — RANKINGS ═════════════════════
    with rt2:
        st.markdown('<div class="pg">', unsafe_allow_html=True)
        results = st.session_state.get("rec_results", [])

        if not results:
            no_result_placeholder("Run the analysis on the <b>Setup & Upload</b> tab first.")
        else:
            # Summary stats
            scores  = [r["result"]["final_score"] for r in results]
            avg_sc  = round(sum(scores) / len(scores))
            shortlist = sum(1 for s in scores if s >= 55)
            st.markdown(
                '<div class="stats-row">'
                + stat_pill(str(len(results)), "Candidates", "#6366f1")
                + stat_pill(str(shortlist), "Shortlisted (≥55)", "#22c55e")
                + stat_pill(f"{max(scores):.0f}", "Top Score", score_color(max(scores)))
                + stat_pill(f"{avg_sc}", "Avg Score", score_color(avg_sc))
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown('<hr class="sep">', unsafe_allow_html=True)
            st.markdown('<div class="h1">Candidate Ranking</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub">Sorted by ATS score. Click "View Details" on any candidate to drill in.</div>', unsafe_allow_html=True)

            for rank, cand in enumerate(results, 1):
                r = cand["result"]
                fs = r["final_score"]
                col_s = score_color(fs)
                dec = r["recruiter_verdict"]["decision"]
                dec_cls = {"YES": "rank-dec-y", "MAYBE": "rank-dec-m", "NO": "rank-dec-n"}.get(dec, "")
                dec_ic  = {"YES": "✅", "MAYBE": "⚠️", "NO": "❌"}.get(dec, "")
                sim = r["similarity"]
                matched_n = len(sim.get("matched_skills", []))
                missing_n = len(sim.get("missing_skills", []))
                top_matched = ", ".join(list(sim.get("matched_skills", []))[:5])

                col_rank, col_detail = st.columns([6, 1])
                with col_rank:
                    st.markdown(
                        f'<div class="rank-card">'
                        f'<div class="rank-num" style="color:#94a3b8;">#{rank}</div>'
                        f'<div class="rank-score" style="color:{col_s};">{fs:.0f}</div>'
                        f'<div class="rank-details">'
                        f'<div class="rank-name">{cand["name"]}'
                        f'  <span class="rank-dec {dec_cls}">{dec_ic} {dec}</span></div>'
                        f'<div class="rank-meta">{matched_n} skills matched · {missing_n} missing'
                        + (f' · Top: {top_matched}' if top_matched else '') +
                        f'</div>'
                        f'{prog_bar(fs, col_s, 5)}'
                        f'</div>'
                        f'<div style="text-align:right;min-width:68px;">'
                        f'<div style="font-size:.68rem;color:#94a3b8;margin-bottom:3px;">{cand["filename"]}</div>'
                        f'<div style="font-size:.7rem;font-weight:600;color:{col_s};">'
                        + ("ATS Ready" if fs >= 75 else "Good Match" if fs >= 55 else "Needs Work" if fs >= 35 else "Poor Match") +
                        f'</div></div></div>',
                        unsafe_allow_html=True,
                    )
                with col_detail:
                    if st.button("View →", key=f"view_{rank}", use_container_width=True):
                        st.session_state["rec_selected"] = rank - 1
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════ RECRUITER TAB 3 — DETAILED VIEW ═══════════════
    with rt3:
        st.markdown('<div class="pg">', unsafe_allow_html=True)
        results = st.session_state.get("rec_results", [])

        if not results:
            no_result_placeholder("Run the analysis on the <b>Setup & Upload</b> tab first.")
        else:
            idx    = st.session_state.get("rec_selected", 0)
            names  = [f"#{i+1} — {r['name']}" for i, r in enumerate(results)]
            chosen = st.selectbox("Select Candidate", names, index=idx, key="cand_sel")
            idx    = names.index(chosen)
            st.session_state["rec_selected"] = idx

            cand  = results[idx]
            r     = cand["result"]
            fs    = r["final_score"]
            fc    = score_color(fs)
            rv3   = r["recruiter_verdict"]
            sim   = r["similarity"]
            comps = r["components"]

            # ── reasoning is a list of bullets (fix display) ──────────
            reasoning_list = rv3.get("reasoning", [])
            if isinstance(reasoning_list, list):
                reasoning_html = "".join(
                    f'<div style="display:flex;gap:7px;margin-bottom:5px;font-size:.78rem;">'
                    f'<span style="color:rgba(255,255,255,.6);flex-shrink:0;">•</span>'
                    f'<span>{item}</span></div>'
                    for item in reasoning_list
                )
            else:
                reasoning_html = f'<div style="font-size:.78rem;">{reasoning_list}</div>'

            dec_icon = "✅" if rv3["decision"] == "YES" else "⚠️" if rv3["decision"] == "MAYBE" else "❌"
            slbl     = ("ATS Ready" if fs >= 75 else "Good Match" if fs >= 55
                        else "Needs Work" if fs >= 35 else "Poor Match")

            # pull all the data we need
            cert_count   = r.get("cert_count", 0)
            certs_dict   = r.get("certifications", {})
            cert_analysis= r.get("cert_analysis", {})
            jd_skills    = r.get("jd_skills", [])
            exp_entries  = r.get("exp_entries", [])
            yrs_disp     = _corrected_years(r)
            domain       = r.get("domain_analysis", {})
            has_quant    = r.get("has_quant", False)
            action_ct    = r.get("action_verbs", 0)
            online       = r.get("online_presence", {})
            patterns     = r.get("patterns", [])
            recs_list    = r.get("recommendations", [])
            matched      = sim.get("matched_skills", [])
            missing_sk   = sim.get("missing_skills", [])
            partial      = sim.get("partial_matches", [])
            bonus        = sim.get("bonus_skills", sim.get("extra_skills", []))

            # contact patch (same logic as student tab)
            ct = r.get("contact_info", {})
            import re as _re
            resume_raw = cand.get("text", "")
            if ct and not ct.get("linkedin", {}).get("found"):
                li_m = _re.search(r"linkedin\.com[/\w\-\.]+", resume_raw, _re.IGNORECASE)
                if li_m:
                    ct["linkedin"]["found"] = True
                    ct["linkedin"]["value"] = li_m.group(0)[:80]
                    ct["score"] = ct.get("score", 0) + 1
            if ct and not ct.get("github", {}).get("found"):
                gh_m = _re.search(r"github\.com[/\w\-\.]+", resume_raw, _re.IGNORECASE)
                if gh_m:
                    ct["github"]["found"] = True
                    ct["github"]["value"] = gh_m.group(0)[:80]
                    ct["score"] = ct.get("score", 0) + 1
            if ct:
                ct["missing_critical"] = [k for k in ["email", "phone"] if not ct[k]["found"]]

            # ── HEADER: candidate name + summary stat row ─────────────
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:14px;margin-top:14px;flex-wrap:wrap;">'
                f'<div class="h1" style="margin:0;font-size:1.2rem;">📋 {cand["name"]}</div>'
                f'<span style="font-size:.72rem;color:#94a3b8;">{cand["filename"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="stats-row">'
                + stat_pill(f'{fs:.0f}', "ATS Score", fc)
                + stat_pill(f'{yrs_disp}yr', "Experience", score_color(min(yrs_disp * 20, 100)))
                + stat_pill(str(len(matched)), "Skills Matched", "#22c55e")
                + stat_pill(str(len(missing_sk)), "Skills Missing", "#ef4444")
                + stat_pill(str(cert_count), "Certifications", "#6366f1")
                + stat_pill(f'{sim["overall"]:.0f}%', "JD Match", score_color(sim["overall"]))
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ── ROW 1: Score ring | Similarity | Verdict ─────────────
            sc1, sc2, sc3 = st.columns([1, 1.1, 2.1], gap="medium")

            with sc1:
                st.markdown(
                    f'<div class="score-ring">'
                    f'<div class="score-num" style="color:{fc};">{fs:.0f}</div>'
                    f'<div class="score-den">/ 100 ATS Score</div>'
                    f'<div class="score-lbl" style="color:{fc};">{slbl}</div>'
                    f'<div class="verdict-badge" style="color:{rv3["dec_color"]};'
                    f'border-color:{rv3["dec_color"]};background:{rv3["dec_bg"]};">'
                    f'{dec_icon} {rv3["decision"]}</div></div>',
                    unsafe_allow_html=True,
                )

            with sc2:
                sim_rows = "".join(
                    f'<div style="margin-bottom:9px;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:.75rem;'
                    f'color:#1e293b;margin-bottom:3px;"><span>{lb}</span>'
                    f'<span style="font-weight:700;color:{score_color(vl)};">{vl:.0f}%</span></div>'
                    f'{prog_bar(vl, score_color(vl), 5)}</div>'
                    for lb, vl in [
                        ("Overall Similarity", sim["overall"]),
                        ("TF-IDF Cosine",      sim.get("tfidf", 0)),
                        ("Keyword Overlap",    sim.get("keyword", 0)),
                        ("Jaccard Score",      sim.get("jaccard", 0)),
                    ]
                )
                st.markdown(
                    f'<div class="card"><span class="lbl">SIMILARITY METRICS</span>{sim_rows}</div>',
                    unsafe_allow_html=True,
                )

            with sc3:
                # Verdict banner with properly-rendered bullet list
                next_act = rv3.get("next_action", rv3.get("next_step", ""))
                st.markdown(
                    f'<div class="rvb" style="border-color:{rv3["dec_color"]};background:{rv3["dec_bg"]};">'
                    f'<div class="rv-d">{dec_icon} Recruiter Verdict: {rv3["decision"]}</div>'
                    f'<div class="rv-c">{rv3["confidence"]} CONFIDENCE</div>'
                    f'<div style="margin-top:8px;">{reasoning_html}</div>'
                    + (f'<div class="rv-n">📌 Next action: {next_act}</div>' if next_act else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ── SECTION 2: SKILLS (matched / missing / partial / bonus) ─
            st.markdown('<div class="h1">🛠️ Skill Analysis vs JD</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub">Complete skill picture — what the candidate has, what they\'re missing, and hidden strengths.</div>', unsafe_allow_html=True)

            total_jd = len(jd_skills) or 1
            cov_pct  = round(len(matched) / total_jd * 100)
            cov_col  = score_color(cov_pct)

            # Coverage bar
            st.markdown(
                f'<div class="card" style="border-left:4px solid {cov_col};">'
                f'<div style="display:flex;align-items:center;gap:16px;">'
                f'<div style="font-size:2rem;font-weight:900;color:{cov_col};letter-spacing:-1px;min-width:56px;">{cov_pct}%</div>'
                f'<div style="flex:1;">{prog_bar(cov_pct, cov_col, 9)}'
                f'<div style="font-size:.74rem;color:#475569;margin-top:5px;">'
                f'{len(matched)} of {total_jd} JD skills matched · {len(missing_sk)} missing · '
                f'{len(partial)} partial · {len(bonus)} bonus skills</div>'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )

            sk1, sk2, sk3 = st.columns(3, gap="medium")
            with sk1:
                st.markdown('<span class="lbl">✅ MATCHED — candidate has these JD skills</span>', unsafe_allow_html=True)
                st.markdown(chips(matched, "g"), unsafe_allow_html=True)
            with sk2:
                st.markdown('<span class="lbl">❌ MISSING — JD requires these, candidate lacks</span>', unsafe_allow_html=True)
                st.markdown(chips(missing_sk, "r"), unsafe_allow_html=True)
            with sk3:
                st.markdown('<span class="lbl">⭐ BONUS — extra skills beyond JD requirements</span>', unsafe_allow_html=True)
                st.markdown(chips(bonus, "b") if bonus else '<span style="font-size:.74rem;color:#94a3b8;font-style:italic;">None</span>', unsafe_allow_html=True)

            if partial:
                st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
                st.markdown('<span class="lbl">🟡 PARTIAL MATCHES — has parent skill but not the specific sub-skill</span>', unsafe_allow_html=True)
                partial_html = "".join(
                    f'{chip(rs, "y")} <span style="font-size:.72rem;color:#94a3b8;margin:0 4px;">→</span> {chip(js, "g")} '
                    for rs, js in partial[:10]
                )
                st.markdown(f'<div class="chips">{partial_html}</div>', unsafe_allow_html=True)

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ── SECTION 3: EXPERIENCE ANALYSIS ───────────────────────
            st.markdown('<div class="h1">💼 Experience Analysis</div>', unsafe_allow_html=True)

            exp1, exp2 = st.columns(2, gap="medium")

            with exp1:
                # Experience timeline entries
                st.markdown('<span class="lbl">WORK HISTORY DETECTED</span>', unsafe_allow_html=True)
                if exp_entries:
                    for entry in exp_entries[:6]:
                        dur = f'{entry["duration_yrs"]}yr' if entry.get("duration_yrs") else ""
                        date_str = f'{entry.get("start", "")} – {entry.get("end", "")}' if entry.get("start") else ""
                        st.markdown(
                            f'<div class="card-sm" style="border-left:3px solid #6366f1;">'
                            f'<div style="font-size:.82rem;font-weight:600;color:#1e293b;margin-bottom:3px;">{entry["raw"][:90]}</div>'
                            f'<div style="display:flex;gap:10px;font-size:.68rem;color:#64748b;">'
                            + (f'<span>📅 {date_str}</span>' if date_str else '')
                            + (f'<span>⏱ {dur}</span>' if dur else '')
                            + f'</div></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown('<div class="callout cw">⚠️ No structured work history entries detected. May be a fresher or uses non-standard formatting.</div>', unsafe_allow_html=True)

            with exp2:
                # Quantitative experience signals
                st.markdown('<span class="lbl">EXPERIENCE QUALITY SIGNALS</span>', unsafe_allow_html=True)

                def signal_row(icon, label, value, ok):
                    color = "#22c55e" if ok else "#ef4444"
                    return (
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:8px 12px;border-radius:8px;margin-bottom:6px;'
                        f'background:{"#f0fdf4" if ok else "#fef2f2"};'
                        f'border:1px solid {"#86efac" if ok else "#fecaca"};">'
                        f'<span style="font-size:.82rem;color:#1e293b;">{icon} {label}</span>'
                        f'<span style="font-size:.8rem;font-weight:700;color:{color};">{value}</span>'
                        f'</div>'
                    )

                st.markdown(
                    signal_row("⏱", "Years of experience", f"{yrs_disp} yr(s)", yrs_disp >= 2)
                    + signal_row("📊", "Quantified achievements", "Yes ✓" if has_quant else "Not found", has_quant)
                    + signal_row("💬", "Action verbs count", str(action_ct), action_ct >= 8)
                    + signal_row("🌐", "GitHub presence", "Found ✓" if online.get("github") else "Not listed", online.get("github", False))
                    + signal_row("🔗", "LinkedIn presence", "Found ✓" if online.get("linkedin") else "Not listed", online.get("linkedin", False))
                    + signal_row("🏗", "Projects detected", "Yes ✓" if r.get("has_projects") else "None found", r.get("has_projects", False)),
                    unsafe_allow_html=True,
                )

                # Domain match
                rel_lbl   = domain.get("relevance_label", "Unknown")
                rel_col   = domain.get("relevance_color", "#94a3b8")
                dom_match = domain.get("domain_match_pct", 0)
                st.markdown(
                    f'<div class="card-sm" style="border-left:3px solid {rel_col};margin-top:8px;">'
                    f'<div style="font-size:.72rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">DOMAIN RELEVANCE</div>'
                    f'<div style="font-size:.9rem;font-weight:800;color:{rel_col};">{rel_lbl} — {dom_match:.0f}% match</div>'
                    f'<div style="font-size:.74rem;color:#475569;margin-top:4px;line-height:1.5;">{domain.get("insight","")}</div>'
                    f'{prog_bar(dom_match, rel_col, 5)}'
                    + (f'<div style="font-size:.7rem;color:#64748b;margin-top:5px;">Candidate: {", ".join(domain.get("candidate_domains",[])[:3])}<br>JD requires: {", ".join(domain.get("jd_domains",[])[:3])}</div>'
                       if domain.get("candidate_domains") else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ── SECTION 4: CERTIFICATIONS ─────────────────────────────
            st.markdown('<div class="h1">🏅 Certifications</div>', unsafe_allow_html=True)

            cert_col1, cert_col2 = st.columns(2, gap="medium")

            with cert_col1:
                st.markdown('<span class="lbl">CERTIFICATIONS FOUND IN RESUME</span>', unsafe_allow_html=True)
                if cert_count == 0:
                    st.markdown('<div class="callout cw">⚠️ No certifications detected. Competitive candidates typically hold 1–2 relevant certs.</div>', unsafe_allow_html=True)
                else:
                    for domain_name, cert_data in certs_dict.items():
                        cert_kws = cert_data.get("keywords", [])
                        for ck in cert_kws:
                            st.markdown(
                                f'<div class="card-sm" style="border-left:3px solid #6366f1;">'
                                f'<div style="display:flex;align-items:center;gap:8px;">'
                                f'<span style="font-size:.9rem;">🏅</span>'
                                f'<div><div style="font-size:.82rem;font-weight:600;color:#1e293b;">{ck.title()}</div>'
                                f'<div style="font-size:.68rem;color:#94a3b8;">{domain_name}</div></div>'
                                f'</div></div>',
                                unsafe_allow_html=True,
                            )

            with cert_col2:
                st.markdown('<span class="lbl">CERT RELEVANCE TO THIS JD</span>', unsafe_allow_html=True)
                verdict_str   = cert_analysis.get("verdict", "")
                cert_relevant = cert_analysis.get("relevant_count", 0)
                cert_total_c  = cert_analysis.get("total", cert_count)

                rel_cert_col = "#22c55e" if cert_relevant >= 2 else "#eab308" if cert_relevant >= 1 else "#ef4444"
                st.markdown(
                    f'<div class="card" style="border-left:4px solid {rel_cert_col};">'
                    f'<div style="font-size:2rem;font-weight:900;color:{rel_cert_col};letter-spacing:-1px;">'
                    f'{cert_relevant} / {cert_total_c}</div>'
                    f'<div style="font-size:.68rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Relevant certs for this role</div>'
                    + (f'<div style="font-size:.78rem;color:#475569;line-height:1.55;">{verdict_str}</div>' if verdict_str else "")
                    + f'{prog_bar(round(cert_relevant / max(cert_total_c, 1) * 100), rel_cert_col, 6)}'
                    + "</div>",
                    unsafe_allow_html=True,
                )

                # JD cert suggestions
                jd_text_lower = st.session_state.get("rec_jd", "").lower()
                jd_cert_hints = []
                for kw in ["aws", "azure", "gcp", "kubernetes", "pmp", "cfa", "cpa", "tableau",
                           "power bi", "databricks", "snowflake", "tensorflow", "cissp", "csm"]:
                    if kw in jd_text_lower:
                        jd_cert_hints.append(kw.upper())
                if jd_cert_hints:
                    st.markdown(
                        f'<div class="callout ci" style="margin-top:8px;">'
                        f'💡 JD signals these cert domains: {", ".join(jd_cert_hints[:6])}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ── SECTION 5: CONTACT INFORMATION ───────────────────────
            st.markdown('<div class="h1">📇 Contact Information</div>', unsafe_allow_html=True)
            if not ct:
                st.markdown('<div class="callout cw">⚠️ Contact info could not be extracted.</div>', unsafe_allow_html=True)
            else:
                c_score  = ct.get("score", 0)
                c_max    = ct.get("max_score", 4)
                c_pct    = round(c_score / c_max * 100)
                c_col    = score_color(c_pct)
                c_lbl    = ("Complete" if c_pct >= 80 else "Mostly complete" if c_pct >= 60
                            else "Incomplete" if c_pct >= 40 else "Critical gaps")
                st.markdown(
                    f'<div class="card" style="border-left:4px solid {c_col};">'
                    f'<div style="display:flex;align-items:center;gap:18px;">'
                    f'<div style="font-size:2rem;font-weight:900;color:{c_col};letter-spacing:-1px;min-width:52px;">{c_score}/{c_max}</div>'
                    f'<div style="flex:1;">{prog_bar(c_pct, c_col, 8)}'
                    f'<div style="font-weight:700;font-size:.82rem;color:{c_col};margin-top:5px;">{c_lbl}</div>'
                    + (f'<div style="font-size:.73rem;color:#b91c1c;margin-top:3px;">⚠️ Missing required: {", ".join(ct["missing_critical"])}</div>'
                       if ct.get("missing_critical") else
                       '<div style="font-size:.73rem;color:#22c55e;margin-top:3px;">✅ Required contact fields present</div>')
                    + "</div></div></div>",
                    unsafe_allow_html=True,
                )

                cc1, cc2, cc3, cc4, cc5 = st.columns(5, gap="small")
                for col_w, fk in zip([cc1, cc2, cc3, cc4, cc5],
                                      ["email", "phone", "linkedin", "github", "portfolio"]):
                    with col_w:
                        f   = ct.get(fk, {})
                        ok  = f.get("found", False)
                        cls = "coc-ok" if ok else ("coc-miss" if fk in ["email", "phone"] else "coc-soft")
                        st.markdown(
                            f'<div class="coc {cls}">'
                            f'<div class="coc-ic">{f.get("icon","")}</div>'
                            f'<div class="coc-lbl">{f.get("label", fk.title())}</div>'
                            + (f'<div class="coc-val">{f["value"][:30]}…</div>'
                               if f.get("value") and len(f.get("value","")) > 30
                               else f'<div class="coc-val">{f.get("value","")}</div>'
                               if f.get("value") else
                               f'<div style="font-size:.71rem;color:{"#b91c1c" if fk in ["email","phone"] else "#94a3b8"};margin-top:4px;">Not found</div>')
                            + "</div>",
                            unsafe_allow_html=True,
                        )

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ── SECTION 6: ATS COMPONENT BREAKDOWN ───────────────────
            st.markdown('<div class="h1">📊 ATS Score Breakdown</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub">Each component weighted by hiring priority — Skills 40% · Experience 25% · Projects 15% · Keywords 15% · Education 5%</div>', unsafe_allow_html=True)
            cc1, cc2 = st.columns(2, gap="medium")
            for i, (cname, comp) in enumerate(comps.items()):
                with (cc1 if i % 2 == 0 else cc2):
                    st.markdown(component_card(cname, comp), unsafe_allow_html=True)

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ── SECTION 7: DETAILED RESUME ANALYSIS (written) ────────
            st.markdown('<div class="h1">📝 Detailed Candidate Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub">Recruiter-grade written assessment — strengths, concerns, interview focus areas, and hiring recommendation.</div>', unsafe_allow_html=True)

            # Build the written analysis dynamically
            pos_signals = rv3.get("pos_signals", 0)
            neg_signals = rv3.get("neg_signals", 0)
            decision    = rv3["decision"]

            # Strengths block
            strengths = []
            if len(matched) >= 5:              strengths.append(f"Strong technical alignment — {len(matched)} of {len(jd_skills)} required skills matched.")
            if yrs_disp >= 3:                  strengths.append(f"{yrs_disp} years of experience places this candidate comfortably at the required level.")
            if has_quant:                       strengths.append("Quantified achievements present — resume demonstrates measurable impact, not just duties.")
            if cert_count >= 1:                 strengths.append(f"{cert_count} certification(s) detected — demonstrates commitment to professional development.")
            if bonus:                           strengths.append(f"Brings additional skills beyond JD requirements: {', '.join(list(bonus)[:4])}.")
            if online.get("github"):            strengths.append("GitHub profile detected — recruiter can independently verify project quality.")
            if domain.get("domain_match_pct", 0) >= 66: strengths.append(f"Experience domain is highly relevant ({domain.get('relevance_label','')}).")
            if not strengths:                   strengths.append("Candidate meets basic application criteria.")

            # Concerns block
            concerns = []
            if len(missing_sk) >= 5:           concerns.append(f"Significant skill gaps — {len(missing_sk)} JD-required skills not found in resume.")
            if yrs_disp == 0:                   concerns.append("No work experience detected. This appears to be a fresher — evaluate project quality closely.")
            elif yrs_disp < 2:                  concerns.append(f"Limited experience ({yrs_disp} yr) — may need additional coaching for the role.")
            if not has_quant and yrs_disp >= 2: concerns.append("No quantified results despite experience — resume reads as task-oriented rather than outcome-oriented.")
            if cert_count == 0:                 concerns.append("No certifications — in competitive pools, certified candidates typically ranked higher.")
            if sim.get("tfidf", 0) < 15:        concerns.append("Very low keyword alignment (TF-IDF ≈ 0%) — resume vocabulary diverges significantly from JD language.")
            if domain.get("domain_match_pct", 0) < 33: concerns.append(f"Domain mismatch: candidate's background in {', '.join(domain.get('candidate_domains',[])[:2])} may not directly transfer.")
            if not ct.get("email", {}).get("found"): concerns.append("Email not detected — contact-ability risk.")
            if not concerns:                    concerns.append("No major concerns identified at this stage.")

            # Interview questions
            interview_qs = []
            if missing_sk:
                interview_qs.append(f"You listed {missing_sk[0].title()} as required — can you walk me through your experience with it?")
            if not has_quant:
                interview_qs.append("Can you give me a specific example where you measured the impact of your work?")
            if yrs_disp < 2:
                interview_qs.append("Can you walk me through the most complex project you've built end-to-end?")
            if partial:
                interview_qs.append(f"You show experience with {partial[0][0].title()} — have you worked specifically with {partial[0][1].title()}?")
            if cert_count == 0:
                interview_qs.append("Do you have any ongoing certifications or professional development planned?")
            interview_qs = interview_qs[:4]

            # Render the written analysis
            an1, an2 = st.columns(2, gap="medium")

            with an1:
                # Strengths
                s_items = "".join(
                    f'<div style="display:flex;gap:8px;margin-bottom:7px;">'
                    f'<span style="color:#22c55e;font-size:.9rem;flex-shrink:0;">✓</span>'
                    f'<span style="font-size:.8rem;color:#1e293b;line-height:1.55;">{s}</span></div>'
                    for s in strengths
                )
                st.markdown(
                    f'<div class="card" style="border-left:4px solid #22c55e;">'
                    f'<div class="h2" style="margin-bottom:10px;color:#15803d;">💪 Strengths</div>'
                    f'{s_items}</div>',
                    unsafe_allow_html=True,
                )

                # Interview focus areas
                if interview_qs:
                    q_items = "".join(
                        f'<div style="display:flex;gap:8px;margin-bottom:8px;">'
                        f'<span style="color:#6366f1;font-weight:800;font-size:.8rem;flex-shrink:0;">{qi+1}.</span>'
                        f'<span style="font-size:.78rem;color:#1e293b;line-height:1.55;font-style:italic;">"{q}"</span></div>'
                        for qi, q in enumerate(interview_qs)
                    )
                    st.markdown(
                        f'<div class="card" style="border-left:4px solid #6366f1;">'
                        f'<div class="h2" style="margin-bottom:10px;color:#4338ca;">🎯 Interview Focus Questions</div>'
                        f'{q_items}</div>',
                        unsafe_allow_html=True,
                    )

            with an2:
                # Concerns
                c_items = "".join(
                    f'<div style="display:flex;gap:8px;margin-bottom:7px;">'
                    f'<span style="color:#ef4444;font-size:.9rem;flex-shrink:0;">⚠</span>'
                    f'<span style="font-size:.8rem;color:#1e293b;line-height:1.55;">{c}</span></div>'
                    for c in concerns
                )
                st.markdown(
                    f'<div class="card" style="border-left:4px solid #ef4444;">'
                    f'<div class="h2" style="margin-bottom:10px;color:#b91c1c;">⚠️ Concerns</div>'
                    f'{c_items}</div>',
                    unsafe_allow_html=True,
                )

                # Hiring recommendation
                hire_bg  = {"YES": "#f0fdf4", "MAYBE": "#fefce8", "NO": "#fef2f2"}.get(decision, "#f8fafc")
                hire_bdr = {"YES": "#86efac", "MAYBE": "#fde68a", "NO": "#fecaca"}.get(decision, "#e2e8f0")
                hire_col = {"YES": "#15803d", "MAYBE": "#854d0e", "NO": "#b91c1c"}.get(decision, "#475569")
                hire_msg = {
                    "YES":   "Recommended for next round. Schedule a technical interview and verify project portfolio.",
                    "MAYBE": "Borderline — conduct a short 15-min screening call to clarify skill gaps before committing to a full interview slot.",
                    "NO":    "Does not meet minimum requirements for this role. Consider keeping profile on file for entry-level openings.",
                }.get(decision, "")
                next_act = rv3.get("next_action", rv3.get("next_step", hire_msg))
                st.markdown(
                    f'<div style="background:{hire_bg};border:1px solid {hire_bdr};border-radius:12px;padding:18px 20px;">'
                    f'<div style="font-size:.72rem;font-weight:700;color:{hire_col};text-transform:uppercase;letter-spacing:1.2px;margin-bottom:8px;">📋 HIRING RECOMMENDATION</div>'
                    f'<div style="font-size:1.1rem;font-weight:900;color:{hire_col};margin-bottom:8px;">{dec_icon} {decision}</div>'
                    f'<div style="font-size:.82rem;color:#475569;line-height:1.6;">{next_act}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Patterns
            if patterns:
                st.markdown('<hr class="sep">', unsafe_allow_html=True)
                st.markdown('<div class="h2">🔍 Detected Resume Patterns</div>', unsafe_allow_html=True)
                for pat in patterns:
                    st.markdown(pattern_card(pat), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)