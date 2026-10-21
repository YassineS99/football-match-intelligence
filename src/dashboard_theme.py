"""Read-only presentation helpers for the Football Match Intelligence dashboard.

No external assets, browser scripts, or accesses to generated analysis artifacts.
"""

from html import escape


NAV_LABELS = {
    "Overview": "◫  Command center",
    "Video analysis": "▣  Match footage",
    "Frame explorer": "⌗  Frame explorer",
    "Interactive pitch explorer": "◇  Tactical pitch",
    "Player tracking": "◉  Player tracking",
    "Team classification": "♧  Team classification",
    "Movement and heatmaps": "▦  Movement & density",
    "Pitch mapping": "⊞  Pitch calibration",
    "Technical diagnostics": "⚙  Diagnostics",
}


CSS = r"""
<style>
/* Stage 3 · Night Match interface. Pure presentation; no source-data mutation. */
:root { color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse 72% 45% at 78% -5%, rgba(39,127,118,.14), transparent 72%),
              radial-gradient(ellipse 48% 30% at 20% 18%, rgba(55,83,145,.08), transparent 78%),
              #080f1b !important;
  color: #e8eef8;
}
[data-testid="stHeader"] {background: rgba(8,15,27,.83) !important;}
[data-testid="stToolbar"] {color: #a8b8cf;}
[data-testid="stSidebar"] {
  background: #0d1726 !important;
  border-right: 1px solid #25334a !important;
}
[data-testid="stSidebar"] > div:first-child {padding-top: 1.25rem;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label p {color: #8fa5bc !important;}
[data-testid="stSidebar"] div[role="radiogroup"] label {
  border-radius: 10px; padding: 7px 8px; transition: background .15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background: #172a3d;}
.block-container {max-width: 1680px; padding: 1.5rem 2.25rem 4rem;}
h1,h2,h3 {color:#f5f8fd !important; letter-spacing:-.035em !important;}
h2 {font-weight:750 !important;}
h3 {font-weight:720 !important;}
[data-testid="stCaptionContainer"], .stCaption {color:#93a5ba !important;}
[data-testid="stMetric"] {
  background: linear-gradient(148deg, #152438, #111c2c) !important;
  border: 1px solid #2b3c51 !important;
  border-radius: 15px !important;
  padding: 1rem 1.12rem !important;
  min-height: 122px;
  box-shadow: 0 10px 28px rgba(0,0,0,.14);
}
[data-testid="stMetricLabel"] {font-size: .83rem !important; color:#a7bad0 !important;}
[data-testid="stMetricValue"] {font-weight:790 !important; letter-spacing:-.055em !important;
 color:#f7fcff !important; font-variant-numeric:tabular-nums;}
[data-testid="stMetricDelta"] {color:#6ed6c7 !important;}
[data-testid="stAlert"] {border-radius: 12px !important; border-width: 1px !important;}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid #2b3c51 !important; border-radius: 13px !important;
  overflow: hidden;
}
[data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stNumberInput"] {
  border-radius: 10px !important;
}
[data-testid="stPlotlyChart"], [data-testid="stImage"], [data-testid="stVideo"] {
  border-radius: 16px; overflow: hidden;
}
hr {border-color:#28374a !important;}
.fmi-brand {display:flex;align-items:center;gap:12px;margin-bottom:1.4rem;}
.fmi-mark {width:42px;height:42px;border-radius:13px;
  background:linear-gradient(140deg,#56e2c1,#2d8c9c); color:#071420;
  display:flex;align-items:center;justify-content:center;font-size:23px;font-weight:850;
  box-shadow: 0 8px 24px rgba(74,212,186,.15);}
.fmi-brand-name {font-size:.95rem;color:#fff;font-weight:800;letter-spacing:-.025em;}
.fmi-brand-sub {font-size:.72rem;color:#8aa0b7;letter-spacing:.12em;text-transform:uppercase;}
.fmi-side-caption {font-size:.65rem;font-weight:750;letter-spacing:.17em;
 text-transform:uppercase;color:#7189a3;margin:1.45rem 0 .55rem;}
.fmi-side-note {padding:.8rem .9rem;border:1px solid #24394c;border-radius:12px;
 background:#132436;color:#9cb3c7;font-size:.78rem;line-height:1.55;margin-top:1.6rem;}
.fmi-online {display:inline-block;width:7px;height:7px;border-radius:50%;
 background:#58d9b4;box-shadow:0 0 10px #58d9b455;margin-right:7px;}
.fmi-hero {position:relative;overflow:hidden;border:1px solid #274c50;border-radius:20px;
 background:linear-gradient(116deg,rgba(23,79,80,.86),rgba(16,36,57,.97) 60%,#131d33);
 padding:1.65rem 1.9rem;margin:0 0 1.65rem;}
.fmi-hero::after {content:"";position:absolute;width:300px;height:300px;border-radius:50%;
 border:1px solid rgba(113,232,204,.10);top:-220px;right:60px;box-shadow:0 0 0 65px rgba(113,232,204,.028);}
.fmi-eyebrow {color:#75e4cb;font-size:.68rem;font-weight:800;letter-spacing:.22em;
 text-transform:uppercase;margin-bottom:.5rem;}
.fmi-title {font-weight:850;font-size:clamp(1.6rem,2.8vw,2.65rem);letter-spacing:-.055em;
 color:#f7fcff;line-height:1.16;margin:0 0 .6rem;}
.fmi-lead {color:#bed0e0;font-size:.96rem;max-width:830px;line-height:1.56;margin:0 0 1.15rem;}
.fmi-pills {display:flex;flex-wrap:wrap;gap:8px;}
.fmi-pill {font-size:.71rem;color:#c3d9e9;border:1px solid #426071;border-radius:30px;
 background:rgba(13,30,46,.5);padding:.32rem .7rem;}
.fmi-pill b {color:#d9fff2;}
.fmi-section-header {display:flex;align-items:baseline;gap:.6rem;margin:1.5rem 0 .6rem;
 color:#f0f5fc;font-size:1.06rem;font-weight:790;letter-spacing:-.025em;}
.fmi-section-header small {font-size:.7rem;font-weight:500;color:#91a5ba;}
.fmi-panel {background:linear-gradient(150deg,#152437,#101b2a); border:1px solid #2a3d50;
 border-radius:16px;padding:1.25rem 1.4rem;margin:.2rem 0 1rem;}
.fmi-panel-kicker {font-size:.66rem;letter-spacing:.17em;text-transform:uppercase;
 color:#83a6b8;font-weight:760;margin-bottom:.55rem;}
.fmi-panel-value {font-size:1.75rem;font-weight:830;color:#ecf9ff;letter-spacing:-.04em;}
.fmi-bar {width:100%;height:9px;border-radius:20px;background:#2b3c4e;
 overflow:hidden;margin:.8rem 0 .65rem;}
.fmi-bar-fill {height:100%;border-radius:20px;
 background:linear-gradient(90deg,#21b999,#83e7ba);}
.fmi-footnote {color:#a5b7cb;font-size:.78rem;line-height:1.55;}
.fmi-pipeline {height:100%;min-height:120px;padding:1rem 1.05rem;border-radius:14px;
 background:#142336;border:1px solid #2a3e50;border-top:3px solid #46bfae;}
.fmi-pipeline b {display:block;color:#f0f7fc;margin-bottom:.35rem;font-size:.85rem;}
.fmi-pipeline span {display:block;color:#9baec3;font-size:.78rem;line-height:1.5;}
.fmi-warning {padding:.92rem 1.08rem;margin:.45rem 0 1.1rem;
 border:1px solid #695432;border-left:4px solid #d9a55b;border-radius:12px;
 background:rgba(109,77,29,.21);color:#f3d9aa;line-height:1.5;}
.fmi-warning b {color:#ffdb9d;}
@media (max-width:900px) {
 .block-container {padding:1rem .9rem 3rem;}
 [data-testid="stMetric"] {min-height:105px;padding:.75rem !important;}
 .fmi-hero {padding:1.3rem;}
}
</style>
"""


def page_hero(page, *, run_id, frames, duration, directory):
    """Escape run metadata before displaying a read-only summary in custom HTML."""
    title = {
        "Overview": "Match intelligence overview",
        "Video analysis": "Match video studio",
        "Frame explorer": "Frame-by-frame intelligence",
        "Interactive pitch explorer": "Tactical pitch explorer",
        "Player tracking": "Tracking diagnostics",
        "Team classification": "Team & role intelligence",
        "Movement and heatmaps": "Movement observations",
        "Pitch mapping": "Pitch calibration review",
        "Technical diagnostics": "Technical diagnostics",
    }.get(page, "Football Match Intelligence")
    text = {
        "Overview": "One source clip. Verifiable processing. Every number traceable to a selected tracking run.",
        "Interactive pitch explorer": "Interactive spatial evidence, bounded by your actual calibration—not extrapolated full-pitch analytics.",
        "Frame explorer": "Explore confirmed player observations on the original frame, without rerunning inference.",
    }.get(page, "Explore your computer-vision pipeline with provenance-aware, experimental analysis.")
    run_label = escape(str(directory))
    safe_run = escape(str(run_id))
    safe_frames = escape(str(frames))
    duration_text = f"{duration:.2f}s" if isinstance(duration, (int, float)) and duration >= 0 else "unknown"
    return (
        '<section class="fmi-hero">'
        '<div class="fmi-eyebrow">FOOTBALL MATCH INTELLIGENCE / RESEARCH CONSOLE</div>'
        f'<h1 class="fmi-title">{escape(title)}</h1>'
        f'<p class="fmi-lead">{escape(text)}</p>'
        '<div class="fmi-pills">'
        '<span class="fmi-pill"><span class="fmi-online"></span>LOCAL · READ-ONLY</span>'
        f'<span class="fmi-pill">FRAMES <b>{safe_frames}</b></span>'
        f'<span class="fmi-pill">CLIP <b>{duration_text}</b></span>'
        f'<span class="fmi-pill">RUN <b title="{safe_run}">{safe_run[:12]}…</b></span>'
        f'<span class="fmi-pill">SOURCE <b>{run_label}</b></span>'
        '</div></section>'
    )


def mapping_coverage_panel(valid_count, total_count):
    """Present sample acceptance, never an estimate of calibrated pitch area."""
    valid = max(0, int(valid_count))
    total = max(0, int(total_count))
    if total == 0:
        ratio = 0.0
        display = "Unavailable"
    else:
        ratio = max(0.0, min(100.0, 100 * valid / total))
        display = f"{ratio:.1f}%"
    return (
        '<section class="fmi-panel">'
        '<div class="fmi-panel-kicker">Mapped observation acceptance</div>'
        f'<div class="fmi-panel-value">{display}</div>'
        f'<div class="fmi-bar"><div class="fmi-bar-fill" style="width:{ratio:.2f}%"></div></div>'
        f'<div class="fmi-footnote">{valid:,} of {total:,} observations are accepted. '
        'This is not the percentage of pitch area calibrated. Unmapped positions remain excluded.</div>'
        '</section>'
    )
