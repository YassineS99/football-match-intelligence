import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import dashboard_data as data
import frame_explorer as explorer
import pitch_explorer as pitch_view
import dashboard_theme as ui


st.set_page_config(
    page_title="Football Match Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

BURGUNDY = "#8b1e3f"
BLUE = "#2878b8"
YELLOW = "#e3ad1b"
GREEN = "#2e8b57"
GRAY = "#7f8790"
TEAM_COLORS = {
    "team_a_outfield": BURGUNDY,
    "team_b_outfield": BLUE,
    "referee_candidate": YELLOW,
    "goalkeeper_candidate": GREEN,
    "unknown": GRAY,
}
TEAM_LABELS = {
    "team_a_outfield": "Team A outfield",
    "team_b_outfield": "Team B outfield",
    "referee_candidate": "Referee candidate",
    "goalkeeper_candidate": "Goalkeeper candidate",
    "unknown": "Unknown",
}


st.markdown(ui.CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading local analysis outputs…")
def load_dataset(output_root, selected_tracking_dir, file_fingerprint):
    # This argument is intentionally part of Streamlit's cache key.
    # It changes when an input CSV/JSON is edited, added, or removed.
    return data.load_dashboard_dataset(Path(output_root), Path(selected_tracking_dir))


@st.cache_data(show_spinner=False)
def cached_read_csv(path, mtime_ns):
    del mtime_ns
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, max_entries=24)
def cached_source_frame(video_path, frame_number, source_mtime_ns, source_size):
    # The source video's metadata participates in the cache key. Cache only a
    # bounded number of decoded frames; never write anything to data/output.
    del source_mtime_ns, source_size
    return explorer.read_source_frame(video_path, frame_number)


def metric_columns(items):
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        column.metric(label, value, help=help_text)


def safe_int(value, default=0):
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def bool_series(series):
    return (
        series.astype("string").str.strip().str.lower()
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
        .fillna(False)
        .astype(bool)
    )


def style_matplotlib_axes(fig, axes):
    """Apply night-match styling without altering underlying observations."""
    fig.set_facecolor("#132237")
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor("#122536")
        ax.tick_params(colors="#afc2d5")
        ax.xaxis.label.set_color("#c3d6e5")
        ax.yaxis.label.set_color("#c3d6e5")
        ax.title.set_color("#edf4fc")
        for spine in ax.spines.values():
            spine.set_color("#35516a")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#182b40")
            legend.get_frame().set_edgecolor("#35516a")
            for text in legend.get_texts():
                text.set_color("#e6f1fc")


def show_frame_scatter(frame, x, y, title, x_label, y_label, color_column=None):
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    if color_column and color_column in frame.columns:
        for value, group in frame.groupby(color_column, dropna=False):
            label = TEAM_LABELS.get(str(value), str(value))
            ax.scatter(group[x], group[y], s=12, alpha=0.38,
                       color=TEAM_COLORS.get(str(value), GRAY), label=label)
        ax.legend(loc="best", frameon=True)
    else:
        ax.scatter(frame[x], frame[y], s=12, alpha=0.42, color=BLUE)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2)
    ax.invert_yaxis()
    style_matplotlib_axes(fig, ax)
    st.pyplot(fig, clear_figure=True)


def show_not_validated():
    st.info(
        "**Not yet validated:** ball possession, pass detection, shot detection, "
        "expected goals (xG), distance covered in meters, and player speed in km/h. "
        "The current project outputs do not provide ground-truth events, calibrated "
        "whole-pitch coordinates, or validated event detectors."
    )


paths = data.ProjectPaths.from_environment(ROOT)
with st.sidebar:
    st.markdown(
        '<div class="fmi-brand"><span class="fmi-mark">⚽</span>'
        '<div><div class="fmi-brand-name">MATCH INTELLIGENCE</div>'
        '<div class="fmi-brand-sub">Vision analytics lab</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="fmi-side-caption">DATA SOURCE</div>', unsafe_allow_html=True)
    output_text = st.text_input(
        "Output directory",
        data.display_path(paths.output_dir, ROOT),
    )
    output_root = Path(output_text).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root = output_root.resolve()
    runs = data.discover_tracking_runs(output_root)
    if not runs:
        st.error(f"No confirmed tracking runs found under {output_root}.")
        st.stop()
    selected_label = st.selectbox(
        "Confirmed tracking run",
        options=[item["label"] for item in runs],
        index=0,
    )
    selected_run = next(item for item in runs if item["label"] == selected_label)
    st.markdown('<div class="fmi-side-caption">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Video analysis",
            "Frame explorer",
            "Interactive pitch explorer",
            "Player tracking",
            "Team classification",
            "Movement and heatmaps",
            "Pitch mapping",
            "Technical diagnostics",
        ],
        format_func=lambda item: ui.NAV_LABELS.get(item, item),
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="fmi-side-note"><span class="fmi-online"></span>Read-only local workspace'
        '<div style="margin-top:6px">Run IDs are provisional. Pitch positions appear only where calibration supports them.</div></div>',
        unsafe_allow_html=True,
    )

try:
    file_fingerprint = data.dataset_cache_fingerprint(output_root, selected_run["directory"])
    dataset = load_dataset(str(output_root), str(selected_run["directory"]), file_fingerprint)
except (data.DataValidationError, FileNotFoundError, OSError) as exc:
    st.error(f"Unable to load selected dataset safely: {exc}")
    st.stop()


tracking = dataset["tracking_summary"]
confirmed = dataset["confirmed_tracks"]
raw = dataset["raw_tracks"]
assignments = dataset["assignments"]
compensated = dataset["compensated_tracks"]
mapping = dataset["mapping"]
run_id = dataset["run_id"]
duration = data.duration_seconds(tracking)
confirmed_stats = data.track_run_summary(confirmed)
raw_stats = data.track_run_summary(raw)
st.markdown(
    ui.page_hero(
        page,
        run_id=run_id,
        frames=safe_int(tracking.get("frames_processed")),
        duration=duration,
        directory=Path(selected_run["directory"]).name,
    ),
    unsafe_allow_html=True,
)
for message in dataset["warnings"]:
    st.warning(message)
for message in dataset["errors"]:
    st.error(message)

if page == "Overview":
    st.markdown('<div class="fmi-section-header">Run overview <small>observed, not inferred</small></div>', unsafe_allow_html=True)
    st.caption(
        f"Tracking run `{run_id}` · source `{data.display_path(dataset['video_path'], ROOT)}` · "
        f"track IDs are provisional tracker assignments, not verified player identities."
    )
    if duration is not None:
        st.warning(
            f"Short demonstration only: {safe_int(tracking.get('frames_processed'))} "
            f"analyzed frames at {float(tracking.get('video_fps', 0)):.2f} FPS "
            f"({duration:.2f} seconds). These are not full-match statistics."
        )
    team_counts = (
        assignments["predicted_role"].fillna("unknown").astype(str).value_counts()
        if assignments is not None else pd.Series(dtype="int64")
    )
    mapped_valid = data.valid_pitch_rows(mapping)
    map_total = int(len(mapping)) if mapping is not None else 0
    mapping_rate = len(mapped_valid) / map_total * 100 if map_total else 0.0
    person_detections = data.person_detection_count(dataset["detections"])
    metric_columns([
        ("Frames analyzed", f"{safe_int(tracking.get('frames_processed')):,}", "Frame range is read from tracking_summary.json."),
        ("Video duration", f"{duration:.2f} s" if duration is not None else "Unavailable", "Duration = analyzed frame span / video FPS."),
        ("Person detections", f"{person_detections:,}" if person_detections is not None else "Unavailable", "Matched YOLO person detections in the selected frame range."),
        ("Confirmed samples", f"{confirmed_stats['observation_count']:,}", "Confirmed person observations in tracking rows with class_id 0."),
    ])
    unmapped_count = max(map_total - len(mapped_valid), 0)
    metric_columns([
        ("Provisional track IDs", f"{raw_stats['unique_track_count']:,}", "Unique tracker IDs in the raw tracking run."),
        ("Confirmed tracks", f"{confirmed_stats['unique_track_count']:,}", "Confirmed candidate IDs; not unique real-world players."),
        ("Team A / Team B", f"{safe_int(team_counts.get('team_a_outfield'))} / {safe_int(team_counts.get('team_b_outfield'))}", "Run-scoped provisional team labels."),
        ("Valid pitch mapped", f"{len(mapped_valid):,} ({mapping_rate:.1f}%)", "Only explicitly valid, finite mapped CSV rows."),
    ])
    metric_columns([
        ("Unmapped observations", f"{unmapped_count:,}", "No pitch point is shown for invalid or out-of-region observations."),
        ("Referee / goalkeeper candidates", f"{safe_int(team_counts.get('referee_candidate'))} / {safe_int(team_counts.get('goalkeeper_candidate'))}", "Provisional roles from jersey-color classification."),
    ])
    if dataset["mapping_summary"]:
        st.markdown(
            f'<div class="fmi-warning"><b>Pitch coverage warning:</b> '
            f'{len(mapped_valid):,} of {map_total:,} observations are validly mapped. '
            "Current calibration covers only a small region of the assumed pitch; "
            "do not interpret this as full-pitch or full-match activity.</div>",
            unsafe_allow_html=True,
        )
    st.markdown(ui.mapping_coverage_panel(len(mapped_valid), map_total), unsafe_allow_html=True)
    st.markdown('<div class="fmi-section-header">Processing pipeline <small>5 processing stages</small></div>', unsafe_allow_html=True)
    pipeline = [
        ("01 · Detect", "YOLO person and ball detections"),
        ("02 · Track", "ByteTrack IDs; run-scoped, provisional"),
        ("03 · Classify", "Team and role candidates"),
        ("04 · Compensate", "Camera motion; valid segments only"),
        ("05 · Map", "Manual pitch calibration; limited region"),
    ]
    pipeline_columns = st.columns(len(pipeline))
    for column, (title, detail) in zip(pipeline_columns, pipeline):
        column.markdown(
            f'<div class="fmi-pipeline"><b>{title}</b><span>{detail}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "This is a short-clip demonstration, not full-match analytics. "
        "Team and role labels are provisional; pitch dimensions are assumed and "
        "mapping coverage is limited to the calibrated region."
    )
    st.markdown('<div class="fmi-section-header">Analysis provenance <small>selected-run artifacts</small></div>', unsafe_allow_html=True)
    source_rows = [
        {
            "Artifact": key.title(),
            "Summary": data.display_path(value, ROOT) if value else "Not available",
        }
        for key, value in dataset["source_summaries"].items()
    ]
    st.dataframe(pd.DataFrame(source_rows), width="stretch", hide_index=True)
    show_not_validated()

elif page == "Video analysis":
    st.subheader("Annotated video outputs")
    video_options = data.compatible_video_options(output_root, dataset)
    if not video_options:
        st.info("No local video files are available in the configured input/output directories.")
    else:
        paths_by_label = {
            option["path"]: option["label"] for option in video_options
        }
        selected_video = st.selectbox(
            "Choose a local video",
            options=list(paths_by_label),
            index=0,
            format_func=lambda video: paths_by_label[video],
        )
        video_path = selected_video
        st.caption(
            f"{paths_by_label[video_path]} · "
            f"{data.display_path(video_path, ROOT)} · playback retains the encoded video timing."
        )
        st.caption(
            "Browser-compatible H.264 copies are preferred when available; original "
            "annotated files and the source video remain available as fallbacks."
        )
        try:
            st.video(str(video_path))
        except (OSError, RuntimeError) as exc:
            st.error(f"Could not display {video_path}: {exc}")
    st.markdown("**Analysis represented by available videos**")
    labels = (
        "YOLO detection",
        "ByteTrack tracking",
        "Team classification",
        "Camera-compensation comparison",
        "Pitch mapping",
    )
    available_by_analysis = {}
    for option in video_options:
        analysis, variant = option["label"].rsplit(" · ", 1) if " · " in option["label"] else (
            "Source video", "source"
        )
        available_by_analysis.setdefault(analysis, {})[variant] = option["path"]
    present = []
    for title in labels:
        variants = available_by_analysis.get(title, {})
        if "H.264" in variants and "original" in variants:
            status, video_path = "H.264 preferred; original fallback", variants["H.264"]
        elif "H.264" in variants:
            status, video_path = "H.264 available", variants["H.264"]
        elif "original" in variants:
            status, video_path = "Original fallback", variants["original"]
        else:
            status, video_path = "Not available for this run", None
        present.append({
            "Analysis": title,
            "Status": status,
            "Path": data.display_path(video_path, ROOT) if video_path else "Not available",
        })
    st.dataframe(pd.DataFrame(present), hide_index=True, width="stretch")
    show_not_validated()

elif page == "Frame explorer":
    st.subheader("Frame-by-frame player explorer")
    st.caption(
        f"Tracking run `{run_id}` · decoded frames come from the original source video. "
        "Annotations show provisional tracker IDs and jersey-color role candidates, "
        "not confirmed individual players. No YOLO processing is performed."
    )
    frame_span = data._frame_span(tracking)
    if frame_span is None or frame_span[1] <= frame_span[0]:
        st.error("The selected tracking summary has no valid source-frame range.")
    else:
        start_frame, end_exclusive = frame_span
        frame_number = st.slider(
            "Source frame (zero-based)",
            min_value=start_frame,
            max_value=end_exclusive - 1,
            value=start_frame,
            step=1,
            key=f"frame_explorer_{run_id}",
        )
        all_roles = sorted(
            confirmed.get("predicted_role", pd.Series(dtype="str"))
            .fillna("unknown").astype(str).unique().tolist()
        )
        roles = st.multiselect(
            "Show provisional roles", options=all_roles, default=all_roles,
            key=f"frame_roles_{run_id}",
            format_func=lambda role: TEAM_LABELS.get(role, role),
        )
        people_on_frame = data.frame_explorer_rows(
            confirmed, run_id, frame_number, roles=roles
        )
        all_track_ids = sorted(people_on_frame["track_id"].astype(str).unique().tolist()) if not people_on_frame.empty else []
        chosen_track = st.selectbox(
            "Show track ID (current frame only)",
            options=["All visible tracks"] + all_track_ids,
            key=f"frame_track_{run_id}",
        )
        if chosen_track != "All visible tracks":
            people_on_frame = data.frame_explorer_rows(
                confirmed, run_id, frame_number, roles=roles, track_id=chosen_track
            )
        show_boxes = st.checkbox("Bounding boxes (when available)", value=True)
        show_feet = st.checkbox("Foot-point markers", value=True)
        fps = data._finite_positive(tracking.get("video_fps", tracking.get("source_video_fps")))
        timestamp = f"{frame_number / fps:.2f} seconds" if fps else "timestamp unavailable"
        st.caption(
            f"Frame {frame_number} · source timestamp {timestamp} · "
            f"{len(people_on_frame)} confirmed person observation(s) after filters."
        )
        video_path = Path(dataset["video_path"])
        if not video_path.is_file():
            st.error(f"The source video is missing: {data.display_path(video_path, ROOT)}")
        else:
            try:
                source_stat = video_path.stat()
                image = cached_source_frame(
                    str(video_path), frame_number,
                    source_stat.st_mtime_ns, source_stat.st_size,
                )
                annotated = explorer.draw_frame_annotations(
                    image, people_on_frame, boxes=show_boxes, foot_points=show_feet
                )
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"Unable to read the selected source frame: {exc}")
            else:
                st.image(annotated, caption=f"Original source frame {frame_number}", width="stretch")
        st.caption(
            "If bounding-box columns are absent from this tracking CSV, only foot points "
            "are drawn. No missing positions, identities, or camera-compensated coordinates "
            "are invented."
        )
        if people_on_frame.empty:
            st.info("No confirmed person observations match this frame and the selected filters.")
        else:
            visible_columns = [
                col for col in (
                    "track_id", "predicted_role", "confidence", "foot_x", "foot_y",
                    "x1", "y1", "x2", "y2", "timestamp",
                ) if col in people_on_frame.columns
            ]
            st.dataframe(people_on_frame[visible_columns], hide_index=True, width="stretch")
    show_not_validated()

elif page == "Interactive pitch explorer":
    st.subheader("Interactive pitch-coordinate explorer")
    st.caption(f"Tracking run `{run_id}` · track IDs and team roles are provisional.")
    if mapping is None:
        st.warning("No uniquely compatible pitch mapping is selected. Resolve any run/compensation warnings first.")
    elif mapping.empty:
        st.info("No mapped observations exist in this run.")
    else:
        try:
            length, width = pitch_view.safe_pitch_geometry(dataset.get("mapping_summary"))
        except (TypeError, ValueError) as exc:
            st.error(f"Invalid pitch geometry: {exc}")
            st.stop()
        st.warning(
            f"Assumed, unverified pitch dimensions: {length:g} × {width:g} m. "
            "Only individually valid mapped positions are shown; this is not whole-pitch tracking."
        )
        valid = data.valid_pitch_rows(mapping)
        segments = sorted(valid["camera_segment_id"].dropna().astype(str).unique().tolist())
        if not segments:
            st.info("No valid pitch-mapped player observations are available.")
        else:
            segment = st.selectbox(
                "Camera segment (isolated reference)", segments, key=f"pitch_segment_{run_id}"
            )
            try:
                all_rows = pitch_view.valid_run_positions(
                    mapping, run_id=run_id, segment_id=segment, length=length, width=width,
                )
            except ValueError as exc:
                st.error(f"Run/segment mismatch: {exc}")
                st.stop()
            available_roles = sorted(all_rows["predicted_role"].unique().tolist())
            roles = st.multiselect(
                "Provisional teams / roles", available_roles, default=available_roles,
                format_func=lambda role: TEAM_LABELS.get(role, role),
                key=f"pitch_roles_{run_id}_{segment}",
            )
            available_tracks = sorted(
                all_rows.loc[all_rows["predicted_role"].isin(roles), "track_id"].unique(),
                key=lambda value: (not str(value).isdigit(), int(value) if str(value).isdigit() else str(value)),
            )
            selected_tracks = st.multiselect(
                "Track ID filter (empty = all selected roles)", available_tracks,
                key=f"pitch_tracks_{run_id}_{segment}",
            )
            frame_span = data._frame_span(tracking)
            first = frame_span[0] if frame_span else int(all_rows["frame"].min())
            last = frame_span[1] - 1 if frame_span else int(all_rows["frame"].max())
            labels = {"Frame positions": "frame", "Trajectories": "trajectories",
                      "Observation density": "samples"}
            mode_name = st.radio(
                "Visualization", list(labels), horizontal=True, key=f"pitch_mode_{run_id}"
            )
            mode = labels[mode_name]
            frame_number = None
            if mode == "frame":
                frame_number = st.slider(
                    "Source frame", min_value=first, max_value=last,
                    value=int(all_rows["frame"].min()), key=f"pitch_frame_{run_id}",
                )
                start_frame = end_frame = frame_number
            elif first == last:
                start_frame = end_frame = first
                st.caption(f"Only source frame {first} is available.")
            else:
                start_frame, end_frame = st.slider(
                    "Source frame interval (inclusive)", min_value=first, max_value=last,
                    value=(first, last), key=f"pitch_range_{run_id}",
                )
            try:
                rows = pitch_view.valid_run_positions(
                    mapping, run_id=run_id, segment_id=segment, length=length, width=width,
                    roles=roles, track_ids=selected_tracks if selected_tracks else None,
                    start_frame=start_frame, end_frame=end_frame,
                )
            except ValueError as exc:
                st.error(f"Cannot plot these observations: {exc}")
                st.stop()
            # Only the loader's explicitly matched calibration can provide a hull.
            # Never create one by taking the convex hull of mapped player positions.
            hull = pitch_view.supported_hull(
                dataset.get("calibration"), camera_segment_id=segment,
                length=length, width=width,
            )
            if hull is None:
                st.info(
                    "No verified calibration hull is linked to this run. No coverage region "
                    "is drawn or inferred from player positions."
                )
            else:
                st.caption("Yellow shading: original calibration hull, without extrapolation.")
            metric_columns([
                ("Valid plotted samples", f"{len(rows):,}", "Only explicitly valid person rows."),
                ("Track IDs", f"{rows['track_id'].nunique():,}", "Provisional IDs within the selected run."),
                ("Segment", segment, "Other camera reference segments are excluded."),
            ])
            try:
                fig = pitch_view.make_pitch_figure(
                    rows, length=length, width=width, segment_id=segment,
                    mode=mode, frame_number=frame_number, hull=hull,
                )
            except ModuleNotFoundError as exc:
                if exc.name != "plotly":
                    raise
                st.error("Plotly is required. Run: python -m pip install plotly")
            else:
                st.plotly_chart(fig, width="stretch", key=f"pitch_plot_{run_id}_{segment}")
                if rows.empty:
                    st.info("No mapped observations match these filters and frames.")
                elif mode == "trajectories":
                    st.caption(
                        "Lines join consecutive mapped frames only. Missing frames split paths; "
                        "no positions are interpolated, and camera segments are never joined."
                    )
                elif mode == "samples":
                    st.caption(
                        "Transparent sample markers show observed concentration without smoothing, "
                        "occupancy inference, or claiming coverage between mapped positions."
                    )
                if not rows.empty:
                    columns = [col for col in (
                        "frame", "track_id", "predicted_role", "camera_segment_id",
                        "pitch_x_m", "pitch_y_m", "pitch_mapping_valid",
                    ) if col in rows.columns]
                    with st.expander("Inspect plotted observations"):
                        st.dataframe(rows[columns], hide_index=True, width="stretch")
    show_not_validated()

elif page == "Player tracking":
    st.subheader("Track-level observations")
    if confirmed.empty:
        st.info("There are no confirmed player-candidate observations in this run.")
    else:
        people = confirmed[confirmed["class_id"] == 0].copy()
        track_ids = sorted(people["track_id"].astype(str).unique(), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
        selected_track = st.selectbox("Track ID (scoped to selected run)", track_ids)
        result = data.player_track_analytics(people, selected_track)
        details = result["summary"]
        if details:
            metric_columns([
                ("Observations", f"{details['observations']:,}", "Confirmed person observations for this provisional tracker ID."),
                ("Frame span", f"{details['first_frame']}–{details['last_frame']}", "Original zero-based source frame indices."),
                ("Missing frames", f"{details['missing_frame_count']:,}", "Missing samples within the observed frame range."),
                ("Gaps / largest gap", f"{details['gap_count']} / {details['largest_gap_frames']}", details["fragmentation_note"]),
            ])
            metric_columns([
                ("Mean confidence", f"{details['mean_confidence']:.3f}" if details["mean_confidence"] is not None else "Unavailable", "YOLO detection confidence for this track."),
                ("Median confidence", f"{details['median_confidence']:.3f}" if details["median_confidence"] is not None else "Unavailable", "Median over available detections."),
                ("Confidence range", f"{details['minimum_confidence']:.3f}–{details['maximum_confidence']:.3f}" if details["minimum_confidence"] is not None else "Unavailable", "Minimum to maximum."),
                ("Run UUID", details["run_id"], "Track IDs must not be compared across different run UUIDs."),
            ])
            show_frame_scatter(result["observations"], "foot_x", "foot_y",
                              "Observed foot point in original image coordinates",
                              "Image x (pixels)", "Image y (pixels)")
            st.caption("Gaps/fragmentation describe missing observations only; they do not establish an identity switch.")
            st.dataframe(result["observations"][[
                column for column in ("frame", "timestamp", "track_id", "confidence", "foot_x", "foot_y")
                if column in result["observations"].columns
            ]], width="stretch", hide_index=True)
    show_not_validated()

elif page == "Team classification":
    st.subheader("Run-scoped team and role assignments")
    if assignments is None or assignments.empty:
        st.info("No compatible team-assignment CSV is available for this tracking run.")
    else:
        role_counts = assignments["predicted_role"].fillna("unknown").astype(str).value_counts()
        metric_columns([
            ("Team A candidates", str(safe_int(role_counts.get("team_a_outfield"))), "Provisional track classification, not unique players."),
            ("Team B candidates", str(safe_int(role_counts.get("team_b_outfield"))), "Provisional track classification, not unique players."),
            ("Unknown", str(safe_int(role_counts.get("unknown"))), "No confident supported classification."),
            ("Referee / goalkeeper", f"{safe_int(role_counts.get('referee_candidate'))} / {safe_int(role_counts.get('goalkeeper_candidate'))}", "Provisional role candidates."),
        ])
        confidence = pd.to_numeric(assignments["classification_confidence"], errors="coerce").dropna()
        if not confidence.empty:
            fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
            ax.hist(confidence, bins=np.linspace(0, 1, 11), color="#566b82", edgecolor="white")
            ax.set(title="Classification confidence distribution (track assignments)", xlabel="Confidence", ylabel="Track count")
            ax.set_xlim(0, 1)
            style_matplotlib_axes(fig, ax)
            st.pyplot(fig, clear_figure=True)
        if not confirmed.empty:
            st.markdown("**Team-colored observations**")
            role_data = confirmed[confirmed["class_id"] == 0].copy()
            role_data = role_data[role_data["track_id"].isin(assignments["track_id"].astype(str))]
            show_frame_scatter(role_data, "foot_x", "foot_y",
                              "Player foot observations by provisional role",
                              "Image x (pixels)", "Image y (pixels)", "predicted_role")
        st.dataframe(assignments, width="stretch", hide_index=True)
        st.caption("Team labels derive from aggregated jersey-color features; no club names are inferred.")
    show_not_validated()

elif page == "Movement and heatmaps":
    st.subheader("Observed movement density")
    st.warning("These are observed sample positions, not full-match activity, physical distance, or verified movement intensity.")
    mode = st.radio("Coordinate system", ["Original image coordinates", "Camera-compensated segment coordinates"], horizontal=True)
    if mode == "Original image coordinates":
        source = confirmed[confirmed["class_id"] == 0].copy()
        columns = ("foot_x", "foot_y")
        label = "Original image pixel coordinates"
    elif compensated is not None and not compensated.empty:
        segments = sorted(compensated["camera_segment_id"].astype(str).unique())
        segment = st.selectbox("Camera segment/reference", segments)
        source = data.camera_segment_rows(compensated, segment)
        source = source[source["class_id"] == 0]
        columns = ("compensated_x", "compensated_y")
        refs = source["reference_frame"].dropna().unique().tolist()
        label = f"Segment-local compensated pixels · {segment} · reference {refs[0] if refs else 'unknown'}"
    else:
        source = pd.DataFrame()
        columns = ("foot_x", "foot_y")
        label = "Camera-compensated data unavailable"
    heat, extent, sample_count = data.image_heatmap(source, *columns, bins=(64, 36))
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    if sample_count:
        ax.imshow(heat, extent=(extent[0], extent[1], extent[3], extent[2]),
                  cmap="magma", aspect="auto", interpolation="nearest")
        ax.set_xlabel(f"{label} · x")
        ax.set_ylabel(f"{label} · y")
        ax.set_title(f"Observed position counts (n={sample_count:,}); no probability normalization")
        style_matplotlib_axes(fig, ax)
        st.pyplot(fig, clear_figure=True)
    else:
        plt.close(fig)
        st.info("No finite valid positions are available for this coordinate selection.")
    st.caption("Compensated heatmaps are isolated by camera segment; disconnected segment references are never combined.")
    show_not_validated()

elif page == "Pitch mapping":
    st.subheader("Experimental pitch-coordinate positions")
    if mapping is None:
        st.info("No compatible pitch-mapping CSV is available.")
    else:
        valid = data.valid_pitch_rows(mapping)
        total_persons = int((mapping["class_id"] == 0).sum())
        unmapped = max(total_persons - len(valid), 0)
        unique_tracks = int(valid["track_id"].nunique()) if not valid.empty else 0
        percent = len(valid) / total_persons * 100 if total_persons else 0
        calibration = dataset["calibration"] or {}
        diagnostics = calibration.get("estimation_diagnostics", {})
        hull_ratio = diagnostics.get("pitch_hull_ratio")
        hull_display = f"{float(hull_ratio) * 100:.2f}%" if hull_ratio is not None else "Unavailable"
        st.error(
            f"Limited calibration: {len(valid):,} of {total_persons:,} person observations "
            f"({percent:.1f}%) have valid pitch coordinates. Invalid, extrapolated, and "
            "out-of-region rows are excluded from this view."
        )
        metric_columns([
            ("Mapped samples", f"{len(valid):,}", "Only rows with pitch_mapping_valid=True and finite coordinates."),
            ("Unmapped samples", f"{unmapped:,}", "No position is drawn for invalid or unsupported person observations."),
            ("Track IDs", f"{unique_tracks:,}", "Run-scoped supporting IDs; not verified player identities."),
            ("Hull coverage", hull_display, "Supported hull as a fraction of the assumed pitch area."),
        ])
        geometry = (dataset["mapping_summary"] or {}).get("pitch_geometry", {})
        length = float(geometry.get("length_m", 105))
        width = float(geometry.get("width_m", 68))
        st.caption(
            f"Pitch geometry: assumed {length:g} × {width:g} m; "
            "these are not verified stadium measurements."
        )
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            ax.set_facecolor("#1c5c2c")
            ax.add_patch(plt.Rectangle((0, 0), length, width, fill=False, color="white", linewidth=1.5))
            ax.plot([length / 2, length / 2], [0, width], color="white", linewidth=1)
            ax.add_patch(plt.Circle((length / 2, width / 2), 9.15, fill=False, color="white", linewidth=1))
            pitch_hull = calibration.get("pitch_calibration_hull")
            if pitch_hull and len(pitch_hull) >= 3:
                hull = np.asarray(pitch_hull, dtype=float)
                if hull.ndim == 2 and hull.shape[1] == 2 and np.isfinite(hull).all():
                    ax.add_patch(plt.Polygon(
                        hull, closed=True, facecolor="#b8d76a", edgecolor="#f4d35e",
                        linewidth=2, alpha=0.33, hatch="///", label="Supported calibration region",
                    ))
            for role, group in valid.groupby("predicted_role", dropna=False):
                key = str(role) if pd.notna(role) else "unknown"
                ax.scatter(group["pitch_x_m"], group["pitch_y_m"], s=19, alpha=0.8,
                           color=TEAM_COLORS.get(key, GRAY), label=TEAM_LABELS.get(key, key))
            ax.set(xlim=(0, length), ylim=(width, 0), xlabel="Assumed pitch x (m)", ylabel="Assumed pitch y (m)",
                   title=f"Valid mapped observations only · n={len(valid):,} · {unique_tracks} track IDs")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
            ax.set_aspect("equal", adjustable="box")
            style_matplotlib_axes(fig, ax)
            st.pyplot(fig, clear_figure=True)
            if pitch_hull:
                st.caption("Shaded region: supported calibration hull. Unshaded pitch areas are outside the calibrated coverage.")
            else:
                st.caption("Calibration hull coordinates are unavailable; only valid mapped observations are shown.")
        else:
            st.info("No valid mapped rows are available; no pitch positions are drawn.")
        reasons = data.pitch_mapping_reason_counts(mapping)
        if not reasons.empty:
            st.dataframe(reasons, width="stretch", hide_index=True)
        else:
            st.caption("No invalid person observations with a reason are present.")
    show_not_validated()

elif page == "Technical diagnostics":
    st.subheader("Pipeline provenance and limitations")
    st.caption(f"Tracking run UUID: `{run_id}`")
    detection_summary = dataset["detection_summary"] or {}
    quality = dataset.get("camera_quality")
    failed_rows = pd.DataFrame()
    if quality is not None:
        quality_valid = bool_series(quality["valid"])
        failed_rows = quality[(~quality_valid) | (quality["method"].astype(str).str.lower() == "failed")]
    calibration = dataset.get("calibration")
    calibration_diag = (calibration or {}).get("estimation_diagnostics", {})
    mapping_summary = dataset.get("mapping_summary") or {}
    pitch_mapping = dataset.get("mapping")
    valid_mapping = data.valid_pitch_rows(pitch_mapping)
    mapping_people = (
        int((pitch_mapping["class_id"] == 0).sum())
        if pitch_mapping is not None and "class_id" in pitch_mapping else 0
    )
    metric_columns([
        ("Analyzed frames", f"{safe_int(tracking.get('frames_processed')):,}", "Selected tracking output."),
        ("Failed transitions", f"{len(failed_rows):,}", "Invalid or failed camera-quality rows."),
        ("Calibration inliers", f"{safe_int(calibration_diag.get('inlier_count')):,}" if calibration else "Unavailable", "RANSAC inliers among manually identified correspondences."),
        ("Hull coverage", f"{float(calibration_diag['pitch_hull_ratio']) * 100:.2f}%" if calibration and calibration_diag.get("pitch_hull_ratio") is not None else "Unavailable", "Supported hull as a fraction of assumed pitch area."),
    ])
    metric_columns([
        ("Mapped / unmapped", f"{len(valid_mapping):,} / {max(mapping_people - len(valid_mapping), 0):,}", "Valid rows / remaining person observations."),
        ("Mapped track IDs", f"{safe_int(mapping_summary.get('mapped_track_ids')):,}", "Run-scoped IDs with at least one valid pitch sample."),
        ("Camera segments", f"{len((dataset.get('compensation_summary') or {}).get('camera_segments', [])):,}", "Segments recorded for this compensation result."),
    ])
    if detection_summary:
        st.caption(
            f"Detector: {Path(str(detection_summary.get('model', 'unavailable'))).name} · "
            f"confidence threshold {detection_summary.get('confidence_threshold', 'unavailable')}"
        )

    st.markdown("**Camera transformation quality**")
    if quality is None:
        st.info("Camera-quality transition CSV is unavailable.")
    else:
        if not quality.empty:
            camera_valid = bool_series(quality["valid"])
            fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
            ax.scatter(quality.loc[camera_valid, "frame"], np.ones(int(camera_valid.sum())),
                       s=12, color=GREEN, label="Valid")
            ax.scatter(quality.loc[~camera_valid, "frame"], np.zeros(int((~camera_valid).sum())),
                       s=14, color=BURGUNDY, label="Invalid / failed")
            ax.set(yticks=[0, 1], yticklabels=["Failed", "Valid"], xlabel="Transition frame",
                   title=f"Camera transition status · {len(failed_rows):,} failed/invalid of {len(quality):,}")
            ax.grid(axis="x", alpha=0.2)
            ax.legend(loc="upper right", frameon=True)
            style_matplotlib_axes(fig, ax)
            st.pyplot(fig, clear_figure=True)
        with st.expander("Advanced camera details"):
            st.write({
                "Camera-quality transitions": int(len(quality)),
                "Failed/invalid transitions": int(len(failed_rows)),
                "Segment boundaries": (dataset["compensation_summary"] or {}).get("camera_segments", []),
            })
            if not failed_rows.empty:
                st.dataframe(failed_rows, width="stretch", hide_index=True)

    st.markdown("**Landmark validation history**")
    latest_runs, validation_history, review_metadata_available = (
        data.landmark_validation_views(dataset.get("landmark_validation"))
    )
    if validation_history.empty:
        st.info("No landmark-validation summary is available.")
    else:
        if review_metadata_available:
            st.caption("Trend shows the latest reviewed run per test frame; frames without a reviewed run use the latest recorded run.")
        else:
            st.warning("Review metadata is unavailable. The trend shows the latest recorded run per frame and labels it unreviewed; it is not ground truth.")
        trend_title = (
            "Latest reviewed run per test frame"
            if review_metadata_available else
            "Latest recorded run per test frame — unreviewed"
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
        axes[0].plot(latest_runs["test_frame"], latest_runs["median_error_px"],
                     marker="o", color=BURGUNDY, label="Median")
        axes[0].plot(latest_runs["test_frame"], latest_runs["max_error_px"],
                     marker="s", color=BLUE, label="Maximum")
        axes[0].set(title=trend_title, xlabel="Test frame", ylabel="Alignment error (pixels)")
        axes[0].grid(alpha=0.2)
        axes[0].legend()
        run_ids = validation_history["run_id"].drop_duplicates().tolist()
        run_labels = {value: f"R{index + 1:02d}" for index, value in enumerate(run_ids)}
        run_colors = {
            value: plt.get_cmap("tab20")(index % 20)
            for index, value in enumerate(run_ids)
        }
        for value, group in validation_history.groupby("run_id", sort=False):
            axes[1].scatter(
                group["test_frame"], group["median_error_px"],
                color=[run_colors[value]], marker="o", s=34,
                label=run_labels[value],
            )
            axes[1].scatter(
                group["test_frame"], group["max_error_px"],
                color=[run_colors[value]], marker="x", s=38,
            )
        axes[1].set(title="All historical runs · circles median, x maximum",
                    xlabel="Test frame", ylabel="Alignment error (pixels)")
        axes[1].grid(alpha=0.2)
        axes[1].legend(title="Run ID", loc="center left", bbox_to_anchor=(1.0, 0.5),
                       frameon=False, fontsize=7, title_fontsize=8)
        style_matplotlib_axes(fig, axes)
        st.pyplot(fig, clear_figure=True)
        st.caption("Runs remain separate. Chart labels R01, R02, etc. map to full run IDs below; status distinguishes latest, reviewed, superseded, and unreviewed measurements.")
        validation_history = validation_history.copy()
        validation_history.insert(
            0, "chart_id", validation_history["run_id"].map(run_labels)
        )
        display_columns = [
            name for name in (
                "chart_id", "run_id", "created_at", "test_frame", "landmark_count",
                "median_error_px", "max_error_px", "p90_error_px", "review_state",
            ) if name in validation_history.columns
        ]
        st.dataframe(validation_history[display_columns], width="stretch", hide_index=True)

    st.markdown("**Calibration and mapping quality**")
    if calibration:
        st.caption(
            f"Calibration `{calibration.get('calibration_id', 'ID unavailable')}` · "
            f"segment `{calibration.get('camera_segment_id', 'unknown')}` · "
            f"reference frame {calibration.get('reference_frame', 'unknown')}"
        )
        metric_columns([
            ("Status", str(calibration.get("status", "Unavailable")), "Saved calibration status."),
            ("Correspondences", f"{safe_int(calibration.get('correspondence_count')):,}", "Manually identified landmarks."),
            ("Median reprojection", f"{float(calibration_diag['median_reprojection_error_m']):.3f} assumed m" if calibration_diag.get("median_reprojection_error_m") is not None else "Unavailable", "Error on assumed pitch coordinate scale."),
            ("Held-out error", f"{float(calibration_diag['held_out_error_m']):.3f} assumed m" if calibration_diag.get("held_out_error_m") is not None else "Unavailable", "Held-out point error on assumed dimensions."),
        ])
        st.caption(
            f"Landmarks: {', '.join(map(str, calibration.get('landmark_ids', [])))} · "
            f"well distributed: {calibration_diag.get('well_distributed', 'unknown')} · "
            f"dimensions assumed: {calibration.get('pitch_dimensions_are_assumed', 'unknown')}"
        )
    else:
        st.info("A valid matching calibration file was not found.")
    if mapping_summary:
        st.caption(
            f"Mapping result: {safe_int(mapping_summary.get('successfully_mapped_observations')):,} valid · "
            f"{safe_int(mapping_summary.get('unmapped_observations')):,} unmapped · "
            f"{safe_int(mapping_summary.get('mapped_track_ids')):,} supporting track IDs."
        )
    with st.expander("Advanced provenance and raw diagnostics"):
        if detection_summary:
            st.markdown("**Detection summary**")
            st.json(detection_summary)
        if calibration:
            st.markdown("**Calibration metadata**")
            st.json(calibration)
        if mapping_summary:
            st.markdown("**Mapping summary**")
            st.json(mapping_summary)
        compensation_summary = dataset.get("compensation_summary") or {}
        if compensation_summary:
            st.markdown("**Camera compensation provenance**")
            st.json(compensation_summary)
        if raw is not None:
            raw_ids = raw.loc[raw["class_id"] == 0, ["run_id", "track_id"]].drop_duplicates()
            conf_ids = confirmed.loc[confirmed["class_id"] == 0, ["run_id", "track_id"]].drop_duplicates()
            st.write({
                "Raw person observations": raw_stats["observation_count"],
                "Raw provisional IDs": raw_stats["unique_track_count"],
                "Confirmed candidate observations": confirmed_stats["observation_count"],
                "Confirmed candidate IDs": confirmed_stats["unique_track_count"],
                "IDs excluded by confirmation filter": int(max(len(raw_ids) - len(conf_ids), 0)),
                "Identity caveat": "Tracker IDs can fragment or switch and do not verify real-world identity.",
            })
    show_not_validated()
