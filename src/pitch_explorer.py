"""Read-only, run-isolated, calibration-bounded pitch exploration.

Plotly graphics are presentation only. Never infer absent observations, interpolate
across unmapped frames, or concatenate disconnected camera reference segments.
"""

import math

import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
except ImportError:  # Plotly is optional outside the interactive pitch page.
    go = None


ROLE_COLORS = {
    "team_a_outfield": "#ae3051", "team_b_outfield": "#51a6e9",
    "goalkeeper_candidate": "#51cf84", "referee_candidate": "#f2cf63",
    "unknown": "#adb5bd",
}


def _valid_boolean(column):
    return column.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def safe_pitch_geometry(mapping_summary):
    """Use recorded assumed dimensions; never silently expand them."""
    geometry = (mapping_summary or {}).get("pitch_geometry") or {}
    length = float(geometry.get("length_m", 105))
    width = float(geometry.get("width_m", 68))
    if not (math.isfinite(length) and math.isfinite(width) and 20 <= length <= 150 and 15 <= width <= 100):
        raise ValueError("Invalid pitch dimensions in mapping summary")
    return length, width


def valid_run_positions(mapping, *, run_id, segment_id, length, width,
                        roles=None, track_ids=None, start_frame=None, end_frame=None):
    """Reject mixed-run inputs; keep only original explicitly valid person samples."""
    if not isinstance(run_id, str) or not run_id or not isinstance(segment_id, str) or not segment_id:
        raise ValueError("A tracking run and camera segment are required")
    if mapping is None or mapping.empty:
        return pd.DataFrame(columns=["run_id", "frame", "track_id", "camera_segment_id",
                                     "pitch_x_m", "pitch_y_m", "predicted_role"])
    required = {"run_id", "frame", "track_id", "class_id", "camera_segment_id",
                "pitch_mapping_valid", "pitch_x_m", "pitch_y_m"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Missing pitch-mapping columns: {', '.join(sorted(missing))}")
    found_runs = set(mapping["run_id"].dropna().astype(str).unique())
    if found_runs != {run_id}:
        raise ValueError("Pitch mapping contains multiple or mismatched tracking run IDs")
    x = pd.to_numeric(mapping["pitch_x_m"], errors="coerce")
    y = pd.to_numeric(mapping["pitch_y_m"], errors="coerce")
    frames = pd.to_numeric(mapping["frame"], errors="coerce")
    classes = pd.to_numeric(mapping["class_id"], errors="coerce")
    selected = (
        _valid_boolean(mapping["pitch_mapping_valid"])
        & classes.eq(0)
        & mapping["camera_segment_id"].astype(str).eq(segment_id)
        & np.isfinite(x) & np.isfinite(y) & x.between(0, length) & y.between(0, width)
        & np.isfinite(frames) & frames.ge(0) & frames.eq(np.floor(frames))
    )
    if start_frame is not None:
        selected &= frames.ge(start_frame)
    if end_frame is not None:
        selected &= frames.le(end_frame)
    if roles is not None:
        roles_column = mapping.get("predicted_role", pd.Series("unknown", index=mapping.index))
        selected &= roles_column.fillna("unknown").astype(str).isin([str(role) for role in roles])
    if track_ids is not None:
        selected &= mapping["track_id"].astype(str).isin([str(track) for track in track_ids])
    rows = mapping.loc[selected].copy()
    rows["pitch_x_m"] = x.loc[selected].astype(float)
    rows["pitch_y_m"] = y.loc[selected].astype(float)
    rows["frame"] = frames.loc[selected].astype(int)
    rows["track_id"] = rows["track_id"].astype(str)
    if "predicted_role" not in rows:
        rows["predicted_role"] = "unknown"
    rows["predicted_role"] = rows["predicted_role"].fillna("unknown").astype(str)
    if rows.duplicated(["run_id", "camera_segment_id", "frame", "track_id"]).any():
        raise ValueError("Duplicate run/segment/frame/track position samples")
    return rows.sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)


def supported_hull(calibration, *, camera_segment_id, length, width):
    """Use only a verified dataset calibration for the requested camera segment."""
    if not calibration or calibration.get("valid_for_mapping") is not True:
        return None
    if calibration.get("camera_segment_id") != camera_segment_id:
        return None
    points = calibration.get("pitch_calibration_hull")
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        return None
    try:
        polygon = np.asarray(points, dtype=float)
    except (ValueError, TypeError):
        return None
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
        return None
    if (polygon[:, 0] < 0).any() or (polygon[:, 0] > length).any() or (polygon[:, 1] < 0).any() or (polygon[:, 1] > width).any():
        return None
    area = abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
               - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))) / 2
    if area <= 1e-8:
        return None
    return polygon


def continuous_trajectories(rows):
    """Split a track whenever a mapped source frame is absent; never bridge gaps."""
    if rows is None or rows.empty:
        return []
    result = []
    for (run, segment, track_id), track in rows.groupby(
        ["run_id", "camera_segment_id", "track_id"], sort=False
    ):
        track = track.sort_values("frame", kind="stable")
        # Duplicate frame keys are rejected at ingestion, not interpreted as a jump.
        if track["frame"].duplicated().any():
            raise ValueError("Duplicate trajectory frames")
        groups = track["frame"].diff().fillna(1).ne(1).cumsum()
        for _, piece in track.groupby(groups, sort=False):
            result.append({"run_id": run, "segment_id": segment, "track_id": str(track_id),
                           "role": str(piece.iloc[0]["predicted_role"]), "rows": piece.copy()})
    return result


def _pitch_figure(length, width, title):
    fig = go.Figure()
    # The surrounding canvas is dark; grass and stripes are clipped to the
    # *assumed* pitch rectangle, never used as a calibration/coverage signal.
    fig.add_shape(type="rect", x0=0, y0=0, x1=length, y1=width,
                  line=dict(width=0), fillcolor="#115441", layer="below")
    for index in range(10):
        if index % 2 == 0:
            fig.add_shape(type="rect", x0=length * index / 10, y0=0,
                          x1=length * (index + 1) / 10, y1=width,
                          line=dict(width=0), fillcolor="rgba(155,232,184,0.055)",
                          layer="below")
    # Field markings are assumed reference geometry, NOT detected/calibrated evidence.
    markings = [
        dict(type="rect", x0=0, y0=0, x1=length, y1=width),
        dict(type="line", x0=length / 2, y0=0, x1=length / 2, y1=width),
        dict(type="circle", x0=length / 2 - 9.15, y0=width / 2 - 9.15,
             x1=length / 2 + 9.15, y1=width / 2 + 9.15),
    ]
    for x0, x1 in ((0, 16.5), (length - 16.5, length)):
        markings.append(dict(type="rect", x0=x0, x1=x1,
                             y0=(width - 40.32) / 2, y1=(width + 40.32) / 2))
    for x0, x1 in ((0, 5.5), (length - 5.5, length)):
        markings.append(dict(type="rect", x0=x0, x1=x1,
                             y0=(width - 18.32) / 2, y1=(width + 18.32) / 2))
    for mark in markings:
        fig.add_shape(**mark, line=dict(color="rgba(245,255,245,0.80)", width=1.35))
    # Penalty arcs are reference-field markings, not inferred camera geometry.
    radius = 9.15
    arc_half_angle = math.acos(5.5 / radius)
    angles = np.linspace(-arc_half_angle, arc_half_angle, 45)
    for goal_x, sign in ((11, 1), (length - 11, -1)):
        fig.add_trace(go.Scatter(
            x=goal_x + sign * radius * np.cos(angles),
            y=width / 2 + radius * np.sin(angles),
            mode="lines", line=dict(color="rgba(245,255,245,0.80)", width=1.35),
            hoverinfo="skip", showlegend=False,
        ))
    for x in (11, length - 11, length / 2):
        fig.add_trace(go.Scatter(x=[x], y=[width / 2], mode="markers",
                                 marker=dict(size=5, color="#ffffff"),
                                 hoverinfo="skip", showlegend=False))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e6f7f2"), x=0.02),
        paper_bgcolor="#101c2b", plot_bgcolor="#0b1724",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#b4c9d8", size=12),
        margin=dict(l=28, r=25, t=55, b=65), height=475,
        hoverlabel=dict(bgcolor="#17283a", font_color="#f3f7ff"),
        legend=dict(orientation="h", y=-0.17, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(range=[-2, length + 2], showgrid=False, zeroline=False,
                   showline=False, fixedrange=False,
                   title="Assumed pitch x (m)", color="#9fb1c0", constrain="domain"),
        yaxis=dict(range=[width + 2, -2], scaleanchor="x", scaleratio=1,
                   showgrid=False, zeroline=False, title="Assumed pitch y (m)",
                   color="#9fb1c0", constrain="domain"),
    )
    return fig


def _add_hull(fig, hull):
    if hull is None:
        return
    closed = np.vstack([hull, hull[0]])
    fig.add_trace(go.Scatter(
        x=closed[:, 0], y=closed[:, 1], mode="lines", fill="toself",
        line=dict(color="#f7e187", width=3), fillcolor="rgba(246,225,135,0.25)",
        name="Verified calibration hull", hoverinfo="skip",
    ))


def _add_points(fig, points, *, opacity=0.88, label_suffix=""):
    if points is None or points.empty:
        return
    for role, role_rows in points.groupby("predicted_role", sort=True):
        custom = list(zip(role_rows["track_id"].astype(str), role_rows["frame"].astype(int)))
        fig.add_trace(go.Scattergl(
            x=role_rows["pitch_x_m"], y=role_rows["pitch_y_m"], mode="markers",
            customdata=custom,
            hovertemplate="Track %{customdata[0]} · frame %{customdata[1]}<extra></extra>",
            marker=dict(size=8, color=ROLE_COLORS.get(role, ROLE_COLORS["unknown"]),
                        opacity=opacity, line=dict(color="#ffffff", width=0.3)),
            name=f"{role.replace('_', ' ')}{label_suffix}",
        ))


def make_pitch_figure(rows, *, length, width, segment_id, mode,
                      frame_number=None, hull=None):
    """Plotly rendering: raw valid samples only, trajectories split at every gap."""
    if go is None:
        raise ModuleNotFoundError("Interactive pitch explorer requires Plotly", name="plotly")
    labels = {"frame": "Frame positions", "trajectories": "Contiguous mapped trajectories",
              "samples": "Mapped observation density (no interpolation)"}
    if mode not in labels:
        raise ValueError("Unknown pitch visualization mode")
    fig = _pitch_figure(length, width, f"{labels[mode]} · {segment_id}")
    _add_hull(fig, hull)
    if mode == "frame":
        if frame_number is None:
            raise ValueError("Frame number required for a snapshot")
        _add_points(fig, rows.loc[rows["frame"].eq(frame_number)])
    elif mode == "samples":
        # This is intentionally an overlay of *observed valid points*, not a
        # smoothed raster: density interpolation could imply uncalibrated areas.
        _add_points(fig, rows, opacity=0.24)
    else:
        used_roles = set()
        for segment in continuous_trajectories(rows):
            points = segment["rows"]
            if len(points) < 2:
                continue
            role = segment["role"]
            used_roles.add(role)
            fig.add_trace(go.Scattergl(
                x=points["pitch_x_m"], y=points["pitch_y_m"],
                mode="lines+markers", marker=dict(size=5),
                line=dict(color=ROLE_COLORS.get(role, ROLE_COLORS["unknown"]), width=2),
                name=f"Track {segment['track_id']} · {role.replace('_', ' ')}",
                showlegend=False,  # Role-only legend below; track IDs remain in hover.
                customdata=list(zip(points["track_id"].astype(str), points["frame"].astype(int))),
                hovertemplate="Track %{customdata[0]} · frame %{customdata[1]}<extra></extra>",
            ))
        # Also show singleton samples that cannot form a supported trajectory.
        singles = [segment["rows"] for segment in continuous_trajectories(rows)
                   if len(segment["rows"]) == 1]
        if singles:
            isolated = pd.concat(singles, ignore_index=True)
            used_roles.update(isolated["predicted_role"].dropna().astype(str).unique())
            # No fabricated positions: these are the original singleton samples.
            _add_points(fig, isolated, opacity=0.65,
                        label_suffix=" · isolated sample")
        # Compact legend: one key per provisional role, not one per fragment.
        # Empty legend traces never introduce coordinates or synthesized evidence.
        for role in sorted(used_roles):
            fig.add_trace(go.Scatter(
                x=[], y=[], mode="markers", showlegend=True,
                marker=dict(size=10, color=ROLE_COLORS.get(role, ROLE_COLORS["unknown"])),
                name=role.replace("_", " "), hoverinfo="skip",
            ))
    return fig
