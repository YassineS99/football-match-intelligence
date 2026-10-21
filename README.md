# Football Match Intelligence

A local Python computer-vision project for exploring football broadcast footage.
It combines object detection, ByteTrack, provisional jersey-color labels, hybrid
camera motion estimation, camera-compensated tracks, manual pitch calibration,
and a read-only Streamlit dashboard.

## Current demonstration and limitations

The checked-in project configuration defaults to the local `data` folder and
expects the source video and generated artifacts to be supplied locally. The
current verified tracking demonstration covers **100 frames at 25 FPS (4
seconds)**. It contains 1,584 confirmed person observations and 23 confirmed
candidate track IDs. These are tracker observations, not verified real-world
player identities or full-match statistics.

For the corresponding mapping run, 94 of 1,584 person observations (about
5.9%) have valid pitch coordinates, across three tracker IDs. The remaining
1,490 are outside the supported calibration region. The calibration uses an
assumed 105 × 68 m pitch model; the dimensions are not surveyed stadium
measurements, and the calibration hull covers only a small area. The dashboard
displays this warning and excludes invalid or extrapolated positions.

Team labels and goalkeeper/referee roles are provisional classifications from
jersey-color features. No club names are inferred. Camera-compensated positions
are segment-local; positions from different camera segments cannot be compared
without independent registration. Track IDs may fragment or switch.

Ball possession, passes, shots, expected goals, distance in meters, and speed
are **not yet validated**. They are not presented as measured statistics.

## Setup (Windows PowerShell)

Use Python 3.11 or a compatible version, then create and activate a virtual
environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your hardware if the default package
does not match your CUDA setup. `check_install.py` reports installed pipeline
versions. Download/provide the YOLOv8x weights and source video separately;
large model weights, input videos, and generated outputs are ignored by Git.
Confirm the applicable Ultralytics/model and dataset licenses before
redistributing them.

## Start the dashboard

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard is read-only. Use the sidebar to choose a confirmed tracking run
and an output directory. Project-relative defaults are `data\input` and
`data\output`. Set environment variables to point at other local folders:

The video page prefers the H.264 copies in `data\output\browser_videos` when
their source summaries match the selected tracking run, source video, and frame
range. The five converted videos remain tied to their original analysis
outputs; original annotated videos and the source video stay selectable as
fallbacks. Missing converted copies fall back to the matching original without
transcoding or rewriting any output. Landmark diagnostics preserve every
recorded run and separate a per-frame latest-run view from the historical
comparison. The current validation CSV contains no review metadata, so the
latest-run view is explicitly marked unreviewed rather than treated as ground
truth. Pitch plots shade only the saved calibration hull; all pitch dimensions
remain assumed unless independently verified.

```powershell
$env:FMI_DATA_DIR = "E:\football-data"
$env:FMI_INPUT_DIR = "E:\football-data\input"
$env:FMI_OUTPUT_DIR = "E:\football-data\output"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The selected tracking summary, team assignment, compensation, and mapping
artifacts are checked against tracking run UUID and source-video provenance.
Detection summaries are also matched by source video and exact processed frame
range so a full-video detection file is not mistaken for the tracking
demonstration's 100-frame subset. Incompatible derived runs are not joined.
Large CSV loads are cached within the Streamlit session; use **Rerun** after
replacing an artifact on disk.

## Pipeline architecture

1. `src\detect_objects.py` runs YOLO inference and writes raw detections.
2. `src\track_objects.py` applies ByteTrack and writes raw and confirmed track
   CSVs. IDs are scoped by a tracking-run UUID.
3. `src\classify_teams.py` aggregates upper-body jersey-color observations into
   provisional role/team assignments, joined by `(run_id, track_id)`.
4. `src\hybrid_camera.py` estimates camera transitions. Camera transforms and
   quality output are preserved independently from tracking.
5. `src\compensate_player_tracks.py` maps player foot points into continuous,
   segment-local camera-reference coordinates and does not bridge failed
   transitions.
6. `src\calibrate_pitch.py` manually calibrates named pitch landmarks and maps
   compensated positions only within the supported calibration region.
7. `src\extend_pitch_calibration.py` supports manual landmark observations
   from multiple frames after validating their camera-transform provenance.
8. `app.py` and `src\dashboard_data.py` read compatible outputs and provide
   exploratory analytics; they do not execute inference or modify outputs.

The app defaults to the `data\output\tracking_confirmed_test` run and searches
for compatible team, compensation, and mapping results. Pick another run from
the sidebar to inspect another available tracking run.

## Example commands

Run focused dashboard and calibration tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s src -p test_dashboard_data.py -v
.\.venv\Scripts\python.exe -m unittest discover -s src -p test_calibrate_pitch.py -v
.\.venv\Scripts\python.exe -m unittest discover -s src -p test_compensate_player_tracks.py -v
```

Start supported upstream processing steps only when you explicitly intend to
create new outputs. Each script has `--help`; use a new output directory to
preserve historical runs. Example:

```powershell
.\.venv\Scripts\python.exe src\track_objects.py --help
.\.venv\Scripts\python.exe src\classify_teams.py --help
.\.venv\Scripts\python.exe src\compensate_player_tracks.py --help
.\.venv\Scripts\python.exe src\calibrate_pitch.py --help
```

## Troubleshooting

- **No tracking runs listed:** ensure the selected output directory contains a
  run folder with `tracking_summary.json` and `tracks_confirmed.csv`.
- **Derived results unavailable:** check their summary JSON files, tracking
  UUIDs, source video paths, and input CSV provenance.
- **Missing output warning:** the dashboard continues with available compatible
  data and labels unavailable pages instead of fabricating values.
- **Video does not play:** the dashboard selects the browser-compatible H.264
  copy when present; verify browser codec support or try the original/source
  fallback. The dashboard does not transcode or rewrite files.
- **Low pitch-mapping coverage:** inspect the calibration diagnostic and add
  manually verified, well-distributed landmarks. Do not relax the calibration
  hull or accept extrapolated positions as validated data.
- **Unexpected camera segment:** inspect `camera_quality_hybrid.csv` and
  compensation segment metadata. Segment-local coordinates must not be joined
  across failed transitions.

## Data and model provenance

The existing detection summary records the local YOLOv8x weights path, COCO
class IDs for `person` and `sports ball`, confidence threshold, inference
device, and source-video metadata. Tracking, team, compensation, and mapping
summaries record run IDs and source files. The sample video and model weights
are local project inputs, not redistributed by this documentation. Results
reflect this specific short clip and pipeline configuration only.
