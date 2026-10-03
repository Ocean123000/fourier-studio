# Interactive Drift Wallpaper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve the existing FFT and add seamless drifting motion, fading trails and a local interactive desktop UI.

**Architecture:** A deterministic wrapper integrates the instantaneous frequency and derives periodic radii/phase modulation. A periodic trail cache supplies pre-loop history. A Tkinter launcher and controls prepare models and export videos on workers; Tk widgets are updated only by the main event loop.

**Tech Stack:** Existing NumPy, OpenCV, svgelements and imageio-ffmpeg; standard-library Tkinter, threading and queue.

---

### Task 1: Dynamic motion and trail
- Create `dynamics.py`: seeded initial random model, radius/phase/frequency modulation, analytic frequency integration, derivative-safe viewport, cyclic cached history, age fade and thinning.
- Keep `main.py:FourierModel` unchanged.
- Test `tests/test_dynamics.py`: disabled drift parity, integrated speed, periodic values and derivatives, finite/nonnegative radii, random reproducibility, trail continuity.

### Task 2: Shared render path and command line
- Extend renderer with optional motion and trail; preserve original loop/draw modes.
- Add drift_strength, drift_period, trail_length, fade_power and glow_strength (hyphen and underscore CLI aliases).
- Add headless image/random modes, fixed integer carrier cycles and documented actual drift periods.
- Test existing `python -m unittest discover -s tests -v` plus drifting MP4 roundtrip.

### Task 3: Local GUI
- Create `gui.py` and `run.bat`; no-argument `python main.py` opens launcher.
- Start button opens controls: image picker, random seed/spectral parameters, all five requested controls, preview, pause/apply/regenerate, export with progress/cancellation.
- Initialize and export on workers, use queued results, prevent stale results and preserve responsive Tk loop.
- Test Tk widget invocations, preparation, preview, pause and short export via `tests/test_gui.py`.

### Task 4: Deliver
- Update README with GUI and headless usage, period quantization, periodic initial history and performance.
- Render and inspect PNG/MP4; verify full decode, dimensions, FPS, periodic frame state and boundary derivative continuity.
- No repository commit: this project currently has no Git checkout.
