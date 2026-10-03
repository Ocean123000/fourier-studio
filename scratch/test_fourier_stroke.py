"""
Test Fourier epicycles on single-line stroke order trajectory
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from main import FourierModel, Renderer, resample_closed

# Load continuous loop points
from scratch.test_stroke_pipeline import continuous_loop

# Convert to complex numbers
pts = continuous_loop.copy()
# Normalize scale and center
z = pts[:, 0] - 1j * pts[:, 1]
z -= z.mean()
extent = max(float(np.ptp(z.real)), float(np.ptp(z.imag)))
z = z * (2 / extent)

model = FourierModel(z, 150, "radius")
error = np.sqrt(np.mean(np.abs(model.curve(np.arange(len(z)) / len(z)) - z) ** 2))
print(f"Fourier model: 150 circles, RMSE: {error:.6f}")

renderer = Renderer(model, 1920, 1080, "loop", ring_width=2.0, ring_opacity=0.85, ticks=12)
frame0 = renderer.frame(0.2) # mid-way

import cv2
cv2.imwrite("scratch/fourier_stroke_preview.png", frame0)
print("Saved scratch/fourier_stroke_preview.png")
