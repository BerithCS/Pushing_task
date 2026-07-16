#!/usr/bin/env python3
"""
Lightweight live tactile viewer for Test_pushing.py.
Only reloads the PNG when the file modification time changes —
no unnecessary disk reads, minimal CPU usage.

Run with system Python:
    python3 view_tactile_pushing.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PNG = "/home/berith/Documents/Pushing_task/data/tactile_s1.png"

# Match CELL_PX=50, MAP_W=4, MAP_H=7 → 200 x 350 px
IMG_W_IN = 200 / 100
IMG_H_IN = 350 / 100

plt.rcParams["toolbar"] = "None"
plt.ion()

fig = plt.figure(figsize=(IMG_W_IN, IMG_H_IN), dpi=100)
fig.patch.set_facecolor("#1e1e2e")
fig.canvas.manager.set_window_title("Sensor 1")
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
im = ax.imshow(np.zeros((350, 200, 3), dtype=np.uint8))
plt.show()

last_mtime = 0.0

while True:
    try:
        mtime = os.path.getmtime(PNG)
        if mtime != last_mtime:
            im.set_data(mpimg.imread(PNG))
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            last_mtime = mtime
    except Exception:
        pass

    plt.pause(0.05)   # just keeps the GUI event loop alive — cheap