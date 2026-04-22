import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches

# Parse the SVG Path
# M11.2966 4.12501
# C11.5853 3.62501 12.307 3.62501 12.5957 4.12501
# L21.2787 19.1645
# C21.638 19.7867 21.0013 20.5092 20.3388 20.231
# L12.2365 16.8289
# C12.0508 16.751 11.8415 16.751 11.6558 16.8289
# L3.55346 20.231
# C2.89106 20.5092 2.25435 19.7867 2.61357 19.1645
# L11.2966 4.12501 Z

Path = mpath.Path
verts = [
    (11.2966, 4.12501), # M
    (11.5853, 3.62501), (12.307, 3.62501), (12.5957, 4.12501), # C (Apex)
    (21.2787, 19.1645), # L (Right outer)
    (21.638, 19.7867), (21.0013, 20.5092), (20.3388, 20.231), # C (Right tip)
    (12.2365, 16.8289), # L (Right inner notch)
    (12.0508, 16.751), (11.8415, 16.751), (11.6558, 16.8289), # C (Center notch)
    (3.55346, 20.231),  # L (Left inner notch)
    (2.89106, 20.5092), (2.25435, 19.7867), (2.61357, 19.1645), # C (Left tip)
    (11.2966, 4.12501), # L (Left outer)
    (0, 0) # CLOSE
]
codes = [
    Path.MOVETO,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.LINETO,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.LINETO,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.LINETO,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.LINETO,
    Path.CLOSEPOLY
]
path = mpath.Path(verts, codes)

# Rasterize
COLS = 44
ROWS = 22 # 22 terminal rows = 44 vertical pixels for halfblocks
grid = np.zeros((ROWS * 2, COLS), dtype=bool)

for row in range(ROWS * 2):
    for col in range(COLS):
        # SVG is 24x24. Map col,row to 0-24
        # Add a slight margin (pad 10%)
        px = (col / (COLS - 1)) * 26 - 1
        py = (row / (ROWS * 2 - 1)) * 26 - 1
        grid[row, col] = path.contains_point((px, py))

# Generate half blocks
lines = []
for row in range(ROWS):
    s = ""
    for col in range(COLS):
        top = grid[row*2, col]
        bot = grid[row*2+1, col]
        if top and bot: s += "█"
        elif top: s += "▀"
        elif bot: s += "▄"
        else: s += " "
    lines.append(s)

with open("/tmp/logo.txt", "w", encoding="utf-8") as f:
    for l in lines:
        # Strip trailing spaces for cleanliness
        f.write("        " + l.rstrip() + "\n")
print('Logo generated to /tmp/logo.txt')
