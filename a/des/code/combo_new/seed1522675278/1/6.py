import random

import numpy as np


BLACK = 0
ORANGE = 7
FRAME_COLORS = [1, 2, 3, 4, 6, 8, 9]


def _draw_frame(g, bbox, color, marker=None):
    r0, c0, r1, c1 = bbox
    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color
    if marker:
        side = marker
        if side == "top":
            g[r0, (c0 + c1) // 2] = ORANGE
        elif side == "bottom":
            g[r1, (c0 + c1) // 2] = ORANGE
        elif side == "left":
            g[(r0 + r1) // 2, c0] = ORANGE
        else:
            g[(r0 + r1) // 2, c1] = ORANGE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the orange side marker on each concentric frame and unfold the frames outward from the fixed core.

    Input structure:
    - Black background with a filled central core and several concentric one-cell rectangular frames.
    - Each frame has its own color.
    - One midpoint cell on each frame is replaced by orange.

    Rule:
    - The orange cell indicates which side that frame unfolds toward.
    - Starting from the core, place frames from inner to outer, each one cell away from the current structure in its marked direction.
    - Orange cells are restored to the frame color in the output.

    Key insight:
    - The frames are not resized or rotated; only their nesting relation is replaced by a directional layout.
    """
    g = np.asarray(grid, dtype=int)
    colors = [c for c in np.unique(g) if c not in (BLACK, ORANGE)]
    layers = []
    core = None
    for color in colors:
        pts = np.argwhere(g == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if (r1 - r0 + 1) * (c1 - c0 + 1) == len(pts):
            core = (r0, c0, r1, c1, int(color))
        else:
            side = None
            mids = [((r0, (c0 + c1) // 2), "top"), ((r1, (c0 + c1) // 2), "bottom"), (((r0 + r1) // 2, c0), "left"), (((r0 + r1) // 2, c1), "right")]
            for (r, c), name in mids:
                if g[r, c] == ORANGE:
                    side = name
                    break
            layers.append((r0, c0, r1, c1, int(color), side))
    layers.sort(key=lambda x: (x[2] - x[0], x[3] - x[1]))
    canvas = np.zeros((80, 80), dtype=int)
    cr0, cc0, cr1, cc1, core_color = core
    off_r = 40 - (cr1 - cr0 + 1) // 2
    off_c = 40 - (cc1 - cc0 + 1) // 2
    canvas[off_r : off_r + (cr1 - cr0 + 1), off_c : off_c + (cc1 - cc0 + 1)] = core_color
    cur = [off_r, off_c, off_r + (cr1 - cr0), off_c + (cc1 - cc0)]
    for r0, c0, r1, c1, color, side in layers:
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if side == "top":
            nr1 = cur[0] - 2
            nr0 = nr1 - hh + 1
            nc0 = cur[1] + (cur[3] - cur[1] + 1 - ww) // 2
        elif side == "bottom":
            nr0 = cur[2] + 2
            nr1 = nr0 + hh - 1
            nc0 = cur[1] + (cur[3] - cur[1] + 1 - ww) // 2
        elif side == "left":
            nc1 = cur[1] - 2
            nc0 = nc1 - ww + 1
            nr0 = cur[0] + (cur[2] - cur[0] + 1 - hh) // 2
        else:
            nc0 = cur[3] + 2
            nc1 = nc0 + ww - 1
            nr0 = cur[0] + (cur[2] - cur[0] + 1 - hh) // 2
        if side in ("top", "bottom"):
            nc1 = nc0 + ww - 1
        else:
            nr1 = nr0 + hh - 1
        _draw_frame(canvas, (nr0, nc0, nr1, nc1), color)
        cur = [min(cur[0], nr0), min(cur[1], nc0), max(cur[2], nr1), max(cur[3], nc1)]
    cropped = canvas[cur[0] : cur[2] + 1, cur[1] : cur[3] + 1]
    return cropped


def generate_input() -> np.ndarray:
    while True:
        layers = 3
        inner_h = random.randint(2, 3)
        inner_w = random.randint(2, 3)
        total_h = inner_h + 2 * layers * 2
        total_w = inner_w + 2 * layers * 2
        h = total_h + 6
        w = total_w + 6
        g = np.zeros((h, w), dtype=int)
        r0 = (h - total_h) // 2
        c0 = (w - total_w) // 2
        colors = random.sample(FRAME_COLORS, layers + 1)
        g[r0 + 2 * layers : r0 + 2 * layers + inner_h, c0 + 2 * layers : c0 + 2 * layers + inner_w] = colors[-1]
        sides = random.sample(["top", "bottom", "left", "right"], layers)
        for i in range(layers):
            pad = 2 * (layers - i)
            bbox = (r0 + pad - 1, c0 + pad - 1, r0 + total_h - pad, c0 + total_w - pad)
            _draw_frame(g, bbox, colors[i], sides[i])
        if max(main(g).shape) <= 30:
            return g
