from pathlib import Path
import importlib.util
import random

import numpy as np


def _load_common():
    path = Path(__file__).with_name("_common.py")
    spec = importlib.util.spec_from_file_location("_arc057_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_common()


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor grey region cells by row controls unless covered by marker rectangles.

    Input structure:
    A connected grey region occupies the interior. Rows containing grey have
    left-column control colors. Same-color marker pairs inside the grey extent
    define axis-aligned diagonal rectangles.

    Rule:
    Each grey cell inside one or more marker rectangles takes the color of the
    smallest covering rectangle, with topmost pair as tie-break. Grey cells in
    no rectangle take that row's left control color. Marker pixels are erased;
    controls remain.

    Key insight:
    Two recoloring systems coexist, and marker rectangles have priority over
    row controls only where they geometrically cover grey cells.
    """
    g = np.array(grid, copy=True)
    out = g.copy()
    marker_colors = set(np.unique(g).tolist()) - {0, 5}
    # Controls are the left-column non-grey colors; repeated interior colors are markers.
    controls = {r: int(g[r, 0]) for r in range(g.shape[0]) if g[r, 0] not in (0, 5)}
    marker_colors -= set(controls.values())
    rects = []
    for col in marker_colors:
        pts = list(zip(*np.where(g == col)))
        pts = [(r, c) for r, c in pts if c != 0]
        if len(pts) >= 2:
            pts = sorted(pts)
            r0, c0 = pts[0]
            r1, c1 = pts[1]
            rr0, rr1 = sorted((r0, r1))
            cc0, cc1 = sorted((c0, c1))
            rects.append((col, rr0, cc0, rr1, cc1, (rr1 - rr0 + 1) * (cc1 - cc0 + 1), rr0))
    for r, c in zip(*np.where(g == 5)):
        covers = [x for x in rects if x[1] <= r <= x[3] and x[2] <= c <= x[4]]
        if covers:
            covers.sort(key=lambda x: (x[5], x[6]))
            out[r, c] = covers[0][0]
        else:
            out[r, c] = controls.get(r, 0)
    for col, r0, c0, r1, c1, _, _ in rects:
        for r, c in zip(*np.where(g == col)):
            if c != 0:
                out[r, c] = 0
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((12, 12), dtype=int)
    # Irregular grey blob.
    g[2:10, 3:9] = 5
    g[2, 3] = g[9, 8] = g[4, 8] = 0
    for r in range(2, 10):
        if np.any(g[r] == 5):
            g[r, 0] = random.choice([1, 2, 3])
    pairs = [(7, (3, 4), (7, 8)), (4, (5, 5), (8, 7))]
    if random.random() < 0.5:
        pairs.append((6, (2, 6), (4, 7)))
    for col, a, b in pairs:
        g[a] = col
        g[b] = col
    return g
