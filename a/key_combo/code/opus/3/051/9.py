from pathlib import Path
import importlib.util
import random

import numpy as np


def _load_common():
    path = Path(__file__).with_name("_common.py")
    spec = importlib.util.spec_from_file_location("_arc051_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_common()


def _fill_panel(p):
    out = p.copy()
    ext = U.flood_exterior_not_wall(p, wall=2)
    out[(p == 0) & (~ext)] = 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Flood each red-wall panel locally, then overlay the panels with asymmetric priority.

    Input structure:
    A grey vertical separator splits the grid into two equal panels. Each panel
    contains red walls enclosing some local holes.

    Rule:
    Fill each panel's closed black regions with blue using that panel's own
    border as the flood boundary, remove the separator, and overlay aligned
    panels with priority right red, left blue, left red, right blue, then green
    when both aligned cells are black.

    Key insight:
    Flood-fill classification is panel-local, but the final answer is a
    cross-panel comparison using an intentionally asymmetric priority order.
    """
    g = np.array(grid, copy=True)
    sc = next(c for c in range(g.shape[1]) if np.all(g[:, c] == 5))
    left = _fill_panel(g[:, :sc])
    right = _fill_panel(g[:, sc + 1:])
    out = np.zeros_like(left)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            l, rr = int(left[r, c]), int(right[r, c])
            if rr == 2:
                out[r, c] = 2
            elif l == 1:
                out[r, c] = 1
            elif l == 2:
                out[r, c] = 2
            elif rr == 1:
                out[r, c] = 1
            elif l == 0 and rr == 0:
                out[r, c] = 3
            else:
                out[r, c] = l or rr
    return out


def generate_input() -> np.ndarray:
    ph, pw = random.choice([(11, 7), (12, 7), (11, 8)])
    g = np.zeros((ph, 2 * pw + 1), dtype=int)
    g[:, pw] = 5
    for off in (0, pw + 1):
        closed = random.choice([(1, 1, 4, 4), (2, 1, 6, 5), (1, 2, 5, min(pw - 2, 6))])
        U.draw_rect_border(g, closed[0], off + closed[1], closed[2], off + closed[3], 2)
        open_rect = random.choice([(6, 1, ph - 2, 4), (6, 2, ph - 2, min(pw - 2, 5))])
        U.draw_rect_border(g, open_rect[0], off + open_rect[1], open_rect[2], off + open_rect[3], 2)
        side = random.choice(["top", "left", "right"])
        if side == "top":
            g[open_rect[0], off + open_rect[1] + 1] = 0
        elif side == "left":
            g[open_rect[0] + 1, off + open_rect[1]] = 0
        else:
            g[open_rect[0] + 1, off + open_rect[3]] = 0
    return g
