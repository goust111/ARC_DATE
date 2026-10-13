import random

import numpy as np


BG = 0
SEP = 5
ARROW = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each separator-defined panel, replace the shape by the opposite-side complement of its edge projection and erase the arrow marker.

    Input structure:
    - Grey full rows and full columns partition the grid into equal rectangular panels.
    - Each panel contains one monochrome foreground shape and one red edge marker on a panel border.
    - The marker identifies one of the four projection directions.

    Rule:
    - Read the marker side as the projection direction.
    - Count shape pixels per column for vertical projection or per row for horizontal projection.
    - Replace the shape with the complementary counts on the opposite side of the panel: up projects to a bottom complement, down to a top complement, left to a right complement, and right to a left complement.
    - Remove the red marker and keep the grey separators unchanged.

    Key insight:
    - The arrow chooses a projection axis, and the output uses the complement against the panel dimension on the opposite side rather than a direct moved copy of the shape.
    """
    g = np.asarray(grid, dtype=int)
    out = np.where(g == SEP, SEP, BG)
    row_lines = [-1] + [r for r in range(g.shape[0]) if np.all(g[r, :] == SEP)] + [g.shape[0]]
    col_lines = [-1] + [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)] + [g.shape[1]]
    for ra, rb in zip(row_lines, row_lines[1:]):
        for ca, cb in zip(col_lines, col_lines[1:]):
            panel = g[ra + 1 : rb, ca + 1 : cb]
            if panel.size == 0:
                continue
            arrow_pos = np.argwhere(panel == ARROW)
            if len(arrow_pos) == 0:
                continue
            rr, cc = arrow_pos[0]
            direction = "up" if rr == 0 else "down" if rr == panel.shape[0] - 1 else "left" if cc == 0 else "right"
            colors = [int(v) for v in np.unique(panel) if v not in (BG, SEP, ARROW)]
            if not colors:
                continue
            color = colors[0]
            mask = panel == color
            if direction == "up":
                counts = mask.sum(axis=0)
                for c, cnt in enumerate(counts):
                    panel[panel[:, c] == color, c] = BG
                    panel[panel[:, c] == ARROW, c] = BG
                    if cnt < panel.shape[0]:
                        panel[panel.shape[0] - (panel.shape[0] - cnt) :, c] = color
            elif direction == "down":
                counts = mask.sum(axis=0)
                panel[mask] = BG
                for c, cnt in enumerate(counts):
                    if cnt < panel.shape[0]:
                        panel[: panel.shape[0] - cnt, c] = color
            elif direction == "left":
                counts = mask.sum(axis=1)
                panel[mask] = BG
                for r, cnt in enumerate(counts):
                    if cnt < panel.shape[1]:
                        panel[r, panel.shape[1] - (panel.shape[1] - cnt) :] = color
            else:
                counts = mask.sum(axis=1)
                panel[mask] = BG
                for r, cnt in enumerate(counts):
                    if cnt < panel.shape[1]:
                        panel[r, : panel.shape[1] - cnt] = color
            panel[arrow_pos[:, 0], arrow_pos[:, 1]] = BG
            out[ra + 1 : rb, ca + 1 : cb] = panel
    return out


SHAPES = [
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
]


def generate_input() -> np.ndarray:
    for _ in range(300):
        pr, pc = random.randint(2, 3), random.randint(2, 3)
        ph = pw = random.randint(5, 7)
        h = pr * ph + (pr - 1)
        w = pc * pw + (pc - 1)
        if h > 30 or w > 30:
            continue
        grid = np.full((h, w), BG, dtype=int)
        for r in range(ph, h, ph + 1):
            grid[r, :] = SEP
        for c in range(pw, w, pw + 1):
            grid[:, c] = SEP
        for i in range(pr):
            for j in range(pc):
                top = i * (ph + 1)
                left = j * (pw + 1)
                panel = grid[top : top + ph, left : left + pw]
                color = random.choice([1, 3, 4, 6, 7, 8, 9])
                shape = random.choice(SHAPES)
                sh, sw = shape.shape
                sr = random.randint(1, ph - sh - 1)
                sc = random.randint(1, pw - sw - 1)
                for r in range(sh):
                    for c in range(sw):
                        if shape[r, c]:
                            panel[sr + r, sc + c] = color
                direction = random.choice(["up", "down", "left", "right"])
                if direction == "up":
                    panel[0, pw // 2] = ARROW
                elif direction == "down":
                    panel[ph - 1, pw // 2] = ARROW
                elif direction == "left":
                    panel[ph // 2, 0] = ARROW
                else:
                    panel[ph // 2, pw - 1] = ARROW
        return grid
    raise RuntimeError("failed to generate input")
