import random
from collections import deque

import numpy as np

BLACK = 0
PAIR_COLORS = [1, 2, 3, 4, 5]
LEGEND_COLORS = [6, 7, 8, 9]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _triangle_info(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 != 1 or c1 - c0 != 1 or len(comp) != 3:
        raise ValueError("not a 2x2 triangle")
    cells = {(r - r0, c - c0) for r, c in comp}
    missing = list({(0, 0), (0, 1), (1, 0), (1, 1)} - cells)[0]
    open_dir = {
        (0, 0): "ul",
        (0, 1): "ur",
        (1, 0): "dl",
        (1, 1): "dr",
    }[missing]
    right_angle = {
        "ul": (r0 + 1, c0 + 1),
        "ur": (r0 + 1, c0),
        "dl": (r0, c0 + 1),
        "dr": (r0, c0),
    }[open_dir]
    return open_dir, right_angle


def _relation(a, b):
    if a == b:
        return "same"
    if {a, b} in ({"ul", "ur"}, {"dl", "dr"}):
        return "hmirror"
    if {a, b} in ({"ul", "dl"}, {"ur", "dr"}):
        return "vmirror"
    return "rot180"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color triangle pairs with an L-path whose color comes from the corner legend relation table.

    Input structure:
    - Black background with several same-color pairs of 3-cell right triangles.
    - Each triangle occupies a 2x2 box with exactly one missing corner; that missing corner defines its opening.
    - One corner of the grid contains a full 2x2 legend block whose four colors encode relation classes.

    Rule:
    - Pair the triangles by color and determine each triangle's opening direction and right-angle vertex.
    - Classify the pair as same opening, horizontal mirror, vertical mirror, or 180-degree rotation.
    - Read the corresponding line color from the legend's top-left, top-right, bottom-left, or bottom-right cell.
    - Draw a horizontal-then-vertical L-path between the two right-angle vertices without changing the triangles
      or the legend block itself.

    Key insight:
    - The triangles' own colors only establish the pairing; the line color is controlled entirely by the
      geometric relation between their openings through the 2x2 legend.
    """
    h, w = grid.shape
    corners = {
        "tl": (0, 0),
        "tr": (0, w - 2),
        "bl": (h - 2, 0),
        "br": (h - 2, w - 2),
    }
    legend_pos = None
    legend = None
    for _, (r0, c0) in corners.items():
        block = grid[r0 : r0 + 2, c0 : c0 + 2]
        if block.shape == (2, 2) and np.all(block != BLACK):
            legend_pos = (r0, c0)
            legend = {
                "same": int(block[0, 0]),
                "hmirror": int(block[0, 1]),
                "vmirror": int(block[1, 0]),
                "rot180": int(block[1, 1]),
            }
            break
    out = grid.copy()
    ignore = set()
    if legend_pos is not None:
        r0, c0 = legend_pos
        ignore = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}

    pairs = {}
    for color in np.unique(grid):
        if color == BLACK or color in LEGEND_COLORS:
            continue
        mask = (grid == color).copy()
        for r, c in ignore:
            mask[r, c] = False
        infos = []
        for comp in _components(mask):
            try:
                info = _triangle_info(comp)
            except ValueError:
                continue
            infos.append(info)
        if len(infos) == 2:
            pairs[int(color)] = infos

    for color, infos in pairs.items():
        (dir_a, pt_a), (dir_b, pt_b) = infos
        relation = _relation(dir_a, dir_b)
        line_color = legend[relation]
        r1, c1 = pt_a
        r2, c2 = pt_b
        step = 1 if c2 >= c1 else -1
        for cc in range(c1 + step, c2 + step, step):
            if out[r1, cc] == BLACK:
                out[r1, cc] = line_color
        step = 1 if r2 >= r1 else -1
        for rr in range(r1 + step, r2 + step, step):
            if out[rr, c2] == BLACK:
                out[rr, c2] = line_color
    return out


def _triangle_cells(top, left, open_dir):
    filled = {
        "ul": [(0, 1), (1, 0), (1, 1)],
        "ur": [(0, 0), (1, 0), (1, 1)],
        "dl": [(0, 0), (0, 1), (1, 1)],
        "dr": [(0, 0), (0, 1), (1, 0)],
    }[open_dir]
    return [(top + r, left + c) for r, c in filled]


def _right_angle(top, left, open_dir):
    return {
        "ul": (top + 1, left + 1),
        "ur": (top + 1, left),
        "dl": (top, left + 1),
        "dr": (top, left),
    }[open_dir]


def generate_input() -> np.ndarray:
    rng = random.Random()
    relation_pairs = {
        "same": [("ul", "ul"), ("ur", "ur"), ("dl", "dl"), ("dr", "dr")],
        "hmirror": [("ul", "ur"), ("dl", "dr")],
        "vmirror": [("ul", "dl"), ("ur", "dr")],
        "rot180": [("ul", "dr"), ("ur", "dl")],
    }
    for _ in range(500):
        h = rng.randint(18, 26)
        w = rng.randint(18, 26)
        grid = np.zeros((h, w), dtype=int)
        used = np.zeros((h, w), dtype=bool)

        corner = rng.choice(["tl", "tr", "bl", "br"])
        if corner == "tl":
            lr, lc = 0, 0
        elif corner == "tr":
            lr, lc = 0, w - 2
        elif corner == "bl":
            lr, lc = h - 2, 0
        else:
            lr, lc = h - 2, w - 2
        legend = rng.sample(LEGEND_COLORS, 4)
        block = np.array([[legend[0], legend[1]], [legend[2], legend[3]]], dtype=int)
        grid[lr : lr + 2, lc : lc + 2] = block
        used[max(0, lr - 1) : min(h, lr + 3), max(0, lc - 1) : min(w, lc + 3)] = True

        num_pairs = rng.randint(2, 4)
        colors = rng.sample(PAIR_COLORS, num_pairs)
        ok = True
        for color in colors:
            relation = rng.choice(list(relation_pairs))
            dir1, dir2 = rng.choice(relation_pairs[relation])
            placed = False
            for _ in range(400):
                t1r = rng.randint(1, h - 3)
                t1c = rng.randint(1, w - 3)
                cells1 = _triangle_cells(t1r, t1c, dir1)
                ra1 = _right_angle(t1r, t1c, dir1)
                area1 = used[max(0, t1r - 1) : min(h, t1r + 3), max(0, t1c - 1) : min(w, t1c + 3)]
                if area1.any():
                    continue
                t2r = rng.randint(1, h - 3)
                t2c = rng.randint(1, w - 3)
                cells2 = _triangle_cells(t2r, t2c, dir2)
                ra2 = _right_angle(t2r, t2c, dir2)
                area2 = used[max(0, t2r - 1) : min(h, t2r + 3), max(0, t2c - 1) : min(w, t2c + 3)]
                if area2.any():
                    continue
                if set(cells1) & set(cells2):
                    continue
                expanded_first = {
                    (r + dr, c + dc)
                    for r, c in cells1
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))
                }
                if any(cell in expanded_first for cell in cells2):
                    continue
                path = []
                step = 1 if ra2[1] >= ra1[1] else -1
                path.extend((ra1[0], cc) for cc in range(ra1[1] + step, ra2[1], step))
                step = 1 if ra2[0] >= ra1[0] else -1
                path.extend((rr, ra2[1]) for rr in range(ra1[0], ra2[0], step))
                if any(not (0 <= r < h and 0 <= c < w) for r, c in path):
                    continue
                if any(used[r, c] for r, c in path):
                    continue
                for r, c in cells1 + cells2:
                    grid[r, c] = color
                used[max(0, t1r - 1) : min(h, t1r + 3), max(0, t1c - 1) : min(w, t1c + 3)] = True
                used[max(0, t2r - 1) : min(h, t2r + 3), max(0, t2c - 1) : min(w, t2c + 3)] = True
                for r, c in path:
                    used[r, c] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
