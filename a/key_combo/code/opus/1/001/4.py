import random
from collections import deque

import numpy as np


BG = 5
TEMPLATE = 0
INDICATORS = [1, 2, 3, 4]
QORDER = ["TL", "TR", "BL", "BR"]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _sign(x: int) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def _quadrants(g):
    h, w = g.shape
    qh, qw = h // 2, w // 2
    return {
        "TL": (g[:qh, :qw], 0, 0),
        "TR": (g[:qh, qw:], 0, qw),
        "BL": (g[qh:, :qw], qh, 0),
        "BR": (g[qh:, qw:], qh, qw),
    }


def _tile_quadrant(sub: np.ndarray):
    comps = _components(sub == TEMPLATE)
    assert len(comps) == 1, "each quadrant must contain one black template"
    comp = comps[0]
    inds = [(int(r), int(c), int(sub[r, c])) for r, c in zip(*np.where((sub != BG) & (sub != TEMPLATE)))]
    assert len(inds) == 1, "each quadrant must contain one indicator"
    ir, ic, color = inds[0]
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    cr = int(round(sum(rs) / len(rs)))
    cc = int(round(sum(cs) / len(cs)))
    dr, dc = _sign(ir - cr), _sign(ic - cc)
    assert (dr, dc) != (0, 0)
    r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
    step_r = dr * (r1 - r0 + 1) if dr else 0
    step_c = dc * (c1 - c0 + 1) if dc else 0
    rel = [(r - r0, c - c0) for r, c in comp]
    canvas = np.full_like(sub, BG)
    count = 0
    ar, ac = r0, c0
    while True:
        cells = [(ar + rr, ac + cc) for rr, cc in rel]
        if any(r < 0 or c < 0 or r >= sub.shape[0] or c >= sub.shape[1] for r, c in cells):
            break
        for r, c in cells:
            canvas[r, c] = color
        count += 1
        ar += step_r
        ac += step_c
    return canvas, count


def _fold(name: str, canvas: np.ndarray) -> np.ndarray:
    if name == "TL":
        return canvas
    if name == "TR":
        return np.fliplr(canvas)
    if name == "BL":
        return np.flipud(canvas)
    return np.flipud(np.fliplr(canvas))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each quadrant template in its indicator direction, then fold quadrants with copy-count ordering.

    Input structure:
    - A grey even-sized grid is conceptually split into four equal quadrants.
    - Each quadrant contains one black connected template and one adjacent colored indicator pixel.
    - Indicator colors are distinct and templates remain inside their quadrant.

    Rule:
    - For each quadrant, infer the direction from the rounded template centroid toward the indicator.
    - Recolor the template with the indicator color and repeat it by its bounding-box step until the next copy would leave the quadrant.
    - Fold TR horizontally, BL vertically, and BR both ways into TL coordinates.
    - Overlay folded canvases from fewest copies to most copies, treating grey as transparent.

    Key insight:
    - The indicator direction controls not only the pattern but also the copy count, and that count determines layer priority.
    """
    g = np.asarray(grid, dtype=int)
    layers = []
    for name, (sub, _, _) in _quadrants(g).items():
        canvas, count = _tile_quadrant(sub)
        layers.append((count, QORDER.index(name), _fold(name, canvas)))
    layers.sort()
    out = np.full_like(layers[0][2], BG)
    for _, _, canvas in layers:
        mask = canvas != BG
        out[mask] = canvas[mask]
    return out


SHAPE = np.array([[1, 0], [1, 1]], dtype=bool)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _indicator_for(r0, c0, direction):
    dr, dc = direction
    if direction == (-1, 0):
        return r0 - 1, c0
    if direction == (1, 0):
        return r0 + 2, c0 + 1
    if direction == (0, -1):
        return r0 + 1, c0 - 1
    return r0 + 1, c0 + 2


def _copy_count(qh, qw, r0, c0, direction):
    dr, dc = direction
    step_r = dr * 2 if dr else 0
    step_c = dc * 2 if dc else 0
    count = 0
    ar, ac = r0, c0
    rel = list(zip(*np.where(SHAPE)))
    while all(0 <= ar + int(rr) < qh and 0 <= ac + int(cc) < qw for rr, cc in rel):
        count += 1
        ar += step_r
        ac += step_c
    return count


def _gen_attempt():
    qh = qw = random.randint(7, 9)
    grid = np.full((qh * 2, qw * 2), BG, dtype=int)
    colors = random.sample(INDICATORS, 4)
    used_counts = set()
    for (name, (_, roff, coff)), color in zip(_quadrants(grid).items(), colors):
        for _ in range(100):
            direction = random.choice(DIRS)
            r0 = random.randint(1, qh - 4)
            c0 = random.randint(1, qw - 4)
            ir, ic = _indicator_for(r0, c0, direction)
            if not (0 <= ir < qh and 0 <= ic < qw):
                continue
            count = _copy_count(qh, qw, r0, c0, direction)
            if count in used_counts or count < 1:
                continue
            used_counts.add(count)
            rr, cc = np.where(SHAPE)
            grid[roff + r0 + rr, coff + c0 + cc] = TEMPLATE
            grid[roff + ir, coff + ic] = color
            break
        else:
            return None
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
