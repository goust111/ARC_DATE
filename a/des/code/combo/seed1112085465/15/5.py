import random
from collections import Counter, deque

import numpy as np


BG = 8
GREY = 5
OTHER_COLORS = [0, 1, 2, 3, 4, 6, 7, 9]

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
LEGEND_ROWS_RANGE = (1, 4)
OBJECT_COUNT_RANGE = (2, 5)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _component_from_seed(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    q = deque([start])
    seen = {start}
    pts: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        pts.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return pts


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp = _component_from_seed(mask, (r, c))
            for rr, cc in comp:
                seen[rr, cc] = True
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    comp = _component_from_seed(mask, tuple(map(int, pts[0])))
    return len(comp) == int(mask.sum())


def _has_interior(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if all(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                return True
    return False


def _is_boundary_cell(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for nr, nc in _neighbors4(r, c):
        if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
            return True
    return False


def _find_legend_bbox(grid: np.ndarray, bg: int, grey: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    mask = grid != bg
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    candidates: list[tuple[int, int, int, int]] = []

    for start in corners:
        sr, sc = start
        if not mask[sr, sc]:
            continue
        comp = _component_from_seed(mask, start)
        r0, c0, r1, c1 = _bbox(comp)
        if c1 - c0 + 1 != 2:
            continue
        if len(comp) != (r1 - r0 + 1) * 2:
            continue
        touches_same_corner = (
            (start == (0, 0) and r0 == 0 and c0 == 0 and c1 == 1)
            or (start == (0, w - 1) and r0 == 0 and c0 == w - 2 and c1 == w - 1)
            or (start == (h - 1, 0) and r1 == h - 1 and c0 == 0 and c1 == 1)
            or (start == (h - 1, w - 1) and r1 == h - 1 and c0 == w - 2 and c1 == w - 1)
        )
        if not touches_same_corner:
            continue
        if int(np.count_nonzero(grid[r0 : r1 + 1, c0 : c1 + 1] == grey)) != 1:
            continue
        candidates.append((r0, c0, r1, c1))

    assert len(candidates) == 1, "expected exactly one corner legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object's boundary and interior using the legend pair that contains grey.

    The legend is a small corner block of horizontal 1x2 color pairs. The row containing
    grey defines (boundary_color, interior_color). The legend is erased in the output.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    r0, c0, r1, c1 = _find_legend_bbox(g, bg, GREY)
    legend = g[r0 : r1 + 1, c0 : c1 + 1]
    grey_rows = np.where((legend == GREY).any(axis=1))[0]
    assert len(grey_rows) == 1, "expected exactly one legend row containing grey"
    entry = legend[int(grey_rows[0])]
    c_border, c_interior = int(entry[0]), int(entry[1])

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    grey_mask = out == GREY

    for comp in _components(grey_mask):
        for rr, cc in comp:
            out[rr, cc] = c_border if _is_boundary_cell(grey_mask, rr, cc) else c_interior

    return out


def _random_polyomino(rng: random.Random) -> np.ndarray:
    target = rng.randint(4, 9)
    cells = {(0, 0)}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if -4 <= nr <= 4 and -4 <= nc <= 4 and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(rng.choice(frontier))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    cells = {(r - r0, c - c0) for r, c in cells}
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_thick_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 6)
    w = rng.randint(3, 6)
    mask = np.ones((h, w), dtype=bool)
    boundary = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
    rng.shuffle(boundary)
    removals = rng.randint(0, min(6, len(boundary) // 2))
    removed = 0
    for r, c in boundary:
        if removed >= removals:
            break
        trial = mask.copy()
        trial[r, c] = False
        if int(trial.sum()) < 5:
            continue
        if _is_connected(trial) and _has_interior(trial):
            mask = trial
            removed += 1
    return mask


def _build_legend(rng: random.Random, n_rows: int, grey_side: int) -> np.ndarray:
    legend = np.zeros((n_rows, 2), dtype=int)
    target_row = rng.randrange(n_rows)
    pool = rng.sample(OTHER_COLORS, 2 * n_rows - 1)
    idx = 0
    partner = pool[idx]
    idx += 1
    if grey_side == 0:
        legend[target_row] = [GREY, partner]
    else:
        legend[target_row] = [partner, GREY]
    for r in range(n_rows):
        if r == target_row:
            continue
        legend[r] = [pool[idx], pool[idx + 1]]
        idx += 2
    return legend


def _place_legend(grid: np.ndarray, legend: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    h, w = grid.shape
    lh, lw = legend.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - lw
    elif corner == "bl":
        r0, c0 = h - lh, 0
    else:
        r0, c0 = h - lh, w - lw
    grid[r0 : r0 + lh, c0 : c0 + lw] = legend
    return r0, c0, r0 + lh - 1, c0 + lw - 1


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    hh, ww = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(hh, r0 + h + 1)
    cc1 = min(ww, c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _find_positions(blocked: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = blocked.shape
    mh, mw = mask.shape
    positions = []
    for r0 in range(1, h - mh):
        for c0 in range(1, w - mw):
            if blocked[r0 - 1 : r0 + mh + 1, c0 - 1 : c0 + mw + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _render_expected(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    objects: list[tuple[int, int, np.ndarray]],
    c_border: int,
    c_interior: int,
) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = legend_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    for top, left, mask in objects:
        mh, mw = mask.shape
        for r in range(mh):
            for c in range(mw):
                if not mask[r, c]:
                    continue
                out[top + r, left + c] = c_border if _is_boundary_cell(mask, r, c) else c_interior
    return out


def _bg_is_dominant(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[BG] > max((n for color, n in counts.items() if color != BG), default=0)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_rows = rng.randint(*LEGEND_ROWS_RANGE)
    if h <= n_rows + 6 or w <= 8:
        return None

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    grey_side = rng.randint(0, 1)
    legend = _build_legend(rng, n_rows, grey_side)
    legend_bbox = _place_legend(grid, legend, rng.choice(["tl", "tr", "bl", "br"]))
    _mark_blocked(blocked, legend_bbox[0], legend_bbox[1], legend.shape[0], legend.shape[1])

    c_border, c_interior = int(legend[(legend == GREY).any(axis=1)][0, 0]), int(
        legend[(legend == GREY).any(axis=1)][0, 1]
    )

    n_objects = rng.randint(*OBJECT_COUNT_RANGE)
    objects: list[tuple[int, int, np.ndarray]] = []
    total_interior = 0

    for idx in range(n_objects):
        use_thick = idx == 0 or rng.random() < 0.65
        mask = _random_thick_shape(rng) if use_thick else _random_polyomino(rng)
        positions = _find_positions(blocked, mask)
        if not positions:
            return None
        top, left = rng.choice(positions)
        grid[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = GREY
        _mark_blocked(blocked, top, left, mask.shape[0], mask.shape[1])
        objects.append((top, left, mask))
        total_interior += sum(
            1 for r in range(mask.shape[0]) for c in range(mask.shape[1]) if mask[r, c] and not _is_boundary_cell(mask, r, c)
        )

    if total_interior == 0:
        return None
    if not _bg_is_dominant(grid):
        return None

    expected = _render_expected(grid, legend_bbox, objects, c_border, c_interior)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
