import random
from collections import Counter, deque

import numpy as np


BG = 0
MAROON = 9
DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
GEN_RETRIES = 300
PLACEMENT_RETRIES = 120


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _follow_cycle(start: tuple[int, int], first: tuple[int, int], pts: set[tuple[int, int]]) -> list[tuple[int, int]]:
    order = [start]
    prev = start
    cur = first
    while True:
        order.append(cur)
        nbrs = [n for n in _neighbors4(*cur) if n in pts]
        assert len(nbrs) == 2, "every loop pixel must have degree two"
        nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
        prev, cur = cur, nxt
        if cur == start:
            break
        assert len(order) <= len(pts), "cycle walk exceeded component size"
    assert len(order) == len(pts), "cycle order must visit every loop pixel exactly once"
    return order


def _signed_area(order: list[tuple[int, int]]) -> int:
    area = 0
    for (r1, c1), (r2, c2) in zip(order, order[1:] + order[:1]):
        area += c1 * r2 - c2 * r1
    return area


def _clockwise_cycle(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    for p in pts:
        deg = sum((n in pts) for n in _neighbors4(*p))
        assert deg == 2, "component must be a simple 4-connected cycle"
    start = min(pts)
    nbrs = sorted(n for n in _neighbors4(*start) if n in pts)
    assert len(nbrs) == 2, "cycle start must have two neighbors"
    order = _follow_cycle(start, nbrs[0], pts)
    if _signed_area(order) < 0:
        order = [start] + list(reversed(order[1:]))
    assert _signed_area(order) > 0, "clockwise traversal must have positive signed area"
    return order


def _interior_mask(points: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    for r, c in points:
        mask[r, c] = True
    r0 = max(min(rows) - 1, 0)
    r1 = min(max(rows) + 1, h - 1)
    c0 = max(min(cols) - 1, 0)
    c1 = min(max(cols) + 1, w - 1)
    block = mask[r0 : r1 + 1, c0 : c1 + 1]
    outside = np.zeros_like(block, dtype=bool)
    q = deque()
    hh, ww = block.shape
    border = []
    for rr in range(hh):
        border.append((rr, 0))
        border.append((rr, ww - 1))
    for cc in range(ww):
        border.append((0, cc))
        border.append((hh - 1, cc))
    for rr, cc in border:
        if not block[rr, cc] and not outside[rr, cc]:
            outside[rr, cc] = True
            q.append((rr, cc))
    while q:
        rr, cc = q.popleft()
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < hh and 0 <= nc < ww and not block[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    interior = np.zeros((h, w), dtype=bool)
    interior[r0 : r1 + 1, c0 : c1 + 1] = (~block) & (~outside)
    return interior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each track's beads clockwise by the number of maroon markers inside that closed loop.

    Input structure:
    - The background is black.
    - Each object is a thin 4-connected closed loop whose pixels are mostly one track color, with a few single-pixel beads replacing some loop cells.
    - Maroon marker pixels lie strictly inside a loop, never touching the loop, and different loops do not overlap.

    Rule:
    - Identify each loop as one connected component of all non-black, non-maroon pixels.
    - For each loop, count the maroon pixels in its enclosed interior region to get a step count N.
    - Order the loop pixels by starting at the topmost-leftmost loop cell and traversing clockwise.
    - Move every bead on that loop forward by N positions along the clockwise cycle; vacated cells revert to the track color and maroon markers stay unchanged.

    Key insight:
    - The maroon pixels do not move; they only encode how far the differently colored loop beads should rotate around their enclosing cycle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    loop_mask = (g != BG) & (g != MAROON)
    for comp in _components(loop_mask):
        order = _clockwise_cycle(comp)
        interior = _interior_mask(comp, (h, w))
        steps = int(np.count_nonzero(interior & (g == MAROON)))
        colors = Counter(int(g[r, c]) for r, c in order)
        track_color, _ = max(colors.items(), key=lambda kv: kv[1])
        beads = [(idx, int(g[r, c])) for idx, (r, c) in enumerate(order) if int(g[r, c]) != track_color]

        for r, c in order:
            out[r, c] = track_color
        n = len(order)
        for idx, color in beads:
            rr, cc = order[(idx + steps) % n]
            out[rr, cc] = color

    return out


def _rasterize_cycle(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(vertices, vertices[1:] + vertices[:1]):
        dr = 0 if r1 == r0 else (1 if r1 > r0 else -1)
        dc = 0 if c1 == c0 else (1 if c1 > c0 else -1)
        steps = abs(r1 - r0) + abs(c1 - c0)
        assert steps > 0 and (dr == 0) != (dc == 0), "cycle edges must be axis-aligned"
        for k in range(steps):
            order.append((r0 + dr * k, c0 + dc * k))
    assert len(order) == len(set(order)), "cycle rasterization must not self-intersect"
    return order


def _rotate_order(order: list[tuple[int, int]], h: int, w: int, k: int) -> tuple[list[tuple[int, int]], int, int]:
    pts = list(order)
    hh, ww = h, w
    for _ in range(k % 4):
        pts = [(c, hh - 1 - r) for r, c in pts]
        hh, ww = ww, hh
    return pts, hh, ww


def _rectangle_loop(rng: random.Random) -> tuple[list[tuple[int, int]], int, int]:
    h = rng.randint(5, 10)
    w = rng.randint(5, 11)
    order = _rasterize_cycle([(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)])
    return order, h, w


def _notched_loop(rng: random.Random) -> tuple[list[tuple[int, int]], int, int]:
    h = rng.randint(7, 12)
    w = rng.randint(7, 12)
    notch_top = rng.randint(2, h - 5)
    notch_bottom = rng.randint(notch_top + 2, h - 3)
    notch_depth = rng.randint(2, w - 3)
    order = _rasterize_cycle(
        [
            (0, 0),
            (0, w - 1),
            (h - 1, w - 1),
            (h - 1, 0),
            (notch_bottom, 0),
            (notch_bottom, notch_depth),
            (notch_top, notch_depth),
            (notch_top, 0),
        ]
    )
    return _rotate_order(order, h, w, rng.randrange(4))


def _sample_non_adjacent_positions(rng: random.Random, length: int, count: int) -> list[int]:
    for _ in range(80):
        order = list(range(length))
        rng.shuffle(order)
        chosen: list[int] = []
        banned: set[int] = set()
        for idx in order:
            if idx in banned:
                continue
            chosen.append(idx)
            banned.add(idx)
            banned.add((idx - 1) % length)
            banned.add((idx + 1) % length)
            if len(chosen) == count:
                return sorted(chosen)
    raise RuntimeError("failed to sample non-adjacent bead positions")


def _build_patch(
    rng: random.Random,
    track_color: int,
    bead_colors: list[int],
) -> np.ndarray:
    if rng.random() < 0.5:
        order, h, w = _rectangle_loop(rng)
    else:
        order, h, w = _notched_loop(rng)

    patch = np.zeros((h, w), dtype=int)
    for r, c in order:
        patch[r, c] = track_color

    interior = _interior_mask(order, patch.shape)
    interior_cells = [(int(r), int(c)) for r, c in np.argwhere(interior)]
    assert interior_cells, "loop must enclose at least one cell"

    max_markers = min(6, max(1, len(interior_cells) // 2))
    marker_count = rng.randint(1, max_markers)
    for r, c in rng.sample(interior_cells, marker_count):
        patch[r, c] = MAROON

    max_beads = min(4, max(1, len(order) // 5))
    bead_count = rng.randint(1, max_beads)
    bead_positions = _sample_non_adjacent_positions(rng, len(order), bead_count)
    if bead_count <= len(bead_colors):
        colors = rng.sample(bead_colors, bead_count)
    else:
        colors = [rng.choice(bead_colors) for _ in range(bead_count)]
    for idx, color in zip(bead_positions, colors):
        r, c = order[idx]
        patch[r, c] = color

    cycle_points = [(int(r), int(c)) for r, c in np.argwhere((patch != BG) & (patch != MAROON))]
    assert len(cycle_points) == len(order)
    assert _clockwise_cycle(cycle_points)
    return patch


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    rr0 = max(r0 - 1, 0)
    cc0 = max(c0 - 1, 0)
    rr1 = min(r0 + ph, h - 1)
    cc1 = min(c0 + pw, w - 1)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _stamp(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    sub = patch != BG
    window = grid[r0 : r0 + ph, c0 : c0 + pw]
    window[sub] = patch[sub]
    rr0 = max(r0 - 1, 0)
    cc0 = max(c0 - 1, 0)
    rr1 = min(r0 + ph, grid.shape[0] - 1)
    cc1 = min(c0 + pw, grid.shape[1] - 1)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 30)
    w = rng.randint(16, 30)
    loop_count = rng.choice([1, 2, 3])
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    track_colors = rng.sample(COLOR_POOL, loop_count)
    bead_palette = [c for c in COLOR_POOL if c not in track_colors]
    if not bead_palette:
        bead_palette = COLOR_POOL[:]

    for track_color in track_colors:
        patch = _build_patch(rng, track_color, bead_palette)
        ph, pw = patch.shape
        if ph + 2 > h or pw + 2 > w:
            return None
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(1, h - ph - 1)
            c0 = rng.randint(1, w - pw - 1)
            if not _can_place(occupied, patch, r0, c0):
                continue
            _stamp(grid, occupied, patch, r0, c0)
            placed = True
            break
        if not placed:
            return None

    comps = _components((grid != BG) & (grid != MAROON))
    if len(comps) != loop_count:
        return None
    for comp in comps:
        order = _clockwise_cycle(comp)
        if not order:
            return None
        interior = _interior_mask(comp, grid.shape)
        if np.count_nonzero(interior & (grid == MAROON)) == 0:
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid loop-and-bead puzzle")
