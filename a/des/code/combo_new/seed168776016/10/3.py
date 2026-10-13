from __future__ import annotations

import numpy as np
from numpy.random import default_rng


RNG = default_rng()
MAX_GENERATION_ATTEMPTS = 400
BG_COLOR = 0
GRID_SIZE_RANGE = (18, 30)
NUM_CORRIDORS_RANGE = (2, 5)
COUNT_RANGE = (1, 5)
PANEL_CELL_SIZE = 3
PANEL_STEP = PANEL_CELL_SIZE + 1
PLACEMENT_ATTEMPTS = 160
ARC_COLORS = tuple(range(10))

BASE_SHAPES: dict[int, list[tuple[tuple[int, int], ...]]] = {
    1: [((0, 0),)],
    2: [((0, 0), (0, 1))],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
    ],
    5: [
        ((1, 1), (0, 1), (1, 0), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ],
}


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_panel_bbox(grid: np.ndarray, wall: int, bg: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    best: tuple[tuple[int, int, int], tuple[int, int, int, int]] | None = None
    comps = _components_mask(grid == wall)
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if not touches_corner:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colored = int(np.count_nonzero((patch != bg) & (patch != wall)))
        if colored == 0:
            continue
        score = (colored, len(comp), (r1 - r0 + 1) * (c1 - c0 + 1))
        if best is None or score > best[0]:
            best = (score, (r0, c0, r1, c1))
    assert best is not None, "expected a corner counter panel"
    return best[1]


def _counter_mapping(grid: np.ndarray, panel_bbox: tuple[int, int, int, int], wall: int, bg: int) -> dict[int, int]:
    r0, c0, r1, c1 = panel_bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    comps = _components_mask((patch != bg) & (patch != wall))
    mapping: dict[int, int] = {}
    for comp in comps:
        colors = {int(patch[r, c]) for r, c in comp}
        assert len(colors) == 1, "counter cell must be monochromatic"
        color = next(iter(colors))
        mapping[color] = len(comp)
    assert mapping, "counter panel must contain at least one color patch"
    return mapping


def _axis_run(grid: np.ndarray, wall: int, r: int, c: int, horizontal: bool) -> list[tuple[int, int]]:
    h, w = grid.shape
    dr, dc = (0, 1) if horizontal else (1, 0)
    sr, sc = r, c
    while True:
        nr, nc = sr - dr, sc - dc
        if not (0 <= nr < h and 0 <= nc < w) or int(grid[nr, nc]) == wall:
            break
        sr, sc = nr, nc

    run: list[tuple[int, int]] = []
    cr, cc = sr, sc
    while 0 <= cr < h and 0 <= cc < w and int(grid[cr, cc]) != wall:
        run.append((int(cr), int(cc)))
        cr += dr
        cc += dc
    return run


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    values, counts = np.unique(g, return_counts=True)
    non_bg = [(int(v), int(c)) for v, c in zip(values.tolist(), counts.tolist(), strict=False) if int(v) != bg]
    assert non_bg, "expected non-background objects"
    wall = max(non_bg, key=lambda item: item[1])[0]

    panel_bbox = _find_panel_bbox(g, wall, bg)
    r0, c0, r1, c1 = panel_bbox
    color_to_count = _counter_mapping(g, panel_bbox, wall, bg)

    out = g.copy()

    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            color = int(g[r, c])
            if color in (bg, wall):
                continue
            assert color in color_to_count, "seed color must appear in counter panel"
            horiz = _axis_run(g, wall, r, c, horizontal=True)
            vert = _axis_run(g, wall, r, c, horizontal=False)
            if len(horiz) == len(vert) == 1:
                continue
            assert (len(horiz) == 1) != (len(vert) == 1), "seed must lie in a straight corridor"
            run = horiz if len(horiz) > len(vert) else vert
            idx = run.index((r, c))
            left = idx
            right = len(run) - 1 - idx
            n = color_to_count[color]
            if left == right:
                directions = (-1, 1)
            elif left < right:
                directions = (1,)
            else:
                directions = (-1,)
            for direction in directions:
                for step in range(1, n + 1):
                    j = idx + direction * step
                    if not (0 <= j < len(run)):
                        break
                    rr, cc = run[j]
                    if int(out[rr, cc]) == bg:
                        out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread each colored seed along its corridor by a length read from the same-colored counter patch in a corner panel.

    Input structure:
    - The background is black.
    - One non-background wall color forms thin horizontal or vertical corridor frames.
    - Each corridor contains exactly one colored seed pixel inside its 1-cell-wide passage.
    - A grey-framed counter panel sits in one corner; each compartment contains a connected patch of one seed color with 1 to 5 pixels.

    Rule:
    - Find the corner counter panel and build a color-to-count mapping from the size of each colored patch inside it.
    - For each seed, identify the straight corridor passage it lies in and measure how many open cells it has to each end.
    - If the seed is centered, extend its color up to N cells in both directions; otherwise extend only toward the farther end by N cells, stopping at walls or the grid edge.
    - Keep the wall frames, counter panel, and original seed pixels unchanged.

    Key insight:
    - The corner panel is a visible lookup table from color to length, and the corridor geometry decides whether that length is applied symmetrically or only away from the nearer end.
    """
    return _solve(grid)


def _shape_variants(size: int) -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for base in BASE_SHAPES[size]:
        for swap in (False, True):
            for sr in (-1, 1):
                for sc in (-1, 1):
                    pts = []
                    for r, c in base:
                        rr, cc = (c, r) if swap else (r, c)
                        pts.append((sr * rr, sc * cc))
                    min_r = min(r for r, _ in pts)
                    min_c = min(c for _, c in pts)
                    norm = [(r - min_r, c - min_c) for r, c in pts]
                    max_r = max(r for r, _ in norm)
                    max_c = max(c for _, c in norm)
                    if max_r >= PANEL_CELL_SIZE or max_c >= PANEL_CELL_SIZE:
                        continue
                    for off_r in range(PANEL_CELL_SIZE - max_r):
                        for off_c in range(PANEL_CELL_SIZE - max_c):
                            placed = tuple(sorted((r + off_r, c + off_c) for r, c in norm))
                            variants.add(placed)
    return sorted(variants)


SHAPE_VARIANTS = {size: _shape_variants(size) for size in range(1, COUNT_RANGE[1] + 1)}


def _mark_box(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _box_is_clear(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _draw_panel(
    grid: np.ndarray,
    wall: int,
    entries: list[tuple[int, int]],
    orientation: str,
    corner: str,
) -> tuple[int, int, int, int]:
    count = len(entries)
    rows = 1 if orientation == "horizontal" else count
    cols = count if orientation == "horizontal" else 1
    panel_h = rows * PANEL_STEP + 1
    panel_w = cols * PANEL_STEP + 1
    h, w = grid.shape

    r0 = 0 if corner[0] == "t" else h - panel_h
    c0 = 0 if corner[1] == "l" else w - panel_w

    for rr in range(rows + 1):
        grid[r0 + rr * PANEL_STEP, c0 : c0 + panel_w] = wall
    for cc in range(cols + 1):
        grid[r0 : r0 + panel_h, c0 + cc * PANEL_STEP] = wall

    positions = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    for (rr, cc), (color, amount) in zip(positions, entries, strict=False):
        top = r0 + rr * PANEL_STEP + 1
        left = c0 + cc * PANEL_STEP + 1
        shape = SHAPE_VARIANTS[amount][int(RNG.integers(len(SHAPE_VARIANTS[amount])))]
        for dr, dc in shape:
            grid[top + dr, left + dc] = color

    return r0, c0, r0 + panel_h - 1, c0 + panel_w - 1


def _seed_index(length: int, mode: str) -> int:
    if mode == "endpoint":
        return 0 if int(RNG.integers(2)) == 0 else length - 1
    if mode == "center":
        return length // 2
    candidates = [idx for idx in range(1, length - 1) if idx != length - 1 - idx]
    assert candidates, "skew mode needs a non-central interior position"
    return int(RNG.choice(np.array(candidates)))


def _place_corridor(
    grid: np.ndarray,
    occupied: np.ndarray,
    wall: int,
    color: int,
    mode: str,
    orientation: str,
) -> bool:
    h, w = grid.shape
    if orientation == "horizontal":
        max_len = min(11, w - 4)
        lengths = list(range(4, max_len + 1))
        if mode == "center":
            lengths = [length for length in lengths if length % 2 == 1]
        if not lengths or h < 5:
            return False
        for _ in range(PLACEMENT_ATTEMPTS):
            length = int(RNG.choice(np.array(lengths)))
            r0 = int(RNG.integers(1, h - 3))
            c0 = int(RNG.integers(1, w - length - 2))
            r1, c1 = r0 + 2, c0 + length + 1
            if not _box_is_clear(occupied, r0, c0, r1, c1, pad=1):
                continue
            grid[r0, c0 : c1 + 1] = wall
            grid[r1, c0 : c1 + 1] = wall
            grid[r0 + 1, c0] = wall
            grid[r0 + 1, c1] = wall
            idx = _seed_index(length, mode)
            grid[r0 + 1, c0 + 1 + idx] = color
            _mark_box(occupied, r0, c0, r1, c1, pad=1)
            return True
        return False

    max_len = min(11, h - 4)
    lengths = list(range(4, max_len + 1))
    if mode == "center":
        lengths = [length for length in lengths if length % 2 == 1]
    if not lengths or w < 5:
        return False
    for _ in range(PLACEMENT_ATTEMPTS):
        length = int(RNG.choice(np.array(lengths)))
        r0 = int(RNG.integers(1, h - length - 2))
        c0 = int(RNG.integers(1, w - 3))
        r1, c1 = r0 + length + 1, c0 + 2
        if not _box_is_clear(occupied, r0, c0, r1, c1, pad=1):
            continue
        grid[r0 : r1 + 1, c0] = wall
        grid[r0 : r1 + 1, c1] = wall
        grid[r0, c0 + 1] = wall
        grid[r1, c0 + 1] = wall
        idx = _seed_index(length, mode)
        grid[r0 + 1 + idx, c0 + 1] = color
        _mark_box(occupied, r0, c0, r1, c1, pad=1)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1] + 1))
    w = int(RNG.integers(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1] + 1))
    corridor_count = int(RNG.integers(NUM_CORRIDORS_RANGE[0], NUM_CORRIDORS_RANGE[1] + 1))
    wall = int(RNG.choice(np.array(ARC_COLORS[1:])))
    usable_colors = [color for color in ARC_COLORS[1:] if color != wall]
    if len(usable_colors) < corridor_count:
        return None
    seed_colors = RNG.choice(np.array(usable_colors), size=corridor_count, replace=False).tolist()
    counts = RNG.integers(COUNT_RANGE[0], COUNT_RANGE[1] + 1, size=corridor_count).tolist()

    orientation = "horizontal" if int(RNG.integers(2)) == 0 else "vertical"
    corner = ["tl", "tr", "bl", "br"][int(RNG.integers(4))]
    panel_h = PANEL_STEP + 1 if orientation == "horizontal" else corridor_count * PANEL_STEP + 1
    panel_w = corridor_count * PANEL_STEP + 1 if orientation == "horizontal" else PANEL_STEP + 1
    if panel_h > h or panel_w > w:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    entries = list(zip(seed_colors, counts, strict=False))
    RNG.shuffle(entries)
    panel_bbox = _draw_panel(grid, wall, entries, orientation, corner)
    _mark_box(occupied, *panel_bbox, pad=1)

    modes = ["endpoint", "center", "skew"]
    for color in seed_colors:
        mode = modes[int(RNG.integers(len(modes)))]
        corridor_orientation = "horizontal" if int(RNG.integers(2)) == 0 else "vertical"
        if not _place_corridor(grid, occupied, wall, int(color), mode, corridor_orientation):
            return None

    try:
        out = _solve(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
