from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
START = 5
GLYPH_COLORS = (1, 2, 3, 7)
AREA_TO_DELTA = {
    3: (0, 2),   # right 2
    4: (2, 0),   # down 2
    5: (0, -3),  # left 3
    6: (-3, 0),  # up 3
}
GEN_ATTEMPTS = 600


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _run_starts(grid: np.ndarray, axis: int) -> list[int]:
    starts = [0]
    length = int(grid.shape[axis])
    for i in range(1, length):
        prev = grid[i - 1, :] if axis == 0 else grid[:, i - 1]
        cur = grid[i, :] if axis == 0 else grid[:, i]
        if not np.array_equal(cur, prev):
            starts.append(i)
    return starts


def _compress_grid(grid: np.ndarray) -> np.ndarray:
    row_starts = _run_starts(grid, axis=0)
    row_compressed = grid[row_starts, :]
    col_starts = _run_starts(row_compressed, axis=1)
    return row_compressed[:, col_starts].copy()


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _ordered_glyphs(grid: np.ndarray) -> list[dict[str, int]]:
    glyphs: list[dict[str, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, START):
            continue
        for coords in _component_coords(grid == color):
            r0, c0 = coords.min(axis=0)
            glyphs.append(
                {
                    "color": color,
                    "area": int(coords.shape[0]),
                    "r0": int(r0),
                    "c0": int(c0),
                }
            )
    glyphs.sort(key=lambda g: (g["r0"], g["c0"]))
    return glyphs


def _trace_segment(start: tuple[int, int], delta: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = delta
    step_r = _sign(dr)
    step_c = _sign(dc)
    steps = max(abs(dr), abs(dc))
    return [(r + step_r * k, c + step_c * k) for k in range(steps + 1)]


def _simulate_output(shape: tuple[int, int], start: tuple[int, int], glyphs: list[dict[str, int]]) -> np.ndarray | None:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    cur = start
    for glyph in glyphs:
        area = int(glyph["area"])
        if area not in AREA_TO_DELTA:
            return None
        color = int(glyph["color"])
        cells = _trace_segment(cur, AREA_TO_DELTA[area])
        if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
            return None
        for r, c in cells:
            out[r, c] = color
        cur = cells[-1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a run-expanded grid, decode hidden glyph areas as moves, and draw the resulting colored polyline.

    Input structure:
    - The input is a black background grid created by repeating every row and every column
      of a smaller grid 2-4 times, so consecutive identical rows/columns appear in runs.
    - In the hidden smaller grid there is one gray start pixel and 3-4 separated colored
      glyph components, each living inside its own 3x3 region and using a distinct color.
    - Each glyph is a monochromatic 4-connected shape whose area is between 3 and 6.

    Rule:
    - First collapse consecutive identical rows, then collapse consecutive identical columns.
    - In the compressed grid, find all non-black, non-gray 4-connected monochromatic glyphs
      and sort them by the top-left corner of their bounding boxes in row-major order.
    - Decode each glyph by area: 3 -> move right 2, 4 -> move down 2, 5 -> move left 3,
      6 -> move up 3.
    - Starting from the compressed gray pixel, draw each segment on a black output grid of
      the compressed size, coloring every visited cell (including both endpoints) with that
      glyph's color; later segments overwrite earlier ones.

    Key insight:
    - The visible large blobs are warped only because rows and columns were duplicated; after
      removing those duplicate runs, the restored 3x3 glyph areas reveal the instruction list.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    compressed = _compress_grid(g)
    start_cells = np.argwhere(compressed == START)
    assert start_cells.shape[0] == 1
    start = tuple(map(int, start_cells[0]))

    glyphs = _ordered_glyphs(compressed)
    out = _simulate_output(compressed.shape, start, glyphs)
    assert out is not None
    return out


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    shifted = sorted((r - min_r, c - min_c) for r, c in coords)
    return tuple(shifted)


def _connected_shape(coords: list[tuple[int, int]]) -> bool:
    pts = set(coords)
    q = deque([coords[0]])
    seen = {coords[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(pts)


def _build_glyph_library() -> dict[int, list[np.ndarray]]:
    lib: dict[int, set[tuple[tuple[int, int], ...]]] = {area: set() for area in AREA_TO_DELTA}
    for bits in range(1 << 9):
        area = bits.bit_count()
        if area not in lib:
            continue
        coords = [(idx // 3, idx % 3) for idx in range(9) if (bits >> idx) & 1]
        if not _connected_shape(coords):
            continue
        lib[area].add(_normalize_coords(coords))

    out: dict[int, list[np.ndarray]] = {}
    for area, shapes in lib.items():
        arrays: list[np.ndarray] = []
        for coords in sorted(shapes):
            mask = np.zeros((3, 3), dtype=int)
            for r, c in coords:
                mask[r, c] = 1
            arrays.append(mask)
        out[area] = arrays
    return out


GLYPH_LIBRARY = _build_glyph_library()


def _no_adjacent_duplicate_lines(grid: np.ndarray) -> bool:
    for axis in (0, 1):
        length = int(grid.shape[axis])
        for i in range(1, length):
            prev = grid[i - 1, :] if axis == 0 else grid[:, i - 1]
            cur = grid[i, :] if axis == 0 else grid[:, i]
            if np.array_equal(prev, cur):
                return False
    return True


def _expanded_component_areas(grid: np.ndarray) -> list[int]:
    areas: list[int] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, START):
            continue
        areas.extend(int(comp.shape[0]) for comp in _component_coords(grid == color))
    return sorted(areas)


def _slot_positions(h: int, w: int) -> list[tuple[int, int]]:
    base_r = random.randint(0, h - 7)
    base_c = random.randint(0, w - 7)
    return [
        (base_r, base_c),
        (base_r, base_c + 4),
        (base_r + 4, base_c),
        (base_r + 4, base_c + 4),
    ]


def _candidate_start_cells(h: int, w: int, slots: list[tuple[int, int]]) -> list[tuple[int, int]]:
    blocked = np.zeros((h, w), dtype=bool)
    for r0, c0 in slots:
        blocked[r0 : r0 + 3, c0 : c0 + 3] = True
    cells = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
    random.shuffle(cells)
    return cells


def _row_col_repeats(h: int, w: int, slots: list[tuple[int, int]]) -> tuple[list[int], list[int]] | None:
    for _ in range(200):
        row_rep = [random.randint(2, 4) for _ in range(h)]
        col_rep = [random.randint(2, 4) for _ in range(w)]
        if sum(row_rep) > 30 or sum(col_rep) > 30:
            continue
        values = set(row_rep + col_rep)
        if 2 not in values or 4 not in values:
            continue
        ok = True
        for r0, c0 in slots:
            row_span = row_rep[r0 : r0 + 3]
            col_span = col_rep[c0 : c0 + 3]
            if len(set(row_span)) == 1 and len(set(col_span)) == 1:
                ok = False
                break
        if ok:
            return row_rep, col_rep
    return None


def _visible_color_counts(grid: np.ndarray, colors: list[int]) -> list[int]:
    return [int(np.count_nonzero(grid == color)) for color in colors]


def _gen_attempt() -> np.ndarray | None:
    h = random.choice((8, 9))
    w = random.choice((8, 9))
    n_glyphs = 4 if random.random() < 0.6 else 3

    slots = _slot_positions(h, w)
    used_slots = random.sample(slots, n_glyphs)
    used_slots.sort()

    areas = random.sample(list(AREA_TO_DELTA), n_glyphs)
    random.shuffle(areas)
    colors = random.sample(list(GLYPH_COLORS), n_glyphs)
    random.shuffle(colors)

    seed = np.zeros((h, w), dtype=int)
    glyph_specs: list[dict[str, int]] = []
    for (r0, c0), area, color in zip(used_slots, areas, colors):
        mask = random.choice(GLYPH_LIBRARY[area])
        seed[r0 : r0 + 3, c0 : c0 + 3] = mask * color
        glyph_specs.append({"r0": r0, "c0": c0, "area": area, "color": color})

    ordered = sorted(glyph_specs, key=lambda g: (g["r0"], g["c0"]))
    start = None
    expected = None
    for cell in _candidate_start_cells(h, w, used_slots):
        trial_seed = seed.copy()
        trial_seed[cell] = START
        if not _no_adjacent_duplicate_lines(trial_seed):
            continue
        expected = _simulate_output(trial_seed.shape, cell, ordered)
        if expected is None:
            continue
        if min(_visible_color_counts(expected, [g["color"] for g in ordered])) < 2:
            continue
        start = cell
        seed = trial_seed
        break
    if start is None or expected is None:
        return None

    repeats = _row_col_repeats(h, w, used_slots)
    if repeats is None:
        return None
    row_rep, col_rep = repeats

    expanded = np.repeat(np.repeat(seed, row_rep, axis=0), col_rep, axis=1)
    if max(expanded.shape) > 30:
        return None

    compressed = _compress_grid(expanded)
    if not np.array_equal(compressed, seed):
        return None

    decoded = _ordered_glyphs(compressed)
    if [(g["r0"], g["c0"], g["area"], g["color"]) for g in decoded] != [
        (g["r0"], g["c0"], g["area"], g["color"]) for g in ordered
    ]:
        return None

    raw_areas = _expanded_component_areas(expanded)
    if len(raw_areas) != n_glyphs or any(area <= 6 for area in raw_areas):
        return None

    actual = main(expanded)
    if not np.array_equal(actual, expected):
        return None

    return expanded


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-glyph compression puzzle")
