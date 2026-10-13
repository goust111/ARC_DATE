from __future__ import annotations

import numpy as np

from arc_utils import Color, find_components, get_bg, in_bounds
import random


_L_CORNERS = {(0, 0), (0, 1), (1, 0), (1, 1)}


def _missing_corner_direction(comp: dict) -> tuple[int, int]:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    assert comp["area"] == 3, "each source object must be an L triomino"
    assert int(r1 - r0) == 1 and int(c1 - c0) == 1, "L triomino must occupy a 2x2 bbox"

    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    missing = _L_CORNERS - rel
    assert len(missing) == 1, "2x2 L triomino must have exactly one missing corner"
    mr, mc = next(iter(missing))
    return (-1 if mr == 0 else 1, -1 if mc == 0 else 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a recolored diagonal copy from each L triomino source object.

    Input structure:
    - The background is the dominant color.
    - The only foreground objects are 4-connected red and grey L triominoes,
      each occupying three pixels of a 2x2 box.

    Rule:
    - Grey L objects paint a yellow copy translated one diagonal step toward
      their missing 2x2 corner.
    - Red L objects paint a green copy translated one diagonal step away from
      their missing 2x2 corner.
    - Copies are clipped by the grid boundary; red-derived green copies are
      painted after grey-derived yellow copies where the projections collide.

    Key insight:
    - The missing corner of each 2x2 L object identifies its diagonal axis, and
      the source color determines which side of that axis receives the copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    source_colors = {Color.GREY, Color.RED}
    assert bg not in source_colors, "background must be distinct from source colors"
    assert set(map(int, np.unique(g))) == source_colors | {bg}, "expected only bg, red, and grey"

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=source_colors)
    assert comps, "expected source L triominoes"
    out = g.copy()

    # Layering follows train[0]: the red-derived green copy remains intact where
    # it overlaps a grey-derived yellow copy.
    for src_color, dst_color, toward_missing in (
        (Color.GREY, Color.YELLOW, True),
        (Color.RED, Color.GREEN, False),
    ):
        src_comps = [comp for comp in comps if int(comp["color"]) == int(src_color)]
        assert src_comps, "expected at least one source object of each source color"
        for comp in src_comps:
            dr, dc = _missing_corner_direction(comp)
            if not toward_missing:
                dr, dc = -dr, -dc

            for r, c in comp["coords"]:
                nr, nc = int(r + dr), int(c + dc)
                if in_bounds(g.shape, nr, nc):
                    assert int(g[nr, nc]) == int(bg), "projection must land on background"
                    out[nr, nc] = int(dst_color)

    return out

# --------------- generator (o5x) ---------------

PlacementRecord = tuple[int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
# Palette.
BG = Color.ORANGE
RED = Color.RED
GREEN = Color.GREEN
YELLOW = Color.YELLOW
GREY = Color.GREY
COPY_COLOR_BY_SOURCE = {RED: GREEN, GREY: YELLOW}
COPY_PAINT_ORDER = (GREY, RED)

# Canvas and object-count sampling.
GRID_SIZES = (8, 10, 12, 14, 16)
MIN_OBJECTS = 3
MAX_OBJECTS = 16
MIN_DENSITY = 0.14
MAX_DENSITY = 0.19
FALLBACK_DENSITY = 0.165
CELLS_PER_OBJECT = 3

# L-triomino geometry and adjacency constraints.
MISSING_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
NEIGHBOR_DELTAS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Rejection sampling.
MAX_GENERATION_ATTEMPTS = 300


def _source_cells(r0: int, c0: int, missing: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r0 + rr, c0 + cc)
        for rr in (0, 1)
        for cc in (0, 1)
        if (rr, cc) != missing
    )


def _copy_delta(missing: tuple[int, int], color: int) -> tuple[int, int]:
    mr, mc = missing
    dr = -1 if mr == 0 else 1
    dc = -1 if mc == 0 else 1
    if color == RED:
        dr, dc = -dr, -dc
    return dr, dc


def _copy_cells(src: tuple[tuple[int, int], ...], H: int, dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r + dr, c + dc)
        for r, c in src
        if 0 <= r + dr < H and 0 <= c + dc < H
    )


def _valid_candidates(
    H: int,
    missing: tuple[int, int],
    src_color: int,
    source_occ: set[tuple[int, int]],
    source_by_color: dict[int, set[tuple[int, int]]],
    copy_by_color: dict[int, set[tuple[int, int]]],
) -> list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]]:
    dr, dc = _copy_delta(missing, src_color)
    dst_color = COPY_COLOR_BY_SOURCE[src_color]
    copy_occ = copy_by_color[GREEN] | copy_by_color[YELLOW]
    same_sources = source_by_color[src_color]
    candidates = []

    for r0 in range(H - 1):
        for c0 in range(H - 1):
            src = _source_cells(r0, c0, missing)
            src_set = set(src)
            if any(cell in source_occ or cell in copy_occ for cell in src):
                continue
            if any((r + dr, c + dc) in same_sources for r, c in src for dr, dc in NEIGHBOR_DELTAS_4):
                continue

            copied = _copy_cells(src, H, dr, dc)
            if any(cell in source_occ for cell in copied):
                continue
            if any(cell in copy_by_color[dst_color] for cell in copied):
                continue
            if any(cell in src_set for cell in copied):
                continue
            candidates.append((src, copied))
    return candidates


def _paint(records: list[PlacementRecord], H: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((H, H), BG, dtype=int)
    out = np.full((H, H), BG, dtype=int)

    for src_color, src, _copied in records:
        for r, c in src:
            inp[r, c] = src_color
            out[r, c] = src_color

    for paint_src_color in COPY_PAINT_ORDER:
        dst_color = COPY_COLOR_BY_SOURCE[paint_src_color]
        for src_color, _src, copied in records:
            if src_color == paint_src_color:
                for r, c in copied:
                    out[r, c] = dst_color
    return inp, out


def _choose_count(H: int) -> int:
    area = H * H
    lo = max(MIN_OBJECTS, int(np.ceil(MIN_DENSITY * area / CELLS_PER_OBJECT)))
    hi = min(MAX_OBJECTS, int(np.floor(MAX_DENSITY * area / CELLS_PER_OBJECT)))
    if lo > hi:
        lo, hi = MIN_OBJECTS, min(MAX_OBJECTS, max(MIN_OBJECTS, int(round(FALLBACK_DENSITY * area / CELLS_PER_OBJECT))))
    return random.randint(lo, hi)


def _choose_source_colors(N: int) -> list[int]:
    red_count = random.randint(1, N - 1)
    colors = [RED] * red_count + [GREY] * (N - red_count)
    random.shuffle(colors)
    return colors


def _place_objects(H: int, colors: list[int]) -> list[PlacementRecord] | None:
    source_occ: set[tuple[int, int]] = set()
    source_by_color = {RED: set(), GREY: set()}
    copy_by_color = {GREEN: set(), YELLOW: set()}
    records: list[PlacementRecord] = []

    for src_color in colors:
        missing = random.choice(MISSING_CORNERS)
        candidates = _valid_candidates(H, missing, src_color, source_occ, source_by_color, copy_by_color)
        if not candidates:
            return None
        src, copied = random.choice(candidates)
        dst_color = COPY_COLOR_BY_SOURCE[src_color]
        source_occ.update(src)
        source_by_color[src_color].update(src)
        copy_by_color[dst_color].update(copied)
        records.append((src_color, src, copied))

    return records


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(GRID_SIZES)
    colors = _choose_source_colors(_choose_count(H))
    records = _place_objects(H, colors)
    if records is None:
        return None

    inp, out = _paint(records, H)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 1e5d6875 input")
