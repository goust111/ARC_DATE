from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
SMALL_SEGMENT_CHOICES = (2, 3)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _plus_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in PLUS_OFFSETS}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_plus_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for comp in _components(grid == GREEN):
        if len(comp) != 5:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 != 2 or c1 - c0 != 2:
            continue
        center = (r0 + 1, c0 + 1)
        if set(comp) == _plus_cells(center):
            centers.append(center)
    centers.sort()
    return centers


def _pair_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _diagonal_gap(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if abs(dr) != abs(dc) or dr == 0:
        return None
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    return [(a[0] + k * sr, a[1] + k * sc) for k in range(1, abs(dr))]


def _draw_connectors(
    grid: np.ndarray,
    *,
    skip_pair: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    centers = _detect_plus_centers(g)
    gaps: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]] = {}
    for i, a in enumerate(centers):
        for b in centers[i + 1 :]:
            gap = _diagonal_gap(a, b)
            if gap is None:
                continue
            if all(g[r, c] == BLACK for r, c in gap):
                key = _pair_key(a, b)
                gaps[key] = gap
                if key == skip_pair:
                    continue
                for r, c in gap:
                    out[r, c] = RED
    return out, gaps


def _enclosed_black_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    enclosed: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                enclosed.append(comp)
    return enclosed


def _area_fill_color(area: int) -> int:
    if 1 <= area <= 3:
        return PINK
    if 4 <= area <= 8:
        return ORANGE
    return TEAL


def _fill_enclosed_regions(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for comp in _enclosed_black_components(out):
        color = _area_fill_color(len(comp))
        for r, c in comp:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw diagonal connectors between aligned green pluses, then fill the newly enclosed black regions by area.
    
    Input structure:
    - The grid has black background, 4-6 green 5-pixel plus objects, and grey one-pixel orthogonal outline segments.
    - Some plus centers lie on the same 45-degree diagonal, and some of those center-to-center diagonals are clear black gaps.
    - The grey segments alone are only partially closed; they need the red diagonal connectors to finish the boundaries.
    
    Rule:
    - Keep every existing input pixel unchanged.
    - For every pair of green plus centers with equal row and column separation, draw a red diagonal connector through the interior cells only if every interior cell is black.
    - Treat the red connectors together with the grey outline pixels as blocking boundary cells, and find every 4-connected black region that cannot reach the grid border.
    - Fill enclosed black regions of area 1-3 with pink, 4-8 with orange, and 9+ with teal.
    
    Key insight:
    - The diagonal connectors are functional boundaries: only after drawing them do the grey half-frames become closed regions whose areas determine the final fill colors.
    """
    with_connectors, _ = _draw_connectors(np.asarray(grid, dtype=int))
    return _fill_enclosed_regions(with_connectors)


def _segment_area(n: int) -> int:
    return n * (n - 1) // 2


def _canonical_chain(segment_specs: list[tuple[int, str]]) -> dict:
    centers = [(0, 0)]
    grey: set[tuple[int, int]] = set()
    segments: list[dict] = []
    cur = 0
    for n, side in segment_specs:
        start = (cur, cur)
        gap_span = n + 3
        end = (cur + gap_span, cur + gap_span)
        segments.append({"start": start, "end": end, "n": n, "side": side})

        base = cur
        if side == "lower_left":
            for r in range(base + 2, base + n + 3):
                grey.add((r, base + 1))
            for c in range(base + 1, base + n + 2):
                grey.add((base + n + 2, c))
        else:
            for c in range(base + 2, base + n + 3):
                grey.add((base + 1, c))
            for r in range(base + 1, base + n + 2):
                grey.add((r, base + n + 2))

        centers.append(end)
        cur += gap_span

    return {"centers": centers, "grey": grey, "segments": segments}


def _transform_group(group: dict, sr: int, sc: int) -> dict:
    def tr(coord: tuple[int, int]) -> tuple[int, int]:
        r, c = coord
        return sr * r, sc * c

    centers = [tr(center) for center in group["centers"]]
    grey = {tr(cell) for cell in group["grey"]}
    segments = [
        {"start": tr(seg["start"]), "end": tr(seg["end"]), "n": seg["n"], "side": seg["side"]}
        for seg in group["segments"]
    ]

    occupied = set(grey)
    for center in centers:
        occupied.update(_plus_cells(center))

    return {"centers": centers, "grey": grey, "segments": segments, "occupied": occupied}


def _translate_group(group: dict, dr: int, dc: int) -> dict:
    def shift(coord: tuple[int, int]) -> tuple[int, int]:
        return coord[0] + dr, coord[1] + dc

    centers = [shift(center) for center in group["centers"]]
    grey = {shift(cell) for cell in group["grey"]}
    segments = [
        {"start": shift(seg["start"]), "end": shift(seg["end"]), "n": seg["n"], "side": seg["side"]}
        for seg in group["segments"]
    ]

    occupied = set(grey)
    for center in centers:
        occupied.update(_plus_cells(center))

    return {"centers": centers, "grey": grey, "segments": segments, "occupied": occupied}


def _group_bounds(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _expanded(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _place_groups(groups: list[dict], h: int, w: int) -> list[dict] | None:
    placed: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()

    order = list(groups)
    random.shuffle(order)
    for group in order:
        min_r, min_c, max_r, max_c = _group_bounds(group["occupied"])
        row_lo = 1 - min_r
        row_hi = (h - 2) - max_r
        col_lo = 1 - min_c
        col_hi = (w - 2) - max_c
        if row_lo > row_hi or col_lo > col_hi:
            return None

        candidates: list[tuple[int, int]] = []
        for _ in range(120):
            tr = random.randint(row_lo, row_hi)
            tc = random.randint(col_lo, col_hi)
            candidates.append((tr, tc))
        candidates.extend((r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1))
        random.shuffle(candidates)

        chosen = None
        for tr, tc in candidates[:220]:
            shifted = _translate_group(group, tr, tc)
            occ = shifted["occupied"]
            if occ & forbidden:
                continue
            chosen = shifted
            occupied |= occ
            forbidden = _expanded(occupied)
            placed.append(shifted)
            break
        if chosen is None:
            return None

    return placed


def _build_input_from_groups(h: int, w: int, placed_groups: list[dict]) -> tuple[np.ndarray, set]:
    grid = np.zeros((h, w), dtype=int)
    intended_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for group in placed_groups:
        for r, c in group["grey"]:
            grid[r, c] = GREY
        for center in group["centers"]:
            for r, c in _plus_cells(center):
                if grid[r, c] != BLACK:
                    raise ValueError("overlapping plus/grey placement")
                grid[r, c] = GREEN
        for seg in group["segments"]:
            intended_pairs.add(_pair_key(seg["start"], seg["end"]))
    return grid, intended_pairs


def _planned_fill_colors(segment_ns: list[int]) -> set[int]:
    return {_area_fill_color(_segment_area(n)) for n in segment_ns}


def _validate_instance(grid: np.ndarray, intended_pairs: set, segment_ns: list[int]) -> bool:
    centers = _detect_plus_centers(grid)
    if not (4 <= len(centers) <= 6):
        return False

    if _enclosed_black_components(grid):
        return False

    with_connectors, actual_gaps = _draw_connectors(grid)
    actual_pairs = set(actual_gaps)
    if actual_pairs != intended_pairs:
        return False

    enclosed = _enclosed_black_components(with_connectors)
    if not enclosed:
        return False

    colors_from_areas = {_area_fill_color(len(comp)) for comp in enclosed}
    if colors_from_areas != _planned_fill_colors(segment_ns):
        return False
    if colors_from_areas != {PINK, ORANGE, TEAL}:
        return False

    full_output = _fill_enclosed_regions(with_connectors)
    for pair in actual_pairs:
        weaker_grid, gap_map = _draw_connectors(grid, skip_pair=pair)
        if set(gap_map) != actual_pairs:
            return False
        weaker_output = _fill_enclosed_regions(weaker_grid)
        if np.array_equal(weaker_output, full_output):
            return False

    return True


def _random_layout() -> tuple[list[dict], list[int]]:
    pattern = random.choice(("chain4", "chain3_pair", "two_chains3"))
    small_n = random.choice(SMALL_SEGMENT_CHOICES)

    if pattern == "chain4":
        segment_ns = [small_n, 4, 5]
        random.shuffle(segment_ns)
        chains = [segment_ns]
    elif pattern == "chain3_pair":
        segment_ns = [small_n, 4, 5]
        random.shuffle(segment_ns)
        chains = [segment_ns[:2], segment_ns[2:]]
    else:
        extra_n = random.choice((2, 3, 4, 5))
        segment_ns = [small_n, 4, 5, extra_n]
        random.shuffle(segment_ns)
        chains = [segment_ns[:2], segment_ns[2:]]

    groups = []
    for chain in chains:
        sr, sc = random.choice(DIAGONALS)
        specs = [(n, random.choice(("lower_left", "upper_right"))) for n in chain]
        groups.append(_transform_group(_canonical_chain(specs), sr, sc))
    return groups, segment_ns


def generate_input() -> np.ndarray:
    for _ in range(500):
        groups, segment_ns = _random_layout()

        max_h = max(_group_bounds(group["occupied"])[2] - _group_bounds(group["occupied"])[0] + 1 for group in groups)
        max_w = max(_group_bounds(group["occupied"])[3] - _group_bounds(group["occupied"])[1] + 1 for group in groups)
        min_dim = max(22, max_h + 2, max_w + 2)
        if min_dim > 30:
            continue

        for _ in range(10):
            h = random.randint(min_dim, 30)
            w = random.randint(min_dim, 30)
            placed = _place_groups(groups, h, w)
            if placed is None:
                continue

            grid, intended_pairs = _build_input_from_groups(h, w, placed)
            if _validate_instance(grid, intended_pairs, segment_ns):
                return grid

    raise RuntimeError("failed to generate a valid input")
