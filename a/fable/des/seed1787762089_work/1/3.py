import random
from typing import Optional

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7
AZURE = 8

PIVOT_COLORS = (BLUE, RED, GREEN, YELLOW, AZURE)
L_PATTERNS = {
    "top_left": {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)},
    "top_right": {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)},
}

GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
PAIR_COUNT_RANGE = (3, 5)
SATELLITE_DISTANCES = (2, 3, 4, 5)
GRID_CORNERS = ("grid_tl", "grid_tr", "grid_bl", "grid_br")
PLACEMENT_TRIES = 200
GENERATION_RETRIES = 400


def _in_bounds(pos: tuple[int, int], h: int, w: int) -> bool:
    r, c = pos
    return 0 <= r < h and 0 <= c < w


def _rotate_offset(dr: int, dc: int, clockwise: bool) -> tuple[int, int]:
    return (dc, -dr) if clockwise else (-dc, dr)


def _expand(cells: set[tuple[int, int]], h: int, w: int, margin: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _find_l_kind(grid: np.ndarray) -> str:
    orange = np.argwhere(grid == ORANGE)
    assert len(orange) == 5, "expected exactly five orange L-marker pixels"
    r0, c0 = orange.min(axis=0)
    r1, c1 = orange.max(axis=0)
    assert r1 - r0 == 2 and c1 - c0 == 2, "orange L-marker must fit a 3x3 box"
    cells = {(int(r - r0), int(c - c0)) for r, c in orange}
    if cells == L_PATTERNS["top_left"]:
        return "top_left"
    if cells == L_PATTERNS["top_right"]:
        return "top_right"
    raise AssertionError("orange marker is not a supported L orientation")


def _pivot_records(grid: np.ndarray) -> list[tuple[int, tuple[int, int]]]:
    pivots: list[tuple[int, tuple[int, int]]] = []
    for color in PIVOT_COLORS:
        coords = np.argwhere(grid == color)
        assert len(coords) in (0, 1), f"pivot color {color} must appear at most once"
        if len(coords) == 1:
            r, c = map(int, coords[0])
            pivots.append((color, (r, c)))
    assert 3 <= len(pivots) <= 5, "expected 3 to 5 pivots"
    return pivots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each magenta satellite 90 degrees around its uniquely aligned pivot.

    The orange L marker encodes the global rotation direction:
    - ┌ (corner at top-left)  -> clockwise
    - ┐ (corner at top-right) -> counterclockwise
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    allowed = {BG, MAGENTA, ORANGE, *PIVOT_COLORS}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected colors in grid"

    l_kind = _find_l_kind(g)
    clockwise = l_kind == "top_left"
    pivots = _pivot_records(g)
    satellites = [tuple(map(int, coord)) for coord in np.argwhere(g == MAGENTA)]
    assert len(satellites) == len(pivots), "satellite count must match pivot count"

    pivot_positions = {pos for _, pos in pivots}
    orange_positions = {tuple(map(int, coord)) for coord in np.argwhere(g == ORANGE)}
    pair_counts = {pos: 0 for _, pos in pivots}
    landings: dict[tuple[int, int], int] = {}

    for sat in satellites:
        matches = [(color, pos) for color, pos in pivots if sat[0] == pos[0] or sat[1] == pos[1]]
        assert len(matches) == 1, "each satellite must align with exactly one pivot"
        color, pivot = matches[0]
        pair_counts[pivot] += 1

        dr = sat[0] - pivot[0]
        dc = sat[1] - pivot[1]
        assert (dr == 0) ^ (dc == 0), "satellite must share exactly one axis with its pivot"
        dist = abs(dr) + abs(dc)
        assert 2 <= dist <= 5, "satellite distance must be between 2 and 5"

        rr_off, cc_off = _rotate_offset(dr, dc, clockwise)
        target = (pivot[0] + rr_off, pivot[1] + cc_off)
        if not _in_bounds(target, *g.shape):
            continue

        assert target not in pivot_positions, "rotated satellite may not overwrite a pivot"
        assert target not in orange_positions, "rotated satellite may not overwrite the L-marker"
        assert int(g[target]) in (BG, MAGENTA), "rotated satellite must land on background or a cleared satellite"
        assert target not in landings, "rotated satellite landings must be unique"
        landings[target] = color

    assert all(count == 1 for count in pair_counts.values()), "pivot/satellite pairing must be bijective"

    out = g.copy()
    out[out == MAGENTA] = BG
    for (r, c), color in landings.items():
        out[r, c] = color
    return out


def _place_l_marker(h: int, w: int, l_kind: str, corner: str) -> set[tuple[int, int]]:
    if corner == "grid_tl":
        r0 = random.randint(0, 1)
        c0 = random.randint(0, 1)
    elif corner == "grid_tr":
        r0 = random.randint(0, 1)
        c0 = w - 3 - random.randint(0, 1)
    elif corner == "grid_bl":
        r0 = h - 3 - random.randint(0, 1)
        c0 = random.randint(0, 1)
    else:
        r0 = h - 3 - random.randint(0, 1)
        c0 = w - 3 - random.randint(0, 1)
    return {(r0 + dr, c0 + dc) for dr, dc in L_PATTERNS[l_kind]}


def _sample_pivots(
    h: int, w: int, n_pairs: int, forbidden: set[tuple[int, int]]
) -> Optional[list[tuple[int, int]]]:
    rows = list(range(h))
    cols = list(range(w))
    for _ in range(PLACEMENT_TRIES):
        chosen_rows = random.sample(rows, n_pairs)
        chosen_cols = random.sample(cols, n_pairs)
        random.shuffle(chosen_cols)
        pivots = list(zip(chosen_rows, chosen_cols))
        if any(pivot in forbidden for pivot in pivots):
            continue
        clear = True
        for i in range(n_pairs):
            for j in range(i + 1, n_pairs):
                pr0, pc0 = pivots[i]
                pr1, pc1 = pivots[j]
                if max(abs(pr0 - pr1), abs(pc0 - pc1)) < 2:
                    clear = False
                    break
            if not clear:
                break
        if clear:
            return pivots
    return None


def _candidate_moves(
    pivot: tuple[int, int],
    pivots: list[tuple[int, int]],
    h: int,
    w: int,
    clockwise: bool,
    forbidden: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], Optional[tuple[int, int]], bool]]:
    pivot_set = set(pivots)
    candidates = []
    for dist in SATELLITE_DISTANCES:
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sat = (pivot[0] + dr * dist, pivot[1] + dc * dist)
            if not _in_bounds(sat, h, w):
                continue
            if sat in forbidden or sat in pivot_set:
                continue
            aligned_pivots = sum(sat[0] == pr or sat[1] == pc for pr, pc in pivots)
            if aligned_pivots != 1:
                continue

            rr_off, cc_off = _rotate_offset(dr * dist, dc * dist, clockwise)
            target = (pivot[0] + rr_off, pivot[1] + cc_off)
            if not _in_bounds(target, h, w):
                candidates.append((sat, None, True))
                continue
            if target in forbidden or target in pivot_set:
                continue
            candidates.append((sat, target, False))
    random.shuffle(candidates)
    return candidates


def _assign_moves(
    candidates: list[list[tuple[tuple[int, int], Optional[tuple[int, int]], bool]]],
    required_oob: int,
) -> Optional[list[tuple[tuple[int, int], Optional[tuple[int, int]], bool]]]:
    n_pairs = len(candidates)
    order = sorted(range(n_pairs), key=lambda idx: len(candidates[idx]))
    used_satellites: set[tuple[int, int]] = set()
    used_targets: set[tuple[int, int]] = set()
    assignment: list[Optional[tuple[tuple[int, int], Optional[tuple[int, int]], bool]]] = [None] * n_pairs

    def backtrack(step: int, used_oob: int) -> bool:
        if step == n_pairs:
            return used_oob == required_oob
        remaining = n_pairs - step
        if used_oob > required_oob or used_oob + remaining < required_oob:
            return False

        idx = order[step]
        random.shuffle(candidates[idx])
        for sat, target, out_of_bounds in candidates[idx]:
            next_oob = used_oob + int(out_of_bounds)
            if next_oob > required_oob:
                continue
            if sat in used_satellites or sat in used_targets:
                continue
            if target is not None and (target in used_satellites or target in used_targets):
                continue

            used_satellites.add(sat)
            if target is not None:
                used_targets.add(target)
            assignment[idx] = (sat, target, out_of_bounds)
            if backtrack(step + 1, next_oob):
                return True
            assignment[idx] = None
            used_satellites.remove(sat)
            if target is not None:
                used_targets.remove(target)
        return False

    if backtrack(0, 0):
        return [item for item in assignment if item is not None]
    return None


def _gen_attempt() -> Optional[np.ndarray]:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_pairs = random.randint(*PAIR_COUNT_RANGE)

    l_kind = random.choice(("top_left", "top_right"))
    clockwise = l_kind == "top_left"
    l_cells = _place_l_marker(h, w, l_kind, random.choice(GRID_CORNERS))
    forbidden = _expand(l_cells, h, w, margin=1)

    pivots = _sample_pivots(h, w, n_pairs, forbidden)
    if pivots is None:
        return None

    move_options = [_candidate_moves(pivot, pivots, h, w, clockwise, forbidden) for pivot in pivots]
    if not all(move_options):
        return None

    if random.random() < 0.5:
        oob_capable = sum(any(option[2] for option in options) for options in move_options)
        max_oob = min(2, oob_capable, n_pairs - 1)
        if max_oob < 1:
            return None
        required_oob = random.randint(1, max_oob)
    else:
        required_oob = 0

    chosen_moves = _assign_moves(move_options, required_oob)
    if chosen_moves is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for r, c in l_cells:
        grid[r, c] = ORANGE

    colors = random.sample(PIVOT_COLORS, n_pairs)
    for color, pivot, move in zip(colors, pivots, chosen_moves):
        pr, pc = pivot
        sat, _, _ = move
        grid[pr, pc] = color
        grid[sat] = MAGENTA

    out = main(grid)
    assert isinstance(out, np.ndarray)
    assert out.shape == grid.shape
    assert np.count_nonzero(grid == MAGENTA) == n_pairs
    assert np.count_nonzero(grid == ORANGE) == 5
    assert set(map(int, np.unique(out))) <= {BG, ORANGE, *PIVOT_COLORS}
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
