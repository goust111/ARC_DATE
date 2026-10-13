from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
ORANGE = 7

NW_SE = "nw_se"
NE_SW = "ne_sw"
ORIENTATIONS = (NW_SE, NE_SW)

GEN_ATTEMPTS = 800
GRID_MIN = 12
GRID_MAX = 18


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _expanded_bbox(
    coords: list[tuple[int, int]], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = shape
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _diag_bar_coords(orientation: str, length: int, r0: int, c0: int) -> list[tuple[int, int]]:
    if orientation == NW_SE:
        return [(r0 + k, c0 + k) for k in range(length)]
    return [(r0 + k, c0 - k) for k in range(length)]


def _bar_center(coords: list[tuple[int, int]]) -> tuple[float, float]:
    r0, c0 = coords[0]
    r1, c1 = coords[-1]
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _all_bars(h: int, w: int, orientation: str, length: int) -> list[list[tuple[int, int]]]:
    bars: list[list[tuple[int, int]]] = []
    for r0 in range(h - length + 1):
        if orientation == NW_SE:
            col_range = range(2, w - length + 1)
        else:
            col_range = range(length + 1, w)
        for c0 in col_range:
            bars.append(_diag_bar_coords(orientation, length, r0, c0))
    return bars


def _blocked_halo(coords: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    blocked.add((rr, cc))
    return blocked


def _classify_groups(grid: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    fg = np.zeros_like(g, dtype=bool)
    fg[:, 2:] = g[:, 2:] == ORANGE

    groups = {NW_SE: [], NE_SW: []}
    for r0, c0 in np.argwhere(fg):
        r, c = int(r0), int(c0)
        has_nw_se = (
            (r > 0 and c > 2 and fg[r - 1, c - 1])
            or (r + 1 < h and c + 1 < w and fg[r + 1, c + 1])
        )
        has_ne_sw = (
            (r > 0 and c + 1 < w and fg[r - 1, c + 1])
            or (r + 1 < h and c > 2 and fg[r + 1, c - 1])
        )
        assert has_nw_se ^ has_ne_sw, "each orange pixel must belong to exactly one diagonal orientation"
        groups[NW_SE if has_nw_se else NE_SW].append((r, c))
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """Group diagonal bars by orientation, expand each group's bbox, and fill by legend colors with majority-on-top layering.

    Input structure:
    - A black background grid has a two-row legend in the upper-left two columns: row 0 maps orange to one fill color and row 1 maps orange to another.
    - Outside the legend, orange pixels form one-cell-thick diagonal segments in both diagonal orientations.
    - Every non-legend orange pixel belongs to exactly one segment and therefore to exactly one of the two diagonal orientations.

    Rule:
    - Ignore the legend columns when classifying orange pixels into the NW-SE group and the NE-SW group.
    - Count the orange pixels in both groups; the larger group takes the fill color from legend row 0 and the smaller group takes the fill color from legend row 1.
    - Expand each group's tight bounding box by one cell in every direction, fill the smaller group's expanded box only on black cells, then fill the larger group's expanded box on every non-orange cell so it overwrites the smaller fill in overlaps.
    - Keep all original orange pixels unchanged and clear the whole legend area to black at the end.

    Key insight:
    - The legend rows do not correspond to fixed slash directions; they correspond to majority versus minority orientation, so pixel counting and overlap-aware layering are both necessary.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 2 and w >= 3

    groups = _classify_groups(g)
    assert groups[NW_SE] and groups[NE_SW], "both diagonal orientations must be present"

    counts = {key: len(groups[key]) for key in ORIENTATIONS}
    assert counts[NW_SE] != counts[NE_SW], "orientation populations must differ"

    majority = NW_SE if counts[NW_SE] > counts[NE_SW] else NE_SW
    minority = NE_SW if majority == NW_SE else NW_SE
    fill_for = {
        majority: int(g[0, 1]),
        minority: int(g[1, 1]),
    }
    boxes = {key: _expanded_bbox(groups[key], (h, w)) for key in ORIENTATIONS}

    out = g.copy()

    for key in (minority, majority):
        r0, c0, r1, c1 = boxes[key]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        if key == minority:
            fill_mask = region == BLACK
        else:
            fill_mask = region != ORANGE
        region[fill_mask] = fill_for[key]

    out[:, :2] = BLACK
    return out


def _random_lengths(bar_count: int) -> list[int]:
    return [random.randint(2, 5) for _ in range(bar_count)]


def _sample_group_lengths() -> tuple[str, dict[str, list[int]]]:
    majority = random.choice(ORIENTATIONS)
    for _ in range(200):
        lengths = {
            NW_SE: _random_lengths(random.randint(1, 2)),
            NE_SW: _random_lengths(random.randint(1, 2)),
        }
        if sum(lengths[NW_SE]) == sum(lengths[NE_SW]):
            continue
        actual_majority = NW_SE if sum(lengths[NW_SE]) > sum(lengths[NE_SW]) else NE_SW
        if actual_majority == majority:
            return majority, lengths
    raise RuntimeError("failed to sample unequal group lengths")


def _sample_focus(h: int, w: int) -> tuple[int, int]:
    row_choices = sorted(
        {
            2,
            3,
            max(2, h // 2 - 1),
            min(h - 3, h // 2 + 1),
            h - 4,
            h - 3,
        }
    )
    col_choices = sorted(
        {
            4,
            5,
            max(4, w // 2 - 1),
            min(w - 3, w // 2 + 1),
            w - 5,
            w - 4,
        }
    )
    return random.choice(row_choices), random.choice(col_choices)


def _target_centers(h: int, w: int) -> dict[str, tuple[int, int]]:
    focus_r, focus_c = _sample_focus(h, w)
    row_shift_a = random.choice([-1, 0, 1])
    row_shift_b = random.choice([-1, 0, 1])
    col_gap = random.choice([1, 2])
    return {
        NW_SE: (max(1, min(h - 2, focus_r + row_shift_a)), max(3, min(w - 3, focus_c - col_gap))),
        NE_SW: (max(1, min(h - 2, focus_r + row_shift_b)), max(3, min(w - 3, focus_c + col_gap))),
    }


def _pick_bar_near(
    h: int,
    w: int,
    orientation: str,
    length: int,
    target: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    target_r, target_c = target
    bars = _all_bars(h, w, orientation, length)
    random.shuffle(bars)
    for radius in (2.0, 3.0, 4.5, 7.0):
        candidates = []
        for coords in bars:
            if any(cell in blocked for cell in coords):
                continue
            center_r, center_c = _bar_center(coords)
            if abs(center_r - target_r) + abs(center_c - target_c) <= radius:
                candidates.append(coords)
        if candidates:
            return random.choice(candidates)
    return None


def _render_input(h: int, w: int, placed: list[tuple[str, list[tuple[int, int]]]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[0, 0] = ORANGE
    grid[0, 1] = YELLOW
    grid[1, 0] = ORANGE
    grid[1, 1] = GREEN
    for _orientation, coords in placed:
        for r, c in coords:
            grid[r, c] = ORANGE
    return grid


def _box_to_mask(box: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _semantic_checks(grid: np.ndarray) -> bool:
    groups = _classify_groups(grid)
    if not groups[NW_SE] or not groups[NE_SW]:
        return False

    counts = {key: len(groups[key]) for key in ORIENTATIONS}
    if counts[NW_SE] == counts[NE_SW]:
        return False

    majority = NW_SE if counts[NW_SE] > counts[NE_SW] else NE_SW
    minority = NE_SW if majority == NW_SE else NW_SE
    boxes = {key: _expanded_bbox(groups[key], grid.shape) for key in ORIENTATIONS}

    major_mask = _box_to_mask(boxes[majority], grid.shape)
    minor_mask = _box_to_mask(boxes[minority], grid.shape)
    orange_mask = grid == ORANGE
    overlap_fillable = major_mask & minor_mask & ~orange_mask
    minority_only_fillable = minor_mask & ~major_mask & ~orange_mask

    if not overlap_fillable.any():
        return False
    if not minority_only_fillable.any():
        return False
    if int(overlap_fillable.sum()) > 12:
        return False

    out = main(grid)
    if not np.all(out[:, :2] == BLACK):
        return False
    if not np.any(out == YELLOW) or not np.any(out == GREEN):
        return False
    if not np.any(out[overlap_fillable] == YELLOW):
        return False
    if not np.any(out[minority_only_fillable] == GREEN):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)

    majority, lengths = _sample_group_lengths()
    centers = _target_centers(h, w)

    specs: list[tuple[str, int]] = []
    for orientation in ORIENTATIONS:
        specs.extend((orientation, length) for length in lengths[orientation])
    specs.sort(key=lambda item: (-item[1], random.random()))

    blocked: set[tuple[int, int]] = set()
    placed: list[tuple[str, list[tuple[int, int]]]] = []
    for orientation, length in specs:
        coords = _pick_bar_near(h, w, orientation, length, centers[orientation], blocked)
        if coords is None:
            return None
        placed.append((orientation, coords))
        blocked.update(_blocked_halo(coords, (h, w)))

    grid = _render_input(h, w, placed)

    groups = _classify_groups(grid)
    actual_majority = NW_SE if len(groups[NW_SE]) > len(groups[NE_SW]) else NE_SW
    if actual_majority != majority:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
