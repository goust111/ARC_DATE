from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
Pair = dict[str, object]

BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (12, 16)
GRID_WIDTH_RANGE = (12, 16)
PAIR_COUNT_CHOICES = (3, 4)

CORE_MAX_DELTA = 3
EXTRA_MAX_DELTA = 5
MAX_GENERATION_ATTEMPTS = 500
CORE_ATTEMPTS = 300
EXTRA_ATTEMPTS = 300


def _inclusive_range(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _build_pair(color: int, a: Coord, b: Coord) -> Pair:
    start, end = sorted((tuple(map(int, a)), tuple(map(int, b))))
    r1, c1 = start
    r2, c2 = end
    assert r1 != r2 and c1 != c2, "each pair must require an L path"
    d_v = r2 - r1
    d_h = abs(c2 - c1)
    return {
        "color": int(color),
        "start": start,
        "end": end,
        "d_v": int(d_v),
        "d_h": int(d_h),
        "distance": int(d_v + d_h),
        "mode": "H" if d_h > d_v else "V",
    }


def _pair_sort_key(pair: Pair) -> tuple[int, int, int]:
    r, c = pair["start"]
    return int(pair["distance"]), int(r), int(c)


def _path_cells(pair: Pair, mode_override: str | None = None) -> list[Coord]:
    (r1, c1) = pair["start"]
    (r2, c2) = pair["end"]
    mode = str(mode_override or pair["mode"])

    if mode == "H":
        cells = [(int(r1), int(c)) for c in _inclusive_range(int(c1), int(c2))]
        cells.extend((int(r), int(c2)) for r in _inclusive_range(int(r1), int(r2)))
        return cells

    cells = [(int(r), int(c1)) for r in _inclusive_range(int(r1), int(r2))]
    cells.extend((int(r2), int(c)) for c in _inclusive_range(int(c1), int(c2)))
    return cells


def _marker_map(pairs: list[Pair]) -> dict[Coord, int]:
    markers: dict[Coord, int] = {}
    for pair in pairs:
        color = int(pair["color"])
        markers[tuple(pair["start"])] = color
        markers[tuple(pair["end"])] = color
    return markers


def _render_pairs(
    pairs: list[Pair],
    shape: tuple[int, int],
    *,
    mode_override: str | None = None,
    order: list[Pair] | None = None,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    draw_order = list(order) if order is not None else sorted(pairs, key=_pair_sort_key)
    for pair in draw_order:
        color = int(pair["color"])
        for r, c in _path_cells(pair, mode_override=mode_override):
            out[int(r), int(c)] = color

    for (r, c), color in _marker_map(pairs).items():
        out[int(r), int(c)] = int(color)
    return out


def _extract_pairs(grid: np.ndarray) -> list[Pair]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    assert colors, "expected at least one foreground color"

    pairs: list[Pair] = []
    for color in colors:
        coords = np.argwhere(g == int(color))
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        pair = _build_pair(int(color), tuple(map(int, coords[0])), tuple(map(int, coords[1])))
        pairs.append(pair)
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair with an axis-adaptive L path, drawing shorter paths first so later paths can cover earlier ones.

    Input structure:
    - A black background contains three or four foreground colors.
    - Each foreground color appears exactly twice as single-pixel markers.
    - The two markers of any one color are in different rows and different columns.

    Rule:
    - Pair the markers by color.
    - For each pair, choose the lexicographically smaller marker (row first, then column) as the start.
    - If the pair's horizontal separation is larger than its vertical separation, draw the L path horizontal first and then vertical; otherwise draw it vertical first and then horizontal.
    - Sort all pairs by Manhattan distance from short to long, breaking ties by the start marker's row and then column, and draw them in that order with later paths overwriting earlier ones.
    - After all paths are drawn, restore every original marker pixel to its original color.

    Key insight:
    - The bend direction is not fixed globally: each pair chooses its own first segment from its geometry, and the visible color at crossings is determined by distance-based draw order rather than by color identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pairs = _extract_pairs(g)
    return _render_pairs(pairs, g.shape)


def _core_pairs(height: int, width: int, colors: list[int]) -> list[Pair] | None:
    case = random.choices(("a_later", "b_later", "tie"), weights=(4, 4, 2), k=1)[0]
    for _ in range(CORE_ATTEMPTS):
        cr = random.randint(2, height - 3)
        cc = random.randint(2, width - 3)

        sign_a = random.choice((-1, 1))
        max_a_start = cc if sign_a > 0 else width - 1 - cc
        max_a_end = width - 1 - cc if sign_a > 0 else cc
        max_y1 = height - 1 - cr
        if min(max_a_start, max_a_end, max_y1) < 1:
            continue

        x1 = random.randint(1, min(CORE_MAX_DELTA, max_a_start))
        x2 = random.randint(1, min(CORE_MAX_DELTA, max_a_end))
        max_y1 = min(CORE_MAX_DELTA, max_y1, x1 + x2 - 1)
        if max_y1 < 1:
            continue
        y1 = random.randint(1, max_y1)

        a_start = (cr, cc - x1) if sign_a > 0 else (cr, cc + x1)
        a_end = (cr + y1, cc + x2) if sign_a > 0 else (cr + y1, cc - x2)
        pair_a = _build_pair(int(colors[0]), a_start, a_end)

        sign_b = random.choice((-1, 1))
        max_y2 = min(CORE_MAX_DELTA, cr)
        max_y3 = min(CORE_MAX_DELTA, height - 1 - cr)
        max_x3 = cc if sign_b < 0 else width - 1 - cc
        if min(max_y2, max_y3, max_x3) < 1:
            continue

        y2 = random.randint(1, max_y2)
        y3 = random.randint(1, max_y3)
        max_x3 = min(CORE_MAX_DELTA, max_x3, y2 + y3)
        if max_x3 < 1:
            continue
        x3 = random.randint(1, max_x3)

        b_start = (cr - y2, cc)
        b_end = (cr + y3, cc + sign_b * x3)
        pair_b = _build_pair(int(colors[1]), b_start, b_end)

        if pair_a["distance"] == pair_b["distance"]:
            relation_ok = case == "tie"
        elif int(pair_a["distance"]) > int(pair_b["distance"]):
            relation_ok = case == "a_later"
        else:
            relation_ok = case == "b_later"
        if not relation_ok:
            continue

        return [pair_a, pair_b]
    return None


def _delta_options(
    height: int, width: int, branch: str, target_distance: int | None
) -> list[tuple[int, int]]:
    max_dr = min(EXTRA_MAX_DELTA, height - 1)
    max_dc = min(EXTRA_MAX_DELTA, width - 1)
    options: list[tuple[int, int]] = []
    for dr in range(1, max_dr + 1):
        for dc in range(1, max_dc + 1):
            if target_distance is not None and dr + dc != int(target_distance):
                continue
            if branch == "H" and not (dc > dr):
                continue
            if branch == "V" and not (dr >= dc):
                continue
            if branch == "EQ" and dr != dc:
                continue
            options.append((dr, dc))
    random.shuffle(options)
    return options


def _sample_extra_pair(
    height: int,
    width: int,
    color: int,
    existing_pairs: list[Pair],
    *,
    force_equal: bool,
    prefer_tie: bool,
) -> Pair | None:
    occupied_paths = set()
    for pair in existing_pairs:
        occupied_paths.update(_path_cells(pair))
    occupied_markers = set(_marker_map(existing_pairs))
    distances = sorted({int(pair["distance"]) for pair in existing_pairs})

    branch_pool = ["EQ"] if force_equal else ["H", "V", "EQ", "H", "V"]
    for _ in range(EXTRA_ATTEMPTS):
        branch = random.choice(branch_pool)
        tie_target = None
        if distances and prefer_tie and random.random() < 0.85:
            feasible = [
                dist
                for dist in distances
                if _delta_options(height, width, branch, dist)
            ]
            if feasible:
                tie_target = random.choice(feasible)
        elif distances and random.random() < 0.35:
            feasible = [
                dist
                for dist in distances
                if _delta_options(height, width, branch, dist)
            ]
            if feasible:
                tie_target = random.choice(feasible)

        options = _delta_options(height, width, branch, tie_target)
        if not options:
            continue
        dr, dc = random.choice(options)
        sign = random.choice((-1, 1))

        max_row = height - dr - 1
        if max_row < 0:
            continue
        start_r = random.randint(0, max_row)
        if sign > 0:
            start_c = random.randint(0, width - dc - 1)
            end_c = start_c + dc
        else:
            start_c = random.randint(dc, width - 1)
            end_c = start_c - dc

        candidate = _build_pair(int(color), (start_r, start_c), (start_r + dr, end_c))
        path = set(_path_cells(candidate))
        if path & occupied_paths:
            continue
        if set((candidate["start"], candidate["end"])) & occupied_markers:
            continue
        return candidate
    return None


def _force_render_difference(pairs: list[Pair], shape: tuple[int, int]) -> bool:
    actual = _render_pairs(pairs, shape)
    forced_h = _render_pairs(pairs, shape, mode_override="H")
    forced_v = _render_pairs(pairs, shape, mode_override="V")
    reverse_order = _render_pairs(pairs, shape, order=list(sorted(pairs, key=_pair_sort_key))[::-1])
    return (
        not np.array_equal(actual, forced_h)
        and not np.array_equal(actual, forced_v)
        and not np.array_equal(actual, reverse_order)
    )


def _validate_generated_case(grid: np.ndarray) -> bool:
    try:
        pairs = _extract_pairs(grid)
    except AssertionError:
        return False

    if not (3 <= len(pairs) <= 4):
        return False

    markers = _marker_map(pairs)
    marker_cells = set(markers)
    if len(markers) != 2 * len(pairs):
        return False

    modes = {str(pair["mode"]) for pair in pairs}
    if "H" not in modes or "V" not in modes:
        return False

    path_sets = [set(_path_cells(pair)) for pair in pairs]
    for pair, path in zip(pairs, path_sets):
        own_markers = {tuple(pair["start"]), tuple(pair["end"])}
        if (path - own_markers) & (marker_cells - own_markers):
            return False

    overlap_map: dict[Coord, list[int]] = {}
    for idx, path in enumerate(path_sets):
        for cell in path - marker_cells:
            overlap_map.setdefault(cell, []).append(idx)
    overlap_cells = [cell for cell, owners in overlap_map.items() if len(owners) >= 2]
    if not overlap_cells or len(overlap_cells) > 4:
        return False

    actual = main(grid)
    if actual.shape != grid.shape:
        return False
    if not _force_render_difference(pairs, grid.shape):
        return False

    sorted_pairs = sorted(pairs, key=_pair_sort_key)
    for pair, path in zip(pairs, path_sets):
        visible = [
            cell
            for cell in path - marker_cells
            if int(actual[cell]) == int(pair["color"])
        ]
        if len(visible) < 2:
            return False

    evidence = False
    for cell in overlap_cells:
        contributors = [pair for pair, path in zip(pairs, path_sets) if cell in path]
        last_pair = max(contributors, key=_pair_sort_key)
        if int(actual[cell]) != int(last_pair["color"]):
            return False
        evidence = True
    if not evidence:
        return False

    return True


def _assemble_grid(height: int, width: int, pairs: list[Pair]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for pair in pairs:
        r1, c1 = pair["start"]
        r2, c2 = pair["end"]
        color = int(pair["color"])
        grid[int(r1), int(c1)] = color
        grid[int(r2), int(c2)] = color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(*GRID_HEIGHT_RANGE)
        width = random.randint(*GRID_WIDTH_RANGE)
        pair_count = random.choice(PAIR_COUNT_CHOICES)
        colors = random.sample(PALETTE, pair_count)

        pairs = _core_pairs(height, width, colors[:2])
        if pairs is None:
            continue

        need_tie = random.random() < 0.6
        for idx, color in enumerate(colors[2:], start=2):
            pair = _sample_extra_pair(
                height,
                width,
                int(color),
                pairs,
                force_equal=(idx == 2 and random.random() < 0.6),
                prefer_tie=need_tie,
            )
            if pair is None:
                pairs = []
                break
            pairs.append(pair)
            distances = [int(existing["distance"]) for existing in pairs]
            need_tie = len(distances) == len(set(distances))

        if not pairs:
            continue

        grid = _assemble_grid(height, width, pairs)
        if _validate_generated_case(grid):
            return grid

    raise RuntimeError("failed to generate a valid adaptive L-path ARC input")
