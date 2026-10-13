from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# --------------- shared helpers adapted from arc_utils ---------------


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    cols = set(map(int, np.unique(grid)))
    if exclude is not None:
        cols -= set(map(int, exclude))
    return sorted(cols)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = int(r + dr), int(c + dc)
            yield nr, nc


def _mirror_point(orient: str, axis: int, point: tuple[int, int]) -> tuple[int, int]:
    r, c = point
    if orient == "h":
        return 2 * int(axis) - int(r), int(c)
    return int(r), 2 * int(axis) - int(c)


def _evaluate_axis(
    grid: np.ndarray,
    color: int,
    orient: str,
    axis: int,
    bg: int,
) -> dict | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))]
    coord_set = set(coords)
    if len(coord_set) < 5:
        return None

    pair_keys: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    pair_distances: set[int] = set()
    lonely: list[tuple[int, int]] = []
    targets: list[tuple[int, int]] = []

    for point in coords:
        r, c = point
        if (orient == "h" and r == axis) or (orient == "v" and c == axis):
            return None
        mr, mc = _mirror_point(orient, axis, point)
        if not in_bounds(grid.shape, mr, mc):
            return None

        value = int(grid[mr, mc])
        if value == int(color):
            pair = tuple(sorted((point, (mr, mc))))
            pair_keys.add(pair)
            pair_distances.add(abs(int(r) - int(axis)) if orient == "h" else abs(int(c) - int(axis)))
        elif value == int(bg):
            lonely.append(point)
            targets.append((int(mr), int(mc)))
        else:
            return None

    if len(pair_keys) < 2 or len(pair_distances) < 2 or not lonely:
        return None

    return {
        "orient": orient,
        "axis": int(axis),
        "pairs": tuple(sorted(pair_keys)),
        "pair_count": int(len(pair_keys)),
        "pair_distances": tuple(sorted(pair_distances)),
        "lonely": tuple(sorted(lonely)),
        "targets": tuple(sorted(targets)),
    }


def _all_axis_candidates(grid: np.ndarray, color: int, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    candidates: list[dict] = []
    for axis in range(h):
        cand = _evaluate_axis(grid, color, "h", axis, bg)
        if cand is not None:
            candidates.append(cand)
    for axis in range(w):
        cand = _evaluate_axis(grid, color, "v", axis, bg)
        if cand is not None:
            candidates.append(cand)
    return candidates


def _infer_axis(grid: np.ndarray, color: int, bg: int) -> dict:
    candidates = _all_axis_candidates(grid, color, bg)
    if not candidates:
        raise AssertionError(f"color {color}: no valid hidden axis")
    if len(candidates) != 1:
        desc = sorted((cand["orient"], cand["axis"]) for cand in candidates)
        raise AssertionError(f"color {color}: hidden axis is ambiguous: {desc}")
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each color's hidden row or column mirror axis and complete its missing partners.

    Input structure:
    - A black background contains several foreground colors, each shown only as isolated single-pixel markers.
    - Within one color group, most markers already form mirrored pairs across one hidden full row or one hidden full column.
    - The same color also has one or more lonely markers whose reflected partner is absent, while different colors may use different hidden axes.

    Rule:
    - Group markers by color.
    - For each color, enumerate the full-row and full-column axes that are consistent with that color's existing mirrored pairs and missing partners.
    - The puzzle guarantees exactly one valid axis for that color; reflect every lonely marker across it and paint the missing partner in the same color.
    - Keep all original pixels unchanged.

    Key insight:
    - The symmetry is color-local rather than global: color membership decides which sparse points belong to the same hidden axis, so each color must infer its own unique axis before any completion can be drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    assert colors, "expected at least one foreground color"

    out = g.copy()
    for color in colors:
        axis_data = _infer_axis(g, int(color), bg)
        for r, c in axis_data["targets"]:
            assert int(out[r, c]) == bg, "completion targets must stay background before drawing"
            out[r, c] = int(color)
    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
COLOR_COUNT_RANGE = (2, 4)
PAIR_COUNT_RANGE = (2, 4)
LONELY_COUNT_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 240
MAX_COLOR_ATTEMPTS = 120


def _reserve(blocked: set[tuple[int, int]], shape: tuple[int, int], point: tuple[int, int]) -> None:
    r, c = point
    for rr, cc in _neighbors8(r, c):
        if in_bounds(shape, rr, cc):
            blocked.add((rr, cc))


def _can_claim(
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
    points: list[tuple[int, int]],
) -> bool:
    return all(in_bounds(shape, r, c) and (int(r), int(c)) not in blocked for r, c in points)


def _blank_group_grid(shape: tuple[int, int], color: int, points: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in points:
        grid[int(r), int(c)] = int(color)
    return grid


def _try_place_color(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    color: int,
    orient: str,
    axis: int,
) -> dict | None:
    h, w = shape
    line_values = list(range(1, w - 1)) if orient == "h" else list(range(1, h - 1))
    max_dist = min(axis, h - 1 - axis) if orient == "h" else min(axis, w - 1 - axis)
    if max_dist < 2:
        return None

    max_pairs = min(PAIR_COUNT_RANGE[1], len(line_values), max_dist)
    if max_pairs < PAIR_COUNT_RANGE[0]:
        return None
    pair_count = random.randint(PAIR_COUNT_RANGE[0], max_pairs)
    pair_dists = random.sample(range(1, max_dist + 1), pair_count)

    free_lines = line_values[:]
    random.shuffle(free_lines)
    max_lonely = min(LONELY_COUNT_RANGE[1], len(free_lines) - pair_count)
    if max_lonely < LONELY_COUNT_RANGE[0]:
        return None
    lonely_count = random.randint(LONELY_COUNT_RANGE[0], max_lonely)

    points: list[tuple[int, int]] = []
    targets: list[tuple[int, int]] = []
    used_lines: set[int] = set()
    local_blocked = set(blocked)

    random.shuffle(pair_dists)
    for dist in pair_dists:
        candidate_lines = [line for line in free_lines if line not in used_lines]
        random.shuffle(candidate_lines)
        placed = False
        for line in candidate_lines:
            cells = (
                [(axis - dist, line), (axis + dist, line)]
                if orient == "h"
                else [(line, axis - dist), (line, axis + dist)]
            )
            if not _can_claim(local_blocked, shape, cells):
                continue
            used_lines.add(int(line))
            for cell in cells:
                points.append((int(cell[0]), int(cell[1])))
                _reserve(local_blocked, shape, cell)
            placed = True
            break
        if not placed:
            return None

    lonely_lines = [line for line in free_lines if line not in used_lines]
    random.shuffle(lonely_lines)
    lonely_lines = lonely_lines[:lonely_count]
    for line in lonely_lines:
        choices = []
        for dist in range(1, max_dist + 1):
            for side in (-1, 1):
                if orient == "h":
                    src = (axis + side * dist, line)
                    dst = (axis - side * dist, line)
                else:
                    src = (line, axis + side * dist)
                    dst = (line, axis - side * dist)
                choices.append((src, dst))
        random.shuffle(choices)

        placed = False
        for src, dst in choices:
            if not _can_claim(local_blocked, shape, [src, dst]):
                continue
            points.append((int(src[0]), int(src[1])))
            targets.append((int(dst[0]), int(dst[1])))
            _reserve(local_blocked, shape, src)
            _reserve(local_blocked, shape, dst)
            placed = True
            break
        if not placed:
            return None

    local_grid = _blank_group_grid(shape, color, points)
    try:
        inferred = _infer_axis(local_grid, color, BG)
    except AssertionError:
        return None

    if inferred["orient"] != orient or inferred["axis"] != axis:
        return None

    actual_targets = tuple(sorted(map(tuple, inferred["targets"])))
    expected_targets = tuple(sorted(targets))
    if actual_targets != expected_targets:
        return None

    return {
        "color": int(color),
        "orient": orient,
        "axis": int(axis),
        "points": tuple(sorted(points)),
        "targets": expected_targets,
        "pair_distances": inferred["pair_distances"],
    }


def _build_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        for r, c in spec["targets"]:
            out[int(r), int(c)] = int(spec["color"])
    return out


def _validate_example(grid: np.ndarray, specs: list[dict]) -> bool:
    if not specs:
        return False
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    bg = int(get_bg(grid))
    if bg != BG:
        return False

    colors = colors_present(grid, exclude={bg})
    if sorted(colors) != sorted(int(spec["color"]) for spec in specs):
        return False

    orient_set = {spec["orient"] for spec in specs}
    if orient_set != {"h", "v"}:
        return False
    if len({(spec["orient"], spec["axis"]) for spec in specs}) < 2:
        return False

    expected = _build_expected(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    for spec in specs:
        color = int(spec["color"])
        candidates = _all_axis_candidates(grid, color, bg)
        if len(candidates) != 1:
            return False
        recovered = candidates[0]
        if recovered["orient"] != spec["orient"] or recovered["axis"] != spec["axis"]:
            return False
        if tuple(sorted(recovered["targets"])) != tuple(sorted(spec["targets"])):
            return False
        if tuple(sorted(recovered["pair_distances"])) != tuple(sorted(spec["pair_distances"])):
            return False
        if len(recovered["pairs"]) < 2 or len(recovered["pair_distances"]) < 2:
            return False
        if len(recovered["lonely"]) < 1:
            return False
        for src in recovered["lonely"]:
            mirrored = _mirror_point(recovered["orient"], recovered["axis"], src)
            if int(grid[mirrored]) != bg:
                return False
            if int(actual[mirrored]) != color:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    color_count = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(PALETTE, color_count)

    orientations = ["h", "v"]
    while len(orientations) < color_count:
        orientations.append(random.choice(["h", "v"]))
    random.shuffle(orientations)

    blocked: set[tuple[int, int]] = set()
    specs: list[dict] = []

    for color, orient in zip(colors, orientations):
        axis_values = list(range(3, h - 3)) if orient == "h" else list(range(3, w - 3))
        random.shuffle(axis_values)
        spec = None
        tries = 0
        while axis_values and tries < MAX_COLOR_ATTEMPTS:
            tries += 1
            axis = axis_values.pop()
            spec = _try_place_color((h, w), blocked, int(color), orient, int(axis))
            if spec is not None:
                break
        if spec is None:
            return None

        for r, c in spec["points"]:
            if grid[int(r), int(c)] != BG:
                return None
            grid[int(r), int(c)] = int(color)
        for cell in list(spec["points"]) + list(spec["targets"]):
            _reserve(blocked, (h, w), cell)
        specs.append(spec)

    if not _validate_example(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-axis-completion input")
