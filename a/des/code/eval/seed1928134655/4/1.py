import random
from collections import Counter
from itertools import combinations

import numpy as np


BG = 0
DIVIDER_COLOR = 5
MARKER_COLOR = 3
PALETTE = [1, 2, 4, 6, 7, 8, 9]

DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)
ALL_DIRS = [DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT]

# Each legend entry is a single connected component: one square pixel on the left
# plus a 4-pixel arrow glyph on the right.
LEGEND_SHAPES = {
    DIR_RIGHT: frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}),
    DIR_LEFT: frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
    DIR_UP: frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (1, 3)}),
    DIR_DOWN: frozenset({(0, 0), (0, 1), (0, 2), (0, 3), (1, 2)}),
}
SHAPE_TO_DIR = {shape: direction for direction, shape in LEGEND_SHAPES.items()}

GEN_ATTEMPTS = 400
FLAG_ATTEMPTS = 200


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray, row_offset: int = 0, col_offset: int = 0) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr + row_offset, cc + col_offset))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _norm_shape(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _rle(values: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))
    return runs


def _find_divider(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(w):
        column = grid[:, c]
        if int(column[0]) != bg and np.all(column == column[0]):
            candidates.append(c)
    assert len(candidates) == 1, "expected one full-height divider column"
    divider = int(candidates[0])
    assert 0 < divider < w - 1
    return divider


def _trace_flag(sequence: list[int], color_to_dir: dict[int, tuple[int, int]]) -> dict | None:
    runs = _rle(sequence)
    cur_r, cur_c = 0, 0
    visited = {(0, 0)}
    cells: list[tuple[int, int, int]] = []
    for color, length in runs:
        dr, dc = color_to_dir[color]
        for _ in range(length):
            cur_r += dr
            cur_c += dc
            if (cur_r, cur_c) in visited:
                return None
            visited.add((cur_r, cur_c))
            cells.append((cur_r, cur_c, color))
    rows = [0] + [r for r, _c, _color in cells]
    cols = [0] + [c for _r, c, _color in cells]
    return {
        "runs": runs,
        "cells": cells,
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the color legend, read each vertical flag as run-length commands,
    and draw the corresponding colored paths from the sorted start markers.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    divider = _find_divider(g, bg)

    left = g[:, :divider]
    right = g[:, divider + 1 :]

    legend: dict[int, tuple[int, int]] = {}
    flags: list[dict] = []
    for comp in _components(left != bg):
        shape = _norm_shape(comp)
        r0, c0, r1, c1 = _bbox(comp)
        color = int(g[int(comp[0, 0]), int(comp[0, 1])])
        if shape in SHAPE_TO_DIR:
            legend[color] = SHAPE_TO_DIR[shape]
            continue
        width = c1 - c0 + 1
        height = r1 - r0 + 1
        if width == 1 and len(comp) == height and 2 <= height <= 5:
            flags.append({"bbox": (r0, c0, r1, c1), "coords": comp})
            continue
        raise AssertionError("unrecognized left-side object")

    assert legend, "legend is required"
    assert flags, "at least one flag is required"

    markers = []
    for comp in _components(right != bg, col_offset=divider + 1):
        assert len(comp) == 1, "right side should contain only singleton markers"
        markers.append((int(comp[0, 0]), int(comp[0, 1])))

    flags.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    markers.sort(key=lambda rc: (rc[0], rc[1]))
    assert len(flags) == len(markers)

    marker_cells = set(markers)
    out = g.copy()
    for flag, (start_r, start_c) in zip(flags, markers):
        col = int(flag["bbox"][1])
        rows = sorted(int(r) for r in flag["coords"][:, 0])
        sequence = [int(g[r, col]) for r in rows]
        cur_r, cur_c = start_r, start_c
        for color, length in _rle(sequence):
            assert color in legend, "flag color missing from legend"
            dr, dc = legend[color]
            for _ in range(length):
                cur_r += dr
                cur_c += dc
                assert 0 <= cur_r < h and divider < cur_c < w
                assert (cur_r, cur_c) not in marker_cells
                out[cur_r, cur_c] = color

    return out


def _sample_lengths(rng: random.Random, total: int) -> list[int]:
    max_runs = min(4, total)
    if total == 2:
        run_count = rng.choice([1, 2])
    else:
        run_count = rng.choices(
            population=list(range(1, max_runs + 1)),
            weights=[1] + [3] * (max_runs - 1),
            k=1,
        )[0]
    cuts = sorted(rng.sample(range(1, total), run_count - 1))
    points = [0] + cuts + [total]
    return [points[i + 1] - points[i] for i in range(len(points) - 1)]


def _sample_flag_sequence(rng: random.Random, colors: list[int]) -> list[int]:
    total = rng.randint(2, 5)
    lengths = _sample_lengths(rng, total)
    run_colors = [rng.choice(colors)]
    for _ in range(1, len(lengths)):
        choices = [color for color in colors if color != run_colors[-1]]
        run_colors.append(rng.choice(choices))
    sequence: list[int] = []
    for color, length in zip(run_colors, lengths):
        sequence.extend([color] * length)
    return sequence


def _choose_flag_columns(rng: random.Random, columns: list[int], count: int) -> list[int] | None:
    valid = [combo for combo in combinations(columns, count) if all(b - a >= 2 for a, b in zip(combo, combo[1:]))]
    if not valid:
        return None
    return list(rng.choice(valid))


def _choose_flag_rows(rng: random.Random, heights: list[int], legend_top: int) -> list[int] | None:
    count = len(heights)
    max_starts = [legend_top - height for height in heights]
    rows: list[int] = []

    def rec(i: int, prev: int) -> bool:
        if i == count:
            return True
        low = 0 if i == 0 else prev + 1
        remaining = count - i - 1
        high = min(max_starts[i], legend_top - remaining - 1)
        if low > high:
            return False
        choices = list(range(low, high + 1))
        rng.shuffle(choices)
        for value in choices:
            rows.append(value)
            if rec(i + 1, value):
                return True
            rows.pop()
        return False

    return rows if rec(0, -1) else None


def _band_bounds(height: int, count: int) -> list[tuple[int, int]]:
    return [(i * height // count, (i + 1) * height // count - 1) for i in range(count)]


def _place_markers(
    rng: random.Random,
    path_infos: list[dict],
    divider: int,
    width: int,
    height: int,
) -> list[tuple[int, int]] | None:
    markers = []
    right_min = divider + 1
    right_max = width - 1
    for band, info in zip(_band_bounds(height, len(path_infos)), path_infos):
        min_r, min_c, max_r, max_c = info["bbox"]
        obj_min_r = min(0, min_r)
        obj_max_r = max(0, max_r)
        obj_min_c = min(0, min_c)
        obj_max_c = max(0, max_c)

        row_lo = band[0] - obj_min_r
        row_hi = band[1] - obj_max_r
        col_lo = right_min - obj_min_c
        col_hi = right_max - obj_max_c
        if row_lo > row_hi or col_lo > col_hi:
            return None

        row = rng.randint(row_lo, row_hi)
        col = rng.randint(col_lo, col_hi)
        markers.append((row, col))
    return markers


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    flag_count = rng.randint(2, 4)
    height = rng.randint(max(24, flag_count * 6), 30)
    divider = rng.randint(14, 16)
    width = rng.randint(divider + 11, 30)
    legend_top = height - 16
    if legend_top < 5:
        return None

    legend_col = rng.randint(1, 2)
    flag_columns = list(range(legend_col + 5, divider - 1))
    chosen_columns = _choose_flag_columns(rng, flag_columns, flag_count)
    if chosen_columns is None:
        return None

    legend_colors = rng.sample(PALETTE, 4)
    directions = rng.sample(ALL_DIRS, 4)
    color_to_dir = dict(zip(legend_colors, directions))

    sequences = []
    path_infos = []
    for _ in range(FLAG_ATTEMPTS):
        sequences = []
        path_infos = []
        used_colors = set()
        has_repeat = False
        has_multi_run = False
        for _flag_idx in range(flag_count):
            sequence = _sample_flag_sequence(rng, legend_colors)
            info = _trace_flag(sequence, color_to_dir)
            if info is None:
                break
            sequences.append(sequence)
            path_infos.append(info)
            used_colors.update(sequence)
            has_repeat = has_repeat or any(length > 1 for _color, length in info["runs"])
            has_multi_run = has_multi_run or len(info["runs"]) > 1
        if len(sequences) != flag_count:
            continue
        if used_colors != set(legend_colors):
            continue
        if not has_repeat or not has_multi_run:
            continue
        if any(max(0, info["bbox"][2]) - min(0, info["bbox"][0]) + 1 > (height // flag_count) for info in path_infos):
            continue
        break
    else:
        return None

    flag_rows = _choose_flag_rows(rng, [len(sequence) for sequence in sequences], legend_top)
    if flag_rows is None:
        return None

    markers = _place_markers(rng, path_infos, divider, width, height)
    if markers is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[:, divider] = DIVIDER_COLOR

    legend_entries = list(color_to_dir.items())
    rng.shuffle(legend_entries)
    for idx, (color, direction) in enumerate(legend_entries):
        row0 = legend_top + idx * 4
        col0 = legend_col
        for dr, dc in LEGEND_SHAPES[direction]:
            grid[row0 + dr, col0 + dc] = color

    for row0, col0, sequence, marker in zip(flag_rows, chosen_columns, sequences, markers):
        for dr, color in enumerate(sequence):
            grid[row0 + dr, col0] = color
        grid[marker[0], marker[1]] = MARKER_COLOR

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
