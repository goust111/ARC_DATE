from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7

SOURCE_COLORS = (BLUE, GREEN, ORANGE)
GEN_HEIGHTS = (19, 21, 23, 25)
GEN_WIDTH_RANGE = (11, 15)
GEN_ATTEMPTS = 400
LOCAL_ATTEMPTS = 600


def _outward_offset(row: int, r0: int, r1: int, bar_height: int) -> int:
    if row < r0:
        return ((r0 - 1 - row) // bar_height) + 1
    if row > r1:
        return ((row - r1 - 1) // bar_height) + 1
    return 0


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == RED)]
    assert len(rows) == 1, "expected exactly one full-width red separator row"
    return int(rows[0])


def _parse_pair(grid: np.ndarray, color: int, separator_r: int) -> dict:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"missing color {color}"

    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    assert len(cols) == 2, "each source color must form exactly two bars"

    r0, r1 = rows[0], rows[-1]
    height = r1 - r0 + 1
    assert rows == list(range(r0, r1 + 1)), "bars must occupy one contiguous row interval"
    assert len(coords) == 2 * height, "bars must be one-pixel-wide and equal-height"

    left, right = cols
    for c in (left, right):
        col_rows = sorted(int(r) for r in np.flatnonzero(grid[:, c] == color))
        assert col_rows == rows, "the two bars must share the same row interval"

    side = "above" if r1 < separator_r else "below" if r0 > separator_r else None
    assert side is not None, "bars must lie entirely on one side of the separator"
    return {
        "color": int(color),
        "r0": int(r0),
        "r1": int(r1),
        "left": int(left),
        "right": int(right),
        "height": int(height),
        "side": side,
    }


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_r = _separator_row(g)
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in {BLACK, RED})
    assert 2 <= len(colors) <= 3, "expected two or three source-color pairs"
    assert YELLOW not in colors and PINK not in colors, "yellow/pink are reserved output colors"
    return [_parse_pair(g, color, separator_r) for color in colors]


def _emit_pair(spec: dict, shape: tuple[int, int], separator_r: int) -> dict[str, set[tuple[int, int]]]:
    height, width = map(int, shape)
    cells = {"top": set(), "bottom": set()}

    for row in range(height):
        offset = _outward_offset(row, spec["r0"], spec["r1"], spec["height"])
        left = spec["left"] - offset
        right = spec["right"] + offset
        for col in (left, right):
            if not (0 <= col < width) or row == separator_r:
                continue
            half = "top" if row < separator_r else "bottom"
            cells[half].add((row, col))
    return cells


def _solve_from_specs(shape: tuple[int, int], separator_r: int, specs: list[dict]) -> tuple[np.ndarray, list[dict], dict]:
    out = np.full(shape, BLACK, dtype=int)
    out[separator_r, :] = RED

    long_claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    short_claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    pair_infos: list[dict] = []

    for spec in specs:
        cells = _emit_pair(spec, shape, separator_r)
        top_count = len(cells["top"])
        bottom_count = len(cells["bottom"])
        assert top_count != bottom_count, "side areas must differ"
        long_side = "top" if top_count > bottom_count else "bottom"
        short_side = "bottom" if long_side == "top" else "top"

        for cell in cells[long_side]:
            long_claims[cell].add(spec["color"])
        for cell in cells[short_side]:
            short_claims[cell].add(spec["color"])

        pair_infos.append(
            {
                **spec,
                "top_cells": cells["top"],
                "bottom_cells": cells["bottom"],
                "top_count": top_count,
                "bottom_count": bottom_count,
                "long_side": long_side,
                "short_side": short_side,
            }
        )

    all_cells = set(long_claims) | set(short_claims)
    for cell in all_cells:
        if cell in long_claims:
            out[cell] = YELLOW
        else:
            colors = short_claims[cell]
            out[cell] = next(iter(colors)) if len(colors) == 1 else PINK

    cell_info = {
        cell: {"long": set(long_claims.get(cell, set())), "short": set(short_claims.get(cell, set()))}
        for cell in all_cells
    }
    return out, pair_infos, cell_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each same-color bar pair into stair rays, repaint the longer separator side yellow, and keep short-side cross-color overlaps pink.
    
    Input structure:
    - The background is black and one full-width red row acts as the separator.
    - Two or three source colors each appear as exactly two equal-height vertical bars sharing one row interval.
    - Every bar pair lies entirely above or entirely below the separator.
    
    Rule:
    - For each pair, extend both bars upward and downward across the whole grid: inside the original row interval the columns stay fixed, and outside it the left bar shifts left and the right bar shifts right after every full bar-height block of rows.
    - Count that pair's emitted pixels above the separator and below the separator; the side with more pixels is the long side.
    - Repaint the entire long-side trajectory yellow and keep the short-side trajectory in the pair's original color.
    - On short-side cells only, any overlap claimed by two or more different source colors becomes pink; if any claimant treats that cell as long-side, it stays yellow.
    - Keep the red separator row unchanged.
    
    Key insight:
    - The bars encode the same stair-stepping rays as the classic outward-extension task, but the separator turns each pair into an asymmetric two-side comparison: only the shorter side still uses overlap-to-pink, while the longer side is completely absorbed into yellow.
    """
    g = np.asarray(grid, dtype=int)
    separator_r = _separator_row(g)
    specs = _extract_pairs(g)
    out, _pair_infos, _cell_info = _solve_from_specs(g.shape, separator_r, specs)
    return out


def _input_cells(spec: dict) -> set[tuple[int, int]]:
    cells = set()
    for row in range(spec["r0"], spec["r1"] + 1):
        cells.add((row, spec["left"]))
        cells.add((row, spec["right"]))
    return cells


def _random_pair(height: int, width: int, separator_r: int, side: str, color: int) -> dict | None:
    gap_options = [gap for gap in range(2, 5) if width - gap - 2 >= 1]
    if not gap_options:
        return None

    bar_height = random.randint(2, 3)
    gap = random.choice(gap_options)
    left = random.randint(1, width - gap - 2)

    if side == "above":
        max_r0 = separator_r - bar_height - 1
        if max_r0 < 1:
            return None
        r0 = random.randint(1, max_r0)
    else:
        min_r0 = separator_r + 2
        max_r0 = height - bar_height - 1
        if min_r0 > max_r0:
            return None
        r0 = random.randint(min_r0, max_r0)

    return {
        "color": int(color),
        "r0": int(r0),
        "r1": int(r0 + bar_height - 1),
        "left": int(left),
        "right": int(left + gap),
        "height": int(bar_height),
        "side": side,
    }


def _build_core_group(height: int, width: int, separator_r: int, side: str, colors: tuple[int, int]) -> list[dict] | None:
    for _ in range(LOCAL_ATTEMPTS):
        first = _random_pair(height, width, separator_r, side, colors[0])
        second = _random_pair(height, width, separator_r, side, colors[1])
        if first is None or second is None:
            continue
        if _input_cells(first) & _input_cells(second):
            continue

        try:
            _out, infos, _cell_info = _solve_from_specs((height, width), separator_r, [first, second])
        except AssertionError:
            continue
        if infos[0]["long_side"] != infos[1]["long_side"]:
            continue
        long_side = infos[0]["long_side"]
        short_side = infos[0]["short_side"]

        short_overlap = infos[0][f"{short_side}_cells"] & infos[1][f"{short_side}_cells"]
        long_overlap = infos[0][f"{long_side}_cells"] & infos[1][f"{long_side}_cells"]
        if not short_overlap or not long_overlap:
            continue
        if len(short_overlap) > 8 or len(long_overlap) > 10:
            continue
        return [first, second]
    return None


def _render_input(height: int, width: int, separator_r: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[separator_r, :] = RED
    for spec in specs:
        for row, col in _input_cells(spec):
            assert grid[row, col] == BLACK
            grid[row, col] = spec["color"]
    return grid


def _pair_matters(shape: tuple[int, int], separator_r: int, specs: list[dict], out: np.ndarray, idx: int) -> bool:
    reduced = specs[:idx] + specs[idx + 1 :]
    try:
        other, _infos, _cell_info = _solve_from_specs(shape, separator_r, reduced)
    except AssertionError:
        return True
    return not np.array_equal(other, out)


def _accept_generated(grid: np.ndarray, expected_pairs: int) -> bool:
    try:
        specs = _extract_pairs(grid)
        separator_r = _separator_row(grid)
        out, infos, cell_info = _solve_from_specs(grid.shape, separator_r, specs)
        solved = main(grid)
    except AssertionError:
        return False

    if len(specs) != expected_pairs:
        return False
    if not np.array_equal(out, solved):
        return False
    if int(np.count_nonzero(out == PINK)) == 0:
        return False
    if int(np.count_nonzero(out == YELLOW)) == 0:
        return False
    if int(np.count_nonzero(out)) < 28:
        return False

    has_magenta_overlap = any(not data["long"] and len(data["short"]) >= 2 for data in cell_info.values())
    has_yellow_overlap = any(data["long"] and (len(data["long"]) + len(data["short"]) >= 2) for data in cell_info.values())
    if not has_magenta_overlap or not has_yellow_overlap:
        return False

    source_colors = {info["color"] for info in infos}
    if any(int(np.count_nonzero(out == color)) == 0 for color in source_colors):
        return False

    for idx in range(len(specs)):
        if not _pair_matters(grid.shape, separator_r, specs, out, idx):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(GEN_HEIGHTS)
    width = random.randint(*GEN_WIDTH_RANGE)
    separator_r = height // 2

    pair_count = random.choice((2, 3))
    colors = random.sample(SOURCE_COLORS, pair_count)

    base_side = random.choice(("above", "below"))
    core = _build_core_group(height, width, separator_r, base_side, (colors[0], colors[1]))
    if core is None:
        return None

    specs = list(core)
    if pair_count == 3:
        extra_side = random.choice((base_side, "below" if base_side == "above" else "above"))
        for _ in range(LOCAL_ATTEMPTS):
            extra = _random_pair(height, width, separator_r, extra_side, colors[2])
            if extra is None:
                continue
            if any(_input_cells(extra) & _input_cells(spec) for spec in specs):
                continue
            candidate = specs + [extra]
            grid = _render_input(height, width, separator_r, candidate)
            if _accept_generated(grid, 3):
                return grid
        return None

    grid = _render_input(height, width, separator_r, specs)
    return grid if _accept_generated(grid, 2) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
