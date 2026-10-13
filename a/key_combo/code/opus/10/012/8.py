import random
from collections import Counter

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
COLOR_COUNT_CHOICES = (3, 4)
SEGMENT_LENGTH_RANGE = (3, 6)
GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 200
SCENARIOS = ("overlap_unique", "overlap_tie", "no_overlap")
SCENARIO_WEIGHTS = (0.45, 0.35, 0.20)
MAX_OVERLAP_RATIO = 0.60


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _rect_from_corner(
    corner_r: int, corner_c: int, width: int, height: int, h_dir: int, v_dir: int
) -> tuple[int, int, int, int]:
    if h_dir == 1:
        c0, c1 = corner_c - width + 1, corner_c
    else:
        c0, c1 = corner_c, corner_c + width - 1
    if v_dir == 1:
        r0, r1 = corner_r - height + 1, corner_r
    else:
        r0, r1 = corner_r, corner_r + height - 1
    return int(r0), int(c0), int(r1), int(c1)


def _line_cells(
    corner_r: int, corner_c: int, width: int, height: int, h_dir: int, v_dir: int
) -> set[tuple[int, int]]:
    cells = {(corner_r, corner_c)}
    cells.update((corner_r, corner_c + step * h_dir) for step in range(width))
    cells.update((corner_r + step * v_dir, corner_c) for step in range(height))
    return cells


def _extract_shape(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, "expected foreground pixels for each color"

    rows, row_counts = np.unique(coords[:, 0], return_counts=True)
    cols, col_counts = np.unique(coords[:, 1], return_counts=True)
    pivot_rows = rows[row_counts > 1]
    pivot_cols = cols[col_counts > 1]
    assert len(pivot_rows) == 1 and len(pivot_cols) == 1, "each color must form one L-shape"

    corner_r = int(pivot_rows[0])
    corner_c = int(pivot_cols[0])
    row_cells = np.sort(coords[coords[:, 0] == corner_r][:, 1])
    col_cells = np.sort(coords[coords[:, 1] == corner_c][:, 0])
    width = int(len(row_cells))
    height = int(len(col_cells))
    assert width >= 2 and height >= 2
    assert int(len(coords)) == width + height - 1
    assert int(row_cells[-1] - row_cells[0] + 1) == width
    assert int(col_cells[-1] - col_cells[0] + 1) == height

    if int(row_cells[0]) == corner_c and int(row_cells[-1]) > corner_c:
        h_dir = 1
    elif int(row_cells[-1]) == corner_c and int(row_cells[0]) < corner_c:
        h_dir = -1
    else:
        raise AssertionError("horizontal segment must extend from the corner in one direction")

    if int(col_cells[0]) == corner_r and int(col_cells[-1]) > corner_r:
        v_dir = 1
    elif int(col_cells[-1]) == corner_r and int(col_cells[0]) < corner_r:
        v_dir = -1
    else:
        raise AssertionError("vertical segment must extend from the corner in one direction")

    rect = _rect_from_corner(corner_r, corner_c, width, height, h_dir, v_dir)
    line_cells = _line_cells(corner_r, corner_c, width, height, h_dir, v_dir)
    return {
        "color": int(color),
        "corner": (corner_r, corner_c),
        "width": width,
        "height": height,
        "area": int(width * height),
        "h_dir": h_dir,
        "v_dir": v_dir,
        "rect": rect,
        "line_cells": line_cells,
    }


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _visible_overlap_cells(spec_a: dict, spec_b: dict) -> set[tuple[int, int]]:
    overlap = _rect_overlap(spec_a["rect"], spec_b["rect"])
    if overlap is None:
        return set()
    r0, c0, r1, c1 = overlap
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    return cells - spec_a["line_cells"] - spec_b["line_cells"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the implicit rectangle encoded by each colored L-shape, with smaller inferred rectangles taking overlap priority.

    Input structure:
    - A black background contains 3-4 foreground colors.
    - Each foreground color forms exactly one connected L-shape made from a horizontal segment and a vertical segment sharing one corner pixel.
    - The arm lengths and directions of that L-shape implicitly encode one axis-aligned rectangle.

    Rule:
    - For each color, find the shared corner pixel, the horizontal arm length, and the vertical arm length.
    - The output rectangle has width equal to the horizontal segment length and height equal to the vertical segment length.
    - The rectangle expands from the corner opposite to the arms: a rightward arm means the rectangle extends leftward, and a downward arm means it extends upward.
    - Paint all inferred rectangles simultaneously; in overlap cells, the rectangle with smaller area wins, breaking ties by smaller corner row and then smaller corner column.
    - Restore every original input line pixel on top so the source L-shapes remain unchanged.

    Key insight:
    - The visible colored pixels are not the target rectangles themselves; they are one-sided horizontal and vertical arms whose shared corner, lengths, and directions completely determine hidden rectangles in the opposite quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = get_bg(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert 3 <= len(colors) <= 4, "expected 3-4 foreground colors"

    specs = [_extract_shape(g, color) for color in colors]
    out = np.full_like(g, bg)

    for spec in sorted(specs, key=lambda item: (item["area"], item["corner"][0], item["corner"][1]), reverse=True):
        r0, c0, r1, c1 = spec["rect"]
        out[r0 : r1 + 1, c0 : c1 + 1] = spec["color"]

    out[g != bg] = g[g != bg]
    return out


def _random_spec(color: int, height: int, width: int, *, area: int | None = None) -> dict | None:
    size_lo, size_hi = SEGMENT_LENGTH_RANGE
    if area is None:
        seg_w = random.randint(size_lo, size_hi)
        seg_h = random.randint(size_lo, size_hi)
    else:
        candidates = [
            (w, h)
            for w in range(size_lo, size_hi + 1)
            for h in range(size_lo, size_hi + 1)
            if w * h == area
        ]
        if not candidates:
            return None
        seg_w, seg_h = random.choice(candidates)

    if width < 2 * seg_w - 1 or height < 2 * seg_h - 1:
        return None

    h_dir = random.choice((-1, 1))
    v_dir = random.choice((-1, 1))
    corner_r = random.randint(seg_h - 1, height - seg_h)
    corner_c = random.randint(seg_w - 1, width - seg_w)
    rect = _rect_from_corner(corner_r, corner_c, seg_w, seg_h, h_dir, v_dir)
    line = _line_cells(corner_r, corner_c, seg_w, seg_h, h_dir, v_dir)
    return {
        "color": int(color),
        "corner": (corner_r, corner_c),
        "width": int(seg_w),
        "height": int(seg_h),
        "area": int(seg_w * seg_h),
        "h_dir": int(h_dir),
        "v_dir": int(v_dir),
        "rect": rect,
        "line_cells": line,
    }


def _spec_fits(spec: dict, existing: list[dict]) -> bool:
    occupied = set()
    for other in existing:
        occupied.update(other["line_cells"])
    return spec["line_cells"].isdisjoint(occupied)


def _place_pair(colors: list[int], height: int, width: int, scenario: str) -> list[dict] | None:
    color_a, color_b = colors[:2]
    tie_areas = [9, 12, 15, 16, 18, 20, 24, 25, 30, 36]
    for _ in range(PLACEMENT_ATTEMPTS):
        if scenario == "overlap_tie":
            area = random.choice(tie_areas)
            spec_a = _random_spec(color_a, height, width, area=area)
            spec_b = _random_spec(color_b, height, width, area=area)
        else:
            spec_a = _random_spec(color_a, height, width)
            spec_b = _random_spec(color_b, height, width)
            if spec_a is not None and spec_b is not None and spec_a["area"] == spec_b["area"]:
                continue
        if spec_a is None or spec_b is None:
            continue
        if not _spec_fits(spec_b, [spec_a]):
            continue
        overlap_cells = _visible_overlap_cells(spec_a, spec_b)
        if not overlap_cells:
            continue
        overlap_ratio = len(overlap_cells) / float(min(spec_a["area"], spec_b["area"]))
        if overlap_ratio > MAX_OVERLAP_RATIO:
            continue
        if spec_a["corner"] == spec_b["corner"]:
            continue
        return [spec_a, spec_b]
    return None


def _place_single(color: int, height: int, width: int, existing: list[dict], *, forbid_overlap: bool) -> dict | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        spec = _random_spec(color, height, width)
        if spec is None or not _spec_fits(spec, existing):
            continue
        if forbid_overlap and any(_rect_overlap(spec["rect"], other["rect"]) is not None for other in existing):
            continue
        return spec
    return None


def _scenario_ok(specs: list[dict], scenario: str) -> bool:
    visible_overlaps = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            cells = _visible_overlap_cells(specs[i], specs[j])
            if cells:
                visible_overlaps.append((specs[i], specs[j], cells))
    if scenario == "no_overlap":
        return not visible_overlaps
    if not visible_overlaps:
        return False
    if scenario == "overlap_tie":
        return any(a["area"] == b["area"] for a, b, _cells in visible_overlaps)
    return any(a["area"] != b["area"] for a, b, _cells in visible_overlaps)


def _build_specs(height: int, width: int, colors: list[int], scenario: str) -> list[dict] | None:
    specs: list[dict] = []
    remaining = colors[:]
    if scenario in {"overlap_unique", "overlap_tie"}:
        pair = _place_pair(remaining, height, width, scenario)
        if pair is None:
            return None
        specs.extend(pair)
        remaining = remaining[2:]

    forbid_overlap = scenario == "no_overlap"
    for color in remaining:
        spec = _place_single(color, height, width, specs, forbid_overlap=forbid_overlap)
        if spec is None:
            return None
        specs.append(spec)

    return specs if _scenario_ok(specs, scenario) else None


def _render_input(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        for r, c in spec["line_cells"]:
            assert grid[r, c] == BACKGROUND
            grid[r, c] = spec["color"]
    return grid


def _validate_generated(grid: np.ndarray, scenario: str) -> bool:
    try:
        colors = sorted(int(color) for color in np.unique(grid) if int(color) != BACKGROUND)
        if len(colors) not in COLOR_COUNT_CHOICES:
            return False
        specs = [_extract_shape(grid, color) for color in colors]
        if not _scenario_ok(specs, scenario):
            return False
        total_line = sum(len(spec["line_cells"]) for spec in specs)
        if total_line >= grid.size // 3:
            return False
        out = main(grid)
        if np.array_equal(out, grid):
            return False
        changed = int(np.sum(out != grid))
        if changed < len(colors):
            return False
    except AssertionError:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        height = random.randint(*GRID_HEIGHT_RANGE)
        width = random.randint(*GRID_WIDTH_RANGE)
        color_count = random.choice(COLOR_COUNT_CHOICES)
        colors = random.sample(PALETTE, color_count)
        scenario = random.choices(SCENARIOS, weights=SCENARIO_WEIGHTS, k=1)[0]
        specs = _build_specs(height, width, colors, scenario)
        if specs is None:
            continue
        grid = _render_input(height, width, specs)
        if _validate_generated(grid, scenario):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
