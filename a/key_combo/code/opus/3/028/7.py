from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask


BG = int(Color.BLACK)
MARKER = int(Color.RED)
WALL = int(Color.GREY)
BAR_COLOR_POOL = (
    int(Color.BLUE),
    int(Color.GREEN),
    int(Color.YELLOW),
    int(Color.PINK),
    int(Color.ORANGE),
    int(Color.TEAL),
    int(Color.MAROON),
)

GEN_MAX_ATTEMPTS = 500
TEMPLATES = ("outer_v", "outer_h", "cross_h", "cross_v")


def _paint_segment(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    value = int(color)
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = value
        return
    assert c0 == c1, "segments must be axis-aligned"
    lo, hi = sorted((r0, r1))
    grid[lo : hi + 1, c0] = value


def _extract_main_and_bar(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a main grid plus one extra color-bar row"

    main = g[:-1].copy()
    bar_row = g[-1]
    bar_cells = np.flatnonzero(bar_row != BG)
    assert 3 <= len(bar_cells) <= 4, "expected a 1x3 or 1x4 bottom color bar"
    assert int(bar_cells[-1] - bar_cells[0] + 1) == len(bar_cells), "color bar must be contiguous"

    bar = [int(v) for v in bar_row[bar_cells[0] : bar_cells[-1] + 1]]
    assert len(set(bar)) == len(bar), "color bar colors must be distinct"
    return main, bar


def _wall_grid_from_markers(main: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    g = np.asarray(main, dtype=int)
    marker_coords = np.argwhere(g == MARKER)
    assert 4 <= len(marker_coords) <= 8, "expected aligned red endpoint markers"

    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in marker_coords:
        rows[int(r)].append(int(c))
        cols[int(c)].append(int(r))

    assert all(len(cs) <= 2 for cs in rows.values()), "each row may contain at most two markers"
    assert all(len(rs) <= 2 for rs in cols.values()), "each column may contain at most two markers"

    paired_rows = sorted(r for r, cs in rows.items() if len(cs) == 2)
    paired_cols = sorted(c for c, rs in cols.items() if len(rs) == 2)
    assert 2 <= len(paired_rows) <= 3, "expected two or three aligned row pairs"
    assert 2 <= len(paired_cols) <= 3, "expected two or three aligned column pairs"

    out = np.zeros_like(g)
    for r in paired_rows:
        c0, c1 = sorted(rows[r])
        _paint_segment(out, (r, c0), (r, c1), WALL)
    for c in paired_cols:
        r0, r1 = sorted(cols[c])
        _paint_segment(out, (r0, c), (r1, c), WALL)

    return out, paired_rows, paired_cols


def _black_regions(wall_grid: np.ndarray) -> list[np.ndarray]:
    return components_mask(np.asarray(wall_grid, dtype=int) == BG, connectivity=4)


def _paint_regions_by_area(wall_grid: np.ndarray, bar: list[int]) -> tuple[np.ndarray, list[np.ndarray], list[int], list[int]]:
    regions = _black_regions(wall_grid)
    areas = [int(region.shape[0]) for region in regions]
    assert len(regions) == len(bar), "color-bar length must match the number of black regions"
    assert len(set(areas)) == len(areas), "region areas must be unique"

    order = sorted(range(len(regions)), key=lambda idx: areas[idx])
    out = np.asarray(wall_grid, dtype=int).copy()
    for rank, idx in enumerate(order):
        out[regions[idx][:, 0], regions[idx][:, 1]] = int(bar[rank])
    return out, regions, areas, order


def _anchor(coords: np.ndarray) -> tuple[int, int]:
    mins = np.min(np.asarray(coords, dtype=int), axis=0)
    return int(mins[0]), int(mins[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect aligned red endpoints into grey walls, split the board into black regions, and recolor those regions by ascending area using the bottom color bar.

    Input structure:
    - The last row is an extra color-bar row containing a contiguous 1x3 or 1x4 strip of distinct non-black colors.
    - Above that row is the main black grid with 6-8 red endpoint markers.
    - Every row and every column contains at most two red markers, yielding exactly 2-3 aligned row pairs and 2-3 aligned column pairs.

    Rule:
    - Discard the extra bottom row from the output.
    - For each row with two red markers, draw a grey horizontal segment between them, including both endpoints.
    - For each column with two red markers, draw a grey vertical segment between them, including both endpoints.
    - Treat the grey segments together with the outer border as walls, find the 4-connected black regions they enclose, sort those regions by area from small to large, and recolor them with the bottom color-bar colors from left to right.

    Key insight:
    - The red markers do not map directly to colors; they first define a wall layout, the wall layout defines irregular regions, and only the region-area ranking determines which color each region receives.
    """
    main_grid, bar = _extract_main_and_bar(grid)
    wall_grid, _paired_rows, _paired_cols = _wall_grid_from_markers(main_grid)
    out, _regions, _areas, _order = _paint_regions_by_area(wall_grid, bar)
    return out


def _render_template_walls(meta: dict, *, include_box: bool = True, include_extra: bool = True) -> np.ndarray:
    wall_grid = np.zeros((meta["H"], meta["W"]), dtype=int)
    r1, r2 = meta["r1"], meta["r2"]
    c1, c2 = meta["c1"], meta["c2"]

    if include_box:
        _paint_segment(wall_grid, (r1, c1), (r1, c2), WALL)
        _paint_segment(wall_grid, (r2, c1), (r2, c2), WALL)
        _paint_segment(wall_grid, (r1, c1), (r2, c1), WALL)
        _paint_segment(wall_grid, (r1, c2), (r2, c2), WALL)

    if include_extra:
        if meta["template"] in ("outer_v", "cross_v"):
            _paint_segment(wall_grid, (0, meta["line"]), (meta["H"] - 1, meta["line"]), WALL)
        else:
            _paint_segment(wall_grid, (meta["line"], 0), (meta["line"], meta["W"] - 1), WALL)

    return wall_grid


def _expected_output_from_meta(meta: dict) -> tuple[np.ndarray, list[np.ndarray], list[int], list[int]]:
    wall_grid = _render_template_walls(meta)
    out, regions, areas, order = _paint_regions_by_area(wall_grid, meta["bar"])
    assert len(regions) == meta["region_count"], "template should produce the intended number of regions"
    return out, regions, areas, order


def _make_input_from_meta(meta: dict) -> np.ndarray:
    main_grid = np.zeros((meta["H"], meta["W"]), dtype=int)
    r1, r2 = meta["r1"], meta["r2"]
    c1, c2 = meta["c1"], meta["c2"]

    main_grid[r1, c1] = MARKER
    main_grid[r1, c2] = MARKER
    main_grid[r2, c1] = MARKER
    main_grid[r2, c2] = MARKER

    if meta["template"] in ("outer_v", "cross_v"):
        main_grid[0, meta["line"]] = MARKER
        main_grid[meta["H"] - 1, meta["line"]] = MARKER
    else:
        main_grid[meta["line"], 0] = MARKER
        main_grid[meta["line"], meta["W"] - 1] = MARKER

    bar_row = np.zeros(meta["W"], dtype=int)
    start = meta["bar_start"]
    bar_row[start : start + len(meta["bar"])] = np.array(meta["bar"], dtype=int)
    return np.vstack([main_grid, bar_row])


def _simple_order_ok(regions: list[np.ndarray], area_order: list[int]) -> bool:
    row_major = sorted(range(len(regions)), key=lambda idx: _anchor(regions[idx]))
    col_major = sorted(range(len(regions)), key=lambda idx: (_anchor(regions[idx])[1], _anchor(regions[idx])[0]))
    return area_order != row_major and area_order != col_major


def _validate_meta_and_grid(meta: dict, grid: np.ndarray) -> bool:
    main_grid, bar = _extract_main_and_bar(grid)
    if bar != meta["bar"]:
        return False

    marker_coords = np.argwhere(main_grid == MARKER)
    if not (6 <= len(marker_coords) <= 8):
        return False

    row_counts = np.count_nonzero(main_grid == MARKER, axis=1)
    col_counts = np.count_nonzero(main_grid == MARKER, axis=0)
    if np.any(row_counts > 2) or np.any(col_counts > 2):
        return False

    paired_rows = int(np.count_nonzero(row_counts == 2))
    paired_cols = int(np.count_nonzero(col_counts == 2))
    if paired_rows not in (2, 3) or paired_cols not in (2, 3):
        return False

    try:
        expected, regions, areas, area_order = _expected_output_from_meta(meta)
    except Exception:
        return False
    if len(set(areas)) != len(areas):
        return False
    if not _simple_order_ok(regions, area_order):
        return False

    only_box = _black_regions(_render_template_walls(meta, include_box=True, include_extra=False))
    only_extra = _black_regions(_render_template_walls(meta, include_box=False, include_extra=True))
    if len(only_box) == meta["region_count"] or len(only_extra) == meta["region_count"]:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _line_choices(template: str, H: int, W: int, r1: int, r2: int, c1: int, c2: int) -> list[int]:
    if template == "outer_v":
        return list(range(1, c1 - 1)) + list(range(c2 + 2, W - 1))
    if template == "outer_h":
        return list(range(1, r1 - 1)) + list(range(r2 + 2, H - 1))
    if template == "cross_h":
        return list(range(r1 + 2, r2 - 1))
    if template == "cross_v":
        return list(range(c1 + 2, c2 - 1))
    raise ValueError(f"unknown template: {template}")


def _sample_meta() -> dict | None:
    template = random.choice(TEMPLATES)
    region_count = 3 if template.startswith("outer") else 4

    H = random.randint(11, 16)
    W = random.randint(11, 16)

    box_h = random.randint(5, min(8, H - 2))
    box_w = random.randint(5, min(8, W - 2))
    if box_h < 5 or box_w < 5:
        return None

    r1_choices = list(range(1, H - box_h))
    c1_choices = list(range(1, W - box_w))
    random.shuffle(r1_choices)
    random.shuffle(c1_choices)

    for r1 in r1_choices:
        r2 = r1 + box_h - 1
        for c1 in c1_choices:
            c2 = c1 + box_w - 1
            choices = _line_choices(template, H, W, r1, r2, c1, c2)
            if not choices:
                continue

            line = random.choice(choices)
            bar = random.sample(BAR_COLOR_POOL, region_count)
            bar_start = random.randint(0, W - region_count)
            return {
                "template": template,
                "region_count": region_count,
                "H": H,
                "W": W,
                "r1": r1,
                "r2": r2,
                "c1": c1,
                "c2": c2,
                "line": line,
                "bar": bar,
                "bar_start": bar_start,
            }
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        meta = _sample_meta()
        if meta is None:
            continue
        grid = _make_input_from_meta(meta)
        if _validate_meta_and_grid(meta, grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
