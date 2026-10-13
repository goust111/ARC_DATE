from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords


BG = Color.BLACK
LATTICE = Color.BLUE
MARKER = Color.RED
LEGEND_ROWS = 2
LEGEND_COLS = 2
BODY_START_ROW = 2
BODY_START_COL = 2
SOURCE_COLORS = (Color.GREEN, Color.ORANGE)
FILL_COLORS = (Color.YELLOW, Color.TEAL)
MAX_GENERATION_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (16, 25)
GRID_WIDTH_RANGE = (16, 27)
PERIOD_CHOICES = (5, 6)
OFFSET_CHOICES = (3, 4, 5)
ACTIVE_COUNT_CHOICES = (1, 2)


def _legend_pairs(grid: np.ndarray) -> list[tuple[int, int]]:
    assert grid.ndim == 2
    h, w = grid.shape
    assert h >= LEGEND_ROWS and w >= LEGEND_COLS

    pairs = [(int(grid[r, 0]), int(grid[r, 1])) for r in range(LEGEND_ROWS)]
    sources = [src for src, _fill in pairs]
    fills = [fill for _src, fill in pairs]
    assert len(set(sources)) == LEGEND_ROWS
    assert len(set(fills)) == LEGEND_ROWS
    assert all(src != BG and fill != BG and src != fill for src, fill in pairs)
    return pairs


def _find_marker_block(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == MARKER)
    assert coords.shape[0] == 4, "expected a unique 2x2 red marker block"
    r0, c0, r1, c1 = bbox_coords(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == MARKER)
    assert r0 >= BODY_START_ROW and c0 >= BODY_START_COL
    return int(r0), int(c0), int(r1), int(c1)


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    pairs = _legend_pairs(g)
    src_to_fill = {src: fill for src, fill in pairs}
    src_to_priority = {src: idx for idx, (src, _fill) in enumerate(pairs)}
    r0, c0, r1, c1 = _find_marker_block(g)

    controls: list[tuple[int, int, int, int]] = []
    for c in range(LEGEND_COLS, w):
        src = int(g[0, c])
        if src in src_to_fill:
            controls.append((int(c), src, int(src_to_fill[src]), int(src_to_priority[src])))

    info = {
        "grid": g,
        "shape": (h, w),
        "pairs": pairs,
        "selected_rows": (r0, r1),
        "selected_cols": (c0, c1),
        "controls": controls,
    }

    band_mask = np.zeros(g.shape, dtype=bool)
    band_mask[r0 : r1 + 1, :] = True
    band_mask[:, c0 : c1 + 1] = True
    assert np.any(band_mask & (g == LATTICE)), "selected bands must cross blue lattice pixels"
    return info


def _transform(
    grid: np.ndarray,
    *,
    fire_mode: str = "selected",
    ray_mode: str = "legend",
) -> np.ndarray:
    info = _analyze(grid)
    g = info["grid"]
    r0, r1 = info["selected_rows"]
    c0, c1 = info["selected_cols"]
    controls = info["controls"]

    out = g.copy()
    selected_mask = np.zeros(g.shape, dtype=bool)
    selected_mask[r0 : r1 + 1, :] = True
    selected_mask[:, c0 : c1 + 1] = True
    out[selected_mask & (g == LATTICE)] = MARKER

    if fire_mode not in {"selected", "all", "none"}:
        raise ValueError(f"unsupported fire_mode: {fire_mode}")
    if ray_mode not in {"legend", "identity"}:
        raise ValueError(f"unsupported ray_mode: {ray_mode}")

    if fire_mode != "none":
        fire_selected_only = fire_mode == "selected"
        # Lower-priority legend rows are painted first so earlier legend rows win.
        for c, src, fill, priority in sorted(controls, key=lambda item: item[3], reverse=True):
            if fire_selected_only and c not in (c0, c1):
                continue
            ray_color = fill if ray_mode == "legend" else src
            out[1:, c] = ray_color

    out[:LEGEND_ROWS, :LEGEND_COLS] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mark the selected lattice bands, then fire legend-mapped rays only from aligned top controls.

    Input structure:
    - A black background contains a blue lattice that starts below and to the right of a 2x2 legend area.
    - A solid 2x2 red marker block is embedded in the lattice and identifies two adjacent rows and two adjacent columns.
    - The upper-left 2x2 legend gives source-to-fill color pairs, and the top row may also contain source-colored control pixels outside the legend.

    Rule:
    - Read the red marker's two rows and two columns, then recolor every blue lattice pixel on those rows or columns to red.
    - Only top-row control pixels whose column matches one of the marker's two columns become active.
    - Each active control fires a downward ray from row 1 to the bottom, using the legend-mapped fill color instead of the control's own color, and the legend area is finally cleared to black.

    Key insight:
    - The output is determined by the intersection of three independent cues: the marker chooses where bands are active, the top controls decide which selected columns actually fire, and the legend decides what color those rays use.
    """
    return _transform(grid, fire_mode="selected", ray_mode="legend")


def _band_starts(limit: int, offset: int, period: int) -> list[int]:
    starts = []
    for start in range(int(offset), int(limit) - 1, int(period)):
        if start + 1 < int(limit):
            starts.append(int(start))
    return starts


def _build_lattice(height: int, width: int, period: int, row_offset: int, col_offset: int) -> tuple[np.ndarray, list[int], list[int]]:
    grid = np.full((height, width), BG, dtype=int)
    row_starts = _band_starts(height, row_offset, period)
    col_starts = _band_starts(width, col_offset, period)
    for r0 in row_starts:
        grid[r0 : r0 + 2, BODY_START_COL:] = LATTICE
    for c0 in col_starts:
        grid[BODY_START_ROW:, c0 : c0 + 2] = LATTICE
    return grid, row_starts, col_starts


def _expected_from_spec(grid: np.ndarray) -> np.ndarray:
    return _transform(grid, fire_mode="selected", ray_mode="legend")


def _looks_like_vertical_lattice_col(grid: np.ndarray, c: int) -> bool:
    blue_count = int(np.count_nonzero(grid[BODY_START_ROW:, int(c)] == LATTICE))
    return blue_count > (grid.shape[0] - BODY_START_ROW) // 2


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
        truth = _expected_from_spec(grid)
        bands_only = _transform(grid, fire_mode="none", ray_mode="legend")
        all_controls = _transform(grid, fire_mode="all", ray_mode="legend")
        identity_rays = _transform(grid, fire_mode="selected", ray_mode="identity")
    except AssertionError:
        return False

    selected_cols = set(info["selected_cols"])
    controls = info["controls"]
    if not any(c in selected_cols for c, _src, _fill, _priority in controls):
        return False
    if not any(c not in selected_cols for c, _src, _fill, _priority in controls):
        return False

    inactive_lattice = any(
        c not in selected_cols and _looks_like_vertical_lattice_col(grid, c)
        for c, _src, _fill, _priority in controls
    )
    inactive_non_lattice = any(
        c not in selected_cols and not _looks_like_vertical_lattice_col(grid, c)
        for c, _src, _fill, _priority in controls
    )
    if not inactive_lattice or not inactive_non_lattice:
        return False

    if np.array_equal(truth, bands_only):
        return False
    if np.array_equal(truth, all_controls):
        return False
    if np.array_equal(truth, identity_rays):
        return False

    active_controls = [(c, src, fill) for c, src, fill, _priority in controls if c in selected_cols]
    if not active_controls:
        return False

    # Rays must visibly extend into black background and selected-band recoloring must remain visible.
    if not all(np.any((grid[1:, c] == BG) & (truth[1:, c] == fill)) for c, _src, fill in active_controls):
        return False
    if np.count_nonzero((truth == MARKER) & (grid != MARKER)) == 0:
        return False
    if np.count_nonzero(truth == FILL_COLORS[0]) + np.count_nonzero(truth == FILL_COLORS[1]) == 0:
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*GRID_HEIGHT_RANGE))
    width = int(rng.integers(*GRID_WIDTH_RANGE))
    period = int(rng.choice(PERIOD_CHOICES))
    row_offset = int(rng.choice(OFFSET_CHOICES))
    col_offset = int(rng.choice(OFFSET_CHOICES))

    grid, row_starts, col_starts = _build_lattice(height, width, period, row_offset, col_offset)
    if len(row_starts) < 2 or len(col_starts) < 2:
        return None

    marker_r = int(rng.choice(row_starts))
    marker_c = int(rng.choice(col_starts))
    grid[marker_r : marker_r + 2, marker_c : marker_c + 2] = MARKER

    legend_pairs = list(zip(SOURCE_COLORS, FILL_COLORS, strict=True))
    for r, (src, fill) in enumerate(legend_pairs):
        grid[r, 0] = int(src)
        grid[r, 1] = int(fill)

    selected_cols = [marker_c, marker_c + 1]
    lattice_cols = sorted({c for c0 in col_starts for c in (c0, c0 + 1)})
    nonselected_lattice_cols = [c for c in lattice_cols if c not in selected_cols]
    nonselected_non_lattice_cols = [
        c for c in range(BODY_START_COL, width) if c not in lattice_cols and c not in selected_cols
    ]
    if not nonselected_lattice_cols or not nonselected_non_lattice_cols:
        return None

    controls: dict[int, int] = {}
    active_count = int(rng.choice(ACTIVE_COUNT_CHOICES))
    if active_count == 2:
        active_cols = selected_cols.copy()
        active_colors = list(SOURCE_COLORS)
        rng.shuffle(active_colors)
        for col, color in zip(active_cols, active_colors, strict=True):
            controls[int(col)] = int(color)
    else:
        active_col = int(rng.choice(selected_cols))
        active_color = int(rng.choice(SOURCE_COLORS))
        controls[active_col] = active_color

    inactive_cols = [int(rng.choice(nonselected_lattice_cols)), int(rng.choice(nonselected_non_lattice_cols))]
    other_source = (
        SOURCE_COLORS[0]
        if SOURCE_COLORS[0] not in controls.values()
        else SOURCE_COLORS[1]
    )
    controls[inactive_cols[0]] = int(other_source)
    controls[inactive_cols[1]] = int(rng.choice(SOURCE_COLORS))

    extra_pool = [
        c
        for c in range(BODY_START_COL, width)
        if c not in controls and c not in selected_cols
    ]
    rng.shuffle(extra_pool)
    extra_count = int(rng.integers(0, min(3, len(extra_pool)) + 1))
    for c in extra_pool[:extra_count]:
        controls[int(c)] = int(rng.choice(SOURCE_COLORS))

    for c, color in controls.items():
        grid[0, c] = int(color)

    if not _validate_generated(grid):
        return None

    expected = _expected_from_spec(grid)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
