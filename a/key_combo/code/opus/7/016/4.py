from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(TOOLBOX_DIR))

from arc_utils import Color, bbox, find_components, get_bg


BG = Color.BLACK
SHAPE_COLOR = Color.RED
PROBE_COLOR = Color.TEAL
STRIPE_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE)

GRID_H_RANGE = (10, 15)
GRID_W_RANGE = (16, 24)
LOCAL_SHAPE_W_RANGE = (3, 5)
SHAPE_AREA_RANGE = (6, 12)
STRIPE_COUNT_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 400
ORTHO_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _red_component(grid: np.ndarray) -> dict:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[SHAPE_COLOR],
    )
    assert len(comps) == 1, "expected exactly one red component"
    comp = comps[0]
    assert SHAPE_AREA_RANGE[0] <= int(comp["area"]) <= SHAPE_AREA_RANGE[1]
    r0, c0, r1, c1 = comp["bbox"]
    touches_left = c0 == 0
    touches_right = c1 == grid.shape[1] - 1
    assert touches_left ^ touches_right, "shape must touch exactly one vertical border"
    assert r0 > 0 and r1 < grid.shape[0] - 1, "shape must not touch horizontal borders"
    return comp


def _stripe_components(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=STRIPE_COLORS,
    )
    stripes = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r0 != r1:
            continue
        stripes.append(
            {
                "row": int(r0),
                "color": int(comp["color"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    stripes.sort(key=lambda item: item["row"])
    rows = [item["row"] for item in stripes]
    colors = [item["color"] for item in stripes]
    assert STRIPE_COUNT_RANGE[0] <= len(stripes) <= STRIPE_COUNT_RANGE[1]
    assert len(set(rows)) == len(rows), "stripe rows must be distinct"
    assert len(set(colors)) == len(colors), "stripe colors must be distinct"
    return stripes


def _probe_mask(grid: np.ndarray) -> np.ndarray:
    shape = grid == SHAPE_COLOR
    probe = np.zeros_like(grid, dtype=int)
    probe[np.fliplr(shape)] = PROBE_COLOR
    return probe


def _selected_stripes(grid: np.ndarray) -> list[dict]:
    probe = _probe_mask(grid)
    probe_box = bbox(probe != BG)
    assert probe_box is not None, "probe cannot be empty"
    r0, _c0, r1, _c1 = probe_box
    probe_rows = set(int(r) for r in np.argwhere(probe == PROBE_COLOR)[:, 0])

    selected = []
    for stripe in _stripe_components(grid):
        if r0 <= stripe["row"] <= r1 and stripe["row"] in probe_rows:
            selected.append(stripe)
    selected.sort(key=lambda item: item["row"])
    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the border-touching red shape into a cyan probe and read out the stripe rows it activates.

    Input structure:
    - A black wide grid contains one irregular 4-connected red shape and several one-cell-high horizontal color stripes.
    - The red shape touches exactly one vertical border, but neither horizontal border.
    - The stripes sit away from the shape, use distinct colors, and occupy distinct rows.

    Rule:
    - Determine whether the red shape touches the left or right border.
    - Reflect that red shape horizontally so the mirrored copy lands against the opposite border, and recolor only the mirrored red pixels to cyan while leaving black inside its bbox black.
    - Inspect the horizontal stripes in the original grid and keep exactly those whose row lies in the cyan probe's row span and contains cyan pixels in that row.
    - Output the selected stripe colors as a single-column grid, ordered from top stripe row to bottom stripe row.

    Key insight:
    - The mirrored shape is not part of the answer itself; it acts as a spatial probe whose occupied rows choose which stripe colors survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30
    assert get_bg(g) == BG, "expected black background"

    _red_component(g)
    selected = _selected_stripes(g)
    assert selected, "at least one stripe must be selected"

    out = np.zeros((len(selected), 1), dtype=int)
    for idx, stripe in enumerate(selected):
        out[idx, 0] = int(stripe["color"])
    return out


def _sample_stripe_layout(height: int) -> tuple[list[int], int, int] | None:
    for _ in range(200):
        n_stripes = random.randint(*STRIPE_COUNT_RANGE)
        gaps = [random.choice((2, 2, 3)) for _ in range(n_stripes - 1)]
        total_span = 1 + sum(gaps)
        max_start = height - 2 - total_span
        if max_start < 1:
            continue
        start_row = random.randint(1, max_start)
        rows = [start_row]
        for gap in gaps:
            rows.append(rows[-1] + gap)

        select_len = random.randint(2, n_stripes - 1)
        select_start = random.randint(0, n_stripes - select_len)
        top = rows[select_start]
        bottom = rows[select_start + select_len - 1]
        if bottom - top + 1 > SHAPE_AREA_RANGE[1]:
            continue
        return rows, select_start, select_len
    return None


def _shape_mask_from_cells(height: int, width: int, cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=int)
    for r, c in cells:
        mask[r, c] = SHAPE_COLOR
    return mask


def _make_border_shape(local_h: int, side: str) -> np.ndarray | None:
    assert side in {"left", "right"}
    for _ in range(300):
        local_w = random.randint(*LOCAL_SHAPE_W_RANGE)
        edge_col = 0 if side == "left" else local_w - 1
        cols = [edge_col]
        for _row in range(1, local_h):
            prev = cols[-1]
            options = {prev}
            if side == "left":
                if prev < local_w - 1:
                    options.add(prev + 1)
                if prev > 0 and random.random() < 0.35:
                    options.add(prev - 1)
            else:
                if prev > 0:
                    options.add(prev - 1)
                if prev < local_w - 1 and random.random() < 0.35:
                    options.add(prev + 1)
            cols.append(random.choice(sorted(options)))

        if side == "left" and max(cols) == 0:
            cols[random.randrange(local_h)] = 1
        if side == "right" and min(cols) == local_w - 1:
            cols[random.randrange(local_h)] = local_w - 2

        cells = {(r, c) for r, c in enumerate(cols)}
        target_area = random.randint(max(SHAPE_AREA_RANGE[0], local_h + 1), min(SHAPE_AREA_RANGE[1], local_h * local_w - 1))

        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in ORTHO_STEPS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < local_h and 0 <= nc < local_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            candidates = list(frontier)
            random.shuffle(candidates)
            placed = False
            for nr, nc in candidates:
                new_cells = cells | {(nr, nc)}
                cols_used = [c for _r, c in new_cells]
                if side == "left" and min(cols_used) != 0:
                    continue
                if side == "right" and max(cols_used) != local_w - 1:
                    continue
                cells = new_cells
                placed = True
                break
            if not placed:
                break

        mask = _shape_mask_from_cells(local_h, local_w, cells)
        shape_box = bbox(mask == SHAPE_COLOR)
        assert shape_box is not None
        r0, c0, r1, c1 = shape_box
        if r0 != 0 or r1 != local_h - 1:
            continue
        if c0 == c1:
            continue
        if int(np.count_nonzero(mask == SHAPE_COLOR)) < SHAPE_AREA_RANGE[0]:
            continue
        if int(np.count_nonzero(mask == SHAPE_COLOR)) > SHAPE_AREA_RANGE[1]:
            continue
        if (r1 - r0 + 1) * (c1 - c0 + 1) == int(np.count_nonzero(mask == SHAPE_COLOR)):
            continue
        comps = find_components(mask, bg=BG, connectivity=4, monochromatic=True, colors=[SHAPE_COLOR])
        if len(comps) != 1:
            continue
        return mask
    return None


def _paint_shape(grid: np.ndarray, shape_mask: np.ndarray, top: int, side: str) -> None:
    h, w = shape_mask.shape
    if side == "left":
        left = 0
    else:
        left = grid.shape[1] - w
    rr, cc = np.where(shape_mask == SHAPE_COLOR)
    grid[top + rr, left + cc] = SHAPE_COLOR


def _stripe_band(width: int, probe_c0: int, probe_c1: int, side: str) -> tuple[int, int]:
    if side == "left":
        band_left = max(width // 2, probe_c0 - random.randint(0, 2))
        band_right = min(width - 1, probe_c1 + random.randint(1, 3))
    else:
        band_left = max(0, probe_c0 - random.randint(1, 3))
        band_right = min(width // 2, probe_c1 + random.randint(0, 2))
    if band_left > probe_c0:
        band_left = probe_c0
    if band_right < probe_c1:
        band_right = probe_c1
    if band_right - band_left + 1 < 4:
        pad = 4 - (band_right - band_left + 1)
        band_left = max(0, band_left - pad)
        band_right = min(width - 1, band_right + pad)
    return int(band_left), int(band_right)


def _paint_stripes(grid: np.ndarray, stripe_rows: list[int], colors: list[int], side: str) -> None:
    probe = _probe_mask(grid)
    probe_box = bbox(probe == PROBE_COLOR)
    assert probe_box is not None
    _r0, probe_c0, _r1, probe_c1 = probe_box
    band_left, band_right = _stripe_band(grid.shape[1], probe_c0, probe_c1, side)

    for row, color in zip(stripe_rows, colors):
        start = random.randint(band_left, probe_c0)
        end = random.randint(probe_c1, band_right)
        if end - start + 1 < 4:
            end = min(grid.shape[1] - 2, start + 3)
        grid[row, start : end + 1] = int(color)


def _validate_semantics(grid: np.ndarray) -> None:
    comp = _red_component(grid)
    stripes = _stripe_components(grid)
    selected = _selected_stripes(grid)
    output = main(grid)

    r0, c0, r1, c1 = comp["bbox"]
    touches_left = c0 == 0
    if touches_left:
        assert all(stripe["bbox"][1] > c1 + 2 for stripe in stripes)
    else:
        assert all(stripe["bbox"][3] < c0 - 2 for stripe in stripes)

    assert 2 <= len(selected) < len(stripes), "must select at least two but not all stripes"
    expected_colors = [stripe["color"] for stripe in selected]
    assert output.shape == (len(expected_colors), 1)
    assert expected_colors == [int(v) for v in output[:, 0]]

    probe = _probe_mask(grid)
    probe_box = bbox(probe == PROBE_COLOR)
    assert probe_box is not None
    pr0, _pc0, pr1, _pc1 = probe_box
    assert sum(pr0 <= stripe["row"] <= pr1 for stripe in stripes) == len(selected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(max(height + 2, GRID_W_RANGE[0]), GRID_W_RANGE[1])

    layout = _sample_stripe_layout(height)
    if layout is None:
        return None
    stripe_rows, select_start, select_len = layout
    top = stripe_rows[select_start]
    bottom = stripe_rows[select_start + select_len - 1]

    side = random.choice(("left", "right"))
    shape = _make_border_shape(bottom - top + 1, side)
    if shape is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    _paint_shape(grid, shape, top, side)

    colors = random.sample(list(STRIPE_COLORS), len(stripe_rows))
    _paint_stripes(grid, stripe_rows, colors, side)

    try:
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
