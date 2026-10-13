from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG_COLOR = 5
YELLOW = 4
WALL_COLOR_CHOICES = (1, 2, 3, 6, 7, 8, 9)

GRID_MIN = 18
GRID_MAX = 24
REGION_COUNT_CHOICES = (2, 3)
INNER_H_RANGE = (5, 7)
INNER_W_RANGE = (5, 8)
REGION_MARGIN = 3
REGION_GAP = 2
MARKER_ROWS = (0, 1, -2, -1)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _edge_band_mask(shape: tuple[int, int], width: int = 2) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[:width, :] = True
    mask[h - width :, :] = True
    mask[:, :width] = True
    mask[:, w - width :] = True
    return mask


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _intersects_with_gap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _stripe_columns(c0: int, c1: int, period: int) -> list[int]:
    cols = []
    col = c0 + 1
    while col < c1:
        cols.append(col)
        col += period
    return cols


def _extract_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    edge_mask = _edge_band_mask(g.shape)
    colors = sorted(int(c) for c in np.unique(g) if int(c) not in (bg, YELLOW))

    regions: list[dict[str, object]] = []
    for color in colors:
        marker_coords = np.argwhere((g == color) & edge_mask)
        wall_coords = np.argwhere((g == color) & ~edge_mask)
        assert marker_coords.shape == (2, 2), "each wall color must have exactly two edge markers"
        assert wall_coords.size > 0, "each wall color must have one enclosed wall rectangle"

        period = abs(int(marker_coords[0, 1]) - int(marker_coords[1, 1]))
        assert period >= 1, "marker columns must differ"

        r0, c0, r1, c1 = _bbox_from_coords(wall_coords)
        wall_mask = np.zeros(g.shape, dtype=bool)
        wall_mask[r0, c0 : c1 + 1] = True
        wall_mask[r1, c0 : c1 + 1] = True
        wall_mask[r0 : r1 + 1, c0] = True
        wall_mask[r0 : r1 + 1, c1] = True
        assert np.all(g[wall_mask] == color), "walls must form a one-cell-thick rectangle"

        stripe_cols = _stripe_columns(c0, c1, period)
        assert len(stripe_cols) >= 1, "period must yield at least one interior stripe"

        regions.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "period": period,
                "stripe_cols": stripe_cols,
            }
        )
    return regions


def _apply_region_gravity(
    out: np.ndarray,
    source: np.ndarray,
    region: dict[str, object],
    bg: int,
) -> None:
    r0, c0, r1, c1 = region["bbox"]  # type: ignore[assignment]
    stripe_cols = set(region["stripe_cols"])  # type: ignore[arg-type]

    yellow_positions = np.argwhere(
        (source == YELLOW)
        & (np.arange(source.shape[0])[:, None] > r0)
        & (np.arange(source.shape[0])[:, None] < r1)
        & (np.arange(source.shape[1])[None, :] > c0)
        & (np.arange(source.shape[1])[None, :] < c1)
    )
    for r, c in yellow_positions:
        assert int(c) not in stripe_cols, "input yellows must not start on stripe columns"
        out[int(r), int(c)] = bg

    counts_by_col: dict[int, int] = {}
    for _r, c in yellow_positions:
        counts_by_col[int(c)] = counts_by_col.get(int(c), 0) + 1

    for c, count in counts_by_col.items():
        for idx in range(count):
            out[r1 - 1 - idx, c] = YELLOW


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    regions = _extract_regions(g)

    out = g.copy()
    for region in regions:
        color = int(region["color"])
        r0, c0, r1, c1 = region["bbox"]  # type: ignore[assignment]
        for c in region["stripe_cols"]:  # type: ignore[union-attr]
            for r in range(r0 + 1, r1):
                if out[r, c] == bg:
                    out[r, c] = color

    for region in regions:
        _apply_region_gravity(out, g, region, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use same-color edge-marker column gaps to place interior vertical stripes that block falling yellow pixels inside matching walled regions.

    Input structure:
    - The background is grey, with two or three enclosed rectangular regions whose one-cell-thick walls each use a distinct color.
    - Each region contains several yellow pixels on the grey interior, and the outer edge contains exactly two same-color marker pixels for each wall color.
    - The two edge markers of a color match the region with the same wall color, and their column separation is the region's hidden stripe period.

    Rule:
    - For each wall color, find its two edge markers and measure the absolute difference between their columns.
    - Inside the matching rectangular region, start at the first interior column to the right of the left wall and draw full-height vertical stripes every period cells, using the wall color and only painting background cells.
    - Then apply downward gravity to the yellow pixels inside each region, so they stack against the bottom wall and against each other while the new stripe columns remain impassable obstacles.

    Key insight:
    - The edge marker pair does not define a fill area directly; it encodes a stripe spacing parameter that only becomes meaningful after linking the pair to the same-colored enclosed region.
    """
    return _transform(grid)


def _sample_region_boxes(
    h: int,
    w: int,
    n_regions: int,
    rng: random.Random,
) -> list[dict[str, object]] | None:
    colors = rng.sample(WALL_COLOR_CHOICES, n_regions)
    regions: list[dict[str, object]] = []
    placed_boxes: list[tuple[int, int, int, int]] = []

    for color in colors:
        placed = False
        for _ in range(120):
            inner_h = rng.randint(INNER_H_RANGE[0], INNER_H_RANGE[1])
            inner_w = rng.randint(INNER_W_RANGE[0], INNER_W_RANGE[1])
            valid_periods = []
            for period in range(2, min(5, inner_w - 1) + 1):
                stripe_count = len(range(0, inner_w, period))
                free_count = inner_w - stripe_count
                if stripe_count >= 2 and free_count >= 2:
                    valid_periods.append(period)
            if not valid_periods:
                continue

            outer_h = inner_h + 2
            outer_w = inner_w + 2
            max_r0 = h - REGION_MARGIN - outer_h
            max_c0 = w - REGION_MARGIN - outer_w
            if max_r0 < REGION_MARGIN or max_c0 < REGION_MARGIN:
                continue

            r0 = rng.randint(REGION_MARGIN, max_r0)
            c0 = rng.randint(REGION_MARGIN, max_c0)
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_intersects_with_gap(box, prev, REGION_GAP) for prev in placed_boxes):
                continue

            period = rng.choice(valid_periods)
            stripe_cols = _stripe_columns(c0, c0 + outer_w - 1, period)
            free_cols = [c for c in range(c0 + 1, c0 + outer_w - 1) if c not in stripe_cols]
            if len(free_cols) < 2:
                continue

            regions.append(
                {
                    "color": color,
                    "bbox": box,
                    "period": period,
                    "stripe_cols": stripe_cols,
                    "free_cols": free_cols,
                    "inner_h": inner_h,
                }
            )
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return regions


def _paint_walls(grid: np.ndarray, region: dict[str, object]) -> None:
    color = int(region["color"])
    r0, c0, r1, c1 = region["bbox"]  # type: ignore[assignment]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_markers(
    grid: np.ndarray,
    regions: list[dict[str, object]],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    occupied: set[tuple[int, int]] = set()

    row_choices = [r if r >= 0 else h + r for r in MARKER_ROWS]
    candidate_rows = [(ra, rb) for ra in row_choices for rb in row_choices]

    for region in regions:
        color = int(region["color"])
        period = int(region["period"])
        placed = False
        starts = list(range(0, w - period))
        rng.shuffle(starts)
        row_pairs = list(candidate_rows)
        rng.shuffle(row_pairs)

        for c0 in starts:
            c1 = c0 + period
            for ra, rb in row_pairs:
                a = (ra, c0)
                b = (rb, c1)
                if a in occupied or b in occupied:
                    continue
                occupied.add(a)
                occupied.add(b)
                grid[a] = color
                grid[b] = color
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _split_total(total: int, parts: int, cap: int, rng: random.Random) -> list[int]:
    counts = [1] * parts
    remaining = total - parts
    while remaining > 0:
        idx = rng.randrange(parts)
        if counts[idx] < cap:
            counts[idx] += 1
            remaining -= 1
    return counts


def _place_yellows(
    grid: np.ndarray,
    regions: list[dict[str, object]],
    rng: random.Random,
) -> bool:
    moved_any = False

    for region in regions:
        r0, _c0, r1, _c1 = region["bbox"]  # type: ignore[assignment]
        inner_h = int(region["inner_h"])
        free_cols: list[int] = list(region["free_cols"])  # type: ignore[assignment]

        n_yellows = rng.randint(2, min(5, len(free_cols) * inner_h))
        active_cols = rng.randint(1, min(len(free_cols), n_yellows))
        chosen_cols = rng.sample(free_cols, active_cols)
        counts = _split_total(n_yellows, active_cols, inner_h, rng)

        for col, count in zip(chosen_cols, counts):
            all_rows = list(range(r0 + 1, r1))
            if len(all_rows) > count:
                preferred = all_rows[:-1]
                if len(preferred) >= count:
                    selected = sorted(rng.sample(preferred, count))
                else:
                    selected = sorted(rng.sample(all_rows, count))
            else:
                selected = sorted(all_rows)

            for row in selected:
                grid[row, col] = YELLOW

            final_rows = set(range(r1 - count, r1))
            if any(row not in final_rows for row in selected):
                moved_any = True

    return moved_any


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(grid, out):
        return False

    if not np.any(grid == YELLOW) or not np.any(out == YELLOW):
        return False

    regions = _extract_regions(grid)
    bg = _get_bg(grid)
    for region in regions:
        color = int(region["color"])
        r0, c0, r1, c1 = region["bbox"]  # type: ignore[assignment]
        stripe_cols = set(region["stripe_cols"])  # type: ignore[arg-type]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if not np.any(interior == color):
            return False
        for c in stripe_cols:
            col = out[r0 + 1 : r1, c]
            if not np.any(col == color):
                return False
        if np.any((grid[r0 + 1 : r1, c0 + 1 : c1] == YELLOW) & (out[r0 + 1 : r1, c0 + 1 : c1] == bg)):
            pass
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    n_regions = rng.choice(REGION_COUNT_CHOICES)

    regions = _sample_region_boxes(h, w, n_regions, rng)
    if regions is None:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for region in regions:
        _paint_walls(grid, region)

    if not _place_markers(grid, regions, rng):
        return None
    if not _place_yellows(grid, regions, rng):
        return None
    if not _validate_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
