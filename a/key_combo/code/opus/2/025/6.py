from __future__ import annotations

import random

import numpy as np


BG = 0
TEMPLATE_H = 3
TEMPLATE_W = 4
TEMPLATE_COLORS = (1, 3)
MARKER_COLORS = (2, 4, 5, 6, 7, 8, 9)
GRID_HEIGHT_RANGE = (15, 22)
GRID_WIDTH_RANGE = (11, 16)
MARKER_COUNT_CHOICES = (3, 4)
MIN_ROW_GAP = 4
MAX_GENERATION_ATTEMPTS = 400


def _extract_template(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid[:TEMPLATE_H, :TEMPLATE_W], dtype=int).copy()


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    coords = np.argwhere(g != BG)
    markers: list[tuple[int, int, int]] = []
    for r, c in coords:
        r = int(r)
        c = int(c)
        if r < TEMPLATE_H and c < TEMPLATE_W:
            continue
        markers.append((r, c, int(g[r, c])))
    markers.sort(key=lambda item: item[0])
    return markers


def _band_bounds(rows: list[int], height: int, *, midpoint_mode: str = "upper") -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for i, row in enumerate(rows):
        if i == 0:
            start = 0
        else:
            pair_sum = rows[i - 1] + row
            if midpoint_mode == "upper":
                start = pair_sum // 2 + 1
            elif midpoint_mode == "lower":
                start = (pair_sum + 1) // 2
            else:
                raise ValueError(f"unknown midpoint_mode: {midpoint_mode}")

        if i == len(rows) - 1:
            end = height - 1
        else:
            pair_sum = row + rows[i + 1]
            if midpoint_mode == "upper":
                end = pair_sum // 2
            elif midpoint_mode == "lower":
                end = (pair_sum - 1) // 2
            else:
                raise ValueError(f"unknown midpoint_mode: {midpoint_mode}")

        bounds.append((int(start), int(end)))
    return bounds


def _near_side_and_depth(col: int, width: int) -> tuple[str, int]:
    left_dist = int(col)
    right_dist = int(width - 1 - col)
    if left_dist <= right_dist:
        return "left", left_dist
    return "right", right_dist


def _make_stamp(
    template: np.ndarray,
    color: int,
    side: str,
    depth: int,
    *,
    flip_vertical: bool,
) -> np.ndarray:
    stamp = np.flipud(template) if flip_vertical else template.copy()
    depth = max(0, min(int(depth), stamp.shape[1]))
    if depth == 0:
        return stamp

    if side == "left":
        view = stamp[:, :depth]
    else:
        view = stamp[:, stamp.shape[1] - depth :]
    view[view != BG] = int(color)
    return stamp


def _render_from_spec(
    template: np.ndarray,
    markers: list[tuple[int, int, int]],
    shape: tuple[int, int],
    *,
    midpoint_mode: str = "upper",
    flip_even: bool = True,
    side_mode: str = "nearest",
    depth_override: int | None = None,
) -> np.ndarray | None:
    height, width = map(int, shape)
    rows = [row for row, _col, _color in markers]
    bounds = _band_bounds(rows, height, midpoint_mode=midpoint_mode)
    out = np.full((height, width), BG, dtype=int)

    for band_index, ((row, col, color), (start, end)) in enumerate(zip(markers, bounds), start=1):
        if not (start <= row <= end):
            return None
        if end - start + 1 < TEMPLATE_H:
            return None

        side, depth = _near_side_and_depth(col, width)
        if side_mode == "left":
            side = "left"
        elif side_mode != "nearest":
            raise ValueError(f"unknown side_mode: {side_mode}")
        if depth_override is not None:
            depth = int(depth_override)

        use_flip = bool(flip_even and band_index % 2 == 0)
        top = end - TEMPLATE_H + 1 if use_flip else start
        if top < 0 or top + TEMPLATE_H > height:
            return None

        stamp = _make_stamp(template, color, side, depth, flip_vertical=use_flip)
        out[top : top + TEMPLATE_H, :TEMPLATE_W] = stamp

    for row, col, color in markers:
        out[int(row), int(col)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the upper-left template into marker-defined horizontal bands, flipping even bands and recoloring edge-depth columns.

    Input structure:
    - A black background grid contains a 3x4 two-color template in the upper-left corner.
    - Outside that template window there are 3 or 4 isolated non-black marker pixels with distinct rows, columns, and colors.
    - Marker columns lie to the right of the template and close enough to a side edge to encode a repaint depth.

    Rule:
    - Ignore the template window when collecting markers; sort the markers by row and split the grid into horizontal bands at the row midpoints, assigning exact midpoint rows to the upper band.
    - In band 1, 3, 5, ... stamp the template at the band's top-left; in band 2, 4, 6, ... stamp a vertical flip at the band's bottom-left.
    - For each band's stamp, find whether its marker is closer to the left or right grid edge, then recolor that many near-side stamp columns with the marker color, changing only non-black template cells.
    - Rebuild the output from scratch so the visible result contains only the band stamps and the original marker pixels.

    Key insight:
    - Each marker is a triple-use control: its row fixes the band partition, the resulting band index fixes the flip direction, and its column-to-edge distance fixes both the repaint side and the repaint depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= TEMPLATE_H and width >= TEMPLATE_W, "grid must fit the 3x4 template window"

    template = _extract_template(g)
    markers = _extract_markers(g)
    assert 3 <= len(markers) <= 4, "expected 3 or 4 markers outside the template window"

    rows = [row for row, _col, _color in markers]
    cols = [col for _row, col, _color in markers]
    colors = [color for _row, _col, color in markers]
    assert len(rows) == len(set(rows)), "marker rows must be unique"
    assert len(cols) == len(set(cols)), "marker columns must be unique"
    assert len(colors) == len(set(colors)), "marker colors must be unique"
    assert min(cols) >= TEMPLATE_W, "markers must lie outside the template columns"

    out = _render_from_spec(template, markers, (height, width))
    assert out is not None, "input must define valid 3-row bands"
    return out


def _random_template() -> np.ndarray:
    for _ in range(200):
        bits = np.random.randint(0, 2, size=(TEMPLATE_H, TEMPLATE_W), dtype=int)
        if bits.min() == bits.max():
            continue
        if np.array_equal(bits, np.flipud(bits)):
            continue
        if np.array_equal(bits[0], bits[-1]):
            continue
        if len({tuple(row.tolist()) for row in bits}) < 2:
            continue
        return np.where(bits == 1, TEMPLATE_COLORS[0], TEMPLATE_COLORS[1]).astype(int)
    raise RuntimeError("failed to sample a non-degenerate template")


def _sample_rows(height: int, count: int) -> list[int] | None:
    candidates = list(range(1, height - 1))
    best_rows: list[int] | None = None
    best_score: tuple[int, int, int, int] | None = None

    for _ in range(300):
        rows = sorted(random.sample(candidates, count))
        gaps = [rows[i + 1] - rows[i] for i in range(count - 1)]
        if any(gap < MIN_ROW_GAP for gap in gaps):
            continue

        bounds = _band_bounds(rows, height)
        band_heights = [end - start + 1 for start, end in bounds]
        if any(h < TEMPLATE_H for h in band_heights):
            continue
        if count >= 3 and gaps[1] % 2 != 0:
            continue

        score = (
            int(any(h > TEMPLATE_H for h in band_heights)),
            int(any((i + 1) % 2 == 0 and h > TEMPLATE_H for i, h in enumerate(band_heights))),
            int(len(set(gap % 2 for gap in gaps)) > 1),
            int(len(set(band_heights)) > 1),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_rows = rows
            if score == (1, 1, 1, 1):
                return best_rows

    return best_rows


def _sample_cols(width: int, count: int) -> list[int]:
    left_col = TEMPLATE_W
    right_cols = [width - 1 - depth for depth in (1, 2, 3, 4) if width - 1 - depth >= TEMPLATE_W]
    right_cols = sorted(dict.fromkeys(right_cols))
    if len(right_cols) < count - 1:
        raise ValueError("grid is too narrow for the required marker columns")
    cols = [left_col, *random.sample(right_cols, count - 1)]
    random.shuffle(cols)
    return cols


def _build_input(
    height: int,
    width: int,
    template: np.ndarray,
    rows: list[int],
    cols: list[int],
    colors: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[:TEMPLATE_H, :TEMPLATE_W] = template
    for row, col, color in zip(rows, cols, colors):
        grid[int(row), int(col)] = int(color)
    return grid


def _is_good_instance(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        height, width = map(int, g.shape)
        template = _extract_template(g)
        markers = _extract_markers(g)
        if len(markers) not in MARKER_COUNT_CHOICES:
            return False

        rows = [row for row, _col, _color in markers]
        cols = [col for _row, col, _color in markers]
        if len(rows) != len(set(rows)) or len(cols) != len(set(cols)):
            return False
        if min(cols) < TEMPLATE_W:
            return False

        gaps = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
        if any(gap < MIN_ROW_GAP for gap in gaps):
            return False
        if len(gaps) >= 2 and gaps[1] % 2 != 0:
            return False

        bounds = _band_bounds(rows, height)
        band_heights = [end - start + 1 for start, end in bounds]
        if any(h < TEMPLATE_H for h in band_heights):
            return False
        if not any(h > TEMPLATE_H for h in band_heights):
            return False
        if not any((i + 1) % 2 == 0 and h > TEMPLATE_H for i, h in enumerate(band_heights)):
            return False

        sides_and_depths = [_near_side_and_depth(col, width) for col in cols]
        sides = {side for side, _depth in sides_and_depths}
        depths = [depth for _side, depth in sides_and_depths]
        if sides != {"left", "right"}:
            return False
        if min(depths) >= TEMPLATE_W or len(set(depths)) < 2:
            return False

        if np.array_equal(template, np.flipud(template)):
            return False

        actual = main(g)
        no_flip = _render_from_spec(template, markers, (height, width), flip_even=False)
        always_left = _render_from_spec(template, markers, (height, width), side_mode="left")
        full_depth = _render_from_spec(template, markers, (height, width), depth_override=TEMPLATE_W)
        midpoint_low = _render_from_spec(template, markers, (height, width), midpoint_mode="lower")
        if no_flip is None or always_left is None or full_depth is None or midpoint_low is None:
            return False
        if np.array_equal(actual, no_flip):
            return False
        if np.array_equal(actual, always_left):
            return False
        if np.array_equal(actual, full_depth):
            return False
        if np.array_equal(actual, midpoint_low):
            return False
        return True
    except (AssertionError, ValueError):
        return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    count = random.choice(MARKER_COUNT_CHOICES)
    template = _random_template()
    rows = _sample_rows(height, count)
    if rows is None:
        return None
    cols = _sample_cols(width, count)
    colors = random.sample(MARKER_COLORS, count)
    grid = _build_input(height, width, template, rows, cols, colors)
    return grid if _is_good_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
