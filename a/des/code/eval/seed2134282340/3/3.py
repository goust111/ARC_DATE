from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

FOLD_COLORS = (BLUE, RED, YELLOW, MAGENTA, ORANGE, AZURE, MAROON)
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for coords in _components(grid == GRAY):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        border_area = 2 * (h + w) - 4
        if len(coords) != border_area:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.all(sub[border] == GRAY) and np.all(sub[~border] != GRAY):
            candidates.append((h * w, (r0, c0, r1, c1)))
    assert candidates, "expected one gray rectangular frame"
    return max(candidates, key=lambda item: item[0])[1]


def _find_order_bar(grid: np.ndarray) -> tuple[list[int], dict[int, tuple[int, int]]]:
    fold_mask = ~np.isin(grid, (BLACK, GREEN, GRAY))
    candidates: list[tuple[int, list[int], dict[int, tuple[int, int]]]] = []
    for coords in _components(fold_mask):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 != r1:
            continue
        ordered = sorted((int(c), int(grid[r, c])) for r, c in coords)
        colors = [color for _c, color in ordered]
        if len(set(colors)) != len(colors) or len(colors) < 2:
            continue
        positions = {color: (r0, c) for c, color in ordered}
        candidates.append((len(colors), colors, positions))
    assert candidates, "expected a multicolor horizontal order bar"
    _, colors, positions = max(candidates, key=lambda item: item[0])
    return colors, positions


def _extract_folds(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    order: list[int],
    order_positions: dict[int, tuple[int, int]],
) -> list[tuple[str, int]]:
    r0, c0, r1, c1 = frame_bbox
    inner_top = r0 + 1
    inner_left = c0 + 1
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1

    folds: list[tuple[str, int]] = []
    for color in order:
        coords = np.argwhere(grid == color)
        bar_pos = order_positions[color]
        seg = np.array(
            [coord for coord in coords if (int(coord[0]), int(coord[1])) != bar_pos],
            dtype=int,
        )
        assert len(seg) > 0, "each order color must match a segment"
        rows = np.unique(seg[:, 0])
        cols = np.unique(seg[:, 1])
        if len(rows) == 1:
            row = int(rows[0])
            assert len(seg) == inner_w, "horizontal segment length must match the frame width"
            assert len(cols) == inner_w, "horizontal segment must be contiguous"
            axis = row - inner_top
            assert 0 < axis < inner_h - 1, "horizontal axis must lie strictly inside the frame"
            folds.append(("H", axis))
        else:
            assert len(cols) == 1, "segment must be horizontal or vertical"
            col = int(cols[0])
            assert len(seg) == inner_h, "vertical segment length must match the frame height"
            assert len(rows) == inner_h, "vertical segment must be contiguous"
            axis = col - inner_left
            assert 0 < axis < inner_w - 1, "vertical axis must lie strictly inside the frame"
            folds.append(("V", axis))
    return folds


def _apply_folds(mask: np.ndarray, folds: list[tuple[str, int]]) -> np.ndarray:
    canvas = mask.astype(bool, copy=True)
    top, bottom = 0, canvas.shape[0]
    left, right = 0, canvas.shape[1]

    for orient, axis in folds:
        if orient == "H":
            axis_abs = int(axis)
            assert top < axis_abs < bottom - 1
            upper = axis_abs - top
            lower = bottom - axis_abs - 1
            if upper <= lower:
                small = canvas[top:axis_abs, left:right]
                canvas[axis_abs + 1 : axis_abs + 1 + upper, left:right] |= small[::-1, :]
                top = axis_abs + 1
            else:
                small = canvas[axis_abs + 1 : bottom, left:right]
                canvas[axis_abs - lower : axis_abs, left:right] |= small[::-1, :]
                bottom = axis_abs
        else:
            axis_abs = int(axis)
            assert left < axis_abs < right - 1
            left_size = axis_abs - left
            right_size = right - axis_abs - 1
            if left_size <= right_size:
                small = canvas[top:bottom, left:axis_abs]
                canvas[top:bottom, axis_abs + 1 : axis_abs + 1 + left_size] |= small[:, ::-1]
                left = axis_abs + 1
            else:
                small = canvas[top:bottom, axis_abs + 1 : right]
                canvas[top:bottom, axis_abs - right_size : axis_abs] |= small[:, ::-1]
                right = axis_abs

    folded = canvas[top:bottom, left:right]
    return np.where(folded, GREEN, BLACK).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat colored line segments as fold instructions and fold the framed green template in the bar's color order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox = _find_frame_bbox(g)
    r0, c0, r1, c1 = frame_bbox
    pattern = g[r0 + 1 : r1, c0 + 1 : c1] == GREEN

    order, order_positions = _find_order_bar(g)
    folds = _extract_folds(g, frame_bbox, order, order_positions)

    return _apply_folds(pattern, folds)


def _sample_folds(rng: random.Random, height: int, width: int) -> tuple[list[tuple[str, int]], tuple[int, int]]:
    for _ in range(120):
        top, bottom = 0, height
        left, right = 0, width
        folds: list[tuple[str, int]] = []
        target_len = rng.randint(2, 4)

        while len(folds) < target_len:
            choices: list[str] = []
            if bottom - top >= 4:
                choices.append("H")
            if right - left >= 4:
                choices.append("V")
            if not choices:
                break

            orient = rng.choice(choices)
            if orient == "H":
                span = bottom - top
                max_small = (span - 2) // 2
                if max_small < 1:
                    break
                side = rng.choice(("top", "bottom"))
                small = rng.randint(1, max_small)
                if side == "top":
                    axis = top + small
                    top = axis + 1
                else:
                    axis = bottom - small - 1
                    bottom = axis
                folds.append(("H", axis))
            else:
                span = right - left
                max_small = (span - 2) // 2
                if max_small < 1:
                    break
                side = rng.choice(("left", "right"))
                small = rng.randint(1, max_small)
                if side == "left":
                    axis = left + small
                    left = axis + 1
                else:
                    axis = right - small - 1
                    right = axis
                folds.append(("V", axis))

        final_h = bottom - top
        final_w = right - left
        if len(folds) == target_len and final_h >= 2 and final_w >= 2:
            return folds, (final_h, final_w)
    raise RuntimeError("failed to sample a valid fold sequence")


def _sample_pattern(rng: random.Random, height: int, width: int) -> np.ndarray:
    area = height * width
    mask = np.zeros((height, width), dtype=bool)
    stamp_count = rng.randint(4, 8)
    for _ in range(stamp_count):
        kind = rng.choice(("point", "hline", "vline", "rect"))
        if kind == "point":
            r = rng.randrange(height)
            c = rng.randrange(width)
            mask[r, c] = True
        elif kind == "hline":
            length = rng.randint(2, min(4, width))
            r = rng.randrange(height)
            c0 = rng.randint(0, width - length)
            mask[r, c0 : c0 + length] = True
        elif kind == "vline":
            length = rng.randint(2, min(4, height))
            r0 = rng.randint(0, height - length)
            c = rng.randrange(width)
            mask[r0 : r0 + length, c] = True
        else:
            rh = rng.randint(2, min(3, height))
            rw = rng.randint(2, min(3, width))
            r0 = rng.randint(0, height - rh)
            c0 = rng.randint(0, width - rw)
            mask[r0 : r0 + rh, c0 : c0 + rw] = True

    count = int(mask.sum())
    min_count = max(5, area // 10)
    max_count = max(min_count, int(area * 0.4))
    if count < min_count or count > max_count:
        return np.zeros((0, 0), dtype=bool)
    return mask


def _build_grid(
    rng: random.Random,
    pattern: np.ndarray,
    folds: list[tuple[str, int]],
    colors: list[int],
) -> np.ndarray:
    height, width = pattern.shape
    placements: list[str] = []
    for orient, _axis in folds:
        if orient == "H":
            placements.append(rng.choice(("L", "R")))
        else:
            placements.append(rng.choice(("T", "B")))

    has_left = any(side == "L" for side in placements)
    has_right = any(side == "R" for side in placements)
    has_top = any(side == "T" for side in placements)
    has_bottom = any(side == "B" for side in placements)
    bar_top = rng.choice((True, False))

    row = 0
    if bar_top:
        bar_row_base = row
        row += 2
    else:
        bar_row_base = None
    top_band_start = row if has_top else None
    if has_top:
        row += height + 1
    rect_top_base = row
    row += height + 2
    if has_bottom:
        row += 1
        bottom_band_start = row
        row += height
    else:
        bottom_band_start = None
    if not bar_top:
        row += 1
        bar_row_base = row
        row += 1
    base_h = row

    col = 0
    left_band_start = col if has_left else None
    if has_left:
        col += width + 1
    rect_left_base = col
    col += width + 2
    if has_right:
        col += 1
        right_band_start = col
        col += width
    else:
        right_band_start = None
    base_w = col

    pad_top = rng.randint(1, 3)
    pad_bottom = rng.randint(1, 3)
    pad_left = rng.randint(1, 3)
    pad_right = rng.randint(1, 3)

    grid = np.zeros((base_h + pad_top + pad_bottom, base_w + pad_left + pad_right), dtype=int)
    rect_top = rect_top_base + pad_top
    rect_left = rect_left_base + pad_left
    bar_row = int(bar_row_base + pad_top)

    if top_band_start is not None:
        top_band_start += pad_top
    if bottom_band_start is not None:
        bottom_band_start += pad_top
    if left_band_start is not None:
        left_band_start += pad_left
    if right_band_start is not None:
        right_band_start += pad_left

    rect_h = height + 2
    rect_w = width + 2
    grid[rect_top, rect_left : rect_left + rect_w] = GRAY
    grid[rect_top + rect_h - 1, rect_left : rect_left + rect_w] = GRAY
    grid[rect_top : rect_top + rect_h, rect_left] = GRAY
    grid[rect_top : rect_top + rect_h, rect_left + rect_w - 1] = GRAY

    interior = grid[rect_top + 1 : rect_top + 1 + height, rect_left + 1 : rect_left + 1 + width]
    interior[pattern] = GREEN

    for (orient, axis), color, side in zip(folds, colors, placements):
        if orient == "H":
            row_idx = rect_top + 1 + axis
            col0 = int(left_band_start if side == "L" else right_band_start)
            grid[row_idx, col0 : col0 + width] = color
        else:
            col_idx = rect_left + 1 + axis
            row0 = int(top_band_start if side == "T" else bottom_band_start)
            grid[row0 : row0 + height, col_idx] = color

    bar_len = len(colors)
    min_col = 1
    max_col = grid.shape[1] - bar_len - 1
    if max_col < min_col:
        min_col = 0
        max_col = grid.shape[1] - bar_len
    bar_col = rng.randint(min_col, max_col)
    grid[bar_row, bar_col : bar_col + bar_len] = np.array(colors, dtype=int)
    return grid


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    height = rng.randint(6, 12)
    width = rng.randint(6, 12)
    folds, final_shape = _sample_folds(rng, height, width)

    pattern = _sample_pattern(rng, height, width)
    if pattern.size == 0:
        return None

    expected = _apply_folds(pattern, folds)
    if expected.shape != final_shape:
        return None
    green_count = int(np.count_nonzero(expected == GREEN))
    if green_count == 0 or green_count == expected.size:
        return None

    colors = rng.sample(FOLD_COLORS, len(folds))
    grid = _build_grid(rng, pattern, folds, colors)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        attempt = _gen_attempt(rng)
        if attempt is None:
            continue
        grid, expected = attempt
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid folding input")
