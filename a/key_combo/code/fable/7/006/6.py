from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import bbox, crop_bbox, paste


GENERATION_ATTEMPTS = 500
DIVIDER_THICKNESS = 1
PALETTE = list(range(10))


def _find_divider_rows_and_cols(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    row_colors: dict[int, list[int]] = {}
    col_colors: dict[int, list[int]] = {}

    for r in range(grid.shape[0]):
        if np.all(grid[r] == grid[r, 0]):
            row_colors.setdefault(int(grid[r, 0]), []).append(int(r))
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == grid[0, c]):
            col_colors.setdefault(int(grid[0, c]), []).append(int(c))

    candidates = sorted(set(row_colors) & set(col_colors))
    assert len(candidates) == 1, "expected one divider color"
    divider = int(candidates[0])
    return divider, row_colors[divider], col_colors[divider]


def _segments(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    assert dividers == sorted(dividers), "dividers must be sorted"
    pieces: list[tuple[int, int]] = []
    start = 0
    for idx in dividers:
        assert 0 < idx < size - 1, "divider cannot be on border"
        assert idx > start, "empty segment before divider"
        pieces.append((start, idx))
        start = idx + DIVIDER_THICKNESS
    assert start < size, "missing tail segment"
    pieces.append((start, size))
    lengths = {b - a for a, b in pieces}
    assert len(lengths) == 1 and next(iter(lengths)) > 0, "cell sizes must match"
    return pieces


def _split_cells(grid: np.ndarray) -> tuple[int, list[list[np.ndarray]]]:
    divider, divider_rows, divider_cols = _find_divider_rows_and_cols(grid)
    row_spans = _segments(grid.shape[0], divider_rows)
    col_spans = _segments(grid.shape[1], divider_cols)
    assert len(row_spans) >= 3 and len(col_spans) >= 3, "need multiple rows and columns"

    cells: list[list[np.ndarray]] = []
    for r0, r1 in row_spans:
        row_cells = []
        for c0, c1 in col_spans:
            row_cells.append(grid[r0:r1, c0:c1].copy())
        cells.append(row_cells)
    return divider, cells


def _key(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _row_backdrop_and_outlier(row_cells: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(_key(cell) for cell in row_cells)
    assert len(counts) == 2, "each row must contain one outlier and one repeated backdrop"
    majority_key, majority_count = counts.most_common(1)[0]
    assert majority_count == len(row_cells) - 1, "expected exactly one outlier cell per row"
    outlier_keys = [k for k, v in counts.items() if v == 1]
    assert len(outlier_keys) == 1

    backdrop = next(cell.copy() for cell in row_cells if _key(cell) == majority_key)
    outlier = next(cell.copy() for cell in row_cells if _key(cell) == outlier_keys[0])
    return backdrop, outlier


def _extract_sprite(backdrop: np.ndarray, outlier: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    diff = outlier != backdrop
    assert np.any(diff), "outlier cell must differ from backdrop"
    box = bbox(diff)
    assert box is not None
    r0, c0, _r1, _c1 = box
    full_patch = crop_bbox(outlier, box)
    sprite_mask = crop_bbox(diff, box).astype(bool)
    sprite_patch = np.zeros_like(full_patch)
    sprite_patch[sprite_mask] = full_patch[sprite_mask]

    rebuilt = backdrop.copy()
    paste(rebuilt, full_patch, int(r0), int(c0), mask=sprite_mask)
    assert np.array_equal(rebuilt, outlier), "difference crop must reconstruct the outlier"
    return sprite_patch, sprite_mask, (int(r0), int(c0))


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    _divider, rows = _split_cells(np.asarray(grid, dtype=int))

    backdrops: list[np.ndarray] = []
    sprite_patches: list[np.ndarray] = []
    sprite_masks: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []

    for row_cells in rows:
        backdrop, outlier = _row_backdrop_and_outlier(row_cells)
        patch, mask, pos = _extract_sprite(backdrop, outlier)
        backdrops.append(backdrop)
        sprite_patches.append(patch)
        sprite_masks.append(mask)
        positions.append(pos)

    reference_backdrop = backdrops[0]
    for backdrop in backdrops[1:]:
        assert np.array_equal(backdrop, reference_backdrop), "backdrop must stay fixed across rows"

    reference_patch = sprite_patches[0]
    reference_mask = sprite_masks[0]
    for patch in sprite_patches[1:]:
        assert np.array_equal(patch, reference_patch), "sprite colors and shape must stay fixed"
    for mask in sprite_masks[1:]:
        assert np.array_equal(mask, reference_mask), "sprite mask must stay fixed"

    deltas = [
        (positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1])
        for i in range(len(positions) - 1)
    ]
    assert deltas, "need at least two frames"
    velocity = deltas[0]
    assert velocity != (0, 0), "motion must be non-zero"
    for delta in deltas[1:]:
        assert delta == velocity, "sprite motion must use one constant velocity"

    next_pos = (positions[-1][0] + velocity[0], positions[-1][1] + velocity[1])
    h, w = reference_backdrop.shape
    sh, sw = reference_patch.shape
    assert 0 <= next_pos[0] and next_pos[0] + sh <= h
    assert 0 <= next_pos[1] and next_pos[1] + sw <= w
    return reference_backdrop, reference_patch, reference_mask, velocity, next_pos


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recover the unique moving sprite from row-wise outlier cells and output its next frame on the fixed backdrop.

    Input structure:
    - The grid is partitioned by one-pixel divider lines into equal-sized rectangular cells arranged in rows.
    - Every cell shows the same multicolor backdrop except for one outlier cell in each row.
    - The outlier contains that row's backdrop plus one extra small sprite at a translated position.

    Rule:
    - Split the grid into cells using the divider color.
    - In each row, identify the unique outlier cell against the repeated backdrop cell, then extract only the added sprite by differencing them.
    - The extracted sprite keeps the same shape and colors while its top-left bbox position shifts by one constant vector from row to row.
    - Return one cell-sized grid: the common backdrop with the sprite moved one more step beyond the last row.

    Key insight:
    - The odd cell is only a snapshot of a moving overlay; the stationary backdrop must be recovered by row-majority first, then the sprite sequence can be extrapolated independently.
    """
    backdrop, sprite_patch, sprite_mask, _velocity, next_pos = _analyze(np.asarray(input_grid, dtype=int))
    out = backdrop.copy()
    paste(out, sprite_patch, int(next_pos[0]), int(next_pos[1]), mask=sprite_mask)
    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    cells = [(r, c) for r in range(height) for c in range(width)]
    start = random.choice(cells)
    used = {start}
    frontier = {start}
    while len(used) < area:
        options = []
        for r, c in frontier:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in used:
                    options.append((rr, cc))
        if not options:
            return _random_connected_mask(height, width, area)
        nxt = random.choice(options)
        used.add(nxt)
        frontier.add(nxt)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in used:
        mask[r, c] = True
    box = bbox(mask)
    assert box is not None
    return crop_bbox(mask, box).astype(bool)


def _add_rect(canvas: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    canvas[top : top + height, left : left + width] = int(color)


def _build_backdrop(height: int, width: int, base_color: int, feature_colors: list[int]) -> np.ndarray:
    grid = np.full((height, width), int(base_color), dtype=int)
    feature_count = random.randint(2, 4)

    for i in range(feature_count):
        color = int(feature_colors[i % len(feature_colors)])
        kind = random.choice(["dot", "rect", "hline", "vline", "plus"])
        if kind == "dot":
            r = random.randrange(height)
            c = random.randrange(width)
            grid[r, c] = color
        elif kind == "rect":
            rh = random.randint(1, max(1, min(3, height - 1)))
            rw = random.randint(1, max(1, min(3, width - 1)))
            r = random.randint(0, height - rh)
            c = random.randint(0, width - rw)
            _add_rect(grid, r, c, rh, rw, color)
        elif kind == "hline":
            r = random.randrange(height)
            c0 = random.randint(0, width - 2)
            c1 = random.randint(c0 + 1, width - 1)
            grid[r, c0 : c1 + 1] = color
        elif kind == "vline":
            c = random.randrange(width)
            r0 = random.randint(0, height - 2)
            r1 = random.randint(r0 + 1, height - 1)
            grid[r0 : r1 + 1, c] = color
        else:
            r = random.randint(1, height - 2)
            c = random.randint(1, width - 2)
            grid[r, c] = color
            grid[r - 1, c] = color
            grid[r + 1, c] = color
            grid[r, c - 1] = color
            grid[r, c + 1] = color

    assert np.any(grid != base_color), "backdrop must not be blank"
    return grid


def _paint_sprite(backdrop: np.ndarray, sprite_colors: np.ndarray, sprite_mask: np.ndarray, pos: tuple[int, int]) -> np.ndarray:
    out = backdrop.copy()
    r0, c0 = pos
    h, w = sprite_mask.shape
    view = out[r0 : r0 + h, c0 : c0 + w]
    view[sprite_mask] = sprite_colors[sprite_mask]
    return out


def _valid_velocity_range(limit: int, span: int, step: int, frames: int) -> tuple[int, int] | None:
    if step >= 0:
        low = 0
        high = limit - span - frames * step
    else:
        low = -frames * step
        high = limit - span
    if low > high:
        return None
    return int(low), int(high)


def _choose_motion(
    cell_h: int, cell_w: int, sprite_h: int, sprite_w: int, rows: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    frames_to_next = rows
    candidates: list[tuple[int, int]] = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            rr = _valid_velocity_range(cell_h, sprite_h, dr, frames_to_next)
            cc = _valid_velocity_range(cell_w, sprite_w, dc, frames_to_next)
            if rr is not None and cc is not None:
                candidates.append((dr, dc))
    if not candidates:
        return None

    velocity = random.choice(candidates)
    row_range = _valid_velocity_range(cell_h, sprite_h, velocity[0], frames_to_next)
    col_range = _valid_velocity_range(cell_w, sprite_w, velocity[1], frames_to_next)
    assert row_range is not None and col_range is not None

    def choose_start(bounds: tuple[int, int]) -> int:
        lo, hi = bounds
        if lo == hi:
            return int(lo)
        if random.random() < 0.45:
            return random.choice([int(lo), int(hi)])
        return random.randint(int(lo), int(hi))

    start = (choose_start(row_range), choose_start(col_range))
    return start, velocity


def _choose_grid_geometry() -> tuple[int, int, int, int]:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    max_cell_h = min(8, (30 - (rows - 1) * DIVIDER_THICKNESS) // rows)
    max_cell_w = min(8, (30 - (cols - 1) * DIVIDER_THICKNESS) // cols)
    assert max_cell_h >= 5 and max_cell_w >= 5
    cell_h = random.randint(5, max_cell_h)
    cell_w = random.randint(5, max_cell_w)
    return rows, cols, cell_h, cell_w


def _gen_attempt() -> np.ndarray | None:
    rows, cols, cell_h, cell_w = _choose_grid_geometry()
    divider_color = random.choice(PALETTE)

    remaining = [c for c in PALETTE if c != divider_color]
    random.shuffle(remaining)
    base_color = remaining.pop()
    backdrop_feature_count = random.randint(2, 3)
    backdrop_colors = [remaining.pop() for _ in range(backdrop_feature_count)]
    sprite_palette_size = random.randint(1, 2)
    sprite_palette = [remaining.pop() for _ in range(sprite_palette_size)]

    backdrop = _build_backdrop(cell_h, cell_w, base_color, backdrop_colors)

    raw_h = random.randint(2, min(4, cell_h - 1))
    raw_w = random.randint(2, min(4, cell_w - 1))
    max_area = min(raw_h * raw_w, 6)
    sprite_mask = _random_connected_mask(raw_h, raw_w, random.randint(2, max_area))
    sprite_h, sprite_w = sprite_mask.shape

    sprite_colors = np.full((sprite_h, sprite_w), int(base_color), dtype=int)
    coords = np.argwhere(sprite_mask)
    order = list(range(len(coords)))
    random.shuffle(order)
    for i, idx in enumerate(order):
        r, c = map(int, coords[idx])
        sprite_colors[r, c] = int(sprite_palette[i % len(sprite_palette)])
    used_sprite_colors = {int(sprite_colors[r, c]) for r, c in coords}
    if len(used_sprite_colors) != len(set(sprite_palette[: len(used_sprite_colors)])):
        for color, (r, c) in zip(sprite_palette, coords):
            sprite_colors[int(r), int(c)] = int(color)

    motion = _choose_motion(cell_h, cell_w, sprite_h, sprite_w, rows)
    if motion is None:
        return None
    start, velocity = motion
    positions = [(start[0] + i * velocity[0], start[1] + i * velocity[1]) for i in range(rows + 1)]

    outlier_cols = [random.randrange(cols) for _ in range(rows)]
    if len(set(outlier_cols)) == 1 and cols > 1:
        outlier_cols[random.randrange(rows)] = (outlier_cols[0] + 1 + random.randrange(cols - 1)) % cols

    grid_h = rows * cell_h + (rows - 1) * DIVIDER_THICKNESS
    grid_w = cols * cell_w + (cols - 1) * DIVIDER_THICKNESS
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), int(divider_color), dtype=int)

    for row in range(rows):
        r0 = row * (cell_h + DIVIDER_THICKNESS)
        for col in range(cols):
            c0 = col * (cell_w + DIVIDER_THICKNESS)
            cell = backdrop.copy()
            if col == outlier_cols[row]:
                cell = _paint_sprite(cell, sprite_colors, sprite_mask, positions[row])
            paste(grid, cell, r0, c0)

    expected = _paint_sprite(backdrop, sprite_colors, sprite_mask, positions[-1])
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(expected, backdrop):
        return None
    if any(np.array_equal(expected, _paint_sprite(backdrop, sprite_colors, sprite_mask, pos)) for pos in positions[:-1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid animated-outlier input")
