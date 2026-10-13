from __future__ import annotations

from collections import deque
import random

import numpy as np


PALETTE = (1, 2, 3, 4, 7)
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        sprite[r - r0, c - c0] = int(grid[r, c])
    return sprite


def _sprite_key(sprite: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return sprite.shape, tuple(int(v) for v in sprite.ravel())


def _column_color(column: np.ndarray) -> int:
    colors = {int(v) for v in column if int(v) != 0}
    assert len(colors) <= 1, "each band column must contain at most one foreground color"
    return next(iter(colors)) if colors else 0


def _detect_bands(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    col_colors = [_column_color(g[:, c]) for c in range(g.shape[1])]
    active_cols = [c for c, color in enumerate(col_colors) if color != 0]
    assert active_cols, "expected at least one foreground band"
    left, right = active_cols[0], active_cols[-1]
    assert all(col_colors[c] != 0 for c in range(left, right + 1)), "bands must be horizontally adjacent"

    bands: list[dict] = []
    c = left
    while c <= right:
        color = int(col_colors[c])
        c0 = c
        while c + 1 <= right and col_colors[c + 1] == color:
            c += 1
        c1 = c
        width = c1 - c0 + 1
        assert 2 <= width <= 3, "band width must be 2 or 3"

        rows, cols = np.where(g[:, c0 : c1 + 1] == color)
        assert rows.size >= 3, "each band needs a nontrivial source pattern"
        r0, r1 = int(rows.min()), int(rows.max())
        band_box = g[r0 : r1 + 1, c0 : c1 + 1]
        assert set(int(v) for v in np.unique(band_box)) <= {0, color}, "band box may contain only its own color and black"
        mask = band_box == color
        assert np.any(~mask), "each band needs black cells to borrow into"
        assert all(mask[:, dc].any() for dc in range(width)), "each band column must contain some source color"
        band_components = _components_from_mask(mask)
        assert len(band_components) == 1, "each band's source pixels must form one 4-connected pattern"
        area = len(band_components[0])
        assert 3 <= area <= 8, "band source area must stay within the intended range"

        bands.append(
            {
                "color": color,
                "c0": c0,
                "c1": c1,
                "r0": r0,
                "r1": r1,
                "mask": mask.copy(),
            }
        )
        c += 1

    assert len(bands) >= 2, "expected multiple bands"
    return bands


def _fill_bands(grid: np.ndarray, bands: list[dict], *, borrow: str = "right", use_bbox_rows: bool = True) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    n = len(bands)
    for i, band in enumerate(bands):
        if borrow == "right":
            fill_color = int(bands[(i + 1) % n]["color"])
        elif borrow == "left":
            fill_color = int(bands[(i - 1) % n]["color"])
        else:
            raise ValueError("borrow must be 'right' or 'left'")

        c0, c1 = int(band["c0"]), int(band["c1"])
        if use_bbox_rows:
            r0, r1 = int(band["r0"]), int(band["r1"])
            view = out[r0 : r1 + 1, c0 : c1 + 1]
        else:
            view = out[:, c0 : c1 + 1]
        view[view == 0] = fill_color
    return out


def _component_crops(grid: np.ndarray) -> list[np.ndarray]:
    comps = _components_from_mask(np.asarray(grid, dtype=int) != 0)
    return [_crop_component(np.asarray(grid, dtype=int), coords) for coords in comps]


def _unique_duplicate_crop(grid: np.ndarray) -> np.ndarray | None:
    samples: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    counts: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}
    for sprite in _component_crops(grid):
        key = _sprite_key(sprite)
        samples[key] = sprite
        counts[key] = counts.get(key, 0) + 1
    dup_keys = [key for key, count in counts.items() if count == 2]
    if len(dup_keys) != 1 or any(count > 2 for count in counts.values()):
        return None
    return samples[dup_keys[0]].copy()


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict], np.ndarray]:
    bands = _detect_bands(grid)
    filled = _fill_bands(grid, bands, borrow="right", use_bbox_rows=True)
    duplicate = _unique_duplicate_crop(filled)
    assert duplicate is not None, "expected exactly one repeated filled sprite"
    assert len({int(v) for v in np.unique(duplicate)} - {0}) >= 2, "the repeated sprite must be multicolor"
    return duplicate, bands, filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each vertical band's internal black gaps with the next band's color, then crop the only filled sprite that appears twice.

    Input structure:
    - The canvas is black except for 5 adjacent vertical bands, each 2-3 columns wide.
    - Each band uses exactly one visible source color; within that band's tight row-bounded box, cells are either that color or black.
    - A band's source-color cells form one 4-connected irregular pattern, and identical source colors may reappear in later non-adjacent bands.

    Rule:
    - Order the bands from left to right by their column spans.
    - For each band, fill only the black cells inside its own tight box with the color of the next band to the right, wrapping from the last band back to the first.
    - In the filled result, extract every 4-connected non-black component, compare their tight crops including exact colors and any internal black holes, and return the unique crop that occurs exactly twice.

    Key insight:
    - The repeated sprite is not present in the raw input: it appears only after each band borrows the next band's color inside its own box.
    """
    duplicate, _bands, _filled = _solve(np.asarray(grid, dtype=int))
    return duplicate


def _random_connected_mask(height: int, width: int) -> np.ndarray | None:
    max_cells = min(8, height * width - 1)
    min_cells = max(3, width + 1)
    if min_cells > max_cells:
        return None

    for _ in range(250):
        target = random.randint(min_cells, max_cells)
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if len(_components_from_mask(mask)) != 1:
            continue
        if not mask[0, :].any() or not mask[-1, :].any():
            continue
        if not all(mask[:, c].any() for c in range(width)):
            continue
        if not np.any(~mask):
            continue
        return mask
    return None


def _different_mask(shape: tuple[int, int], forbidden: list[np.ndarray]) -> np.ndarray | None:
    h, w = shape
    for _ in range(200):
        mask = _random_connected_mask(h, w)
        if mask is None:
            continue
        if any(mask.shape == other.shape and np.array_equal(mask, other) for other in forbidden):
            continue
        return mask
    return None


def _place_intervals(heights: list[int], overlap_unique_pair: bool) -> tuple[list[int], int]:
    h0, h1, h2, h3, h4 = heights
    top = random.randint(0, 2)
    gap01 = random.randint(1, 3)
    if overlap_unique_pair:
        overlap12 = random.randint(1, min(h1, h2) - 1)
        gap12 = -overlap12
    else:
        gap12 = random.randint(1, 3)
    gap23 = random.randint(1, 3)
    gap34 = random.randint(1, 3)
    bottom = random.randint(0, 2)

    starts = [0] * 5
    starts[0] = top
    starts[1] = starts[0] + h0 + gap01
    starts[2] = starts[1] + h1 + gap12
    end12 = max(starts[1] + h1, starts[2] + h2)
    starts[3] = end12 + gap23
    starts[4] = starts[3] + h3 + gap34
    total_height = starts[4] + h4 + bottom
    return starts, total_height


def _build_grid(
    colors: list[int],
    widths: list[int],
    heights: list[int],
    masks: list[np.ndarray],
    row_starts: list[int],
    *,
    left_margin: int,
    right_margin: int,
) -> np.ndarray:
    total_width = left_margin + sum(widths) + right_margin
    total_height = max(start + height for start, height in zip(row_starts, heights))
    grid = np.zeros((total_height, total_width), dtype=int)
    col = left_margin
    for color, width, height, mask, row in zip(colors, widths, heights, masks, row_starts):
        assert mask.shape == (height, width)
        view = grid[row : row + height, col : col + width]
        view[mask] = color
        col += width
    return grid


def _original_duplicate_crops(grid: np.ndarray) -> list[np.ndarray]:
    samples: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    counts: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}
    for sprite in _component_crops(grid):
        key = _sprite_key(sprite)
        samples[key] = sprite
        counts[key] = counts.get(key, 0) + 1
    return [samples[key] for key, count in counts.items() if count >= 2]


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        duplicate, bands, filled = _solve(grid)
    except AssertionError:
        return False

    if grid.ndim != 2 or filled.ndim != 2 or max(grid.shape) > 30 or max(duplicate.shape) > 30:
        return False
    if len(bands) != 5:
        return False

    if np.array_equal(main(grid), duplicate) is False:
        return False

    band_colors = [int(band["color"]) for band in bands]
    pair_keys = [(band_colors[i], band_colors[(i + 1) % len(bands)]) for i in range(len(bands))]
    if sum(pair == pair_keys[0] for pair in pair_keys) != 2:
        return False
    if pair_keys[0] != pair_keys[3]:
        return False
    if any(pair_keys[i] == pair_keys[0] for i in (1, 2, 4)):
        return False

    if not all(np.any(~band["mask"]) for band in bands):
        return False

    original_dups = _original_duplicate_crops(grid)
    if any(np.array_equal(sprite, duplicate) for sprite in original_dups):
        return False

    left_filled = _fill_bands(grid, bands, borrow="left", use_bbox_rows=True)
    left_duplicate = _unique_duplicate_crop(left_filled)
    if left_duplicate is not None and np.array_equal(left_duplicate, duplicate):
        return False

    full_height_filled = _fill_bands(grid, bands, borrow="right", use_bbox_rows=False)
    full_height_duplicate = _unique_duplicate_crop(full_height_filled)
    if full_height_duplicate is not None and np.array_equal(full_height_duplicate, duplicate):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    a, b, c = random.sample(PALETTE, 3)
    colors = [a, b, c, a, b]

    w0 = random.randint(2, 3)
    widths = [w0, random.randint(2, 3), random.randint(2, 3), w0, random.randint(2, 3)]

    h0 = random.randint(3, 5)
    heights = [h0, random.randint(3, 5), random.randint(3, 5), h0, random.randint(3, 5)]

    mask0 = _random_connected_mask(heights[0], widths[0])
    if mask0 is None:
        return None
    mask1 = _random_connected_mask(heights[1], widths[1])
    mask2 = _random_connected_mask(heights[2], widths[2])
    mask4 = _different_mask((heights[4], widths[4]), [mask1] if heights[4] == heights[1] and widths[4] == widths[1] else [])
    if mask1 is None or mask2 is None or mask4 is None:
        return None
    masks = [mask0, mask1, mask2, mask0.copy(), mask4]

    overlap_unique_pair = random.random() < 0.5 and min(heights[1], heights[2]) >= 3
    row_starts, total_height = _place_intervals(heights, overlap_unique_pair)
    if total_height > 29:
        return None

    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    total_width = left_margin + sum(widths) + right_margin
    if total_width > 30:
        return None

    grid = _build_grid(
        colors,
        widths,
        heights,
        masks,
        row_starts,
        left_margin=left_margin,
        right_margin=right_margin,
    )

    if random.random() < 0.5:
        grid = np.flipud(grid).copy()

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
