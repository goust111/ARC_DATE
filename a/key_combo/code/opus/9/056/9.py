from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8

PAYLOAD_COLORS = (BLUE, RED, GREEN, YELLOW)
EXCLUDED_FROM_DIVERSITY = {BLACK, GREY, TEAL}
MAX_GENERATION_ATTEMPTS = 400


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components from a boolean mask as coord arrays."""
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _rail_segments(grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Return grey vertical rail segments as (top_row, col, bottom_row)."""
    segments: list[tuple[int, int, int]] = []
    for coords in _component_coords(grid == GREY):
        rows = np.sort(coords[:, 0])
        cols = np.unique(coords[:, 1])
        assert len(cols) == 1, "each grey object must occupy exactly one column"
        assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)), "grey rail must be contiguous"
        segments.append((int(rows[0]), int(cols[0]), int(rows[-1])))
    return sorted(segments, key=lambda item: item[1])


def _crop_tile(grid: np.ndarray) -> np.ndarray:
    """Crop the rail-bounded tile, including the teal marker rows."""
    g = np.asarray(grid, dtype=int)
    H, _W = g.shape
    rails = _rail_segments(g)
    assert len(rails) == 2, "expected exactly two grey vertical rails"
    (top_l, col_l, bot_l), (top_r, col_r, bot_r) = rails
    assert col_l < col_r, "rails must be in distinct columns"

    for top, col, bottom in rails:
        assert 0 <= top - 1 < H and 0 <= bottom + 1 < H, "teal markers must fit inside the canvas"
        assert g[top - 1, col] == TEAL, "missing top teal marker above a rail"
        assert g[bottom + 1, col] == TEAL, "missing bottom teal marker below a rail"

    r0 = max(0, min(top_l, top_r) - 1)
    r1 = min(H - 1, max(bot_l, bot_r) + 1)
    return g[r0 : r1 + 1, col_l : col_r + 1]


def _diversity_colors(tile: np.ndarray) -> list[int]:
    colors = sorted(int(v) for v in np.unique(tile) if int(v) not in EXCLUDED_FROM_DIVERSITY)
    assert colors, "cropped tile must contain at least one payload color"
    return colors


def _build_output(tile: np.ndarray, row_count: int) -> np.ndarray:
    rows = []
    mirrored = np.fliplr(tile)
    for idx in range(row_count):
        row_tile = tile if idx % 2 == 0 else mirrored
        rows.append(np.tile(row_tile, (1, 3)))
    return np.vstack(rows)


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the rail-bounded tile and tile it into K mirrored rows, where K is the tile's payload-color diversity.

    Input structure:
    - The grid has a black background with scattered payload-colored pixels.
    - Exactly two grey vertical rails mark the left and right crop columns.
    - Each rail has one teal marker directly above its top endpoint and one teal marker directly below its bottom endpoint.

    Rule:
    - Detect the two grey vertical rails and sort them by column.
    - Crop from the left rail column to the right rail column, and from one row above the higher rail top to one row below the lower rail bottom.
    - Count the distinct payload colors inside that crop, excluding black, grey, and teal; call that count K.
    - Build an output of K tile-rows by 3 tile-columns: odd-numbered tile-rows use the crop unchanged, even-numbered tile-rows use its left-right mirror.

    Key insight:
    - The rails and teal markers first define a reusable tile, and the crop's own color diversity then controls how many times that tile is repeated vertically.
    """
    tile = _crop_tile(np.asarray(grid, dtype=int))
    k = len(_diversity_colors(tile))
    return _build_output(tile, k)


def _random_counts(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    """Split total into positive integer counts across parts."""
    if parts == 1:
        return np.array([total], dtype=int)
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total]))).astype(int)


def _choose_start(limit: int, span: int, rng: np.random.Generator) -> int:
    """Choose a start coordinate, often touching a boundary but not always."""
    max_start = limit - span
    assert max_start >= 0
    if max_start == 0:
        return 0
    mode = int(rng.integers(0, 4))
    if mode == 0:
        return 0
    if mode == 1:
        return max_start
    return int(rng.integers(0, max_start + 1))


def _line_offsets(crop_h: int, rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
    """Choose rail extents relative to the cropped tile."""
    line_union = crop_h - 2
    assert 4 <= line_union <= 8

    intervals = [
        (start, end)
        for start in range(line_union - 3)
        for end in range(start + 3, line_union)
    ]
    covering_pairs = [
        (left, right)
        for left in intervals
        for right in intervals
        if min(left[0], right[0]) == 0 and max(left[1], right[1]) == line_union - 1
    ]
    assert covering_pairs, "at least one rail pair must cover the crop height"

    aligned = [pair for pair in covering_pairs if pair[0] == pair[1] == (0, line_union - 1)]
    non_aligned = [pair for pair in covering_pairs if pair[0] != pair[1]]

    if non_aligned and rng.random() >= 0.35:
        return non_aligned[int(rng.integers(0, len(non_aligned)))]
    return aligned[0]


def _fill_crop_payload(
    tile: np.ndarray,
    colors: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Place payload-colored cells into the crop without overwriting rails or teal markers."""
    free = np.argwhere(tile == BLACK)
    assert len(free) >= len(colors) + 2, "crop needs enough free cells for payload colors"
    rng.shuffle(free)

    forced = free[: len(colors)]
    for color, (r, c) in zip(colors.tolist(), forced.tolist()):
        tile[r, c] = int(color)

    remaining = free[len(colors) :]
    extra_cap = min(len(remaining), max(2, (tile.shape[0] * tile.shape[1]) // 3))
    extra = int(rng.integers(len(colors), extra_cap + 1))
    counts = _random_counts(extra, len(colors), rng)

    cursor = 0
    for color, count in zip(colors.tolist(), counts.tolist()):
        for r, c in remaining[cursor : cursor + count]:
            tile[int(r), int(c)] = int(color)
        cursor += count


def _add_outside_noise(
    grid: np.ndarray,
    crop_box: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> None:
    """Scatter payload-colored distractor pixels outside the crop."""
    r0, c0, r1, c1 = crop_box
    outside = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if not (r0 <= r <= r1 and c0 <= c <= c1)
    ]
    rng.shuffle(outside)
    noise_count = int(rng.integers(12, min(40, len(outside)) + 1))
    noise_colors = list(PAYLOAD_COLORS)
    if noise_count > len(noise_colors):
        noise_colors.extend(rng.choice(PAYLOAD_COLORS, size=noise_count - len(noise_colors), replace=True).tolist())
    rng.shuffle(noise_colors)
    for (r, c), color in zip(outside[:noise_count], noise_colors):
        grid[r, c] = int(color)


def _semantic_checks(grid: np.ndarray, expected_k: int) -> None:
    tile = _crop_tile(grid)
    assert len(_diversity_colors(tile)) == expected_k, "crop diversity mismatch"
    assert not np.array_equal(tile, np.fliplr(tile)), "crop must not be left-right symmetric"

    output = main(grid)
    h, w = tile.shape
    assert output.shape == (expected_k * h, 3 * w), "unexpected output size"
    assert max(output.shape) <= 30, "output exceeds ARC size limit"
    for idx in range(expected_k):
        row_band = output[idx * h : (idx + 1) * h]
        expected_tile = tile if idx % 2 == 0 else np.fliplr(tile)
        assert np.array_equal(row_band, np.tile(expected_tile, (1, 3))), "tiling pattern mismatch"

    full_palette = {int(v) for v in np.unique(grid) if int(v) in PAYLOAD_COLORS}
    if expected_k < len(PAYLOAD_COLORS):
        assert len(full_palette) == len(PAYLOAD_COLORS), "global palette should not collapse to local diversity"


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    k = int(rng.choice([2, 3, 4]))
    max_crop_h = min(10, 30 // k)
    crop_h = int(rng.integers(6, max_crop_h + 1))
    rail_gap = int(rng.integers(3, 7))
    crop_w = rail_gap + 1

    H = int(rng.integers(max(18, crop_h + 6), 26))
    W = int(rng.integers(max(18, crop_w + 10), 26))
    grid = np.zeros((H, W), dtype=int)

    crop_r0 = _choose_start(H, crop_h, rng)
    crop_c0 = _choose_start(W, crop_w, rng)
    crop_r1 = crop_r0 + crop_h - 1
    crop_c1 = crop_c0 + crop_w - 1

    (ls, le), (rs, re) = _line_offsets(crop_h, rng)
    left_top = crop_r0 + 1 + ls
    left_bottom = crop_r0 + 1 + le
    right_top = crop_r0 + 1 + rs
    right_bottom = crop_r0 + 1 + re
    left_col = crop_c0
    right_col = crop_c1

    grid[left_top : left_bottom + 1, left_col] = GREY
    grid[right_top : right_bottom + 1, right_col] = GREY
    grid[left_top - 1, left_col] = TEAL
    grid[left_bottom + 1, left_col] = TEAL
    grid[right_top - 1, right_col] = TEAL
    grid[right_bottom + 1, right_col] = TEAL

    tile = grid[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]
    chosen_colors = rng.choice(PAYLOAD_COLORS, size=k, replace=False)
    _fill_crop_payload(tile, chosen_colors, rng)

    # Add scatter outside the crop so global color diversity is not a reliable shortcut.
    _add_outside_noise(grid, (crop_r0, crop_c0, crop_r1, crop_c1), rng)

    try:
        _semantic_checks(grid, k)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        generated = _generate_attempt(rng)
        if generated is not None:
            return generated
    raise RuntimeError("failed to generate a valid input")
