from __future__ import annotations

import random

import numpy as np


GREY = 5
IMAGE_COLORS = (0, 1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
HOLE_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_cols = np.flatnonzero(np.all(g == GREY, axis=0))
    assert divider_cols.size == 1, "expected a unique full grey divider column"
    divider = int(divider_cols[0])
    assert divider * 2 + 1 == g.shape[1], "divider must split the grid into equal panels"

    left = g[:, :divider].copy()
    right = g[:, divider + 1 :].copy()
    assert left.shape == right.shape and left.size > 0
    return left, right


def _overlap_slices(shape: tuple[int, int], dr: int, dc: int):
    height, width = map(int, shape)
    r0 = max(0, dr)
    r1 = min(height, height + dr)
    c0 = max(0, dc)
    c1 = min(width, width + dc)
    if r0 >= r1 or c0 >= c1:
        return None
    left_slice = (slice(r0, r1), slice(c0, c1))
    right_slice = (slice(r0 - dr, r1 - dr), slice(c0 - dc, c1 - dc))
    return left_slice, right_slice


def _match_score(left: np.ndarray, right: np.ndarray, dr: int, dc: int) -> int | None:
    overlap = _overlap_slices(left.shape, dr, dc)
    if overlap is None:
        return None

    left_slice, right_slice = overlap
    left_view = left[left_slice]
    right_view = right[right_slice]
    known = (left_view != GREY) & (right_view != GREY)
    if not np.any(known):
        return None
    if np.any(left_view[known] != right_view[known]):
        return None
    return int(known.sum())


def _holes_are_complementary(left: np.ndarray, right: np.ndarray, dr: int, dc: int) -> bool:
    left_holes = np.argwhere(left == GREY)
    for r, c in left_holes:
        rr = int(r - dr)
        cc = int(c - dc)
        if not (0 <= rr < right.shape[0] and 0 <= cc < right.shape[1]):
            return False
        if int(right[rr, cc]) == GREY:
            return False

    right_holes = np.argwhere(right == GREY)
    for r, c in right_holes:
        rr = int(r + dr)
        cc = int(c + dc)
        if not (0 <= rr < left.shape[0] and 0 <= cc < left.shape[1]):
            return False
        if int(left[rr, cc]) == GREY:
            return False

    return True


def _infer_shift(left: np.ndarray, right: np.ndarray) -> tuple[int, int]:
    candidates: list[tuple[int, int, int]] = []
    height, width = map(int, left.shape)

    for dr in range(-(height - 1), height):
        for dc in range(-(width - 1), width):
            score = _match_score(left, right, dr, dc)
            if score is None:
                continue
            if not _holes_are_complementary(left, right, dr, dc):
                continue
            candidates.append((score, dr, dc))

    assert candidates, "failed to infer a valid panel shift"
    best_score = max(score for score, _, _ in candidates)
    best = [(dr, dc) for score, dr, dc in candidates if score == best_score]
    assert len(best) == 1, "panel shift must be unique"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the left crop by aligning the right crop and copying over its complementary hole pixels.

    Input structure:
    - A single full grey divider column splits the grid into two equal-sized panels.
    - The left and right panels are shifted crops of the same multi-color source image.
    - Grey cells inside the panels are missing-data holes, and the corresponding holes on the two panels do not overlap after the true alignment.

    Rule:
    - Search for the unique row/column shift that aligns the right panel to the left panel so that every overlapping non-grey pixel agrees.
    - Use that shift to map the right panel onto the left panel.
    - Replace each grey hole in the left panel with the aligned pixel from the right panel and return the repaired left panel.

    Key insight:
    - The two panels are not already aligned, so the missing pixels cannot be filled until the hidden translation between the two windows is recovered from their intact overlap.
    """
    left, right = _split_panels(grid)
    dr, dc = _infer_shift(left, right)

    overlap = _overlap_slices(left.shape, dr, dc)
    assert overlap is not None
    left_slice, right_slice = overlap

    out = left.copy()
    left_view = out[left_slice].copy()
    right_view = right[right_slice]
    hole_mask = left_view == GREY
    left_view[hole_mask] = right_view[hole_mask]
    out[left_slice] = left_view

    assert not np.any(out == GREY), "all left-panel holes must be repairable"
    return out


def _random_partition(total: int, parts: int, min_size: int) -> list[int]:
    assert total >= parts * min_size
    remaining = total
    sizes: list[int] = []
    for i in range(parts - 1):
        low = min_size
        high = remaining - min_size * (parts - i - 1)
        size = random.randint(low, high)
        sizes.append(size)
        remaining -= size
    sizes.append(remaining)
    random.shuffle(sizes)
    return sizes


def _random_blob(max_h: int, max_w: int) -> np.ndarray:
    h = random.randint(2, max_h)
    w = random.randint(2, max_w)
    target = random.randint(3, min(h * w, 8))

    mask = np.zeros((h, w), dtype=bool)
    seed_r = random.randrange(h)
    seed_c = random.randrange(w)
    mask[seed_r, seed_c] = True

    while int(mask.sum()) < target:
        frontier: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if (
                    (r > 0 and mask[r - 1, c])
                    or (r + 1 < h and mask[r + 1, c])
                    or (c > 0 and mask[r, c - 1])
                    or (c + 1 < w and mask[r, c + 1])
                ):
                    frontier.append((r, c))
        if not frontier:
            break
        r, c = random.choice(frontier)
        mask[r, c] = True

    return mask


def _make_source_image(height: int, width: int) -> np.ndarray:
    palette = random.sample(IMAGE_COLORS, random.randint(4, 6))
    row_bands = random.randint(2, min(5, height // 2))
    col_bands = random.randint(2, min(5, width // 2))
    row_sizes = _random_partition(height, row_bands, 2)
    col_sizes = _random_partition(width, col_bands, 2)

    grid = np.zeros((height, width), dtype=int)
    block_colors = [[0] * col_bands for _ in range(row_bands)]
    row_start = 0
    for i, band_h in enumerate(row_sizes):
        col_start = 0
        for j, band_w in enumerate(col_sizes):
            forbidden = set()
            if i > 0:
                forbidden.add(block_colors[i - 1][j])
            if j > 0:
                forbidden.add(block_colors[i][j - 1])
            choices = [color for color in palette if color not in forbidden] or list(palette)
            color = random.choice(choices)
            block_colors[i][j] = color
            grid[row_start : row_start + band_h, col_start : col_start + band_w] = color
            col_start += band_w
        row_start += band_h

    for _ in range(random.randint(2, 5)):
        mask = _random_blob(min(5, height), min(5, width))
        blob_h, blob_w = mask.shape
        for _ in range(20):
            r0 = random.randint(0, height - blob_h)
            c0 = random.randint(0, width - blob_w)
            color = random.choice(palette)
            patch = grid[r0 : r0 + blob_h, c0 : c0 + blob_w]
            if np.any(patch[mask] != color):
                patch[mask] = color
                break

    if random.random() < 0.7:
        color = random.choice(palette)
        if random.random() < 0.5:
            row = random.randrange(height)
            c0 = random.randint(0, width - 3)
            c1 = random.randint(c0 + 2, width - 1)
            grid[row, c0 : c1 + 1] = color
        else:
            col = random.randrange(width)
            r0 = random.randint(0, height - 3)
            r1 = random.randint(r0 + 2, height - 1)
            grid[r0 : r1 + 1, col] = color

    return grid


def _sample_shift(panel_h: int, panel_w: int) -> tuple[int, int]:
    max_dr = min(4, panel_h - 4)
    max_dc = min(4, panel_w - 4)
    while True:
        dr = random.randint(-max_dr, max_dr)
        dc = random.randint(-max_dc, max_dc)
        if dr == 0 and dc == 0:
            continue
        if panel_h - abs(dr) >= 4 and panel_w - abs(dc) >= 4:
            return dr, dc


def _mark_halo(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                blocked[rr, cc] = True


def _sample_hole_sets(overlap_h: int, overlap_w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    left_patches = random.randint(1, 3)
    right_patches = random.randint(1, 3)
    order = ["L"] * left_patches + ["R"] * right_patches
    random.shuffle(order)

    blocked = np.zeros((overlap_h, overlap_w), dtype=bool)
    left_cells: list[tuple[int, int]] = []
    right_cells: list[tuple[int, int]] = []

    for side in order:
        candidates: list[list[tuple[int, int]]] = []
        for shape in HOLE_SHAPES:
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            if max_r >= overlap_h or max_c >= overlap_w:
                continue
            for r0 in range(overlap_h - max_r):
                for c0 in range(overlap_w - max_c):
                    cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
                    if any(blocked[r, c] for r, c in cells):
                        continue
                    candidates.append(cells)
        if not candidates:
            return None
        chosen = random.choice(candidates)
        _mark_halo(blocked, chosen)
        if side == "L":
            left_cells.extend(chosen)
        else:
            right_cells.extend(chosen)

    if not left_cells or not right_cells:
        return None
    return left_cells, right_cells


def _build_input_pair() -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = random.randint(8, 15)
    panel_w = random.randint(8, 14)
    dr, dc = _sample_shift(panel_h, panel_w)

    extra_top = random.randint(0, 2)
    extra_bottom = random.randint(0, 2)
    extra_left = random.randint(0, 2)
    extra_right = random.randint(0, 2)
    source_h = panel_h + abs(dr) + extra_top + extra_bottom
    source_w = panel_w + abs(dc) + extra_left + extra_right

    source = _make_source_image(source_h, source_w)

    if dr >= 0:
        right_r0 = extra_top
        left_r0 = extra_top + dr
    else:
        left_r0 = extra_top
        right_r0 = extra_top - dr

    if dc >= 0:
        right_c0 = extra_left
        left_c0 = extra_left + dc
    else:
        left_c0 = extra_left
        right_c0 = extra_left - dc

    left_full = source[left_r0 : left_r0 + panel_h, left_c0 : left_c0 + panel_w].copy()
    right_full = source[right_r0 : right_r0 + panel_h, right_c0 : right_c0 + panel_w].copy()

    overlap = _overlap_slices(left_full.shape, dr, dc)
    assert overlap is not None
    left_slice, _ = overlap
    overlap_h = left_slice[0].stop - left_slice[0].start
    overlap_w = left_slice[1].stop - left_slice[1].start

    sampled = _sample_hole_sets(overlap_h, overlap_w)
    if sampled is None:
        return None
    left_holes_local, right_holes_local = sampled

    left = left_full.copy()
    right = right_full.copy()
    row_off = left_slice[0].start
    col_off = left_slice[1].start

    for lr, lc in left_holes_local:
        left[row_off + lr, col_off + lc] = GREY
    for lr, lc in right_holes_local:
        left_r = row_off + lr
        left_c = col_off + lc
        right[left_r - dr, left_c - dc] = GREY

    divider = np.full((panel_h, 1), GREY, dtype=int)
    inp = np.concatenate([left, divider, right], axis=1)
    expected = left_full

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair = _build_input_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid puzzle input")
