from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty array"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _separator_lines(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = map(int, g.shape)
    candidates = []
    for color in np.unique(g):
        color = int(color)
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one separator color forming full rows and columns"
    return candidates[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave nonempty partitions"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final partition"
    spans.append((start, length - 1))
    return spans


def _seed_patch(cell: np.ndarray, bg: int, sep: int) -> tuple[np.ndarray, int]:
    mask = (cell != int(bg)) & (cell != int(sep))
    assert bool(mask.any()), "each partition must contain a seed"
    colors = {int(v) for v in np.unique(cell[mask])}
    assert len(colors) == 1, "each seed must use one color"

    rows, cols = np.where(mask)
    seed_h = int(rows.max()) + 1
    seed_w = int(cols.max()) + 1
    patch = cell[:seed_h, :seed_w].copy()
    patch_colors = {int(v) for v in np.unique(patch)} - {int(bg)}
    assert patch_colors == colors, "seed must occupy a top-left rectangle over background"
    return patch, int(next(iter(colors)))


def _is_lr_symmetric(patch: np.ndarray) -> bool:
    return bool(np.array_equal(patch, patch[:, ::-1]))


def _is_ud_symmetric(patch: np.ndarray) -> bool:
    return bool(np.array_equal(patch, patch[::-1, :]))


def _tile_patch(patch: np.ndarray, out_shape: tuple[int, int], mode: str, bg: int) -> np.ndarray:
    out_h, out_w = map(int, out_shape)
    patch_h, patch_w = map(int, patch.shape)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    row_offsets = [0] if mode == "h" else list(range(0, out_h, patch_h))
    col_offsets = [0] if mode == "v" else list(range(0, out_w, patch_w))

    for r0 in row_offsets:
        for c0 in col_offsets:
            h = min(patch_h, out_h - r0)
            w = min(patch_w, out_w - c0)
            out[r0 : r0 + h, c0 : c0 + w] = patch[:h, :w]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each top-left seed inside a separator lattice into a periodic tiling chosen by the seed's symmetry.

    Input structure:
    - Full same-color separator rows and columns partition the grid into equal-sized rectangular regions.
    - Each region contains one monochrome seed pattern in a small top-left rectangle drawn on the background.
    - Everything outside the seed rectangle inside a region is background, and separator lines stay intact.

    Rule:
    - Split the grid into regions using the separator lines and extract each region's top-left seed rectangle.
    - If the seed is left-right symmetric, repeat that seed horizontally across the region only.
    - If the seed is up-down symmetric, repeat that seed vertically across the region only.
    - Otherwise repeat the seed in both directions, cropping any partial copy at the region boundary.

    Key insight:
    - The seed's symmetry is not decorative; it encodes whether the periodic extension should run horizontally, vertically, or on both axes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    sep, sep_rows, sep_cols = _separator_lines(g)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)

    row_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(row_heights) == 1 and len(col_widths) == 1, "partitions must be equal-sized"

    bg = _unique_mode(g[g != int(sep)])
    out = g.copy()

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            patch, _seed_color = _seed_patch(cell, bg, sep)
            lr = _is_lr_symmetric(patch)
            ud = _is_ud_symmetric(patch)

            if lr and not ud:
                mode = "h"
            elif ud and not lr:
                mode = "v"
            else:
                mode = "hv"
            out[r0 : r1 + 1, c0 : c1 + 1] = _tile_patch(patch, cell.shape, mode, bg)
    return out


BRANCHES = ("h", "v", "hv")
MAX_ATTEMPTS = 400


def _base_grid(part_rows: int, part_cols: int, cell_h: int, cell_w: int, bg: int, sep: int) -> np.ndarray:
    h = part_rows * cell_h + (part_rows - 1)
    w = part_cols * cell_w + (part_cols - 1)
    grid = np.full((h, w), int(bg), dtype=int)
    for r in range(1, part_rows):
        grid[r * cell_h + (r - 1), :] = int(sep)
    for c in range(1, part_cols):
        grid[:, c * cell_w + (c - 1)] = int(sep)
    return grid


def _part_origin(part_r: int, part_c: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return part_r * (cell_h + 1), part_c * (cell_w + 1)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _mirror_lr(left: np.ndarray, center: np.ndarray | None) -> np.ndarray:
    if center is None:
        return np.concatenate([left, left[:, ::-1]], axis=1)
    return np.concatenate([left, center, left[:, ::-1]], axis=1)


def _mirror_ud(top: np.ndarray, center: np.ndarray | None) -> np.ndarray:
    if center is None:
        return np.concatenate([top, top[::-1, :]], axis=0)
    return np.concatenate([top, center, top[::-1, :]], axis=0)


def _random_density() -> float:
    return random.uniform(0.28, 0.62)


def _gen_lr_mask(h: int, w: int) -> np.ndarray | None:
    assert w >= 3
    left_w = w // 2
    center_w = w % 2
    half_choices = list(range((w + 1) // 2))
    for _ in range(200):
        left = np.random.rand(h, left_w) < _random_density()
        center = (np.random.rand(h, 1) < _random_density()) if center_w else None
        mask = _mirror_lr(left, center)

        side_row = random.randrange(h)
        mask[side_row, 0] = True
        mask[side_row, -1] = True

        top_col = random.choice(half_choices)
        bottom_candidates = [c for c in half_choices if c != top_col]
        if not bottom_candidates:
            continue
        bottom_col = random.choice(bottom_candidates)
        for col in {top_col, w - 1 - top_col}:
            mask[0, col] = True
        for col in {bottom_col, w - 1 - bottom_col}:
            mask[-1, col] = True

        if mask.sum() < 2:
            continue
        if _touches_all_sides(mask) and _is_lr_symmetric(mask) and not _is_ud_symmetric(mask):
            return mask
    return None


def _gen_ud_mask(h: int, w: int) -> np.ndarray | None:
    assert h >= 3
    top_h = h // 2
    center_h = h % 2
    half_choices = list(range((h + 1) // 2))
    for _ in range(200):
        top = np.random.rand(top_h, w) < _random_density()
        center = (np.random.rand(1, w) < _random_density()) if center_h else None
        mask = _mirror_ud(top, center)

        side_col = random.randrange(w)
        mask[0, side_col] = True
        mask[-1, side_col] = True

        left_row = random.choice(half_choices)
        right_candidates = [r for r in half_choices if r != left_row]
        if not right_candidates:
            continue
        right_row = random.choice(right_candidates)
        for row in {left_row, h - 1 - left_row}:
            mask[row, 0] = True
        for row in {right_row, h - 1 - right_row}:
            mask[row, -1] = True

        if mask.sum() < 2:
            continue
        if _touches_all_sides(mask) and _is_ud_symmetric(mask) and not _is_lr_symmetric(mask):
            return mask
    return None


def _gen_asym_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(300):
        mask = np.random.rand(h, w) < _random_density()
        mask[0, random.randrange(w)] = True
        mask[-1, random.randrange(w)] = True
        mask[random.randrange(h), 0] = True
        mask[random.randrange(h), -1] = True
        if mask.sum() < 2:
            continue
        if _touches_all_sides(mask) and not _is_lr_symmetric(mask) and not _is_ud_symmetric(mask):
            return mask
    return None


def _seed_dims(branch: str, cell_h: int, cell_w: int) -> tuple[int, int]:
    if branch == "h":
        h = random.randint(2, min(5, cell_h - 1))
        w = random.randint(3, min(5, cell_w - 1))
    elif branch == "v":
        h = random.randint(3, min(5, cell_h - 1))
        w = random.randint(2, min(5, cell_w - 1))
    else:
        h = random.randint(2, min(5, cell_h - 1))
        w = random.randint(2, min(5, cell_w - 1))
    return int(h), int(w)


def _seed_mask(branch: str, h: int, w: int) -> np.ndarray | None:
    if branch == "h":
        return _gen_lr_mask(h, w)
    if branch == "v":
        return _gen_ud_mask(h, w)
    return _gen_asym_mask(h, w)


def _paint_seed(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    return np.where(mask, int(color), int(bg)).astype(int)


def _transform(g: np.ndarray, rotations: int, mirror: bool) -> np.ndarray:
    out = np.rot90(g, rotations)
    if mirror:
        out = np.fliplr(out)
    return out.copy()


def _gen_attempt() -> np.ndarray | None:
    part_rows, part_cols = random.choice([(2, 3), (3, 2)])
    max_cell_h = min(10, (30 - (part_rows - 1)) // part_rows)
    max_cell_w = min(10, (30 - (part_cols - 1)) // part_cols)
    if max_cell_h < 5 or max_cell_w < 5:
        return None

    cell_h = random.randint(5, max_cell_h)
    cell_w = random.randint(5, max_cell_w)
    bg, sep = random.sample(range(10), 2)
    palette = [c for c in range(10) if c not in {bg, sep}]

    n_parts = part_rows * part_cols
    if len(palette) < n_parts:
        return None
    colors = random.sample(palette, n_parts)
    branches = list(BRANCHES) * (n_parts // len(BRANCHES))
    random.shuffle(branches)

    grid = _base_grid(part_rows, part_cols, cell_h, cell_w, bg, sep)
    for idx, (part_r, part_c) in enumerate(
        (divmod(k, part_cols) for k in range(n_parts))
    ):
        branch = branches[idx]
        seed_h, seed_w = _seed_dims(branch, cell_h, cell_w)
        mask = _seed_mask(branch, seed_h, seed_w)
        if mask is None:
            return None

        seed = _paint_seed(mask, colors[idx], bg)
        r0, c0 = _part_origin(part_r, part_c, cell_h, cell_w)
        grid[r0 : r0 + seed_h, c0 : c0 + seed_w] = seed

    rotations = random.randrange(4)
    mirror = random.random() < 0.5
    grid = _transform(grid, rotations, mirror)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
