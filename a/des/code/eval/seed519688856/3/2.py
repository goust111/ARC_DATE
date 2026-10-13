import random
from collections import Counter

import numpy as np


BG = 8
SEP = 5
PALETTE = [0, 1, 2, 3, 4, 6, 7, 9]
LAYOUTS = [(2, 3), (3, 2), (3, 3)]
MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
MASK_ATTEMPTS = 300


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for idx in separators + [length]:
        if start < idx:
            bands.append((start, idx - 1))
        start = idx + 1
    return bands


def _separator_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    row_colors = {int(grid[r, 0]) for r in range(h) if np.all(grid[r, :] == grid[r, 0])}
    col_colors = {int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])}
    candidates = row_colors & col_colors
    assert len(candidates) == 1
    return next(iter(candidates))


def _background_color(grid: np.ndarray, separator: int) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    counts.pop(separator, None)
    assert counts
    return max(counts, key=counts.get)


def _seed_bbox(cell: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    rows, cols = np.where(cell != bg)
    assert len(rows) > 0
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _classify_symmetry(seed: np.ndarray) -> tuple[bool, bool]:
    horizontal = np.array_equal(seed, np.fliplr(seed))
    vertical = np.array_equal(seed, np.flipud(seed))
    assert horizontal or vertical
    return horizontal, vertical


def _paste_clipped(dst: np.ndarray, tile: np.ndarray, top: int, left: int) -> None:
    h, w = dst.shape
    th, tw = tile.shape
    if top >= h or left >= w:
        return
    rr = min(th, h - top)
    cc = min(tw, w - left)
    dst[top : top + rr, left : left + cc] = tile[:rr, :cc]


def _expand_seed(
    seed: np.ndarray,
    out_shape: tuple[int, int],
    bg: int,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    horizontal, vertical = _classify_symmetry(seed)
    seed_h, seed_w = seed.shape
    out_h, out_w = out_shape
    start_r, start_c = origin

    out = np.full((out_h, out_w), bg, dtype=int)
    row_starts = list(range(start_r, out_h, seed_h)) if vertical else [start_r]
    col_starts = list(range(start_c, out_w, seed_w)) if horizontal else [start_c]

    for rb, top in enumerate(row_starts):
        for cb, left in enumerate(col_starts):
            tile = seed
            if rb % 2 == 1:
                tile = np.flipud(tile)
            if cb % 2 == 1:
                tile = np.fliplr(tile)
            _paste_clipped(out, tile, top, left)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator = _separator_color(g)
    bg = _background_color(g, separator)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == separator)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == separator)]
    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)
    assert row_bands and col_bands

    out = g.copy()
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            sr0, sc0, sr1, sc1 = _seed_bbox(cell, bg)
            seed = cell[sr0 : sr1 + 1, sc0 : sc1 + 1]
            out[r0 : r1 + 1, c0 : c1 + 1] = _expand_seed(
                seed,
                cell.shape,
                bg,
                (sr0, sc0),
            )
    return out


def _reflect_lr(mask: np.ndarray, width: int) -> np.ndarray:
    if width % 2 == 0:
        return np.concatenate([mask, np.fliplr(mask)], axis=1)
    return np.concatenate([mask, np.fliplr(mask[:, :-1])], axis=1)


def _reflect_ud(mask: np.ndarray, height: int) -> np.ndarray:
    if height % 2 == 0:
        return np.concatenate([mask, np.flipud(mask)], axis=0)
    return np.concatenate([mask, np.flipud(mask[:-1, :])], axis=0)


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _build_mask(kind: str, height: int, width: int) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        density = random.uniform(0.4, 0.7)
        if kind == "h":
            left = np.random.random((height, (width + 1) // 2)) < density
            left[0, random.randrange(left.shape[1])] = True
            left[-1, random.randrange(left.shape[1])] = True
            left[random.randrange(height), 0] = True
            mask = _reflect_lr(left, width)
        elif kind == "v":
            top = np.random.random(((height + 1) // 2, width)) < density
            top[random.randrange(top.shape[0]), 0] = True
            top[random.randrange(top.shape[0]), -1] = True
            top[0, random.randrange(width)] = True
            mask = _reflect_ud(top, height)
        else:
            quad = np.random.random(((height + 1) // 2, (width + 1) // 2)) < density
            quad[0, random.randrange(quad.shape[1])] = True
            quad[random.randrange(quad.shape[0]), 0] = True
            mask = _reflect_ud(_reflect_lr(quad, width), height)

        if mask.all():
            continue
        if int(mask.sum()) < max(4, (height + width) // 2):
            continue
        if not _touches_all_edges(mask):
            continue
        if not _connected(mask):
            continue
        horizontal, vertical = _classify_symmetry(mask.astype(int))
        if kind == "h" and horizontal and not vertical:
            return mask
        if kind == "v" and vertical and not horizontal:
            return mask
        if kind == "b" and horizontal and vertical:
            return mask
    raise RuntimeError(f"failed to build {kind}-symmetric mask")


def _sample_band_sizes(count: int) -> list[int]:
    for _ in range(200):
        lo, hi = (7, 12) if count == 2 else (7, 9)
        sizes = [random.randint(lo, hi) for _ in range(count)]
        if sum(sizes) + count + 1 <= MAX_GRID_SIZE:
            return sizes
    raise RuntimeError("failed to sample band sizes")


def _assign_region_colors(rows: int, cols: int) -> list[int]:
    colors: list[int] = []
    for r in range(rows):
        for c in range(cols):
            banned = set()
            if r > 0:
                banned.add(colors[(r - 1) * cols + c])
            if c > 0:
                banned.add(colors[r * cols + c - 1])
            choices = [color for color in PALETTE if color not in banned]
            colors.append(random.choice(choices))
    return colors


def _make_types(total: int) -> list[str]:
    base = ["h", "v", "b"] * (total // 3)
    assert len(base) == total
    random.shuffle(base)
    return base


def _region_slices(sizes: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = 1
    for size in sizes:
        out.append((pos, pos + size))
        pos += size + 1
    return out


def _make_case() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = random.choice(LAYOUTS)
    row_sizes = _sample_band_sizes(rows)
    col_sizes = _sample_band_sizes(cols)

    grid_h = sum(row_sizes) + rows + 1
    grid_w = sum(col_sizes) + cols + 1
    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    inp[0, :] = inp[-1, :] = SEP
    inp[:, 0] = inp[:, -1] = SEP
    out[0, :] = out[-1, :] = SEP
    out[:, 0] = out[:, -1] = SEP

    row_ranges = _region_slices(row_sizes)
    col_ranges = _region_slices(col_sizes)
    for end in [stop for _, stop in row_ranges[:-1]]:
        inp[end, :] = SEP
        out[end, :] = SEP
    for end in [stop for _, stop in col_ranges[:-1]]:
        inp[:, end] = SEP
        out[:, end] = SEP

    types = _make_types(rows * cols)
    colors = _assign_region_colors(rows, cols)

    idx = 0
    for rr, (r0, r1) in enumerate(row_ranges):
        for cc, (c0, c1) in enumerate(col_ranges):
            region_h = r1 - r0
            region_w = c1 - c0
            seed_h = random.randint(3, min(5, region_h - 2))
            seed_w = random.randint(3, min(5, region_w - 2))
            kind = types[idx]
            color = colors[idx]
            mask = _build_mask(kind, seed_h, seed_w)
            seed = np.full((seed_h, seed_w), BG, dtype=int)
            seed[mask] = color

            inp[r0 : r0 + seed_h, c0 : c0 + seed_w] = seed
            out[r0:r1, c0:c1] = _expand_seed(seed, (region_h, region_w), BG)
            idx += 1

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, expected = _make_case()
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate valid input")
