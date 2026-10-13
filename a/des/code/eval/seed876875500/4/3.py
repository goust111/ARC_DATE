import random
from typing import Iterable

import numpy as np


BG = 0
ALL_COLORS = tuple(range(1, 10))
LEGEND_OPS = ("hflip", "vflip", "rot_cw", "rot_ccw")
MAX_GENERATION_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(arr: np.ndarray, bg: int = BG) -> np.ndarray:
    box = _bbox(arr != bg)
    assert box is not None, "expected at least one foreground pixel"
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            coords.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            coords.append((r, c0))
    return coords


def _apply_op(arr: np.ndarray, op_index: int) -> np.ndarray:
    if op_index == 0:
        return np.fliplr(arr)
    if op_index == 1:
        return np.flipud(arr)
    if op_index == 2:
        return np.rot90(arr, -1)
    if op_index == 3:
        return np.rot90(arr, 1)
    raise ValueError(f"unknown op index: {op_index}")


def _apply_sequence(seed: np.ndarray, legend: list[int], frame_colors: Iterable[int]) -> np.ndarray:
    out = seed.copy()
    color_to_index = {color: idx for idx, color in enumerate(legend)}
    for color in frame_colors:
        out = _apply_op(out, color_to_index[int(color)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top-row color legend, peel nested rectangular frames from outside to inside,
    and apply the corresponding transform sequence to the innermost seed pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(np.bincount(g.ravel()).argmax())
    legend_cols = np.flatnonzero(g[0] != bg)
    assert legend_cols.size == 4, "top row must contain exactly four legend pixels"

    legend = [int(g[0, c]) for c in legend_cols]
    assert len(set(legend)) == 4, "legend colors must be distinct"

    work = g.copy()
    work[0, legend_cols] = bg

    frame_colors: list[int] = []
    while True:
        box = _bbox(work != bg)
        assert box is not None, "expected nested frames plus a seed pattern"
        r0, c0, r1, c1 = box
        border = _border_coords(r0, c0, r1, c1)
        border_vals = [int(work[r, c]) for r, c in border]
        border_set = set(border_vals)

        if bg in border_set or len(border_set) != 1:
            break

        color = border_vals[0]
        if color not in legend:
            break

        peeled = work.copy()
        for r, c in border:
            peeled[r, c] = bg
        if _bbox(peeled != bg) is None:
            break

        frame_colors.append(color)
        work = peeled

    assert frame_colors, "expected at least one rectangular frame"
    seed = _crop_non_bg(work, bg)
    return _apply_sequence(seed, legend, frame_colors)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if not bool(mask[nr, nc]) or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _dihedral_variants(arr: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rot = np.rot90(arr, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    return variants


def _seed_signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _all_dihedral_unique(arr: np.ndarray) -> bool:
    sigs = {_seed_signature(variant) for variant in _dihedral_variants(arr)}
    return len(sigs) == 8


def _make_seed(rng: random.Random, usable_colors: list[int]) -> np.ndarray:
    for _ in range(400):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        target = rng.randint(max(3, (h * w) // 2), max(3, h * w - 1))

        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        cells = [start]

        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, h, w):
                    if not bool(mask[nr, nc]) and (nr, nc) not in frontier:
                        frontier.append((nr, nc))
            if not frontier:
                break
            choice = frontier[rng.randrange(len(frontier))]
            mask[choice] = True
            cells.append(choice)

        if not _is_connected(mask):
            continue

        cell_rows, cell_cols = np.where(mask)
        crop = mask[cell_rows.min() : cell_rows.max() + 1, cell_cols.min() : cell_cols.max() + 1]
        if crop.shape[0] < 2 or crop.shape[1] < 2 or crop.shape[0] > 4 or crop.shape[1] > 4:
            continue

        palette_size = rng.randint(1, min(3, len(usable_colors)))
        palette = rng.sample(usable_colors, palette_size)
        seed = np.zeros(crop.shape, dtype=int)
        pts = list(map(tuple, np.argwhere(crop)))
        rng.shuffle(pts)
        for idx, (r, c) in enumerate(pts):
            if idx < palette_size:
                seed[r, c] = palette[idx]
            else:
                seed[r, c] = rng.choice(palette)

        if not _all_dihedral_unique(seed):
            continue

        return seed
    raise RuntimeError("failed to generate a non-degenerate seed")


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend = rng.sample(list(ALL_COLORS), 4)
    frame_colors = rng.sample(legend, len(legend))
    seed_colors = [c for c in ALL_COLORS if c not in legend]
    seed = _make_seed(rng, seed_colors)

    gaps = [rng.randint(1, 2) for _ in range(4)]
    inset_total = sum(gap + 1 for gap in gaps)
    outer_h = int(seed.shape[0] + 2 * inset_total)
    outer_w = int(seed.shape[1] + 2 * inset_total)

    legend_steps = [rng.randint(1, 2) for _ in range(3)]
    legend_span = 1 + sum(legend_steps)

    grid_h = outer_h + rng.randint(4, 8)
    grid_w = max(outer_w + rng.randint(3, 9), legend_span + rng.randint(2, 7))
    if grid_h <= outer_h + 2 or grid_w <= outer_w + 2:
        return None

    outer_r0 = rng.randint(2, grid_h - outer_h - 1)
    outer_c0 = rng.randint(1, grid_w - outer_w - 1)
    outer_box = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)

    legend_c0 = rng.randint(0, grid_w - legend_span)
    legend_cols = [legend_c0]
    for step in legend_steps:
        legend_cols.append(legend_cols[-1] + step)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for color, col in zip(legend, legend_cols):
        grid[0, col] = color

    frame_boxes: list[tuple[int, int, int, int]] = []
    r0, c0, r1, c1 = outer_box
    for gap, color in zip(gaps, frame_colors):
        frame_boxes.append((r0, c0, r1, c1))
        _draw_frame(grid, frame_boxes[-1], color)
        shrink = gap + 1
        r0 += shrink
        c0 += shrink
        r1 -= shrink
        c1 -= shrink

    assert (r1 - r0 + 1, c1 - c0 + 1) == seed.shape
    seed_mask = seed != BG
    grid[r0 : r1 + 1, c0 : c1 + 1][seed_mask] = seed[seed_mask]

    expected = _apply_sequence(seed, legend, frame_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame legend puzzle")
