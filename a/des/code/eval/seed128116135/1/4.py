from __future__ import annotations

import random
from typing import Iterable

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_POOL = tuple(range(1, 10))
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _true_runs(row_mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(np.asarray(row_mask, dtype=bool)):
        if value and start is None:
            start = int(i)
        elif not value and start is not None:
            runs.append((start, int(i) - 1))
            start = None
    if start is not None:
        runs.append((start, int(len(row_mask)) - 1))
    return runs


def _find_legend(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    nonzero_rows = [r for r in range(g.shape[0]) if np.any(g[r] != 0)]
    assert nonzero_rows, "expected at least one non-black row"
    bottom = int(nonzero_rows[-1])
    occupied = g[bottom] != 0
    assert occupied.any(), "legend row must contain non-black cells"
    top = bottom
    while top > 0 and np.array_equal(g[top - 1] != 0, occupied):
        top -= 1

    for r in range(top, bottom + 1):
        assert np.array_equal(g[r] != 0, occupied), "legend rows must share block columns"

    runs = _true_runs(occupied)
    widths = {c1 - c0 + 1 for c0, c1 in runs}
    assert len(runs) >= 2 and len(widths) == 1, "expected equal-sized legend blocks"
    blocks = [g[top : bottom + 1, c0 : c1 + 1].copy() for c0, c1 in runs]
    return top, runs, blocks


def _block_shape_index(blocks: list[np.ndarray]) -> dict[tuple[tuple[int, int], ...], int]:
    owners: dict[tuple[tuple[int, int], ...], int] = {}
    for block_idx, block in enumerate(blocks):
        for color in np.unique(block):
            color = int(color)
            if color == 0:
                continue
            for comp in _components(block == color):
                shape = _normalize(comp)
                prior = owners.get(shape)
                assert prior in (None, block_idx), "seed shape must identify exactly one legend block"
                owners[shape] = block_idx
    return owners


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each upper seed shape to a connected monochrome region inside one bottom legend texture block,
    then stamp the whole matching block at the seed bounding-box origin.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    legend_top, _runs, blocks = _find_legend(g)
    shape_to_block = _block_shape_index(blocks)
    out = g.copy()

    upper = g[:legend_top, :]
    seed_count = 0
    for color in np.unique(upper):
        color = int(color)
        if color == 0:
            continue
        for comp in _components(upper == color):
            shape = _normalize(comp)
            assert shape in shape_to_block, "every seed must match a legend component shape"
            block = blocks[shape_to_block[shape]]
            r0, c0, _, _ = _bbox(comp)
            bh, bw = map(int, block.shape)
            assert r0 + bh <= legend_top and c0 + bw <= g.shape[1], "stamp must stay in upper region"
            out[r0 : r0 + bh, c0 : c0 + bw] = block
            seed_count += 1

    assert seed_count > 0, "expected at least one seed"
    return out


def _random_shape(max_h: int, max_w: int) -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        target = random.randint(2, 6)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells:
                        continue
                    grown = cells | {(nr, nc)}
                    shape = _normalize(grown)
                    h, w = _shape_dims(shape)
                    if h <= max_h and w <= max_w:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) == target:
            return _normalize(cells)
    raise RuntimeError("failed to build a fitting seed shape")


def _shape_mask(shape: tuple[tuple[int, int], ...], color: int, bg: int = 0) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.full((h, w), bg, dtype=int)
    for r, c in shape:
        mask[r, c] = int(color)
    return mask


def _texture_block(
    block_h: int,
    block_w: int,
    shape: tuple[tuple[int, int], ...],
    colors: tuple[int, int],
) -> np.ndarray:
    fg, bg_color = map(int, colors)
    shape_h, shape_w = _shape_dims(shape)
    r_off = random.randint(1, block_h - shape_h - 1)
    c_off = random.randint(1, block_w - shape_w - 1)
    invert = random.random() < 0.5
    block = np.full((block_h, block_w), fg if invert else bg_color, dtype=int)
    paint_color = bg_color if invert else fg
    for r, c in shape:
        block[r_off + r, c_off + c] = paint_color
    return block


def _all_block_shapes(blocks: list[np.ndarray]) -> dict[tuple[tuple[int, int], ...], int] | None:
    owners: dict[tuple[tuple[int, int], ...], int] = {}
    for idx, block in enumerate(blocks):
        seen_here = set()
        for color in np.unique(block):
            color = int(color)
            if color == 0:
                continue
            for comp in _components(block == color):
                shape = _normalize(comp)
                if shape in seen_here:
                    continue
                prior = owners.get(shape)
                if prior is not None and prior != idx:
                    return None
                owners[shape] = idx
                seen_here.add(shape)
    return owners


def _place_legend(grid: np.ndarray, blocks: list[np.ndarray], top_row: int, left_col: int) -> None:
    block_h, block_w = map(int, blocks[0].shape)
    for i, block in enumerate(blocks):
        c0 = left_col + i * (block_w + 1)
        grid[top_row : top_row + block_h, c0 : c0 + block_w] = block


def _slot_anchors(
    seed_count: int,
    block_h: int,
    block_w: int,
) -> tuple[int, int, list[tuple[int, int]]]:
    slot_h = block_h + random.randint(1, 2)
    slot_w = block_w + random.randint(1, 3)
    slot_cols = random.randint(2, min(4, seed_count))
    slot_rows = (seed_count + slot_cols - 1) // slot_cols + random.randint(0, 1)
    top_h = slot_rows * slot_h + random.randint(0, 1)
    top_w = slot_cols * slot_w + random.randint(0, 2)

    all_slots = [(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    chosen_slots = random.sample(all_slots, seed_count)
    anchors = []
    for sr, sc in chosen_slots:
        r0 = sr * slot_h + random.randint(0, slot_h - block_h)
        c0 = sc * slot_w + random.randint(0, slot_w - block_w)
        anchors.append((r0, c0))
    return top_h, top_w, anchors


def _render_expected(
    grid: np.ndarray,
    legend_top: int,
    blocks: list[np.ndarray],
    seed_records: list[tuple[int, tuple[int, int]]],
) -> np.ndarray:
    out = grid.copy()
    for block_idx, (r0, c0) in seed_records:
        block = blocks[block_idx]
        bh, bw = map(int, block.shape)
        assert r0 + bh <= legend_top and c0 + bw <= grid.shape[1]
        out[r0 : r0 + bh, c0 : c0 + bw] = block
    return out


def _gen_attempt() -> np.ndarray | None:
    block_h = random.choice((4, 5))
    block_w = random.choice((5, 6))
    block_count = random.randint(3, 5)

    max_shape_h = block_h - 2
    max_shape_w = block_w - 2
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen_shapes = set()
    while len(shapes) < block_count:
        shape = _random_shape(max_shape_h, max_shape_w)
        if shape in seen_shapes:
            continue
        shapes.append(shape)
        seen_shapes.add(shape)

    blocks = []
    for shape in shapes:
        colors = tuple(random.sample(COLOR_POOL, 2))
        blocks.append(_texture_block(block_h, block_w, shape, colors))

    shape_index = _all_block_shapes(blocks)
    if shape_index is None:
        return None
    for idx, shape in enumerate(shapes):
        if shape_index.get(shape) != idx:
            return None

    extra_seeds = random.randint(0, 2)
    seed_targets = list(range(block_count)) + [random.randrange(block_count) for _ in range(extra_seeds)]
    random.shuffle(seed_targets)
    seed_count = len(seed_targets)

    top_h, top_w, anchors = _slot_anchors(seed_count, block_h, block_w)
    legend_gap = random.randint(1, 3)
    legend_w = block_count * block_w + (block_count - 1)
    grid_w = max(top_w, legend_w + 2) + random.randint(0, 3)
    legend_left = random.randint(0, grid_w - legend_w)
    legend_top = top_h + legend_gap
    grid_h = legend_top + block_h

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _place_legend(grid, blocks, legend_top, legend_left)

    seed_records: list[tuple[int, tuple[int, int]]] = []
    for block_idx, anchor in zip(seed_targets, anchors):
        r0, c0 = anchor
        seed_color = random.choice(COLOR_POOL)
        seed = _shape_mask(shapes[block_idx], seed_color)
        sh, sw = map(int, seed.shape)
        grid[r0 : r0 + sh, c0 : c0 + sw][seed != 0] = seed[seed != 0]
        seed_records.append((block_idx, anchor))

    expected = _render_expected(grid, legend_top, blocks, seed_records)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
