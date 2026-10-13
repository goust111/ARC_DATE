from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(np.array(coords, dtype=int))

    return comps


def _extract_blocks(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    comps = _components(grid != BG)
    assert len(comps) >= 2, "expected at least two pattern blocks"

    blocks: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in comps:
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        assert coords.shape[0] == area, "each block must fill its rectangular bbox"
        assert np.all(patch != BG), "pattern blocks must not contain background holes"
        blocks.append(((r0, c0, r1, c1), patch))

    shapes = {patch.shape for _bbox, patch in blocks}
    assert len(shapes) == 1, "all blocks must have the same size"
    return sorted(blocks, key=lambda item: (item[0][0], item[0][1]))


def _consensus_patch(patches: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(patches, axis=0)
    _count, h, w = stack.shape
    consensus = np.empty((h, w), dtype=int)

    for r in range(h):
        for c in range(w):
            colors, counts = np.unique(stack[:, r, c], return_counts=True)
            winners = colors[counts == counts.max()]
            assert winners.size == 1, "expected a unique consensus color at each local coordinate"
            consensus[r, c] = int(winners[0])

    return consensus


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the consensus motif across repeated rectangular blocks, then repair variant copies and mark their differing pixels red.

    Input structure:
    - The background is black.
    - The foreground consists of several separated same-sized rectangular pattern blocks.
    - Every block is fully filled with non-black colors, and most blocks are exact copies
      of the same motif while a minority contain local color anomalies.

    Rule:
    - Extract every rectangular block and align them by local coordinates.
    - At each local coordinate, take the unique majority color across the blocks to form
      the consensus motif.
    - Leave any block that already matches the consensus unchanged.
    - Replace each non-matching block by the consensus motif, but recolor the cells that
      originally disagreed with the consensus to red.

    Key insight:
    - The correct reference pattern is not given directly by a marker block; it must be
      reconstructed from redundancy across all copies, and anomaly locations are preserved
      as red evidence rather than simply erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blocks = _extract_blocks(g)
    consensus = _consensus_patch([patch for _bbox, patch in blocks])

    out = g.copy()
    for (r0, c0, r1, c1), patch in blocks:
        diff = patch != consensus
        if not np.any(diff):
            continue
        repaired = consensus.copy()
        repaired[diff] = RED
        out[r0 : r1 + 1, c0 : c1 + 1] = repaired

    return out


def _paint_random_feature(patch: np.ndarray, color: int) -> None:
    h, w = patch.shape
    kind = random.choice(("rect", "hseg", "vseg", "cross"))

    if kind == "rect":
        r0 = random.randint(0, h - 1)
        r1 = random.randint(r0, h - 1)
        c0 = random.randint(0, w - 1)
        c1 = random.randint(c0, w - 1)
        patch[r0 : r1 + 1, c0 : c1 + 1] = color
        return

    if kind == "hseg":
        row = random.randint(0, h - 1)
        c0 = random.randint(0, w - 1)
        c1 = random.randint(c0, w - 1)
        patch[row, c0 : c1 + 1] = color
        return

    if kind == "vseg":
        col = random.randint(0, w - 1)
        r0 = random.randint(0, h - 1)
        r1 = random.randint(r0, h - 1)
        patch[r0 : r1 + 1, col] = color
        return

    row = random.randint(0, h - 1)
    col = random.randint(0, w - 1)
    patch[row, :] = color
    patch[:, col] = color


def _make_base_patch(height: int, width: int, palette: list[int]) -> np.ndarray:
    for _ in range(100):
        color_count = random.randint(2, min(4, len(palette)))
        motif_colors = random.sample(palette, color_count)
        patch = np.full((height, width), motif_colors[0], dtype=int)

        feature_colors = motif_colors[1:] + random.choices(motif_colors[1:], k=random.randint(1, 4))
        random.shuffle(feature_colors)
        for color in feature_colors:
            _paint_random_feature(patch, color)

        missing = [color for color in motif_colors if not np.any(patch == color)]
        if missing:
            cells = [(r, c) for r in range(height) for c in range(width)]
            random.shuffle(cells)
            for color, (r, c) in zip(missing, cells):
                patch[r, c] = color

        if len(set(map(int, patch.ravel()))) >= 2:
            return patch

    raise RuntimeError("failed to build a multi-color base patch")


def _mutate_patch(base_patch: np.ndarray, palette: list[int]) -> tuple[np.ndarray, np.ndarray]:
    mutated = base_patch.copy()
    h, w = mutated.shape
    max_diff = min(max(2, (h * w) // 2), h * w - 1, 6)
    diff_count = random.randint(1, max_diff)
    indices = random.sample(range(h * w), diff_count)
    diff_mask = np.zeros((h, w), dtype=bool)

    for idx in indices:
        r, c = divmod(idx, w)
        choices = [color for color in palette if color != int(base_patch[r, c])]
        mutated[r, c] = random.choice(choices)
        diff_mask[r, c] = True

    return mutated, diff_mask


def _layout_options(block_count: int, height: int, width: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for rows in range(1, block_count + 1):
        min_cols = (block_count + rows - 1) // rows
        for cols in range(min_cols, block_count + 1):
            min_h = rows * height + (rows - 1)
            min_w = cols * width + (cols - 1)
            if min_h <= 30 and min_w <= 30:
                options.append((rows, cols))
    return options


def _arrange_blocks(block_count: int, height: int, width: int) -> tuple[int, int, list[tuple[int, int]]]:
    options = _layout_options(block_count, height, width)
    random.shuffle(options)

    for rows, cols in options:
        for _ in range(60):
            row_gaps = [random.randint(1, 3) for _ in range(rows - 1)]
            col_gaps = [random.randint(1, 3) for _ in range(cols - 1)]
            core_h = rows * height + sum(row_gaps)
            core_w = cols * width + sum(col_gaps)
            if core_h > 30 or core_w > 30:
                continue

            top = random.randint(0, min(3, 30 - core_h))
            bottom = random.randint(0, min(3, 30 - core_h - top))
            left = random.randint(0, min(3, 30 - core_w))
            right = random.randint(0, min(3, 30 - core_w - left))
            grid_h = core_h + top + bottom
            grid_w = core_w + left + right

            row_starts = [top]
            for gap in row_gaps:
                row_starts.append(row_starts[-1] + height + gap)
            col_starts = [left]
            for gap in col_gaps:
                col_starts.append(col_starts[-1] + width + gap)

            slots = [(r, c) for r in row_starts for c in col_starts]
            if len(slots) < block_count:
                continue
            chosen = random.sample(slots, block_count)
            return grid_h, grid_w, chosen

    raise RuntimeError("failed to arrange blocks within 30x30")


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _build_example() -> tuple[np.ndarray, np.ndarray]:
    block_count = random.randint(3, 6)
    minority_count = random.randint(1, (block_count - 1) // 2)
    majority_count = block_count - minority_count

    height = random.randint(3, 6)
    width = random.randint(3, 6)
    palette = random.sample(INPUT_COLORS, random.randint(3, 5))
    base_patch = _make_base_patch(height, width, palette)

    patches = [base_patch.copy() for _ in range(majority_count)]
    diff_masks = [np.zeros((height, width), dtype=bool) for _ in range(majority_count)]
    for _ in range(minority_count):
        mutated, diff_mask = _mutate_patch(base_patch, palette)
        patches.append(mutated)
        diff_masks.append(diff_mask)

    order = list(range(block_count))
    random.shuffle(order)
    patches = [patches[i] for i in order]
    diff_masks = [diff_masks[i] for i in order]

    grid_h, grid_w, positions = _arrange_blocks(block_count, height, width)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros((grid_h, grid_w), dtype=int)

    for patch, diff_mask, (top, left) in zip(patches, diff_masks, positions):
        _paste_patch(grid, patch, top, left)
        out_patch = base_patch.copy()
        out_patch[diff_mask] = RED
        if not np.any(diff_mask):
            out_patch = patch
        _paste_patch(expected, out_patch, top, left)

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, expected = _build_example()
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        if np.count_nonzero(actual == RED) == 0:
            continue
        if len({grid.shape, actual.shape}) != 1:
            continue
        return grid
    raise RuntimeError("failed to generate a valid consensus-repair puzzle")
