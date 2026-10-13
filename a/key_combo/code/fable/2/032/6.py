from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLUE = 1
INPUT_COLORS = [0, 2, 3, 4, 5, 6, 7, 8, 9]


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if bool(flag) == bool(value):
            if start is None:
                start = idx
        elif start is not None:
            spans.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        spans.append((int(start), int(len(mask) - 1)))
    return spans


def _span_lengths(spans: list[tuple[int, int]]) -> list[int]:
    return [int(b - a + 1) for a, b in spans]


def _sig(arr: np.ndarray) -> bytes:
    return arr.tobytes() + str(tuple(arr.shape)).encode()


def _majority_color(values: np.ndarray) -> int:
    counts = np.bincount(values.astype(int), minlength=max(10, int(values.max()) + 1))
    best = int(counts.max())
    winners = np.flatnonzero(counts == best)
    assert len(winners) == 1, "majority must be unique at every aligned coordinate"
    return int(winners[0])


def _majority_grid(stack: np.ndarray) -> np.ndarray:
    n, h, w = stack.shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = _majority_color(stack[:, r, c])
    return out


def _surface_vote_recovers_template(blocks: list[np.ndarray], template: np.ndarray) -> bool:
    stack = np.stack(blocks, axis=0)
    h, w = template.shape
    for r in range(h):
        for c in range(w):
            values = stack[:, r, c].astype(int)
            counts = np.bincount(values, minlength=max(10, int(values.max()) + 1))
            best = int(counts.max())
            winners = np.flatnonzero(counts == best)
            if len(winners) != 1 or int(winners[0]) != int(template[r, c]):
                return False
    return True


def _parse_layout(grid: np.ndarray) -> dict:
    candidates: list[dict] = []
    for sep in sorted(set(map(int, np.unique(grid)))):
        row_mask = np.all(grid == int(sep), axis=1)
        col_mask = np.all(grid == int(sep), axis=0)
        if not bool(row_mask.any()) or not bool(col_mask.any()):
            continue

        row_blocks = _runs(row_mask, False)
        col_blocks = _runs(col_mask, False)
        if not row_blocks or not col_blocks:
            continue

        row_lengths = _span_lengths(row_blocks)
        col_lengths = _span_lengths(col_blocks)
        if len(set(row_lengths)) != 1 or len(set(col_lengths)) != 1:
            continue

        if len(row_blocks) * len(col_blocks) < 3:
            continue

        candidates.append(
            {
                "separator": int(sep),
                "row_blocks": row_blocks,
                "col_blocks": col_blocks,
                "block_h": int(row_lengths[0]),
                "block_w": int(col_lengths[0]),
            }
        )

    assert len(candidates) == 1, "separator layout must be uniquely identifiable"
    return candidates[0]


def _extract_aligned_blocks(grid: np.ndarray) -> tuple[dict, np.ndarray]:
    layout = _parse_layout(grid)
    aligned_blocks: list[np.ndarray] = []

    for i, (r0, r1) in enumerate(layout["row_blocks"]):
        for j, (c0, c1) in enumerate(layout["col_blocks"]):
            block = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
            aligned_blocks.append(np.roll(block, shift=(-int(i), -int(j)), axis=(0, 1)))

    return layout, np.stack(aligned_blocks, axis=0)


def _recover(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, dict, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert BLUE not in set(map(int, np.unique(g))), "blue is reserved for the output marker"

    layout, aligned = _extract_aligned_blocks(g)
    consensus = _majority_grid(aligned)

    mismatches = aligned != consensus[None, :, :]
    mismatch_counts = mismatches.reshape(aligned.shape[0], -1).sum(axis=1)
    damaged = np.flatnonzero(mismatch_counts > 0)
    assert len(damaged) == 1, "exactly one aligned block must disagree with the consensus"
    damaged_idx = int(damaged[0])

    damage_mask = mismatches[damaged_idx]
    assert bool(damage_mask.any()), "the damaged block must actually lose information"

    bg = int(Counter(map(int, consensus.flat)).most_common(1)[0][0])
    damaged_fill = aligned[damaged_idx][damage_mask]
    assert len(set(map(int, damaged_fill.tolist()))) == 1, "damage must overwrite with one fill color"
    assert int(damaged_fill[0]) == int(bg), "damage must overwrite with the mother background color"
    assert bool(np.all(consensus[damage_mask] != int(bg))), "only foreground cells may be damaged"

    return consensus, damage_mask, damaged_idx, layout, aligned


def main(grid: np.ndarray) -> np.ndarray:
    """
    Undo position-driven cyclic rolls, vote out a damaged copy, and mark the lost template cells in blue.

    Input structure:
    - The grid is partitioned into equal-sized blocks by full-row and full-column separator lines of one color.
    - Block `(i, j)` is a cyclic copy of one hidden mother tile, rolled down by `i` and right by `j`.
    - Exactly one block has some of its foreground cells replaced by the mother background color.

    Rule:
    - Detect the separator layout, crop every block, and inverse-roll block `(i, j)` by `(-i, -j)` back into mother coordinates.
    - Take a per-cell majority vote across the aligned copies to recover the consensus mother tile.
    - Find the unique block that still disagrees with that consensus, and collect its mismatching mother coordinates.
    - Output the consensus mother tile with those damaged coordinates recolored blue.

    Key insight:
    - Surface appearances differ only because the block grid position encodes the cyclic shift, so redundancy reappears only after undoing that position-dependent roll.
    """
    consensus, damage_mask, _damaged_idx, _layout, _aligned = _recover(grid)
    out = consensus.copy()
    out[damage_mask] = BLUE
    return out


_BASE_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
]


def _shape_variants() -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in _BASE_SHAPES:
        for k in range(4):
            rotated = np.rot90(base, k)
            for variant in (rotated, np.fliplr(rotated)):
                sig = _sig(variant.astype(int))
                if sig not in seen:
                    seen.add(sig)
                    out.append(variant.copy())
    return out


_SHAPES = _shape_variants()


def _can_place(occupied: np.ndarray, shape: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = shape.shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(shape):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        rr0 = max(0, rr - padding)
        rr1 = min(H, rr + padding + 1)
        cc0 = max(0, cc - padding)
        cc1 = min(W, cc + padding + 1)
        if bool(occupied[rr0:rr1, cc0:cc1].any()):
            return False
    return True


def _stamp(template: np.ndarray, occupied: np.ndarray, shape: np.ndarray, color: int, r0: int, c0: int) -> None:
    for dr, dc in np.argwhere(shape):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        template[rr, cc] = int(color)
        occupied[rr, cc] = True


def _rolled(template: np.ndarray, i: int, j: int) -> np.ndarray:
    return np.roll(template, shift=(int(i), int(j)), axis=(0, 1))


def _generate_template(block_h: int, block_w: int, bg: int, colors: list[int], rows: int, cols: int) -> np.ndarray | None:
    padding = 1 if min(block_h, block_w) >= 6 else 0
    for _ in range(300):
        template = np.full((block_h, block_w), int(bg), dtype=int)
        occupied = np.zeros((block_h, block_w), dtype=bool)
        usable_colors = random.sample(colors, random.randint(2, min(4, len(colors))))
        target_shapes = random.randint(3, 6)
        placed = 0

        for _shape_idx in range(target_shapes):
            shape = random.choice(_SHAPES)
            sh, sw = shape.shape
            if sh > block_h or sw > block_w:
                continue
            positions = [(r, c) for r in range(block_h - sh + 1) for c in range(block_w - sw + 1)]
            random.shuffle(positions)
            for r0, c0 in positions:
                if _can_place(occupied, shape, int(r0), int(c0), padding):
                    _stamp(template, occupied, shape, random.choice(usable_colors), int(r0), int(c0))
                    placed += 1
                    break

        fg_mask = template != int(bg)
        fg_count = int(fg_mask.sum())
        if placed < 3 or fg_count < 6 or fg_count > (block_h * block_w) // 2:
            continue

        if len(set(map(int, template[fg_mask].tolist()))) < 2:
            continue

        pristine_blocks = [_rolled(template, i, j) for i in range(rows) for j in range(cols)]
        if len({_sig(block) for block in pristine_blocks}) != rows * cols:
            continue

        if _surface_vote_recovers_template(pristine_blocks, template):
            continue

        return template

    return None


def _compose_input(
    template: np.ndarray,
    sep: int,
    bg: int,
    rows: int,
    cols: int,
    damaged_block: tuple[int, int],
    damage_mask: np.ndarray,
) -> np.ndarray:
    block_h, block_w = template.shape
    grid = np.full((rows * block_h + rows + 1, cols * block_w + cols + 1), int(sep), dtype=int)

    for i in range(rows):
        for j in range(cols):
            block = _rolled(template, i, j)
            if (i, j) == damaged_block:
                block = block.copy()
                block[np.roll(damage_mask, shift=(int(i), int(j)), axis=(0, 1))] = int(bg)
            r0 = 1 + i * (block_h + 1)
            c0 = 1 + j * (block_w + 1)
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = block

    return grid


def _validate_instance(
    grid: np.ndarray,
    template: np.ndarray,
    bg: int,
    sep: int,
    rows: int,
    cols: int,
    damaged_block: tuple[int, int],
    damage_mask: np.ndarray,
) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert template.ndim == 2
    assert BLUE not in set(map(int, np.unique(grid)))
    assert int(bg) != BLUE and int(sep) != BLUE and int(bg) != int(sep)

    layout = _parse_layout(grid)
    assert int(layout["separator"]) == int(sep)
    assert int(layout["block_h"]) == int(template.shape[0])
    assert int(layout["block_w"]) == int(template.shape[1])
    assert len(layout["row_blocks"]) == int(rows)
    assert len(layout["col_blocks"]) == int(cols)

    pristine_blocks: list[np.ndarray] = []
    damaged_index = int(damaged_block[0] * cols + damaged_block[1])
    for i, (r0, r1) in enumerate(layout["row_blocks"]):
        for j, (c0, c1) in enumerate(layout["col_blocks"]):
            expected = _rolled(template, i, j)
            if (i, j) == damaged_block:
                expected = expected.copy()
                expected[np.roll(damage_mask, shift=(int(i), int(j)), axis=(0, 1))] = int(bg)
            observed = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
            assert np.array_equal(observed, expected)
            pristine_blocks.append(_rolled(template, i, j))

    assert len({_sig(block) for block in pristine_blocks}) == rows * cols
    assert not _surface_vote_recovers_template(pristine_blocks, template)

    consensus, found_damage, found_idx, _layout, aligned = _recover(grid)
    assert int(found_idx) == int(damaged_index)
    assert np.array_equal(consensus, template)
    assert np.array_equal(found_damage, damage_mask)
    assert bool(np.all(aligned[damaged_index][damage_mask] == int(bg)))

    expected_output = template.copy()
    expected_output[damage_mask] = BLUE
    assert np.array_equal(main(grid), expected_output)


def generate_input() -> np.ndarray:
    for _ in range(500):
        block_h = random.randint(5, 8)
        block_w = random.randint(5, 8)

        max_rows = min(4, block_h, 29 // (block_h + 1))
        max_cols = min(4, block_w, 29 // (block_w + 1))
        if max_rows < 2 or max_cols < 2:
            continue

        rows = random.randint(2, max_rows)
        cols = random.randint(2, max_cols)
        if rows * cols < 3:
            continue

        bg, sep = random.sample(INPUT_COLORS, 2)
        fg_palette = [color for color in INPUT_COLORS if color not in (int(bg), int(sep))]
        template = _generate_template(block_h, block_w, int(bg), fg_palette, rows, cols)
        if template is None:
            continue

        fg_coords = np.argwhere(template != int(bg))
        if fg_coords.shape[0] < 2:
            continue

        damage_count = random.randint(1, min(5, int(fg_coords.shape[0]) - 1))
        damage_pick = fg_coords[np.random.choice(fg_coords.shape[0], size=damage_count, replace=False)]
        damage_mask = np.zeros_like(template, dtype=bool)
        damage_mask[damage_pick[:, 0], damage_pick[:, 1]] = True

        damaged_block = (random.randrange(rows), random.randrange(cols))
        grid = _compose_input(template, int(sep), int(bg), rows, cols, damaged_block, damage_mask)

        _validate_instance(grid, template, int(bg), int(sep), rows, cols, damaged_block, damage_mask)
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
