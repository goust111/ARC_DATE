from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
SEP = 5
EDGE = 2
BLOCK = 5
INNER_COLORS = [1, 3, 4, 6, 7, 8]
SIGNATURE_POOL = [
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 3),
    (1, 4),
    (2, 4),
    (0, 1, 3),
    (0, 2, 4),
    (1, 2, 4),
    (0, 1, 4),
    (0, 3, 4),
    (1, 2, 3),
    (0, 1, 2, 4),
]
TOP_BOTTOM_POOL = [
    (),
    (1,),
    (2,),
    (3,),
    (1, 2),
    (1, 3),
    (2, 3),
    (1, 2, 3),
]


def _constant_full_lines(grid: np.ndarray, axis: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    length = grid.shape[axis]
    for idx in range(length):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if bool(np.all(line == line[0])):
            out.append((idx, int(line[0])))
    return out


def _find_separator_color(grid: np.ndarray) -> int:
    candidates = {color for _idx, color in _constant_full_lines(grid, 0)}
    candidates |= {color for _idx, color in _constant_full_lines(grid, 1)}
    assert candidates, "expected separator lines"

    scored: list[tuple[int, int]] = []
    for color in sorted(candidates):
        row_spans = _try_intervals(grid.shape[0], _separator_indices(grid, 0, int(color)))
        col_spans = _try_intervals(grid.shape[1], _separator_indices(grid, 1, int(color)))
        if row_spans is None or col_spans is None:
            continue
        row_sizes = {r1 - r0 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 for c0, c1 in col_spans}
        if row_sizes == {BLOCK} and col_sizes == {BLOCK}:
            lines = len(_separator_indices(grid, 0, int(color))) + len(_separator_indices(grid, 1, int(color)))
            scored.append((lines, int(color)))

    assert scored, "failed to identify separator color"
    return max(scored)[1]


def _separator_indices(grid: np.ndarray, axis: int, sep: int) -> list[int]:
    length = grid.shape[axis]
    out: list[int] = []
    for idx in range(length):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if bool(np.all(line == int(sep))):
            out.append(int(idx))
    return out


def _intervals(length: int, separators: list[int]) -> list[tuple[int, int]]:
    if not separators:
        return [(0, int(length))]
    bounds = [-1, *map(int, separators), int(length)]
    out = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in out]
    assert min(sizes) > 0
    assert len(set(sizes)) == 1
    return out


def _try_intervals(length: int, separators: list[int]) -> list[tuple[int, int]] | None:
    if not separators:
        return [(0, int(length))]
    bounds = [-1, *map(int, separators), int(length)]
    out = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in out]
    if not sizes or min(sizes) <= 0 or len(set(sizes)) != 1:
        return None
    return out


def _extract_blocks(grid: np.ndarray) -> list[np.ndarray]:
    sep = _find_separator_color(grid)
    row_spans = _intervals(grid.shape[0], _separator_indices(grid, 0, sep))
    col_spans = _intervals(grid.shape[1], _separator_indices(grid, 1, sep))
    assert {r1 - r0 for r0, r1 in row_spans} == {BLOCK}
    assert {c1 - c0 for c0, c1 in col_spans} == {BLOCK}

    blocks: list[np.ndarray] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            blocks.append(grid[r0:r1, c0:c1].copy())
    return blocks


def _left_signature(block: np.ndarray) -> tuple[int, ...]:
    return tuple(int(r) for r in np.where(block[:, 0] == EDGE)[0].tolist())


def _right_signature(block: np.ndarray) -> tuple[int, ...]:
    return tuple(int(r) for r in np.where(block[:, -1] == EDGE)[0].tolist())


def _inner_mask_info(block: np.ndarray) -> tuple[int, np.ndarray]:
    colors = sorted(set(map(int, np.unique(block))) - {BG, EDGE})
    assert len(colors) == 1, "each block must contain exactly one inner-mask color"
    color = int(colors[0])
    mask = block == color
    assert mask[1:4, 1:4].any()
    assert not mask[0, :].any()
    assert not mask[-1, :].any()
    assert not mask[:, 0].any()
    assert not mask[:, -1].any()
    return color, mask


def _recover_chain(blocks: list[np.ndarray]) -> list[int]:
    n = len(blocks)
    lefts = [_left_signature(block) for block in blocks]
    rights = [_right_signature(block) for block in blocks]

    predecessors: list[int | None] = [None] * n
    successors: list[int | None] = [None] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if rights[i] == lefts[j]:
                assert successors[i] is None, "ambiguous successor"
                assert predecessors[j] is None, "ambiguous predecessor"
                successors[i] = j
                predecessors[j] = i

    starts = [idx for idx, pred in enumerate(predecessors) if pred is None]
    ends = [idx for idx, succ in enumerate(successors) if succ is None]
    assert len(starts) == 1 and len(ends) == 1, "expected one linear chain"

    chain: list[int] = []
    seen: set[int] = set()
    cur = starts[0]
    while cur is not None:
        assert cur not in seen, "cycle detected"
        seen.add(cur)
        chain.append(cur)
        cur = successors[cur]

    assert len(chain) == n, "chain must use every block"
    return chain


def _render_output(blocks: list[np.ndarray], chain: list[int]) -> np.ndarray:
    ordered = [blocks[idx].copy() for idx in chain]
    inner_colors: list[int] = []
    inner_masks: list[np.ndarray] = []
    for block in ordered:
        color, mask = _inner_mask_info(block)
        inner_colors.append(color)
        inner_masks.append(mask)

    rotated = [inner_colors[-1], *inner_colors[:-1]]
    for block, mask, color in zip(ordered, inner_masks, rotated):
        block[mask] = int(color)

    out = np.full((BLOCK, len(ordered) * BLOCK + len(ordered) - 1), SEP, dtype=int)
    for idx, block in enumerate(ordered):
        c0 = idx * (BLOCK + 1)
        out[:, c0 : c0 + BLOCK] = block
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder divided sprite blocks by matched red edge signatures and rotate their inner colors around the recovered chain.

    Input structure:
    - A black background grid is partitioned by full grey separator rows and columns into equal 5x5 blocks.
    - Each block contains red edge pixels on the block border plus one non-red interior mask color that never touches the border.
    - Horizontal connectivity is encoded only by exact matches between one block's right-edge red row offsets and another block's left-edge red row offsets.

    Rule:
    - Split the grid into 5x5 blocks using the grey separators.
    - Recover the unique left-to-right chain whose consecutive blocks satisfy right-edge signature equals next left-edge signature.
    - Treat that linear chain as a ring, keep every block's own geometry unchanged, and recolor each interior mask with the previous block's interior color.
    - Output a single horizontal strip of the blocks in recovered chain order, separated by grey columns.

    Key insight:
    - The visible spatial layout is a distraction: the true order comes from comparing red edge-offset signatures, and that inferred order alone determines the cyclic color transfer.
    """
    g = np.asarray(grid, dtype=int)
    blocks = _extract_blocks(g)
    chain = _recover_chain(blocks)
    return _render_output(blocks, chain)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < 3 and 0 <= cc < 3:
            out.append((rr, cc))
    return out


def _random_inner_mask() -> np.ndarray:
    target = random.randint(2, 5)
    start = (random.randrange(3), random.randrange(3))
    cells = {start}
    frontier = set(_neighbors4(start))
    while len(cells) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(cell):
            if nb not in cells:
                frontier.add(nb)

    mask = np.zeros((3, 3), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_block(
    left_sig: tuple[int, ...],
    right_sig: tuple[int, ...],
    top_sig: tuple[int, ...],
    bottom_sig: tuple[int, ...],
    inner_mask: np.ndarray,
    color: int,
) -> np.ndarray:
    block = np.full((BLOCK, BLOCK), BG, dtype=int)
    for r in left_sig:
        block[int(r), 0] = EDGE
    for r in right_sig:
        block[int(r), BLOCK - 1] = EDGE
    for c in top_sig:
        block[0, int(c)] = EDGE
    for c in bottom_sig:
        block[BLOCK - 1, int(c)] = EDGE
    block[1:4, 1:4][inner_mask] = int(color)
    return block


def _layout_choices(n: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for rows in range(1, n + 1):
        if n % rows != 0:
            continue
        cols = n // rows
        h = rows * BLOCK + rows - 1
        w = cols * BLOCK + cols - 1
        if h <= 30 and w <= 30:
            pairs.append((rows, cols))
    return pairs


def _assemble_grid(blocks: list[np.ndarray], rows: int, cols: int, placement: list[int]) -> np.ndarray:
    h = rows * BLOCK + rows - 1
    w = cols * BLOCK + cols - 1
    grid = np.full((h, w), SEP, dtype=int)
    for slot, block_idx in enumerate(placement):
        r = slot // cols
        c = slot % cols
        r0 = r * (BLOCK + 1)
        c0 = c * (BLOCK + 1)
        grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = blocks[block_idx]
    return grid


def _count_valid_chains(blocks: list[np.ndarray]) -> int:
    lefts = [_left_signature(block) for block in blocks]
    rights = [_right_signature(block) for block in blocks]
    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(rights[perm[i]] == lefts[perm[i + 1]] for i in range(len(perm) - 1)):
            total += 1
    return total


def _distinct_shape_count(blocks: list[np.ndarray]) -> int:
    patterns = set()
    for block in blocks:
        _color, mask = _inner_mask_info(block)
        patterns.add(tuple(map(int, mask[1:4, 1:4].flatten().tolist())))
    return len(patterns)


def _oracle_checks(grid: np.ndarray) -> bool:
    blocks = _extract_blocks(grid)
    if not (3 <= len(blocks) <= 5):
        return False
    if _count_valid_chains(blocks) != 1:
        return False

    colors = [_inner_mask_info(block)[0] for block in blocks]
    if len(set(colors)) != len(colors):
        return False
    if _distinct_shape_count(blocks) < 2:
        return False

    chain = _recover_chain(blocks)
    reading_order = list(range(len(blocks)))
    if chain == reading_order:
        return False

    out = main(grid)
    expected = _render_output(blocks, chain)
    if not np.array_equal(out, expected):
        return False

    ordered = [blocks[idx].copy() for idx in chain]
    strip_without_rotation = np.full_like(out, SEP)
    for idx, block in enumerate(ordered):
        c0 = idx * (BLOCK + 1)
        strip_without_rotation[:, c0 : c0 + BLOCK] = block
    if np.array_equal(out, strip_without_rotation):
        return False

    next_colors = [_inner_mask_info(block)[0] for block in ordered]
    wrong = [color for color in next_colors[1:]] + [next_colors[0]]
    wrong_strip = np.full_like(out, SEP)
    for idx, (block, wrong_color) in enumerate(zip(ordered, wrong)):
        alt = block.copy()
        _color, mask = _inner_mask_info(alt)
        alt[mask] = int(wrong_color)
        c0 = idx * (BLOCK + 1)
        wrong_strip[:, c0 : c0 + BLOCK] = alt
    if np.array_equal(out, wrong_strip):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    n = random.randint(3, 5)
    signatures = random.sample(SIGNATURE_POOL, n + 1)
    colors = random.sample(INNER_COLORS, n)

    masks = [_random_inner_mask() for _ in range(n)]
    if len({tuple(mask.flatten().tolist()) for mask in masks}) < 2:
        return None

    blocks: list[np.ndarray] = []
    for idx in range(n):
        left_sig = signatures[idx]
        right_sig = signatures[idx + 1]
        top_sig = random.choice(TOP_BOTTOM_POOL)
        bottom_sig = random.choice(TOP_BOTTOM_POOL)
        blocks.append(_make_block(left_sig, right_sig, top_sig, bottom_sig, masks[idx], colors[idx]))

    if _count_valid_chains(blocks) != 1:
        return None

    rows, cols = random.choice(_layout_choices(n))
    placement = list(range(n))
    random.shuffle(placement)
    if placement == list(range(n)):
        return None

    grid = _assemble_grid(blocks, rows, cols, placement)
    return grid if _oracle_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
