from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
FG_COLORS = (2, 8)  # red, teal
BLOCK_SIZE = 5
MAX_GENERATE_ATTEMPTS = 500
BLOCK_BUILD_ATTEMPTS = 60
SIGNATURES = [
    sig
    for size in (1, 2)
    for sig in itertools.combinations(range(BLOCK_SIZE), size)
]


def _most_common(a: np.ndarray) -> int:
    return int(Counter(map(int, a.ravel())).most_common(1)[0][0])


def _full_indices(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == color))]
    return [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == color))]


def _segments(length: int, cuts: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for cut in sorted(cuts) + [length]:
        if cut > start:
            spans.append((start, cut))
        start = cut + 1
    return spans


def _find_separator(grid: np.ndarray) -> int:
    candidates: list[tuple[int, int]] = []
    global_bg = _most_common(grid)
    for color in map(int, np.unique(grid)):
        if color == global_bg:
            continue
        row_spans = _segments(grid.shape[0], _full_indices(grid, color, axis=0))
        col_spans = _segments(grid.shape[1], _full_indices(grid, color, axis=1))
        if not row_spans or not col_spans:
            continue
        if all(r1 - r0 == BLOCK_SIZE for r0, r1 in row_spans) and all(
            c1 - c0 == BLOCK_SIZE for c0, c1 in col_spans
        ):
            sep_count = len(_full_indices(grid, color, axis=0)) + len(_full_indices(grid, color, axis=1))
            candidates.append((sep_count, color))
    assert candidates, "failed to identify separator color"
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def _extract_blocks(grid: np.ndarray) -> list[np.ndarray]:
    sep = _find_separator(grid)
    row_spans = _segments(grid.shape[0], _full_indices(grid, sep, axis=0))
    col_spans = _segments(grid.shape[1], _full_indices(grid, sep, axis=1))

    blocks: list[np.ndarray] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = np.array(grid[r0:r1, c0:c1], dtype=int)
            if patch.shape == (BLOCK_SIZE, BLOCK_SIZE):
                blocks.append(patch)
    assert blocks, "no 5x5 blocks found"
    return blocks


def _block_mask(block: np.ndarray) -> np.ndarray:
    mask = np.isin(block, FG_COLORS)
    assert mask.any(), "empty block"
    return mask


def _block_color(block: np.ndarray) -> int:
    mask = _block_mask(block)
    colors = {int(v) for v in block[mask]}
    assert len(colors) == 1, "each block must be monochrome"
    return int(next(iter(colors)))


def _edge_signature(block: np.ndarray, side: str) -> tuple[int, ...]:
    mask = _block_mask(block)
    if side == "left":
        return tuple(map(int, np.flatnonzero(mask[:, 0])))
    if side == "right":
        return tuple(map(int, np.flatnonzero(mask[:, -1])))
    raise ValueError(f"unsupported side: {side}")


def _valid_chains(blocks: list[np.ndarray], *, use_sizes_only: bool = False) -> list[tuple[int, ...]]:
    left = [_edge_signature(block, "left") for block in blocks]
    right = [_edge_signature(block, "right") for block in blocks]

    def connects(i: int, j: int) -> bool:
        if not right[i]:
            return False
        if use_sizes_only:
            return len(right[i]) == len(left[j])
        return right[i] == left[j]

    valid: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(blocks))):
        if all(connects(perm[k], perm[k + 1]) for k in range(len(perm) - 1)):
            valid.append(tuple(map(int, perm)))
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the unique edge-matched chain, alternate add/subtract block areas by chain position, and emit a 1-column bar in the first block's color.

    Input structure:
    - Grey separator lines divide the canvas into several 5x5 sprite blocks.
    - Each block contains background plus one monochrome foreground shape, colored either red or teal.
    - A block's right-edge foreground rows may match another block's left-edge foreground rows.

    Rule:
    - Ignore the input placement order and recover the unique chain whose consecutive blocks have exactly the same non-empty foreground-row set on predecessor-right and successor-left edges.
    - Count foreground pixels in chain positions 1, 3, 5, ... positively and in positions 2, 4, 6, ... negatively.
    - Let S be that algebraic sum, then output an S-by-1 vertical bar filled entirely with the foreground color of the first chain block.

    Key insight:
    - Red and teal are a decoy for arithmetic sign; the true sign is determined only after restoring chain order from exact edge-row matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    blocks = _extract_blocks(g)
    chains = _valid_chains(blocks)
    assert len(chains) == 1, "chain must be unique"
    chain = chains[0]

    signed_total = 0
    for pos, idx in enumerate(chain):
        count = int(_block_mask(blocks[idx]).sum())
        signed_total += count if pos % 2 == 0 else -count
    assert signed_total >= 1

    color = _block_color(blocks[chain[0]])
    return np.full((signed_total, 1), color, dtype=int)


def _alt_sum(values: list[int]) -> int:
    return sum(v if i % 2 == 0 else -v for i, v in enumerate(values))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(mask: np.ndarray) -> bool:
    start = tuple(np.argwhere(mask)[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], *, horiz_first: bool) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    if horiz_first:
        step = 1 if ec >= c else -1
        for cc in range(c, ec + step, step):
            mask[r, cc] = True
        step = 1 if er >= r else -1
        for rr in range(r, er + step, step):
            mask[rr, ec] = True
    else:
        step = 1 if er >= r else -1
        for rr in range(r, er + step, step):
            mask[rr, c] = True
        step = 1 if ec >= c else -1
        for cc in range(c, ec + step, step):
            mask[er, cc] = True


def _forbidden_cells(left_sig: tuple[int, ...], right_sig: tuple[int, ...]) -> set[tuple[int, int]]:
    forbidden = {(r, 0) for r in range(BLOCK_SIZE) if r not in left_sig}
    forbidden |= {(r, BLOCK_SIZE - 1) for r in range(BLOCK_SIZE) if r not in right_sig}
    return forbidden


def _frontier(mask: np.ndarray, forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            cell = (int(nr), int(nc))
            if not (0 <= nr < BLOCK_SIZE and 0 <= nc < BLOCK_SIZE):
                continue
            if cell in forbidden or mask[nr, nc]:
                continue
            out.add(cell)
    return list(out)


def _make_mask(
    left_sig: tuple[int, ...], right_sig: tuple[int, ...], target_area: int
) -> np.ndarray | None:
    forbidden = _forbidden_cells(left_sig, right_sig)
    for _ in range(BLOCK_BUILD_ATTEMPTS):
        mask = _seed_mask(left_sig, right_sig)
        if any(mask[r, 0] for r in range(BLOCK_SIZE) if r not in left_sig):
            continue
        if any(mask[r, BLOCK_SIZE - 1] for r in range(BLOCK_SIZE) if r not in right_sig):
            continue
        if int(mask.sum()) > target_area or not _is_connected(mask):
            continue
        while int(mask.sum()) < target_area:
            choices = _frontier(mask, forbidden)
            if not choices:
                break
            r, c = random.choice(choices)
            mask[r, c] = True
        if (
            int(mask.sum()) == target_area
            and _is_connected(mask)
            and tuple(np.flatnonzero(mask[:, 0])) == left_sig
            and tuple(np.flatnonzero(mask[:, -1])) == right_sig
        ):
            return mask
    return None


def _seed_mask(left_sig: tuple[int, ...], right_sig: tuple[int, ...]) -> np.ndarray:
    hub = (random.randint(1, 3), random.randint(1, 3))
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    for r in left_sig:
        _draw_path(mask, (int(r), 0), hub, horiz_first=bool(random.getrandbits(1)))
    for r in right_sig:
        _draw_path(mask, (int(r), BLOCK_SIZE - 1), hub, horiz_first=bool(random.getrandbits(1)))
    return mask


def _render_block(mask: np.ndarray, color: int) -> np.ndarray:
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


def _assemble_row(blocks: list[np.ndarray]) -> np.ndarray:
    width = len(blocks) * BLOCK_SIZE + (len(blocks) - 1)
    grid = np.full((BLOCK_SIZE, width), BG, dtype=int)
    c = 0
    for i, block in enumerate(blocks):
        grid[:, c : c + BLOCK_SIZE] = block
        c += BLOCK_SIZE
        if i < len(blocks) - 1:
            grid[:, c] = SEP
            c += 1
    return grid


def _choose_signatures(n: int) -> tuple[list[tuple[int, ...]], tuple[int, ...], tuple[int, ...]]:
    while True:
        internal = random.sample(SIGNATURES, n - 1)
        if len({len(sig) for sig in internal}) == len(internal):
            continue
        leftovers = [sig for sig in SIGNATURES if sig not in internal]
        dead_left, dead_right = random.sample(leftovers, 2)
        return internal, dead_left, dead_right


def _counts_by_order(blocks: list[np.ndarray], order: list[int] | tuple[int, ...]) -> list[int]:
    return [int(_block_mask(blocks[i]).sum()) for i in order]


def _shortcut_checks(chain_blocks: list[np.ndarray], input_blocks: list[np.ndarray], chain: tuple[int, ...]) -> bool:
    counts = _counts_by_order(chain_blocks, list(range(len(chain_blocks))))
    colors = [_block_color(block) for block in chain_blocks]
    actual = _alt_sum(counts)

    color_short = sum(count if color == 8 else -count for count, color in zip(counts, colors))
    input_short = _alt_sum([int(_block_mask(block).sum()) for block in input_blocks])
    bad = {color_short, input_short}
    if len(counts) % 2 == 0:
        bad.add(_alt_sum(list(reversed(counts))))
    if len(counts) >= 4:
        bad.add(_alt_sum([counts[i] for i in np.argsort([-v for v in counts])]))
        bad.add(_alt_sum([counts[i] for i in np.argsort(counts)]))

    if actual in bad:
        return False

    size_only_chains = _valid_chains(input_blocks, use_sizes_only=True)
    if len(size_only_chains) == 1 and size_only_chains[0] == chain:
        return False

    positive_red = any(colors[i] == 2 for i in range(0, len(colors), 2))
    negative_teal = any(colors[i] == 8 for i in range(1, len(colors), 2))
    return positive_red and negative_teal


def _generate_one() -> np.ndarray | None:
    n = random.choice([3, 4, 5])
    internal_sigs, dead_left, dead_right = _choose_signatures(n)

    left_sigs = [dead_left] + internal_sigs
    right_sigs = internal_sigs + [dead_right]

    blocks: list[np.ndarray] = []
    for i in range(n):
        built = None
        for _ in range(BLOCK_BUILD_ATTEMPTS):
            probe = _seed_mask(left_sigs[i], right_sigs[i])
            min_area = int(probe.sum())
            if min_area > 14:
                continue
            target_area = random.randint(min_area, min(14, min_area + 4))
            mask = _make_mask(left_sigs[i], right_sigs[i], target_area)
            if mask is None:
                continue
            built = mask
            break
        if built is None:
            return None
        blocks.append(_render_block(built, color=2))

    counts = [int(_block_mask(block).sum()) for block in blocks]
    actual = _alt_sum(counts)
    if not (1 <= actual <= 30):
        return None

    colors: list[int] = []
    for _ in range(BLOCK_BUILD_ATTEMPTS):
        trial = [random.choice(FG_COLORS) for _ in range(n)]
        if len(set(trial)) < 2:
            continue
        if not any(trial[i] == 2 for i in range(0, n, 2)):
            continue
        if not any(trial[i] == 8 for i in range(1, n, 2)):
            continue
        color_short = sum(count if color == 8 else -count for count, color in zip(counts, trial))
        if color_short == actual:
            continue
        colors = trial
        break
    if not colors:
        return None

    for i, color in enumerate(colors):
        blocks[i][blocks[i] != BG] = int(color)

    exact_chains = _valid_chains(blocks)
    if exact_chains != [tuple(range(n))]:
        return None

    perm = list(range(n))
    for _ in range(BLOCK_BUILD_ATTEMPTS):
        random.shuffle(perm)
        if perm != list(range(n)) and perm != list(reversed(range(n))) and perm[0] != 0:
            break
    else:
        return None

    input_blocks = [blocks[i] for i in perm]
    grid = _assemble_row(input_blocks)

    input_chains = _valid_chains(input_blocks)
    if len(input_chains) != 1:
        return None
    chain = input_chains[0]
    if tuple(perm[idx] for idx in chain) != tuple(range(n)):
        return None

    if not _shortcut_checks(blocks, input_blocks, chain):
        return None

    expected = np.full((actual, 1), colors[0], dtype=int)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate chain-order sample")
