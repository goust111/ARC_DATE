"""ARC task 5: repair one corrupted block in a two-scale self-similar pattern.

Mechanism references: 58c02a16, c4d067a0, bb52a14b.
"""

from __future__ import annotations

import math
import random

import numpy as np


def _block_view(grid: np.ndarray, n: int, br: int, bc: int) -> np.ndarray:
    """Return block (br, bc), using the puzzle's block-coordinate frame."""
    return grid[br * n : (br + 1) * n, bc * n : (bc + 1) * n]


def _decode(grid: np.ndarray) -> tuple[int, int, tuple[int, int], np.ndarray, int, int]:
    """Decode n, background, damaged position, mask, mask-color, other-color."""
    g = np.asarray(grid)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("expected a square 2D grid")
    side = int(g.shape[0])
    n = math.isqrt(side)
    if n < 2 or n * n != side:
        raise ValueError("grid side must be the square of the block side")

    blocks = [[_block_view(g, n, r, c) for c in range(n)] for r in range(n)]
    uniform = []
    for r in range(n):
        for c in range(n):
            if np.unique(blocks[r][c]).size == 1:
                uniform.append((r, c))
    if len(uniform) != 1:
        raise ValueError("expected exactly one uniformly corrupted block")
    damaged = uniform[0]
    corrupt_color = int(blocks[damaged[0]][damaged[1]][0, 0])

    intact_sets = [
        set(map(int, np.unique(blocks[r][c])))
        for r in range(n)
        for c in range(n)
        if (r, c) != damaged
    ]
    if not intact_sets or any(len(s) != 2 for s in intact_sets):
        raise ValueError("every intact block must contain background and one main color")
    common = set.intersection(*intact_sets)
    if len(common) != 1:
        raise ValueError("intact blocks must determine one shared background color")
    bg = int(next(iter(common)))
    if corrupt_color == bg:
        raise ValueError("corruption color must differ from the background")

    masks = []
    layout = np.full((n, n), -1, dtype=int)
    for r in range(n):
        for c in range(n):
            if (r, c) == damaged:
                continue
            block = blocks[r][c]
            non_bg = set(map(int, np.unique(block))) - {bg}
            if len(non_bg) != 1:
                raise ValueError("an intact block must have exactly one main color")
            layout[r, c] = int(next(iter(non_bg)))
            masks.append(block != bg)
    mask = masks[0]
    if not all(np.array_equal(mask, candidate) for candidate in masks[1:]):
        raise ValueError("intact blocks do not encode one common mask")
    if not np.any(mask) or np.all(mask):
        raise ValueError("mask must contain foreground and background cells")

    mask_side_colors = {int(layout[r, c]) for r, c in zip(*np.nonzero(mask)) if (r, c) != damaged}
    other_side_colors = {
        int(layout[r, c])
        for r, c in zip(*np.nonzero(~mask))
        if (r, c) != damaged
    }
    if len(mask_side_colors) != 1 or len(other_side_colors) != 1:
        raise ValueError("block colors must reproduce the decoded mask at block scale")
    mask_color = int(next(iter(mask_side_colors)))
    other_color = int(next(iter(other_side_colors)))
    if mask_color == other_color or corrupt_color in (mask_color, other_color):
        raise ValueError("background, two main colors, and corruption color must be distinguishable")

    for r in range(n):
        for c in range(n):
            if (r, c) != damaged:
                expected_color = mask_color if bool(mask[r, c]) else other_color
                if int(layout[r, c]) != expected_color:
                    raise ValueError("block-scale layout does not match the inner mask")
    return n, bg, damaged, mask.copy(), mask_color, other_color


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the uniformly corrupted block in a two-scale self-similar expansion.

    Input structure:
    - A square grid of side n squared is partitioned into an n by n array of
      n by n blocks. Every intact block contains one shared binary mask, drawn
      with that block's main color over a shared background. Exactly one block
      is uniformly covered by a distinct corruption color.

    Rule:
    - Decode the mask from all intact blocks. At block scale, mask positions
      use one main color and non-mask positions use the other. Use the damaged
      block's block-coordinate position to choose its main color, then redraw
      the same mask inside it; preserve every other cell.

    Key insight:
    - The identical pattern is encoded at both pixel and block scales, so the
      intact small copies recover the erased shape while the large layout
      independently recovers its missing main color.
    """
    g = np.asarray(grid)
    n, bg, (br, bc), mask, mask_color, other_color = _decode(g)
    main_color = mask_color if bool(mask[br, bc]) else other_color
    out = g.copy()
    repaired = _block_view(out, n, br, bc)
    repaired[:, :] = bg
    repaired[mask] = main_color
    return out


def _is_asymmetric(mask: np.ndarray) -> bool:
    """Reject masks whose orientation can be mistaken after a dihedral transform."""
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.extend((rot, np.fliplr(rot)))
    return all(not np.array_equal(mask, v) for v in variants[1:])


def _sample_mask(n: int, rng: random.Random) -> np.ndarray:
    """Make a mixed, oriented mask with both block classes redundantly visible."""
    for _ in range(500):
        count = rng.randint(3, n * n - 3)
        chosen = rng.sample(range(n * n), count)
        mask = np.zeros((n, n), dtype=bool)
        mask.flat[chosen] = True
        # Avoid row/column-only rules and retain visible evidence in both axes.
        if min(mask.sum(axis=0)) == 0 or min(mask.sum(axis=1)) == 0:
            continue
        if max(mask.sum(axis=0)) == n or max(mask.sum(axis=1)) == n:
            continue
        if not _is_asymmetric(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a non-degenerate seed mask")


def _render_complete(n: int, mask: np.ndarray, bg: int, a: int, b: int) -> np.ndarray:
    complete = np.full((n * n, n * n), bg, dtype=int)
    for br in range(n):
        for bc in range(n):
            color = a if bool(mask[br, bc]) else b
            block = _block_view(complete, n, br, bc)
            block[mask] = color
    return complete


def generate_input() -> np.ndarray:
    """Generate a varied, semantically validated two-scale repair instance."""
    rng = random.Random()
    for _ in range(300):
        n = rng.choice((3, 4, 5))
        mask = _sample_mask(n, rng)
        bg, mask_color, other_color, corrupt_color = rng.sample(range(10), 4)

        # Equal-probability semantic branches: repair an A block or a B block.
        target_value = bool(rng.getrandbits(1))
        candidates = list(zip(*np.nonzero(mask if target_value else ~mask)))
        # Deliberately cover border and interior positions when either is available.
        want_border = bool(rng.getrandbits(1))
        preferred = [
            (int(r), int(c)) for r, c in candidates
            if (r in (0, n - 1) or c in (0, n - 1)) == want_border
        ]
        pool = preferred or [(int(r), int(c)) for r, c in candidates]
        br, bc = rng.choice(pool)

        # Both layout colors remain multiply witnessed after the erasure.
        remaining_true = int(mask.sum()) - int(target_value)
        remaining_false = int((~mask).sum()) - int(not target_value)
        if remaining_true < 2 or remaining_false < 2:
            continue

        expected = _render_complete(n, mask, bg, mask_color, other_color)
        inp = expected.copy()
        _block_view(inp, n, br, bc)[:, :] = corrupt_color

        # Semantic checks: unique damage, two genuinely necessary scales, exact
        # preservation outside the target, and correct recovery of both shape/color.
        decoded = _decode(inp)
        if decoded[2] != (br, bc) or not np.array_equal(decoded[3], mask):
            continue
        out = main(inp)
        target_cells = np.zeros_like(inp, dtype=bool)
        target_cells[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = True
        if not np.array_equal(out, expected):
            continue
        if not np.array_equal(out[~target_cells], inp[~target_cells]):
            continue
        if np.unique(out[target_cells]).size != 2:
            continue
        return inp
    raise RuntimeError("failed to generate a valid non-degenerate input")
