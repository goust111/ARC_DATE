from __future__ import annotations

import math
import random

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 500


def _hidden_dims(shape: tuple[int, int]) -> tuple[int, int]:
    rows, cols = map(int, shape)
    h = math.isqrt(rows)
    w = math.isqrt(cols)
    assert h * h == rows and w * w == cols, "grid dimensions must be h*h by w*w"
    assert h > 1 and w > 1, "mother pattern must have block dimensions"
    return h, w


def _block_view(grid: np.ndarray, br: int, bc: int, h: int, w: int) -> np.ndarray:
    return grid[br * h : (br + 1) * h, bc * w : (bc + 1) * w]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the one wrong self-similar stamp block by matching block occupancy to the stamp content.

    Input structure:
    - The background is black and all foreground pixels have one shared color.
    - The grid shape is (h*h) by (w*w), partitioned into an h by w array of h by w blocks.
    - Every nonempty block is an identical copy of the hidden h by w mother pattern.
    - Exactly one block-level error exists: one true stamp block was cleared, or one empty
      placeholder block was filled with the mother pattern.

    Rule:
    - Read the mother pattern from the identical content of the nonempty blocks.
    - Compare that h by w pattern with the h by w distribution of nonempty blocks.
    - If the occupancy is missing the one foreground cell of the pattern, repaint that block
      with the mother pattern; if it has one extra occupied cell, clear that block to black.

    Key insight:
    - The same pattern appears as both block content and block occupancy; the single corrupt
      placeholder is the unique disagreement in this self-similar equality.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(g.min()) >= 0 and int(g.max()) <= 9, "ARC colors must be 0..9"

    colors = set(map(int, g.ravel()))
    fg_colors = colors - {BG}
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = next(iter(fg_colors))

    h, w = _hidden_dims(g.shape)
    occupancy = np.zeros((h, w), dtype=bool)
    nonempty_blocks: list[np.ndarray] = []

    for br in range(h):
        for bc in range(w):
            block = _block_view(g, br, bc, h, w)
            nonempty = bool(np.any(block != BG))
            occupancy[br, bc] = nonempty
            if nonempty:
                assert np.all((block == BG) | (block == fg)), "blocks may only use bg and fg"
                nonempty_blocks.append(block.copy())

    assert nonempty_blocks, "at least one intact copy is needed to read the mother pattern"
    prototype = nonempty_blocks[0]
    assert np.any(prototype == fg), "mother pattern must contain foreground"

    for block in nonempty_blocks[1:]:
        assert np.array_equal(block, prototype), "all nonempty blocks must contain the same pattern"

    mother_mask = prototype != BG
    disagreement = occupancy ^ mother_mask
    assert int(disagreement.sum()) == 1, "occupancy and content must differ in exactly one block"
    br, bc = map(int, np.argwhere(disagreement)[0])

    out = g.copy()
    target = _block_view(out, br, bc, h, w)
    if mother_mask[br, bc] and not occupancy[br, bc]:
        target[:, :] = prototype
    elif occupancy[br, bc] and not mother_mask[br, bc]:
        target[:, :] = BG
    else:
        raise AssertionError("unreachable disagreement type")
    return out


def _is_connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
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


def _random_mother_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    min_fill = max(2, math.ceil(area * 0.28))
    max_fill = min(area - 1, math.floor(area * 0.72))
    if min_fill > max_fill:
        return None

    for _ in range(200):
        fill = random.randint(min_fill, max_fill)
        flat = random.sample(range(area), fill)
        mask = np.zeros((h, w), dtype=bool)
        mask.flat[flat] = True
        if not _is_connected4(mask):
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if np.all(mask) or not np.any(~mask):
            continue
        return mask
    return None


def _sprite(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG).astype(int)


def _render_complete(mask: np.ndarray, color: int) -> np.ndarray:
    h, w = mask.shape
    stamp = _sprite(mask, color)
    out = np.zeros((h * h, w * w), dtype=int)
    for br, bc in np.argwhere(mask):
        _block_view(out, int(br), int(bc), h, w)[:, :] = stamp
    return out


def _corrupt_one_block(complete: np.ndarray, mask: np.ndarray, branch: int) -> tuple[np.ndarray, tuple[int, int], str]:
    h, w = mask.shape
    stamp = _block_view(complete, int(np.argwhere(mask)[0][0]), int(np.argwhere(mask)[0][1]), h, w).copy()

    if branch == 0:
        choices = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        br, bc = random.choice(choices)
        inp = complete.copy()
        _block_view(inp, br, bc, h, w)[:, :] = BG
        return inp, (br, bc), "missing"

    choices = [tuple(map(int, rc)) for rc in np.argwhere(~mask)]
    br, bc = random.choice(choices)
    inp = complete.copy()
    _block_view(inp, br, bc, h, w)[:, :] = stamp
    return inp, (br, bc), "extra"


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
    color: int,
    corrupt_at: tuple[int, int],
    kind: str,
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False

    h, w = mask.shape
    stamp = _sprite(mask, color)
    occupancy = np.zeros((h, w), dtype=bool)
    for br in range(h):
        for bc in range(w):
            block = _block_view(inp, br, bc, h, w)
            nonempty = bool(np.any(block != BG))
            occupancy[br, bc] = nonempty
            if nonempty and not np.array_equal(block, stamp):
                return False
            if not nonempty and np.any(block != BG):
                return False

    disagreement = occupancy ^ mask
    if int(disagreement.sum()) != 1:
        return False
    if tuple(map(int, np.argwhere(disagreement)[0])) != corrupt_at:
        return False
    if kind == "missing" and not (mask[corrupt_at] and not occupancy[corrupt_at]):
        return False
    if kind == "extra" and not (occupancy[corrupt_at] and not mask[corrupt_at]):
        return False
    if not (2 <= int(mask.sum()) <= mask.size - 1):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


_GEN_BRANCH_OFFSET = random.randrange(2)
_GEN_CALL_COUNT = 0


def _gen_attempt(branch: int) -> np.ndarray | None:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    mask = _random_mother_mask(h, w)
    if mask is None:
        return None

    color = random.randint(1, 9)
    expected = _render_complete(mask, color)
    inp, corrupt_at, kind = _corrupt_one_block(expected, mask, branch)
    if not _semantic_check(inp, expected, mask, color, corrupt_at, kind):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    preferred = (_GEN_CALL_COUNT + _GEN_BRANCH_OFFSET) % 2
    _GEN_CALL_COUNT += 1

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        branch = preferred if attempt < MAX_GENERATION_ATTEMPTS // 2 else 1 - preferred
        grid = _gen_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
