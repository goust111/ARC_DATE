from __future__ import annotations

import math
from collections import Counter

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _infer_block_size(grid: np.ndarray) -> int:
    h, w = map(int, grid.shape)
    assert h == w and h > 0, "expected a non-empty square grid"
    n = int(round(math.sqrt(h)))
    assert 3 <= n <= 5 and n * n == h, "expected a grid whose side length is n^2 for n in [3, 5]"
    return n


def _extract_blocks(grid: np.ndarray, n: int) -> np.ndarray:
    blocks = np.empty((n, n, n, n), dtype=int)
    for br in range(n):
        for bc in range(n):
            r0 = br * n
            c0 = bc * n
            blocks[br, bc] = grid[r0 : r0 + n, c0 : c0 + n]
    return blocks


def _is_nonempty(block: np.ndarray, bg: int) -> bool:
    return bool(np.any(block != bg))


def _infer_pattern(blocks: np.ndarray, bg: int) -> np.ndarray:
    counts: Counter[bytes] = Counter()
    samples: dict[bytes, np.ndarray] = {}

    for block in blocks.reshape(-1, blocks.shape[-2], blocks.shape[-1]):
        if not _is_nonempty(block, bg):
            continue
        key = block.tobytes()
        counts[key] += 1
        samples.setdefault(key, block.copy())

    assert counts, "expected at least one non-empty block"
    ranked = counts.most_common(2)
    assert ranked[0][1] >= 2, "expected the motif block to appear at least twice"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "expected a unique majority motif block"

    pattern = samples[ranked[0][0]]
    assert pattern.shape[0] == pattern.shape[1], "expected square motif blocks"
    fg = pattern != bg
    assert int(np.count_nonzero(fg)) >= 2, "expected a non-trivial motif"
    assert int(np.count_nonzero(fg)) < pattern.size, "expected the motif to contain background too"
    return pattern


def _find_bad_blocks(blocks: np.ndarray, pattern: np.ndarray, bg: int) -> list[tuple[int, int]]:
    occupancy = pattern != bg
    bad: list[tuple[int, int]] = []
    for br in range(blocks.shape[0]):
        for bc in range(blocks.shape[1]):
            block = blocks[br, bc]
            expected_nonempty = bool(occupancy[br, bc])
            actual_nonempty = _is_nonempty(block, bg)
            if expected_nonempty:
                if not np.array_equal(block, pattern):
                    bad.append((br, bc))
            elif actual_nonempty:
                bad.append((br, bc))
    return bad


def _render_expansion(pattern: np.ndarray, bg: int) -> np.ndarray:
    n = int(pattern.shape[0])
    out = np.full((n * n, n * n), bg, dtype=int)
    for br, bc in np.argwhere(pattern != bg):
        r0 = int(br) * n
        c0 = int(bc) * n
        out[r0 : r0 + n, c0 : c0 + n] = pattern
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the one block that breaks a hidden self-referential stamp expansion.

    Input structure:
    - The square grid side length is n^2 for some n in {3, 4, 5}, so the grid splits into an n x n array of n x n blocks.
    - Every correct non-empty block is the same motif block P, while correct empty blocks are all background.
    - The occupied block positions are themselves the foreground positions of P, except for one corrupted block.

    Rule:
    - Read motif P from the unique majority among the non-empty blocks.
    - Treat P's foreground mask as the required block-occupancy layout.
    - Locate the unique bad block: either a position that should be empty but is occupied, or a position that should contain P but does not equal P.
    - Output the fully repaired self-consistent expansion by stamping P into every foreground position of P and leaving all other blocks background.

    Key insight:
    - The repeated block content and the coarse occupied-block layout are two copies of the same hidden motif; one copy identifies P, and the other exposes the single inconsistency.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    n = _infer_block_size(g)
    bg = _mode_color(g)
    blocks = _extract_blocks(g, n)
    pattern = _infer_pattern(blocks, bg)
    bad = _find_bad_blocks(blocks, pattern, bg)
    assert len(bad) == 1, "expected exactly one inconsistent block"
    return _render_expansion(pattern, bg)


_BG = 0
_COLORS = np.arange(1, 10, dtype=int)
_BRANCHES = ("fill_empty", "erase_filled", "distort_filled")
_MAX_ATTEMPTS = 500
_RNG = np.random.default_rng()


def _max_fg_count(n: int) -> int:
    return min(n * n - 2, int(math.floor((n * n - 1) / math.sqrt(2))))


def _valid_mask(mask: np.ndarray) -> bool:
    fg = int(np.count_nonzero(mask))
    n = int(mask.shape[0])
    if fg < 3 or fg > _max_fg_count(n):
        return False
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if int(rows.sum()) < 2 or int(cols.sum()) < 2:
        return False
    if int((~mask).sum()) < 2:
        return False
    return True


def _random_mask(n: int) -> np.ndarray:
    for _ in range(200):
        fg = int(_RNG.integers(3, _max_fg_count(n) + 1))
        chosen = _RNG.choice(n * n, size=fg, replace=False)
        mask = np.zeros((n, n), dtype=bool)
        mask.reshape(-1)[chosen] = True
        if _valid_mask(mask):
            return mask
    raise RuntimeError("failed to sample a valid motif mask")


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    n = int(mask.shape[0])
    pattern = np.full((n, n), _BG, dtype=int)
    coords = np.argwhere(mask)
    n_fg = len(coords)

    n_colors = 1 if n_fg < 5 else int(_RNG.integers(1, 3))
    palette = _RNG.choice(_COLORS, size=n_colors, replace=False)
    style = str(_RNG.choice(["uniform", "checker", "row", "col", "random"]))

    if n_colors == 1 or style == "uniform":
        assigned = np.full(n_fg, int(palette[0]), dtype=int)
    elif style == "checker":
        assigned = np.array([int(palette[(r + c) % n_colors]) for r, c in coords], dtype=int)
    elif style == "row":
        assigned = np.array([int(palette[r % n_colors]) for r, _c in coords], dtype=int)
    elif style == "col":
        assigned = np.array([int(palette[c % n_colors]) for _r, c in coords], dtype=int)
    else:
        assigned = np.empty(n_fg, dtype=int)
        assigned[:n_colors] = palette
        if n_fg > n_colors:
            assigned[n_colors:] = _RNG.choice(palette, size=n_fg - n_colors, replace=True)
        _RNG.shuffle(assigned)

    if len(np.unique(assigned)) < n_colors:
        assigned = np.empty(n_fg, dtype=int)
        assigned[:n_colors] = palette
        if n_fg > n_colors:
            assigned[n_colors:] = _RNG.choice(palette, size=n_fg - n_colors, replace=True)
        _RNG.shuffle(assigned)

    pattern[coords[:, 0], coords[:, 1]] = assigned
    return pattern


def _make_noise_block(n: int, pattern: np.ndarray) -> np.ndarray:
    for _ in range(200):
        block = np.full((n, n), _BG, dtype=int)
        cells = int(_RNG.integers(1, n * n))
        idx = _RNG.choice(n * n, size=cells, replace=False)
        colors = _RNG.choice(_COLORS, size=cells, replace=True)
        block.reshape(-1)[idx] = colors
        if not np.array_equal(block, pattern):
            return block
    raise RuntimeError("failed to sample a noisy block")


def _distort_pattern(pattern: np.ndarray) -> np.ndarray:
    block = pattern.copy()
    fg_coords = np.argwhere(pattern != _BG)
    r, c = map(int, fg_coords[int(_RNG.integers(len(fg_coords)))])
    choices = [color for color in _COLORS if int(color) != int(block[r, c])]
    block[r, c] = int(_RNG.choice(np.array(choices, dtype=int)))
    assert np.any(block != _BG)
    assert not np.array_equal(block, pattern)
    return block


def _build_instance(n: int, pattern: np.ndarray) -> np.ndarray:
    return _render_expansion(pattern, _BG)


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, branch: str) -> bool:
    if np.array_equal(inp, expected):
        return False
    if _mode_color(inp) != _BG:
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    n = _infer_block_size(inp)
    blocks = _extract_blocks(inp, n)
    pattern = _infer_pattern(blocks, _BG)
    bad = _find_bad_blocks(blocks, pattern, _BG)
    if len(bad) != 1:
        return False

    br, bc = bad[0]
    occupancy = pattern != _BG
    bad_block = blocks[br, bc]
    if branch == "fill_empty":
        if bool(occupancy[br, bc]) or not _is_nonempty(bad_block, _BG):
            return False
    elif branch == "erase_filled":
        if not bool(occupancy[br, bc]) or _is_nonempty(bad_block, _BG):
            return False
    elif branch == "distort_filled":
        if not bool(occupancy[br, bc]) or not _is_nonempty(bad_block, _BG) or np.array_equal(bad_block, pattern):
            return False
    else:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n = int(_RNG.integers(3, 6))
    mask = _random_mask(n)
    pattern = _colorize_mask(mask)
    expected = _build_instance(n, pattern)
    inp = expected.copy()

    available_branches = []
    if int((~mask).sum()) > 0:
        available_branches.append("fill_empty")
    if int(mask.sum()) > 0:
        available_branches.extend(["erase_filled", "distort_filled"])
    branch = str(_RNG.choice(np.array(available_branches, dtype=object)))

    if branch == "fill_empty":
        empties = np.argwhere(~mask)
        br, bc = map(int, empties[int(_RNG.integers(len(empties)))])
        inp[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = _make_noise_block(n, pattern)
    elif branch == "erase_filled":
        filled = np.argwhere(mask)
        br, bc = map(int, filled[int(_RNG.integers(len(filled)))])
        inp[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = _BG
    else:
        filled = np.argwhere(mask)
        br, bc = map(int, filled[int(_RNG.integers(len(filled)))])
        inp[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = _distort_pattern(pattern)

    if not _semantic_checks(inp, expected, branch):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid self-referential repair input")
