from __future__ import annotations

from collections import Counter
import math

import numpy as np


def _block_side(grid: np.ndarray) -> int:
    """Return n when a square grid has side n**2."""
    assert grid.ndim == 2, "expected a 2D grid"
    height, width = map(int, grid.shape)
    assert height == width, "expected a square grid"
    n = math.isqrt(height)
    assert n >= 2 and n * n == height, "grid side must be a square number"
    return n


def _background(grid: np.ndarray) -> int:
    """Infer the uniquely most frequent visible color."""
    colors, counts = np.unique(grid, return_counts=True)
    winners = colors[counts == counts.max()]
    assert len(winners) == 1, "background color must be uniquely most frequent"
    return int(winners[0])


def _blocks(grid: np.ndarray, n: int) -> np.ndarray:
    """View an n**2 grid as blocks[block_r, block_c, inner_r, inner_c]."""
    return grid.reshape(n, n, n, n).transpose(0, 2, 1, 3).copy()


def _candidate_patterns(grid: np.ndarray) -> list[np.ndarray]:
    """Find block patterns whose self-Kronecker image has one bad block."""
    g = np.asarray(grid, dtype=int)
    n = _block_side(g)
    colors = list(map(int, np.unique(g)))
    assert len(colors) == 2, "expected exactly two visible colors"

    blocks = _blocks(g, n)
    seen_outputs: set[tuple[int, ...]] = set()
    candidates: list[np.ndarray] = []

    # Infer which visible color is background from the self-similar structure,
    # rather than from its numeric id or even from a frequency assumption.
    for bg in colors:
        fg = next(color for color in colors if color != bg)
        bg_block = np.full((n, n), bg, dtype=int)
        seen_for_bg: set[tuple[int, ...]] = set()

        # At least one genuine foreground block survives because valid patterns
        # have at least two foreground cells and only one block is corrupted.
        for br in range(n):
            for bc in range(n):
                pattern = blocks[br, bc]
                key = tuple(map(int, pattern.ravel()))
                if key in seen_for_bg:
                    continue
                seen_for_bg.add(key)
                if set(key) != {bg, fg}:
                    continue

                mismatches = 0
                for rr in range(n):
                    for cc in range(n):
                        expected = pattern if int(pattern[rr, cc]) == fg else bg_block
                        if not np.array_equal(blocks[rr, cc], expected):
                            mismatches += 1
                if mismatches == 1 and key not in seen_outputs:
                    seen_outputs.add(key)
                    candidates.append(pattern.copy())
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the small two-color pattern from a once-corrupted self-Kronecker product.

    Input structure:
    - The square grid has side n squared and is partitioned, in the same grid
      coordinates, into n by n blocks of size n by n.
    - For an unknown two-color pattern M, each foreground layout cell normally
      contains M and each background layout cell normally contains a solid
      background block; exactly one whole block differs from that construction.

    Rule:
    - Treat each observed two-color non-solid block as a possible content copy.
    - For each possibility, use its own foreground mask as the block-level layout
      and reconstruct the corresponding self-Kronecker grid.
    - Return the unique candidate whose reconstruction disagrees with the input
      in exactly one n by n block.

    Key insight:
    - M appears both inside foreground blocks and as the layout selecting those
      blocks, so the two self-similar readings cross-check and correct one bad block.
    """
    g = np.asarray(grid, dtype=int)
    candidates = _candidate_patterns(g)
    assert len(candidates) == 1, "expected one pattern consistent with one corrupted block"
    return candidates[0]


_generation_counter = 0
_seen_outputs: set[tuple[int, tuple[int, ...]]] = set()
_BRANCHES = ("spurious_background", "wrong_foreground", "empty_foreground")
_MAX_GENERATION_ATTEMPTS = 800


def _sample_mask(rng: np.random.Generator, n: int, foreground_count: int) -> np.ndarray:
    """Sample a genuinely two-dimensional, non-uniform binary pattern."""
    for _ in range(100):
        indices = rng.choice(n * n, size=foreground_count, replace=False)
        mask = np.zeros((n, n), dtype=bool)
        mask.flat[indices] = True
        rows, cols = np.where(mask)
        if len(set(map(int, rows))) >= 2 and len(set(map(int, cols))) >= 2:
            return mask
    raise RuntimeError("failed to sample a non-degenerate mask")


def _render_ideal(pattern: np.ndarray, bg: int) -> np.ndarray:
    """Render M x M using the visible foreground mask (a Kronecker product)."""
    mask = pattern != bg
    return np.where(np.kron(mask, mask), pattern.flat[np.flatnonzero(mask)[0]], bg).astype(int)


def _replace_block(grid: np.ndarray, n: int, pos: tuple[int, int], patch: np.ndarray) -> None:
    br, bc = pos
    grid[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = patch


def _block_mismatch_count(left: np.ndarray, right: np.ndarray, n: int) -> int:
    left_blocks = _blocks(left, n)
    right_blocks = _blocks(right, n)
    return sum(
        not np.array_equal(left_blocks[r, c], right_blocks[r, c])
        for r in range(n)
        for c in range(n)
    )


def _semantic_check(
    inp: np.ndarray,
    pattern: np.ndarray,
    branch: str,
    bg: int,
) -> bool:
    """Validate the stated rule and explicit counterexamples to one-layer shortcuts."""
    n = int(pattern.shape[0])
    ideal = _render_ideal(pattern, bg)
    if _block_mismatch_count(inp, ideal, n) != 1:
        return False
    try:
        candidates = _candidate_patterns(inp)
    except (AssertionError, ValueError):
        return False
    if len(candidates) != 1 or not np.array_equal(candidates[0], pattern):
        return False

    blocks = _blocks(inp, n)
    occupied = np.any(blocks != bg, axis=(2, 3))
    true_layout = pattern != bg
    layout_errors = int(np.count_nonzero(occupied != true_layout))

    nonempty_keys = [
        tuple(map(int, blocks[r, c].ravel()))
        for r in range(n)
        for c in range(n)
        if occupied[r, c]
    ]
    counts = Counter(nonempty_keys)
    if branch == "wrong_foreground":
        # With two foreground cells, the genuine and corrupted contents occur
        # once each: content alone is ambiguous, while layout disambiguates them.
        if layout_errors != 0 or len(nonempty_keys) != 2:
            return False
        if sorted(counts.values()) != [1, 1]:
            return False
    else:
        # Adding or deleting a block makes raw occupancy differ from M, so a
        # layout-only reading is wrong and the repeated content copies are needed.
        if layout_errors != 1:
            return False

    return np.array_equal(main(inp), pattern)


def _gen_attempt(rng: np.random.Generator, branch: str) -> tuple[np.ndarray, np.ndarray] | None:
    n = int(rng.choice((3, 4, 5)))
    if branch == "wrong_foreground":
        foreground_count = 2
    else:
        foreground_count = int(rng.integers(3, max(4, (n * n) // 2 + 1)))

    mask = _sample_mask(rng, n, foreground_count)
    bg, fg = map(int, rng.choice(10, size=2, replace=False))
    pattern = np.where(mask, fg, bg).astype(int)
    output_key = (n, tuple(map(int, pattern.ravel())))
    if output_key in _seen_outputs:
        return None

    ideal = _render_ideal(pattern, bg)
    inp = ideal.copy()
    fg_positions = [tuple(map(int, p)) for p in np.argwhere(mask)]
    bg_positions = [tuple(map(int, p)) for p in np.argwhere(~mask)]

    if branch == "spurious_background":
        # An extra exact-looking copy defeats the raw nonempty-block layout.
        pos = bg_positions[int(rng.integers(len(bg_positions)))]
        _replace_block(inp, n, pos, pattern)
    elif branch == "empty_foreground":
        pos = fg_positions[int(rng.integers(len(fg_positions)))]
        _replace_block(inp, n, pos, np.full((n, n), bg, dtype=int))
    else:
        pos = fg_positions[int(rng.integers(len(fg_positions)))]
        for _ in range(60):
            decoy_count = int(rng.integers(1, n * n))
            decoy_mask = _sample_mask(rng, n, decoy_count) if decoy_count >= 2 else None
            if decoy_mask is None:
                decoy_mask = np.zeros((n, n), dtype=bool)
                decoy_mask.flat[int(rng.integers(n * n))] = True
            if not np.array_equal(decoy_mask, mask):
                decoy = np.where(decoy_mask, fg, bg).astype(int)
                _replace_block(inp, n, pos, decoy)
                break
        else:
            return None

    # Dominant background is a visible structural fact, not a color-id rule.
    if _background(inp) != bg:
        return None
    if not _semantic_check(inp, pattern, branch, bg):
        return None
    return inp, pattern


def generate_input() -> np.ndarray:
    """Generate varied valid inputs while cycling all three corruption branches."""
    global _generation_counter
    branch = _BRANCHES[_generation_counter % len(_BRANCHES)]
    rng = np.random.default_rng()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        generated = _gen_attempt(rng, branch)
        if generated is None:
            continue
        inp, pattern = generated
        key = (int(pattern.shape[0]), tuple(map(int, pattern.ravel())))
        _seen_outputs.add(key)
        _generation_counter += 1
        return inp
    raise RuntimeError(f"failed to generate a valid {branch} example")
