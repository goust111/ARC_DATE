from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color


BG = Color.BLACK
GENERATOR_RETRIES = 600
MASK_RETRIES = 400


def _divisors_with_nontrivial_complement(n: int) -> list[int]:
    return [d for d in range(2, n) if n % d == 0 and n // d >= 2]


def _has_axis_period(mask: np.ndarray, axis: int) -> bool:
    length = int(mask.shape[axis])
    for period in range(1, length):
        if axis == 0:
            if np.array_equal(mask[:-period, :], mask[period:, :]):
                return True
        else:
            if np.array_equal(mask[:, :-period], mask[:, period:]):
                return True
    return False


def _factor_with_outer_shape(mask: np.ndarray, outer_h: int, outer_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = map(int, mask.shape)
    if h % outer_h != 0 or w % outer_w != 0:
        return None
    inner_h = h // outer_h
    inner_w = w // outer_w
    if inner_h < 2 or inner_w < 2:
        return None

    blocks = mask.reshape(outer_h, inner_h, outer_w, inner_w).transpose(0, 2, 1, 3)
    flat = blocks.reshape(outer_h, outer_w, inner_h * inner_w)
    nonzero = np.argwhere(flat.any(axis=2))
    if nonzero.size == 0:
        return None

    first_r, first_c = map(int, nonzero[0])
    inner = blocks[first_r, first_c].copy()
    outer = np.zeros((outer_h, outer_w), dtype=bool)

    zero_block = np.zeros((inner_h, inner_w), dtype=bool)
    for r in range(outer_h):
        for c in range(outer_w):
            block = blocks[r, c]
            if np.array_equal(block, zero_block):
                continue
            if not np.array_equal(block, inner):
                return None
            outer[r, c] = True

    if not np.array_equal(mask, np.kron(outer.astype(int), inner.astype(int)).astype(bool)):
        return None
    return outer, inner


def _has_nontrivial_kron_factorization(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    for outer_h in _divisors_with_nontrivial_complement(h):
        for outer_w in _divisors_with_nontrivial_complement(w):
            factors = _factor_with_outer_shape(mask, outer_h, outer_w)
            if factors is None:
                continue
            outer, inner = factors
            if outer.any() and inner.any():
                return True
    return False


def _is_candidate_factor(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 2:
        return False
    if not mask.any() or mask.all():
        return False
    if _has_axis_period(mask, axis=0) or _has_axis_period(mask, axis=1):
        return False
    if _has_nontrivial_kron_factorization(mask):
        return False
    return True


def _decomposition_candidates(mask: np.ndarray) -> list[dict[str, np.ndarray | tuple[int, int]]]:
    h, w = map(int, mask.shape)
    candidates: list[dict[str, np.ndarray | tuple[int, int]]] = []
    seen: set[tuple[bytes, bytes, tuple[int, int], tuple[int, int]]] = set()

    for outer_h in _divisors_with_nontrivial_complement(h):
        inner_h = h // outer_h
        if math.gcd(outer_h, inner_h) != 1:
            continue
        for outer_w in _divisors_with_nontrivial_complement(w):
            inner_w = w // outer_w
            if math.gcd(outer_w, inner_w) != 1:
                continue
            factors = _factor_with_outer_shape(mask, outer_h, outer_w)
            if factors is None:
                continue
            outer, inner = factors
            if outer.shape == inner.shape:
                continue
            if not (_is_candidate_factor(outer) and _is_candidate_factor(inner)):
                continue
            key = (outer.tobytes(), inner.tobytes(), outer.shape, inner.shape)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "outer": outer,
                    "inner": inner,
                    "outer_shape": outer.shape,
                    "inner_shape": inner.shape,
                }
            )
    return candidates


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Factor a monochrome Kronecker-product grid into outer layout mask A and inner tile mask B, then swap them to output B⊗A.

    Input structure:
    - The grid uses black as background and exactly one foreground color.
    - The foreground image is a two-level nested block array equal to A⊗B: every 1-cell of binary layout A is replaced by the same binary tile B, while every 0-cell becomes an all-black block of B's size.
    - The row counts of A and B are coprime, the column counts of A and B are coprime, the two mask shapes differ, and each mask is aperiodic and not itself a nontrivial Kronecker product.

    Rule:
    - Recover the unique factorization A and B from the repeated block structure of the foreground mask.
    - Build B⊗A by treating B as the outer layout and placing a full copy of A inside each foreground cell of B.
    - Keep the foreground color unchanged and return the swapped-order product at the same overall size.

    Key insight:
    - The pixels are not being freely permuted; the task is to recognize the hidden two-scale Kronecker construction and reverse the order of its factors.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2 and grid.size > 0, "expected a non-empty 2D grid"

    foreground = [int(color) for color in np.unique(grid) if int(color) != BG]
    assert len(foreground) == 1, "expected exactly one foreground color"
    fg = foreground[0]

    mask = grid == fg
    candidates = _decomposition_candidates(mask)
    assert len(candidates) == 1, "expected a unique A⊗B factorization"
    candidate = candidates[0]

    out_mask = np.kron(candidate["inner"].astype(int), candidate["outer"].astype(int)).astype(bool)
    out = np.zeros_like(grid)
    out[out_mask] = fg
    return out


SIZE_OPTIONS = [
    (outer_h, outer_w, inner_h, inner_w)
    for outer_h in range(2, 7)
    for outer_w in range(2, 7)
    for inner_h in range(2, 7)
    for inner_w in range(2, 7)
    if math.gcd(outer_h, inner_h) == 1
    if math.gcd(outer_w, inner_w) == 1
    if (outer_h, outer_w) != (inner_h, inner_w)
    if outer_h * inner_h <= 30
    if outer_w * inner_w <= 30
]


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _generator_friendly_mask(mask: np.ndarray) -> bool:
    if not _is_candidate_factor(mask):
        return False
    if not _touches_all_sides(mask):
        return False
    density = float(mask.mean())
    if not (0.22 <= density <= 0.78):
        return False
    if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
        return False
    if len({tuple(row.tolist()) for row in mask}) < 2:
        return False
    if len({tuple(col.tolist()) for col in mask.T}) < 2:
        return False
    return True


def _random_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    for _ in range(MASK_RETRIES):
        density = random.uniform(0.28, 0.72)
        mask = np.random.random((h, w)) < density
        if _generator_friendly_mask(mask):
            return mask
    raise RuntimeError(f"failed to sample a valid {shape} mask")


def _semantic_checks(grid: np.ndarray) -> bool:
    foreground = [int(color) for color in np.unique(grid) if int(color) != BG]
    if len(foreground) != 1:
        return False
    fg = foreground[0]
    mask = grid == fg
    candidates = _decomposition_candidates(mask)
    if len(candidates) != 1:
        return False

    swapped = main(grid)
    if swapped.shape != grid.shape or np.array_equal(swapped, grid):
        return False
    if not np.array_equal(main(swapped), grid):
        return False

    swapped_mask = swapped == fg
    swapped_candidates = _decomposition_candidates(swapped_mask)
    if len(swapped_candidates) != 1:
        return False

    original = candidates[0]
    recovered = swapped_candidates[0]
    if original["outer_shape"] != recovered["inner_shape"]:
        return False
    if original["inner_shape"] != recovered["outer_shape"]:
        return False
    if not np.array_equal(original["outer"], recovered["inner"]):
        return False
    if not np.array_equal(original["inner"], recovered["outer"]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATOR_RETRIES):
        outer_h, outer_w, inner_h, inner_w = random.choice(SIZE_OPTIONS)
        outer = _random_mask((outer_h, outer_w))
        inner = _random_mask((inner_h, inner_w))
        color = int(random.choice(Color.NOT_BLACK))
        grid = np.kron(outer.astype(int), inner.astype(int)).astype(int) * color
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid Kronecker-swap input")
