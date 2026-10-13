from __future__ import annotations

import random

import numpy as np


def _extract_runs(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[1] != 1 or g.shape[0] == 0:
        raise ValueError("expected a non-empty Nx1 column grid")

    colors: list[int] = []
    lengths: list[int] = []
    for value in map(int, g[:, 0]):
        if not colors or value != colors[-1]:
            colors.append(value)
            lengths.append(1)
        else:
            lengths[-1] += 1
    return colors, lengths


def _build_base_block(colors: list[int]) -> np.ndarray:
    seq = np.asarray(colors, dtype=int)
    return np.vstack([np.roll(seq, -shift) for shift in range(len(seq))])


def _build_output(colors: list[int]) -> np.ndarray:
    base = _build_base_block(colors)
    upper_left = np.rot90(base, 2)
    upper_right = np.rot90(base, -1)
    lower_left = np.rot90(base, 1)
    lower_right = base
    return np.block([[upper_left, upper_right], [lower_left, lower_right]])


def main(grid: np.ndarray) -> np.ndarray:
    """Compress a stripe column, cycle-shift it into a square, then assemble rotated quadrants.

    Input structure:
    - The input is a single-column grid whose cells form contiguous vertical color bands.
    - Every band has one solid non-black color and adjacent bands have different colors.
    - Band thickness may vary, but the ordered band colors are the only information source.

    Rule:
    - Collapse each maximal same-color run into one entry, keeping the top-to-bottom order.
    - If the compressed color sequence has length K, build a KxK base block whose row i is the
      sequence cyclically shifted left by i positions.
    - Form the 2Kx2K output from four KxK quadrants: upper-left is the base rotated 180 degrees,
      upper-right is the base rotated 90 degrees clockwise, lower-left is the base rotated
      90 degrees counterclockwise, and lower-right is the unrotated base.

    Key insight:
    - The compressed stripe order simultaneously determines the output size, the cyclic row pattern,
      and the content of all four rotated quadrants.
    """
    colors, _lengths = _extract_runs(grid)
    return _build_output(colors)


_RNG = np.random.default_rng()
_USED_OUTPUT_SIGNATURES: set[tuple[int, ...]] = set()
_GENERATION_INDEX = 0
_MAX_ATTEMPTS = 400
_K_CYCLE = (3, 4, 5)
_COLOR_POOL = tuple(range(1, 10))


def _composition(total: int, parts: int, min_part: int) -> list[int]:
    remaining = total - parts * min_part
    cuts = sorted(_RNG.integers(0, remaining + 1, size=parts - 1).tolist())
    values = [cuts[0], *[cuts[i] - cuts[i - 1] for i in range(1, parts - 1)], remaining - cuts[-1]]
    return [min_part + int(v) for v in values]


def _sample_lengths(k: int, family: int) -> list[int]:
    for _ in range(300):
        total = int(_RNG.integers(max(9, 2 * k), 16))
        lengths = _composition(total, k, 2)
        if len(set(lengths)) == 1:
            continue

        if family == 0:
            if max(lengths) - min(lengths) < 2:
                continue
        elif family == 1:
            peak = random.choice([0, k - 1])
            max_index = int(np.argmax(lengths))
            lengths[peak], lengths[max_index] = lengths[max_index], lengths[peak]
        elif family == 2:
            if lengths[0] == lengths[-1]:
                continue
        else:
            if len(set(lengths)) < min(3, k):
                continue
            if lengths == sorted(lengths) or lengths == sorted(lengths, reverse=True):
                continue

        return lengths
    raise RuntimeError("failed to sample stripe lengths")


def _render_input(colors: list[int], lengths: list[int]) -> np.ndarray:
    column = np.concatenate(
        [np.full(length, color, dtype=int) for color, length in zip(colors, lengths)]
    )
    return column[:, None]


def _validate_input(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] == 1
    assert 9 <= g.shape[0] <= 15
    assert np.all(g != 0)

    colors, lengths = _extract_runs(g)
    assert 3 <= len(colors) <= 5
    assert len(colors) == len(set(colors))
    assert all(length >= 2 for length in lengths)
    assert len(set(lengths)) >= 2
    return colors, lengths


def _validate_semantics(grid: np.ndarray) -> None:
    colors, _lengths = _validate_input(grid)
    out = main(grid)
    k = len(colors)
    assert out.shape == (2 * k, 2 * k)

    base = _build_base_block(colors)
    upper_left = out[:k, :k]
    upper_right = out[:k, k:]
    lower_left = out[k:, :k]
    lower_right = out[k:, k:]

    assert np.array_equal(lower_right, base)
    assert np.array_equal(upper_left, np.rot90(base, 2))
    assert np.array_equal(upper_right, np.rot90(base, -1))
    assert np.array_equal(lower_left, np.rot90(base, 1))

    for shift in range(k):
        assert np.array_equal(base[shift], np.roll(base[0], -shift))
    assert tuple(map(int, base[0])) == tuple(colors)
    assert len(set(map(int, out.flatten()))) == k


def _output_signature(colors: list[int]) -> tuple[int, ...]:
    out = _build_output(colors)
    return (out.shape[0], out.shape[1], *map(int, out.flatten()))


def _sample_colors(k: int) -> list[int]:
    colors = [int(c) for c in _RNG.choice(_COLOR_POOL, size=k, replace=False)]
    if random.random() < 0.5:
        colors.reverse()
    return colors


def _gen_attempt() -> np.ndarray | None:
    global _GENERATION_INDEX

    k = _K_CYCLE[_GENERATION_INDEX % len(_K_CYCLE)]
    family = (_GENERATION_INDEX // len(_K_CYCLE)) % 4
    _GENERATION_INDEX += 1

    colors = _sample_colors(k)
    signature = _output_signature(colors)
    if signature in _USED_OUTPUT_SIGNATURES:
        return None

    lengths = _sample_lengths(k, family)
    grid = _render_input(colors, lengths)
    _validate_semantics(grid)
    _USED_OUTPUT_SIGNATURES.add(signature)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a fresh valid input")
