from __future__ import annotations

import random

import numpy as np


BLACK = 0
AVAILABLE_COLORS = tuple(range(1, 10))

# Generator limits chosen so the full grid always stays within ARC's 30x30 cap.
RING_COUNT_CHOICES = (1, 2, 2, 3, 3, 4, 4)
MAX_AXIS_TOTAL = 12
GEN_RETRY_LIMIT = 300


def _parse_instruction_bar(values: np.ndarray) -> list[tuple[int, int]]:
    """Read consecutive non-black runs from the start of an edge bar."""
    runs: list[tuple[int, int]] = []
    idx = 0
    n = int(values.size)
    while idx < n and int(values[idx]) != BLACK:
        color = int(values[idx])
        start = idx
        idx += 1
        while idx < n and int(values[idx]) == color:
            idx += 1
        runs.append((color, idx - start))

    assert runs, "expected at least one non-black instruction segment"
    assert np.all(values[idx:] == BLACK), "instruction bars must be contiguous prefixes"
    colors = [color for color, _length in runs]
    assert len(colors) == len(set(colors)), "instruction colors must be distinct"
    return runs


def _paint_ring(
    out: np.ndarray,
    top: int,
    left: int,
    bottom: int,
    right: int,
    color: int,
    vertical: int,
    horizontal: int,
) -> None:
    """Paint one rectangular ring inside the inclusive bbox."""
    hh = bottom - top + 1
    ww = right - left + 1
    rr = np.arange(hh, dtype=int)[:, None]
    cc = np.arange(ww, dtype=int)[None, :]
    mask = (rr < vertical) | (rr >= hh - vertical) | (cc < horizontal) | (cc >= ww - horizontal)
    window = out[top : bottom + 1, left : right + 1]
    window[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read top and left instruction bars, then draw nested colored rectangular rings on the black canvas.

    Input structure:
    - Black is the background color.
    - The used non-black prefix of the leftmost column is a top-anchored color sequence bar.
      Each contiguous run has one color and encodes a ring's vertical thickness by its run length.
    - The used non-black prefix of the top row, excluding the top-left overlap cell, is a
      left-anchored color sequence bar whose run lengths encode the matching rings' horizontal
      thicknesses.
    - The remaining cells of the top row and left column are black, and the interior canvas
      (everything except the top row and left column) starts fully black.

    Rule:
    - Parse the contiguous color runs from the start of the left edge and from the start of the
      top edge.
    - The two bars must contain the same number of runs, and corresponding runs must have the same
      color; that shared color is the ring color.
    - Starting from the full interior canvas, draw rectangular rings from outermost to innermost.
      For each ring, the top and bottom thickness equal the left-bar run length, and the left and
      right thickness equal the top-bar run length.
    - After drawing one ring, continue from the remaining inner rectangle. Keep the instruction
      bars unchanged in the output.

    Key insight:
    - The edge bars are not decorative legends: each run length is a numeric drawing command, and
      the two bars cooperate to define anisotropic nested rectangle rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 2, "grid must contain both bars and a canvas"
    assert np.all(g[1:, 1:] == BLACK), "input canvas must start black"

    left_runs = _parse_instruction_bar(g[:, 0])
    top_runs = _parse_instruction_bar(g[0, 1:])
    assert len(left_runs) == len(top_runs), "left and top bars must describe the same number of rings"

    out = g.copy()
    out[1:, 1:] = BLACK

    top = 1
    left = 1
    bottom = height - 1
    right = width - 1

    for idx, ((left_color, vertical), (top_color, horizontal)) in enumerate(zip(left_runs, top_runs, strict=True)):
        assert left_color == top_color, "corresponding edge runs must share the same color"
        assert top <= bottom and left <= right, "no canvas remains for the next ring"
        _paint_ring(out, top, left, bottom, right, left_color, vertical, horizontal)
        top += vertical
        bottom -= vertical
        left += horizontal
        right -= horizontal
        if idx + 1 < len(left_runs):
            assert top <= bottom and left <= right, "instructions require another ring but no inner canvas remains"

    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    """Sample a random ordered composition of `total` into `parts` positive integers."""
    assert parts >= 1
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _promote_first_segment(values: list[int]) -> None:
    """Bias the first segment to be more visually obvious when possible."""
    if values[0] > 1:
        return
    donors = [idx for idx in range(1, len(values)) if values[idx] > 1]
    if not donors:
        return
    donor = random.choice(donors)
    values[0] += 1
    values[donor] -= 1


def _sample_axis_thicknesses(ring_count: int) -> list[int]:
    total = random.randint(ring_count, MAX_AXIS_TOTAL)
    values = _positive_composition(total, ring_count)
    _promote_first_segment(values)
    return values


def _sample_center(max_h: int, max_w: int) -> tuple[int, int]:
    """Return either a positive black center or a fully covered final canvas."""
    assert max_h >= 1 and max_w >= 1
    if random.random() < 0.5:
        return random.randint(1, max_h), random.randint(1, max_w)

    mode = random.choice(("fill", "flat_h", "flat_w"))
    if mode == "flat_h":
        return 0, random.randint(1, max_w)
    if mode == "flat_w":
        return random.randint(1, max_h), 0
    return 0, 0


def _build_input(colors: list[int], verticals: list[int], horizontals: list[int], center_h: int, center_w: int) -> np.ndarray:
    canvas_h = 2 * sum(verticals) + center_h
    canvas_w = 2 * sum(horizontals) + center_w
    grid = np.zeros((canvas_h + 1, canvas_w + 1), dtype=int)

    row = 0
    for color, length in zip(colors, verticals, strict=True):
        grid[row : row + length, 0] = int(color)
        row += length

    col = 1
    for color, length in zip(colors, horizontals, strict=True):
        grid[0, col : col + length] = int(color)
        col += length

    return grid


def _is_valid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape:
        return False
    if not np.array_equal(out[0, :], grid[0, :]) or not np.array_equal(out[:, 0], grid[:, 0]):
        return False
    if not np.any(out[1:, 1:] != BLACK):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    ring_count = random.choice(RING_COUNT_CHOICES)
    colors = random.sample(AVAILABLE_COLORS, ring_count)
    verticals = _sample_axis_thicknesses(ring_count)
    horizontals = _sample_axis_thicknesses(ring_count)

    max_center_h = 29 - 2 * sum(verticals)
    max_center_w = 29 - 2 * sum(horizontals)
    if max_center_h < 1 or max_center_w < 1:
        return None

    center_h, center_w = _sample_center(max_center_h, max_center_w)
    grid = _build_input(colors, verticals, horizontals, center_h, center_w)
    if not _is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRY_LIMIT} attempts")
