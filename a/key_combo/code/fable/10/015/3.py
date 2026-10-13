from __future__ import annotations

import numpy as np


HOLE = 0
CYCLE = (1, 2, 3)  # blue -> red -> green -> blue
_CYCLE_INDEX = {color: idx for idx, color in enumerate(CYCLE)}
BG_CHOICES = (4, 5, 6, 7, 8, 9)

# Generator ranges.
HEIGHT_RANGE = (6, 12)
SEED_WIDTH_RANGE = (4, 8)
BLOCK_COUNT_RANGE = (4, 7)
MAX_ATTEMPTS = 600


def _shift_cycle(color: int, steps: int) -> int:
    color = int(color)
    if color not in _CYCLE_INDEX:
        return color
    return CYCLE[(_CYCLE_INDEX[color] + int(steps)) % 3]


def _seed_col(local_col: int, period: int, block_idx: int) -> int:
    local_col = int(local_col)
    period = int(period)
    return local_col if int(block_idx) % 2 == 0 else period - 1 - local_col


def _inverse_observation(color: int, block_idx: int) -> int:
    return _shift_cycle(int(color), -int(block_idx))


def _render_from_seed(seed: np.ndarray, block_count: int) -> np.ndarray:
    seed = np.asarray(seed, dtype=int)
    height, period = seed.shape
    out = np.empty((height, period * int(block_count)), dtype=int)
    for block_idx in range(int(block_count)):
        tile = np.fliplr(seed) if block_idx % 2 else seed
        out[:, block_idx * period : (block_idx + 1) * period] = np.vectorize(
            lambda color: _shift_cycle(int(color), block_idx), otypes=[int]
        )(tile)
    return out


def _reconstruct_period(grid: np.ndarray, period: int) -> np.ndarray | None:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    if period <= 0 or width % int(period) != 0:
        return None

    seed = np.full((height, int(period)), -1, dtype=int)
    for row in range(height):
        for col in range(width):
            value = int(g[row, col])
            if value == HOLE:
                continue
            block_idx = col // int(period)
            local_col = col % int(period)
            seed_col = _seed_col(local_col, period, block_idx)
            base_value = _inverse_observation(value, block_idx)
            if seed[row, seed_col] == -1:
                seed[row, seed_col] = base_value
            elif seed[row, seed_col] != base_value:
                return None

    if np.any(seed == -1):
        return None

    out = _render_from_seed(seed, width // int(period))
    visible = g != HOLE
    if not np.array_equal(out[visible], g[visible]):
        return None
    return out


def _valid_period_outputs(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    _, width = grid.shape
    valid: list[tuple[int, np.ndarray]] = []
    for period in range(1, width + 1):
        if width % period != 0:
            continue
        out = _reconstruct_period(grid, period)
        if out is not None:
            valid.append((period, out))
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the black rectangle in a horizontal flip-and-recolor tile strip.

    Input structure:
    - The grid contains one non-black background color plus the three cycle colors
      blue, red, and green.
    - The full picture is made by repeating one seed block horizontally; odd-indexed
      copies are left-right mirrored, and copy k recolors every foreground pixel by
      k steps of the blue->red->green color wheel.
    - One axis-aligned rectangular region is erased to black, and it may cross tile
      boundaries.

    Rule:
    - Try each width that divides the canvas and interpret the strip as blocks of
      that width.
    - For every visible pixel, map its column back into the seed using the block
      parity, then undo the block's color-wheel shift to recover the seed color.
    - The smallest width whose recovered seed explains every visible pixel and every
      hidden seed cell is the true seed width; rerender all blocks from that seed to
      fill the black rectangle.

    Key insight:
    - A valid source pixel for a missing cell must match in two ways at once: odd
      blocks reverse the column inside the seed, and different blocks advance the
      three-color wheel by different phase offsets.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == HOLE), "expected a black missing rectangle"

    for period in range(1, g.shape[1] + 1):
        if g.shape[1] % period != 0:
            continue
        out = _reconstruct_period(g, period)
        if out is not None:
            return out

    raise AssertionError("no valid seed width found")


def _make_seed(rng: np.random.Generator, height: int, period: int, bg: int) -> np.ndarray:
    probs = np.array([0.34, 0.22, 0.22, 0.22], dtype=float)
    palette = np.array((bg, *CYCLE), dtype=int)
    for _ in range(200):
        seed = rng.choice(palette, size=(height, period), p=probs).astype(int)
        if not all(color in seed for color in CYCLE):
            continue
        if not np.any(seed == bg):
            continue
        if np.array_equal(seed, np.fliplr(seed)):
            continue
        if any(len(set(map(int, seed[row]))) == 1 for row in range(height)):
            continue
        return seed
    raise RuntimeError("failed to build a non-degenerate seed")


def _hole_mask(shape: tuple[int, int], top: int, left: int, hole_h: int, hole_w: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[int(top) : int(top) + int(hole_h), int(left) : int(left) + int(hole_w)] = True
    return mask


def _pick_hole(
    rng: np.random.Generator,
    output: np.ndarray,
    period: int,
    cross_boundary: bool,
) -> tuple[int, int, int, int] | None:
    height, width = output.shape
    max_h = min(height - 1, max(2, min(5, height)))
    min_w = 2
    max_w = min(width - 1, max(3, min(period + 3, width - 1)))
    if max_h < 2 or max_w < min_w:
        return None

    for _ in range(120):
        hole_h = int(rng.integers(2, max_h + 1))
        if cross_boundary:
            hole_w = int(rng.integers(max(2, min(period, max_w // 2 + 1)), max_w + 1))
            if width // period < 2 or hole_w < 2:
                continue
            boundary = int(rng.integers(1, width // period)) * period
            left_lo = max(0, boundary - hole_w + 1)
            left_hi = min(boundary - 1, width - hole_w)
            if left_lo > left_hi:
                continue
            left = int(rng.integers(left_lo, left_hi + 1))
            if not (left < boundary < left + hole_w):
                continue
        else:
            hole_w = int(rng.integers(min_w, max_w + 1))
            block_idx = int(rng.integers(0, width // period))
            block_left = block_idx * period
            if hole_w > period:
                continue
            left = int(rng.integers(block_left, block_left + period - hole_w + 1))
        top = int(rng.integers(0, height - hole_h + 1))

        patch = output[top : top + hole_h, left : left + hole_w]
        if len(set(map(int, patch.flat))) < 2:
            continue
        if not np.any(np.isin(patch, CYCLE)):
            continue
        return top, left, hole_h, hole_w
    return None


def _transfer_same_local(source_color: int, from_block: int, to_block: int) -> int:
    return _shift_cycle(int(source_color), int(to_block) - int(from_block))


def _has_flip_critical_hole(inp: np.ndarray, out: np.ndarray, period: int) -> bool:
    height, width = inp.shape
    block_count = width // int(period)
    for row in range(height):
        for col in range(width):
            if inp[row, col] != HOLE:
                continue
            block_idx = col // int(period)
            local_col = col % int(period)
            correct = int(out[row, col])
            for other in range(block_count):
                if other == block_idx:
                    continue
                other_col = other * int(period) + local_col
                if inp[row, other_col] == HOLE:
                    continue
                predicted = _transfer_same_local(int(inp[row, other_col]), other, block_idx)
                if predicted != correct:
                    return True
    return False


def _has_recolor_critical_hole(inp: np.ndarray, out: np.ndarray, period: int) -> bool:
    height, width = inp.shape
    block_count = width // int(period)
    for row in range(height):
        for col in range(width):
            if inp[row, col] != HOLE:
                continue
            block_idx = col // int(period)
            local_col = col % int(period)
            correct = int(out[row, col])
            seed_col = _seed_col(local_col, period, block_idx)
            for other in range(block_count):
                if other == block_idx:
                    continue
                other_local = seed_col if other % 2 == 0 else int(period) - 1 - seed_col
                other_col = other * int(period) + other_local
                visible = int(inp[row, other_col])
                if visible == HOLE:
                    continue
                if visible != correct and visible in _CYCLE_INDEX:
                    return True
    return False


def _validate_instance(inp: np.ndarray, out: np.ndarray, period: int, cross_boundary: bool) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    valid = _valid_period_outputs(inp)
    if len(valid) != 1 or valid[0][0] != int(period):
        return False

    visible_colors = set(map(int, inp[inp != HOLE].flat))
    if not set(CYCLE) <= visible_colors:
        return False

    if not _has_flip_critical_hole(inp, out, period):
        return False
    if not _has_recolor_critical_hole(inp, out, period):
        return False

    hole_cols = np.where(inp == HOLE)[1]
    if hole_cols.size == 0:
        return False
    actually_crosses = np.any((hole_cols // int(period)) != (hole_cols[0] // int(period)))
    if bool(actually_crosses) != bool(cross_boundary):
        return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*HEIGHT_RANGE))
    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    valid_periods = [period for period in range(*SEED_WIDTH_RANGE) if period * block_count <= 30]
    if not valid_periods:
        return None
    period = int(rng.choice(valid_periods))
    bg = int(rng.choice(BG_CHOICES))

    seed = _make_seed(rng, height, period, bg)
    output = _render_from_seed(seed, block_count)

    full_valid = _valid_period_outputs(output)
    if not full_valid or full_valid[0][0] != period:
        return None

    cross_boundary = bool(rng.integers(0, 2))
    hole = _pick_hole(rng, output, period, cross_boundary)
    if hole is None:
        return None
    top, left, hole_h, hole_w = hole

    inp = output.copy()
    inp[_hole_mask(inp.shape, top, left, hole_h, hole_w)] = HOLE

    if not _validate_instance(inp, output, period, cross_boundary):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
