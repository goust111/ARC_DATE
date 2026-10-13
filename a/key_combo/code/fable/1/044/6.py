from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, colors_present
except Exception:  # pragma: no cover - fallback only if shared helpers are unavailable
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        blocked = set() if exclude is None else set(map(int, exclude))
        return sorted(int(v) for v in np.unique(grid) if int(v) not in blocked)


_GEN_RETRIES = 500
_WIDTH_CHOICES = [2, 3, 4, 5, 6]
_HEIGHT_RANGE = (5, 12)
_BAND_COUNT_RANGE = (4, 7)
_CYCLE_LENGTHS = (3, 4)
_FIXED_COUNTS = (1, 2)
_MIN_HOLES = 5


def _advance_color(color: int, steps: int, forward: dict[int, int], backward: dict[int, int]) -> int:
    current = int(color)
    if steps >= 0:
        for _ in range(int(steps)):
            current = int(forward[current])
        return current
    for _ in range(int(-steps)):
        current = int(backward[current])
    return current


def _infer_forward_permutation(grid: np.ndarray, width: int) -> dict[int, int] | None:
    height, total_width = grid.shape
    if width <= 0 or width >= total_width:
        return None

    observed_colors = set(colors_present(grid, exclude={Color.BLACK}))
    if not observed_colors:
        return None

    forward: dict[int, int] = {}
    backward: dict[int, int] = {}
    pair_count = 0

    for r in range(height):
        for c in range(total_width - width):
            left = int(grid[r, c])
            right = int(grid[r, c + width])
            if left == Color.BLACK or right == Color.BLACK:
                continue
            pair_count += 1
            if left in forward and forward[left] != right:
                return None
            if right in backward and backward[right] != left:
                return None
            forward[left] = right
            backward[right] = left

    if pair_count == 0:
        return None
    if set(forward) != observed_colors or set(backward) != observed_colors:
        return None
    return forward


def _complete_for_width(grid: np.ndarray, width: int, forward: dict[int, int]) -> np.ndarray | None:
    height, total_width = grid.shape
    backward = {value: key for key, value in forward.items()}
    out = np.asarray(grid, dtype=int).copy()

    for r in range(height):
        for residue in range(width):
            chain: list[tuple[int, int, int]] = []
            band = 0
            c = residue
            while c < total_width:
                color = int(grid[r, c])
                if color != Color.BLACK:
                    chain.append((band, c, color))
                band += 1
                c += width

            if not chain:
                return None

            ref_band, _ref_col, ref_color = chain[0]
            for band, _col, color in chain[1:]:
                expected = _advance_color(ref_color, band - ref_band, forward, backward)
                if expected != color:
                    return None

            band = 0
            c = residue
            while c < total_width:
                predicted = _advance_color(ref_color, band - ref_band, forward, backward)
                if int(out[r, c]) == Color.BLACK:
                    out[r, c] = predicted
                elif int(out[r, c]) != predicted:
                    return None
                band += 1
                c += width

    if np.any(out == Color.BLACK):
        return None
    return out


def _solve(grid: np.ndarray) -> tuple[np.ndarray, int, dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, total_width = g.shape
    assert 0 < height <= 30 and 0 < total_width <= 30, "grid must stay within ARC size limits"
    assert np.any(g != Color.BLACK), "expected at least one non-black pixel"

    for width in range(1, total_width):
        forward = _infer_forward_permutation(g, width)
        if forward is None:
            continue
        completed = _complete_for_width(g, width, forward)
        if completed is not None:
            return completed, width, forward

    raise AssertionError("failed to infer a valid periodic band width and color cycle")


def main(grid: np.ndarray) -> np.ndarray:
    """Repair black holes in a repeated vertical-band pattern whose colors rotate one step per band.

    Input structure:
    - The whole grid is formed by repeating a base vertical band of some unknown width `w`.
    - When the band is copied one step to the right, every non-black color is advanced by the same permutation; some colors may be fixed points.
    - Black cells are missing pixels and do not belong to the color permutation.

    Rule:
    - Infer the smallest band width `w` for which same-row cells `w` columns apart obey one shared color permutation.
    - Recover that permutation from the visible cells, including fixed-point colors.
    - For each black cell, look along its row at cells with the same column residue modulo `w`; advance or reverse the visible color by the number of crossed bands to fill the hole.

    Key insight:
    - The pattern is not plain horizontal periodicity: each repeated strip keeps the same geometry but globally rotates colors, so repairing a hole needs both the strip period and the accumulated permutation step count.
    """
    completed, _width, _forward = _solve(grid)
    return completed


def _build_forward(colors: list[int], cycle_len: int, fixed_count: int) -> tuple[dict[int, int], list[int], list[int]]:
    moving = random.sample(colors, cycle_len)
    fixed_pool = [color for color in colors if color not in moving]
    fixed = random.sample(fixed_pool, fixed_count)
    forward: dict[int, int] = {}
    for i, color in enumerate(moving):
        forward[int(color)] = int(moving[(i + 1) % len(moving)])
    for color in fixed:
        forward[int(color)] = int(color)
    return forward, moving, fixed


def _render_full_grid(base_band: np.ndarray, band_count: int, forward: dict[int, int]) -> np.ndarray:
    height, width = base_band.shape
    out = np.zeros((height, width * band_count), dtype=int)
    backward = {value: key for key, value in forward.items()}
    for band in range(band_count):
        for r in range(height):
            for c in range(width):
                out[r, band * width + c] = _advance_color(int(base_band[r, c]), band, forward, backward)
    return out


def _candidate_band_count(width: int) -> int:
    low, high = _BAND_COUNT_RANGE
    max_bands = min(high, 30 // width)
    return random.randint(low, max_bands)


def _make_base_band(height: int, width: int, palette: list[int], moving: list[int], fixed: list[int]) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    base = np.zeros((height, width), dtype=int)
    weighted_palette = list(moving) * 2 + list(fixed) * 3
    for r in range(height):
        for c in range(width):
            base[r, c] = int(random.choice(weighted_palette))

    coords = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(coords)
    witnesses: dict[int, tuple[int, int]] = {}
    for color in palette:
        if not coords:
            break
        r, c = coords.pop()
        base[r, c] = int(color)
        witnesses[int(color)] = (int(r), int(c))

    for r in range(height):
        if len(set(map(int, base[r]))) == 1:
            c = random.randrange(width)
            options = [color for color in palette if int(color) != int(base[r, c])]
            base[r, c] = int(random.choice(options))

    moving_positions = [(r, c) for r in range(height) for c in range(width) if int(base[r, c]) in set(moving)]
    if len(moving_positions) < 2:
        r, c = random.randrange(height), random.randrange(width)
        base[r, c] = int(random.choice(moving))

    return base, witnesses


def _choose_deep_chain(base_band: np.ndarray, witnesses: dict[int, tuple[int, int]], moving: list[int]) -> tuple[int, int]:
    protected_positions = set(witnesses.values())
    candidates = [
        (r, c)
        for r in range(base_band.shape[0])
        for c in range(base_band.shape[1])
        if int(base_band[r, c]) in set(moving) and (int(r), int(c)) not in protected_positions
    ]
    if not candidates:
        for r in range(base_band.shape[0]):
            for c in range(base_band.shape[1]):
                if (r, c) not in protected_positions:
                    base_band[r, c] = int(random.choice(moving))
                    candidates.append((r, c))
                    break
            if candidates:
                break
    return random.choice(candidates)


def _make_holes(
    full: np.ndarray,
    base_band: np.ndarray,
    width: int,
    band_count: int,
    witnesses: dict[int, tuple[int, int]],
    moving: list[int],
    fixed: list[int],
) -> np.ndarray:
    height = full.shape[0]
    holes = np.zeros_like(full, dtype=bool)
    protected = np.zeros_like(full, dtype=bool)

    for r, c in witnesses.values():
        protected[r, c] = True
        if band_count > 1:
            protected[r, c + width] = True

    deep_r, deep_c = _choose_deep_chain(base_band, witnesses, moving)
    if band_count == 4:
        deep_bands = list(range(1, band_count))
    else:
        deep_bands = list(range(1, band_count - 1))
    for band in deep_bands:
        holes[deep_r, deep_c + band * width] = True

    fixed_positions = [
        (r, c)
        for r in range(base_band.shape[0])
        for c in range(base_band.shape[1])
        if int(base_band[r, c]) in set(fixed) and (r, c) not in set(witnesses.values())
    ]
    if fixed_positions and random.random() < 0.7:
        fr, fc = random.choice(fixed_positions)
        band = random.randint(1, band_count - 1)
        holes[fr, fc + band * width] = True

    base_prob = random.uniform(0.10, 0.22)
    for r in range(height):
        for c in range(full.shape[1]):
            if protected[r, c]:
                continue
            if random.random() < base_prob:
                holes[r, c] = True

    holes &= ~protected

    for r in range(height):
        for residue in range(width):
            cols = [residue + band * width for band in range(band_count)]
            if np.all(holes[r, cols]):
                keep = random.choice(cols)
                holes[r, keep] = False

    if np.sum(holes) < _MIN_HOLES:
        choices = [(r, c) for r in range(height) for c in range(full.shape[1]) if not protected[r, c] and not holes[r, c]]
        random.shuffle(choices)
        for r, c in choices:
            residue = c % width
            cols = [residue + band * width for band in range(band_count)]
            if np.sum(~holes[r, cols]) <= 1:
                continue
            holes[r, c] = True
            if np.sum(holes) >= _MIN_HOLES:
                break

    if not np.any(holes & np.isin(full, moving)):
        choices = [(r, c) for r in range(height) for c in range(full.shape[1]) if int(full[r, c]) in set(moving) and not protected[r, c]]
        random.shuffle(choices)
        for r, c in choices:
            residue = c % width
            cols = [residue + band * width for band in range(band_count)]
            if np.sum(~holes[r, cols]) > 1:
                holes[r, c] = True
                break

    return holes


def _gen_attempt() -> np.ndarray | None:
    width = random.choice(_WIDTH_CHOICES)
    band_count = _candidate_band_count(width)
    cycle_len = random.choice(_CYCLE_LENGTHS)
    fixed_count = random.choice(_FIXED_COUNTS)
    total_colors = cycle_len + fixed_count
    height_low = max(_HEIGHT_RANGE[0], (total_colors + width - 1) // width)
    height = random.randint(height_low, _HEIGHT_RANGE[1])

    palette = random.sample(list(Color.NOT_BLACK), total_colors)
    forward, moving, fixed = _build_forward(palette, cycle_len, fixed_count)
    base_band, witnesses = _make_base_band(height, width, palette, moving, fixed)
    full = _render_full_grid(base_band, band_count, forward)
    holes = _make_holes(full, base_band, width, band_count, witnesses, moving, fixed)

    inp = full.copy()
    inp[holes] = Color.BLACK

    try:
        solved, solved_width, solved_forward = _solve(inp)
    except Exception:
        return None

    if solved_width != width:
        return None
    if solved_forward != forward:
        return None
    if not np.array_equal(solved, full):
        return None
    if not np.any(inp == Color.BLACK):
        return None
    if not np.any(holes & np.isin(full, moving)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-band color-cycle input")
