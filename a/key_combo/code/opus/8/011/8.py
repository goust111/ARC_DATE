from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.insert(0, str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, get_bg
except Exception:  # pragma: no cover - fallback for standalone loading
    class Color:
        BLACK = 0

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


PERIOD_CHOICES = (2, 3, 4)
SIZE_CHOICES = (11, 13, 15, 17, 19, 21)
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _find_cross_center(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = map(int, grid.shape)
    row_candidates = [
        r
        for r in range(h)
        if int(np.count_nonzero(grid[r, :] != bg)) == w - 1
    ]
    col_candidates = [
        c
        for c in range(w)
        if int(np.count_nonzero(grid[:, c] != bg)) == h - 1
    ]

    centers: list[tuple[int, int]] = []
    for r in row_candidates:
        for c in col_candidates:
            if int(grid[r, c]) != bg:
                continue
            if not np.all(grid[r, np.arange(w) != c] != bg):
                continue
            if not np.all(grid[np.arange(h) != r, c] != bg):
                continue
            outside = np.ones_like(grid, dtype=bool)
            outside[r, :] = False
            outside[:, c] = False
            if np.all(grid[outside] == bg):
                centers.append((r, c))

    assert len(centers) == 1, "expected exactly one full-row/full-column cross center"
    return centers[0]


def _infer_periodic_sequence(offset_to_color: dict[int, int]) -> tuple[int, tuple[int, ...]]:
    assert offset_to_color, "need at least one observed offset"
    max_period = max(abs(offset) for offset in offset_to_color)
    for period in range(1, max_period + 1):
        residue_to_color: dict[int, int] = {}
        ok = True
        for offset, color in offset_to_color.items():
            residue = offset % period
            if residue in residue_to_color and residue_to_color[residue] != color:
                ok = False
                break
            residue_to_color[residue] = color
        if ok and len(residue_to_color) == period:
            seq = tuple(int(residue_to_color[idx]) for idx in range(period))
            return period, seq
    raise AssertionError("failed to infer a periodic color sequence")


def _period_from_arm(grid: np.ndarray, center: tuple[int, int], bg: int, axis: int) -> tuple[int, tuple[int, ...]]:
    cr, cc = center
    offsets: dict[int, int] = {}
    if axis == 0:
        for r in range(grid.shape[0]):
            if r == cr:
                continue
            color = int(grid[r, cc])
            assert color != bg, "vertical arm must be foreground away from the center"
            offsets[r - cr] = color
    else:
        for c in range(grid.shape[1]):
            if c == cc:
                continue
            color = int(grid[cr, c])
            assert color != bg, "horizontal arm must be foreground away from the center"
            offsets[c - cc] = color
    return _infer_periodic_sequence(offsets)


def _smallest_period(seq: tuple[int, ...]) -> int:
    n = len(seq)
    for period in range(1, n + 1):
        if n % period != 0:
            continue
        if all(seq[idx] == seq[idx % period] for idx in range(n)):
            return period
    return n


def _render_output(shape: tuple[int, int], center: tuple[int, int], h_seq: tuple[int, ...], v_seq: tuple[int, ...], bg: int) -> np.ndarray:
    h, w = map(int, shape)
    cr, cc = center
    row_colors = np.array([v_seq[(r - cr) % len(v_seq)] for r in range(h)], dtype=int)
    col_colors = np.array([h_seq[(c - cc) % len(h_seq)] for c in range(w)], dtype=int)
    row_grid = np.repeat(row_colors[:, None], w, axis=1)
    col_grid = np.repeat(col_colors[None, :], h, axis=0)
    return np.where(row_grid == col_grid, row_grid, int(bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the cross arms as independent periodic row and column labels, then fill only their same-color intersections.

    Input structure:
    - The background is black.
    - Exactly one central row and one central column contain foreground pixels everywhere except their shared center cell.
    - The horizontal arm encodes a repeating color sequence across columns, and the vertical arm encodes another repeating color sequence across rows.
    - Outside that cross, all cells are background.

    Rule:
    - Find the unique cross center where the full foreground row and full foreground column meet on a background cell.
    - Recover the shortest periodic horizontal sequence from arm colors as a function of column offset from the center.
    - Recover the shortest periodic vertical sequence from arm colors as a function of row offset from the center.
    - Extend those two sequences across the whole grid and color a cell iff its row label and column label are equal; otherwise leave it background.

    Key insight:
    - The cross is only a pair of one-dimensional periodic seeds; the output is the two-dimensional interference pattern formed by equality matches between the row-period and column-period labels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert int(bg) == Color.BLACK, "expected black background"

    center = _find_cross_center(g, bg)
    h_period, h_seq = _period_from_arm(g, center, bg, axis=1)
    v_period, v_seq = _period_from_arm(g, center, bg, axis=0)

    assert h_period >= 1 and v_period >= 1
    return _render_output(g.shape, center, h_seq, v_seq, bg)


def _random_primitive_sequence(period: int, allowed: tuple[int, ...], required: set[int]) -> tuple[int, ...] | None:
    for _ in range(200):
        seq = tuple(random.choice(allowed) for _ in range(period))
        if len(set(seq)) < 2:
            continue
        if not required.issubset(seq):
            continue
        if _smallest_period(seq) != period:
            continue
        return seq
    return None


def _build_cross_grid(shape: tuple[int, int], h_seq: tuple[int, ...], v_seq: tuple[int, ...]) -> np.ndarray:
    h, w = map(int, shape)
    cr, cc = h // 2, w // 2
    grid = np.full((h, w), Color.BLACK, dtype=int)
    for c in range(w):
        if c != cc:
            grid[cr, c] = int(h_seq[(c - cc) % len(h_seq)])
    for r in range(h):
        if r != cr:
            grid[r, cc] = int(v_seq[(r - cr) % len(v_seq)])
    return grid


def _validate_generated_case(grid: np.ndarray, h_seq: tuple[int, ...], v_seq: tuple[int, ...]) -> bool:
    h, w = map(int, grid.shape)
    center = (h // 2, w // 2)
    bg = Color.BLACK

    try:
        inferred_center = _find_cross_center(grid, bg)
        hp, inferred_h = _period_from_arm(grid, center, bg, axis=1)
        vp, inferred_v = _period_from_arm(grid, center, bg, axis=0)
        out = main(grid)
    except AssertionError:
        return False

    if inferred_center != center:
        return False
    if hp != len(h_seq) or vp != len(v_seq):
        return False
    if inferred_h != h_seq or inferred_v != v_seq:
        return False

    expected = _render_output(grid.shape, center, h_seq, v_seq, bg)
    if not np.array_equal(out, expected):
        return False

    filled = int(np.count_nonzero(out))
    density = filled / float(h * w)
    if not (0.05 <= density <= 0.45):
        return False

    cross_mask = np.zeros_like(grid, dtype=bool)
    cross_mask[center[0], :] = True
    cross_mask[:, center[1]] = True
    cross_mask[center] = False
    outside_mask = ~cross_mask

    if not np.any(out[cross_mask] != bg):
        return False
    if not np.any((grid[cross_mask] != bg) & (out[cross_mask] == bg)):
        return False
    if not np.any(out[outside_mask] != bg):
        return False
    if not np.any(expected[outside_mask] == bg):
        return False
    if np.array_equal(grid, out):
        return False

    row_colors = np.array([v_seq[(r - center[0]) % len(v_seq)] for r in range(h)], dtype=int)
    col_colors = np.array([h_seq[(c - center[1]) % len(h_seq)] for c in range(w)], dtype=int)
    match_grid = np.repeat(row_colors[:, None], w, axis=1) == np.repeat(col_colors[None, :], h, axis=0)
    if not np.any(match_grid[outside_mask]):
        return False
    if not np.any(~match_grid[outside_mask]):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.choice(SIZE_CHOICES)
    width = random.choice(SIZE_CHOICES)
    h_period, v_period = random.sample(PERIOD_CHOICES, 2)

    shared_count = random.choice((1, 1, 2))
    h_unique_count = random.choice((1, 2))
    v_unique_count = random.choice((1, 2))
    total_colors = shared_count + h_unique_count + v_unique_count
    if total_colors > len(PALETTE):
        return None

    sampled = random.sample(PALETTE, total_colors)
    shared = tuple(sampled[:shared_count])
    h_only = tuple(sampled[shared_count : shared_count + h_unique_count])
    v_only = tuple(sampled[shared_count + h_unique_count :])

    h_allowed = shared + h_only
    v_allowed = shared + v_only
    h_required = {random.choice(shared), random.choice(h_only)}
    v_required = {random.choice(shared), random.choice(v_only)}

    h_seq = _random_primitive_sequence(h_period, h_allowed, h_required)
    v_seq = _random_primitive_sequence(v_period, v_allowed, v_required)
    if h_seq is None or v_seq is None:
        return None

    if set(h_seq) == set(v_seq):
        return None

    grid = _build_cross_grid((height, width), h_seq, v_seq)
    if not _validate_generated_case(grid, h_seq, v_seq):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-cross input")
