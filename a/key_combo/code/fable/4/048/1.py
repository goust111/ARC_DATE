from __future__ import annotations

import random

import numpy as np


_SIDES = ("top", "right", "bottom", "left")
_ROT_K_BY_WIDTH = {1: 3, 2: 2, 3: 1}
_PALETTE = list(range(1, 10))
_GEN_RETRIES = 300


def _allowed_notch_positions(side_len: int, width: int) -> list[tuple[int, ...]]:
    mid = side_len // 2
    if width == 1:
        return [(mid,)]
    if width == 2:
        return [(mid - 1, mid), (mid, mid + 1)]
    if width == 3:
        return [(mid - 1, mid, mid + 1)]
    return []


def _extract_side(grid: np.ndarray, depth: int, side: str) -> np.ndarray:
    n = int(grid.shape[0])
    lo = depth
    hi = n - depth - 1
    if side == "top":
        return grid[lo, lo : hi + 1]
    if side == "bottom":
        return grid[hi, lo : hi + 1]
    if side == "left":
        return grid[lo : hi + 1, lo]
    if side == "right":
        return grid[lo : hi + 1, hi]
    raise AssertionError(f"unknown side: {side}")


def _ring_width_at_depth(grid: np.ndarray, bg: int, depth: int) -> int | None:
    n = int(grid.shape[0])
    side_len = n - 2 * depth
    if side_len < 5:
        return None

    sides = {side: _extract_side(grid, depth, side) for side in _SIDES}
    border_vals = np.concatenate([arr.reshape(-1) for arr in sides.values()])
    non_bg_colors = set(map(int, np.unique(border_vals))) - {int(bg)}
    if len(non_bg_colors) != 1:
        return None
    ring_color = int(next(iter(non_bg_colors)))

    notch_width = None
    notch_side = None
    for side, arr in sides.items():
        bg_pos = tuple(int(i) for i in np.flatnonzero(arr == bg))
        if not bg_pos:
            if not np.all(arr == ring_color):
                return None
            continue
        width = len(bg_pos)
        if width not in (1, 2, 3):
            return None
        if bg_pos not in _allowed_notch_positions(side_len, width):
            return None
        if notch_side is not None:
            return None
        if np.any(arr[np.array([i for i in range(side_len) if i not in bg_pos], dtype=int)] != ring_color):
            return None
        notch_side = side
        notch_width = width

    if notch_side is None or notch_width is None:
        return None
    return int(notch_width)


def _detect_rings(grid: np.ndarray) -> tuple[int, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 1, "expected an odd square grid"

    best: tuple[int, list[int]] | None = None
    best_count = -1
    tie = False

    for bg in map(int, np.unique(g)):
        widths: list[int] = []
        for depth in range(h // 2):
            width = _ring_width_at_depth(g, bg, depth)
            if width is None:
                break
            widths.append(width)
        if len(widths) > best_count:
            best = (bg, widths)
            best_count = len(widths)
            tie = False
        elif len(widths) == best_count and len(widths) > 0:
            tie = True

    assert best is not None and best_count >= 1 and not tie, "failed to identify a unique ring/background layout"
    bg, widths = best
    core_depth = len(widths)
    core = g[core_depth : h - core_depth, core_depth : w - core_depth]
    assert core.size > 0, "expected a non-empty core"
    assert len(set(map(int, np.unique(core)))) >= 2, "expected a multicolor core"
    return int(bg), [int(w) for w in widths]


def _apply_cascade(grid: np.ndarray, widths: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    n = int(out.shape[0])
    for depth, width in enumerate(widths):
        side_len = n - 2 * depth
        sub = out[depth : depth + side_len, depth : depth + side_len]
        out[depth : depth + side_len, depth : depth + side_len] = np.rot90(sub, k=_ROT_K_BY_WIDTH[int(width)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cascade-rotate each concentric square ring and everything inside it by the angle encoded by that ring's notch width.

    Input structure:
    - The grid is an odd square canvas made of one-cell-thick concentric square rings.
    - Each ring is a single color and has one centered background-color notch of width one, two, or three on exactly one side.
    - The innermost region is a multicolor core pattern without quarter-turn symmetry.

    Rule:
    - Decode each ring's notch width as a rotation angle: width 1 -> clockwise 90 degrees, width 2 -> 180 degrees, width 3 -> clockwise 270 degrees.
    - Process rings from outermost to innermost; at each step rotate that ring together with the entire square inside it around the canvas center.
    - Return the final grid after all nested rotations have been applied in sequence.

    Key insight:
    - The rotations accumulate inward, so each inner ring and the core inherit every outer ring's turn instead of rotating independently.
    """
    _bg, widths = _detect_rings(grid)
    return _apply_cascade(grid, widths)


def _fill_ring(grid: np.ndarray, depth: int, color: int, side: str, width: int, bg: int) -> None:
    n = int(grid.shape[0])
    lo = depth
    hi = n - depth - 1
    grid[lo, lo : hi + 1] = color
    grid[hi, lo : hi + 1] = color
    grid[lo : hi + 1, lo] = color
    grid[lo : hi + 1, hi] = color

    side_len = hi - lo + 1
    notch = random.choice(_allowed_notch_positions(side_len, width))
    if side == "top":
        grid[lo, lo + np.array(notch, dtype=int)] = bg
    elif side == "bottom":
        grid[hi, lo + np.array(notch, dtype=int)] = bg
    elif side == "left":
        grid[lo + np.array(notch, dtype=int), lo] = bg
    elif side == "right":
        grid[lo + np.array(notch, dtype=int), hi] = bg
    else:
        raise AssertionError(f"unknown side: {side}")


def _make_core(size: int, colors: list[int]) -> np.ndarray:
    for _ in range(200):
        core = np.random.choice(colors, size=(size, size)).astype(int)
        if len(set(map(int, np.unique(core)))) < 3:
            continue
        border = np.concatenate((core[0, :], core[-1, :], core[1:-1, 0], core[1:-1, -1]))
        if len(set(map(int, border))) < 2:
            continue
        if any(np.array_equal(core, np.rot90(core, k=k)) for k in (1, 2, 3)):
            continue
        return core
    raise RuntimeError("failed to build a non-symmetric multicolor core")


def _sample_widths(ring_count: int) -> list[int]:
    if ring_count == 1:
        return [random.randint(1, 3)]
    widths = [1, 2, 3][: min(3, ring_count)]
    while len(widths) < ring_count:
        widths.append(random.randint(1, 3))
    random.shuffle(widths)
    return widths


def _sample_sides(ring_count: int) -> list[str]:
    sides = [random.choice(_SIDES) for _ in range(ring_count)]
    if ring_count > 1 and len(set(sides)) == 1:
        choices = [side for side in _SIDES if side != sides[-1]]
        sides[-1] = random.choice(choices)
    return sides


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    ring_count = random.randint(2, 5)
    core_size = random.choice([3, 5, 7])
    n = core_size + 2 * ring_count
    if n > 29:
        return None

    ring_colors = random.sample(_PALETTE, ring_count)
    core_palette = random.sample([c for c in _PALETTE if c not in ring_colors], 3)
    widths = _sample_widths(ring_count)
    sides = _sample_sides(ring_count)

    grid = np.full((n, n), bg, dtype=int)
    core = _make_core(core_size, core_palette)
    grid[ring_count : ring_count + core_size, ring_count : ring_count + core_size] = core

    for depth, (color, side, width) in enumerate(zip(ring_colors, sides, widths, strict=False)):
        _fill_ring(grid, depth, color, side, width, bg)

    try:
        detected_bg, detected_widths = _detect_rings(grid)
        out = main(grid)
    except Exception:
        return None

    if detected_bg != bg or detected_widths != widths:
        return None
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out, _apply_cascade(grid, widths)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring-cascade rotation input")
