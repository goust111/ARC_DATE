from __future__ import annotations

from collections import Counter
import random

import numpy as np


GUIDE = 5
BACKGROUND = 0
ARC_COLORS = tuple(range(10))
FOREGROUND_COLORS = tuple(c for c in ARC_COLORS if c not in (BACKGROUND, GUIDE))


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    best_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == best_count]
    assert len(winners) == 1, "background color must be uniquely most common"
    return int(winners[0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scale2(sprite: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(sprite, 2, axis=0), 2, axis=1)


def _find_guide(g: np.ndarray) -> tuple[str, int]:
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GUIDE)]
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GUIDE)]
    assert len(rows) + len(cols) == 1, "expected exactly one full grey guide row or column"

    if cols:
        assert int(np.count_nonzero(g == GUIDE)) == g.shape[0], "grey must appear only on the guide column"
        return "vertical", int(cols[0])

    assert int(np.count_nonzero(g == GUIDE)) == g.shape[1], "grey must appear only on the guide row"
    return "horizontal", int(rows[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Slide scattered pixels back to the grey guide line, crop the revealed pattern, and enlarge it two times.

    Input structure:
    - The grid has one dominant background color and exactly one full grey guide row or guide column.
    - All non-background, non-grey pixels are scattered on one side of that guide line.
    - Along each row for a vertical guide, or each column for a horizontal guide, the scattered pixels preserve the
      color order of the hidden compact pattern while leaving gaps between them.

    Rule:
    - If the guide is a column, read each occupied row from left to right and slide those pixels horizontally toward
      the guide so they become a contiguous run touching the guide or the previous slid pixel.
    - If the guide is a row, read each occupied column from top to bottom and slide those pixels vertically toward
      the guide in the same order.
    - Remove the guide line, crop the tight bounding box of the assembled colored pattern, and replace every output
      cell by a 2x2 block of the same color.

    Key insight:
    - The visible pixels are only the exploded residue of the picture; order-preserving compression toward the grey
      guide reveals the compact pattern before the crop and scale operations are meaningful.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    assert bg != GUIDE, "the guide color cannot be the background"
    orientation, line = _find_guide(g)

    fg = (g != bg) & (g != GUIDE)
    assert np.any(fg), "expected scattered foreground pixels"

    assembled = np.full_like(g, bg)
    if orientation == "vertical":
        cols = np.argwhere(fg)[:, 1]
        if np.all(cols < line):
            side = -1
        else:
            assert np.all(cols > line), "all scattered pixels must lie on one side of the guide column"
            side = 1

        for r in range(g.shape[0]):
            positions = [c for c in range(g.shape[1]) if bool(fg[r, c])]
            if not positions:
                continue
            colors = [int(g[r, c]) for c in sorted(positions)]
            if side < 0:
                c0 = line - len(colors)
                assert c0 >= 0, "compressed run must fit before the guide column"
            else:
                c0 = line + 1
                assert c0 + len(colors) <= g.shape[1], "compressed run must fit after the guide column"
            for i, color in enumerate(colors):
                assembled[r, c0 + i] = color
    else:
        rows = np.argwhere(fg)[:, 0]
        if np.all(rows < line):
            side = -1
        else:
            assert np.all(rows > line), "all scattered pixels must lie on one side of the guide row"
            side = 1

        for c in range(g.shape[1]):
            positions = [r for r in range(g.shape[0]) if bool(fg[r, c])]
            if not positions:
                continue
            colors = [int(g[r, c]) for r in sorted(positions)]
            if side < 0:
                r0 = line - len(colors)
                assert r0 >= 0, "compressed run must fit above the guide row"
            else:
                r0 = line + 1
                assert r0 + len(colors) <= g.shape[0], "compressed run must fit below the guide row"
            for i, color in enumerate(colors):
                assembled[r0 + i, c] = color

    box = _bbox(assembled != bg)
    assert box is not None, "assembled pattern must be non-empty"
    r0, c0, r1, c1 = box
    cropped = assembled[r0 : r1 + 1, c0 : c1 + 1]
    return _scale2(cropped)


# --------------- generator ---------------

GEN_RETRIES = 500
TARGET_SIDE_RANGE = (3, 7)
MARGIN_RANGE = (0, 4)
OTHER_SIDE_RANGE = (1, 5)
BRANCHES = (
    ("vertical", -1),
    ("vertical", 1),
    ("horizontal", -1),
    ("horizontal", 1),
)
_GEN_CALL_COUNT = 0


def _sample_lengths(count: int, max_len: int) -> list[int]:
    assert count >= 3 and max_len >= 3
    for _ in range(80):
        lengths = [random.randint(1, max_len) for _ in range(count)]
        full_idx, short_idx = random.sample(range(count), 2)
        lengths[full_idx] = max_len
        lengths[short_idx] = random.randint(1, max_len - 1)
        if len(set(lengths)) >= 2 and any(n >= 2 for n in lengths):
            return lengths
    raise RuntimeError("failed to sample row or column lengths")


def _color_stream(n: int) -> list[int]:
    palette_size = random.randint(3, min(6, len(FOREGROUND_COLORS), n))
    palette = random.sample(FOREGROUND_COLORS, palette_size)
    values = palette + [random.choice(palette) for _ in range(n - palette_size)]
    random.shuffle(values)
    return values


def _make_target(orientation: str, side: int) -> tuple[np.ndarray, list[int]]:
    height = random.randint(*TARGET_SIDE_RANGE)
    width = random.randint(*TARGET_SIDE_RANGE)

    if orientation == "vertical":
        lengths = _sample_lengths(height, width)
        target = np.full((height, width), BACKGROUND, dtype=int)
        values = _color_stream(sum(lengths))
        k = 0
        for r, n in enumerate(lengths):
            c0 = width - n if side < 0 else 0
            for c in range(c0, c0 + n):
                target[r, c] = values[k]
                k += 1
    else:
        lengths = _sample_lengths(width, height)
        target = np.full((height, width), BACKGROUND, dtype=int)
        values = _color_stream(sum(lengths))
        k = 0
        for c, n in enumerate(lengths):
            r0 = height - n if side < 0 else 0
            for r in range(r0, r0 + n):
                target[r, c] = values[k]
                k += 1

    assert _bbox(target != BACKGROUND) == (0, 0, target.shape[0] - 1, target.shape[1] - 1)
    return target, lengths


def _sample_spaced_indices(capacity: int, n: int) -> list[int]:
    assert capacity >= 2 * n - 1
    if n == 1:
        return [random.randint(0, capacity - 2)]

    for _ in range(200):
        positions = sorted(random.sample(range(capacity), n))
        if all(positions[i + 1] - positions[i] >= 2 for i in range(n - 1)):
            return positions

    base = list(range(0, 2 * n - 1, 2))
    slack = capacity - (2 * n - 1)
    shift = random.randint(0, slack)
    return [p + shift for p in base]


def _scatter_vertical(target: np.ndarray, side: int, lengths: list[int]) -> np.ndarray | None:
    height, width = target.shape
    max_count = max(lengths)
    other = random.randint(*OTHER_SIDE_RANGE)
    capacity_min = 2 * max_count - 1
    capacity_max = 30 - other - 1
    if capacity_min > capacity_max:
        return None
    capacity = random.randint(capacity_min, capacity_max)

    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    grid_h = height + top + bottom
    if grid_h > 30:
        return None

    if side < 0:
        line = capacity
        grid_w = capacity + 1 + other
    else:
        line = other
        grid_w = other + 1 + capacity
    if grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    grid[:, line] = GUIDE
    for rr in range(height):
        colors = [int(v) for v in target[rr, :] if int(v) != BACKGROUND]
        rel_positions = _sample_spaced_indices(capacity, len(colors))
        row = top + rr
        for rel, color in zip(rel_positions, colors):
            col = rel if side < 0 else line + 1 + rel
            grid[row, col] = color
    return grid


def _scatter_horizontal(target: np.ndarray, side: int, lengths: list[int]) -> np.ndarray | None:
    height, width = target.shape
    max_count = max(lengths)
    other = random.randint(*OTHER_SIDE_RANGE)
    capacity_min = 2 * max_count - 1
    capacity_max = 30 - other - 1
    if capacity_min > capacity_max:
        return None
    capacity = random.randint(capacity_min, capacity_max)

    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)
    grid_w = width + left + right
    if grid_w > 30:
        return None

    if side < 0:
        line = capacity
        grid_h = capacity + 1 + other
    else:
        line = other
        grid_h = other + 1 + capacity
    if grid_h > 30:
        return None

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    grid[line, :] = GUIDE
    for cc in range(width):
        colors = [int(v) for v in target[:, cc] if int(v) != BACKGROUND]
        rel_positions = _sample_spaced_indices(capacity, len(colors))
        col = left + cc
        for rel, color in zip(rel_positions, colors):
            row = rel if side < 0 else line + 1 + rel
            grid[row, col] = color
    return grid


def _direct_scatter_crop(grid: np.ndarray) -> np.ndarray | None:
    fg = (grid != BACKGROUND) & (grid != GUIDE)
    box = _bbox(fg)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _semantic_accepts(grid: np.ndarray, target: np.ndarray, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    try:
        orientation, line = _find_guide(grid)
        bg = _most_common_color(grid)
        solved = main(grid)
    except AssertionError:
        return False

    if bg != BACKGROUND or not np.array_equal(solved, expected):
        return False

    fg = (grid != BACKGROUND) & (grid != GUIDE)
    coords = np.argwhere(fg)
    if coords.size == 0:
        return False

    if orientation == "vertical":
        cols = coords[:, 1]
        if not (np.all(cols < line) or np.all(cols > line)):
            return False
        active_lines = [np.flatnonzero(fg[r, :]).tolist() for r in range(grid.shape[0]) if np.any(fg[r, :])]
    else:
        rows = coords[:, 0]
        if not (np.all(rows < line) or np.all(rows > line)):
            return False
        active_lines = [np.flatnonzero(fg[:, c]).tolist() for c in range(grid.shape[1]) if np.any(fg[:, c])]

    has_gap = any(len(line_cells) >= 2 and any(b - a > 1 for a, b in zip(line_cells, line_cells[1:])) for line_cells in active_lines)
    if not has_gap:
        return False

    direct = _direct_scatter_crop(grid)
    if direct is not None and direct.shape == target.shape and np.array_equal(direct, target):
        return False

    if expected.shape != (target.shape[0] * 2, target.shape[1] * 2):
        return False
    if len(set(map(int, target[target != BACKGROUND]))) < 3:
        return False
    return True


def _gen_attempt(orientation: str, side: int) -> np.ndarray | None:
    target, lengths = _make_target(orientation, side)
    expected = _scale2(target)

    if orientation == "vertical":
        grid = _scatter_vertical(target, side, lengths)
    else:
        grid = _scatter_horizontal(target, side, lengths)
    if grid is None:
        return None

    return grid if _semantic_accepts(grid, target, expected) else None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    branch = BRANCHES[_GEN_CALL_COUNT % len(BRANCHES)]
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(*branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid guide-compression input")
