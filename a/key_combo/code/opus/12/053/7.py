import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
TEAL = 8

GRID_SIZE = 12
PANEL_SIZE = 12
SEPARATOR_COL = 12
INPUT_WIDTH = PANEL_SIZE * 2 + 1
MAX_GENERATION_ATTEMPTS = 600


def _find_separator(grid: np.ndarray) -> int:
    h, w = map(int, grid.shape)
    candidates = []
    for c in range(w):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != BLACK:
            candidates.append(c)
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    sep = int(candidates[0])
    assert sep > 0 and sep < w - 1
    assert sep * 2 + 1 == w, "separator must split equal-width panels"
    return sep


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep = _find_separator(grid)
    left = np.asarray(grid[:, :sep], dtype=int)
    right = np.asarray(grid[:, sep + 1 :], dtype=int)
    assert left.shape == right.shape
    return left, right


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _red_segments_by_col(red_mask: np.ndarray) -> dict[int, tuple[int, int]]:
    segments: dict[int, tuple[int, int]] = {}
    for c in np.flatnonzero(np.any(red_mask, axis=0)):
        rows = np.flatnonzero(red_mask[:, c])
        assert rows.size > 0
        assert np.all(np.diff(rows) == 1), "each red column must contain one contiguous segment"
        segments[int(c)] = (int(rows[0]), int(rows.size))
    return segments


def _compose_output(union_mask: np.ndarray, bar_heights: dict[int, int]) -> np.ndarray:
    out = np.full((GRID_SIZE, PANEL_SIZE), BLACK, dtype=int)
    out[union_mask] = TEAL
    for c, height in sorted(bar_heights.items()):
        teal_rows = np.flatnonzero(union_mask[:, c])
        assert teal_rows.size > 0
        dest_top = int(teal_rows.max()) + 1
        dest_bottom = dest_top + int(height)
        assert dest_bottom <= GRID_SIZE
        assert not np.any(union_mask[dest_top:dest_bottom, c])
        out[dest_top:dest_bottom, c] = RED
    return out


def _solve_from_masks(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    union_mask = (left == YELLOW) | (right == YELLOW)
    red_mask = ((left == RED) | (right == RED)) & ~union_mask
    out = np.full(left.shape, BLACK, dtype=int)
    out[union_mask] = TEAL
    for c, (top, height) in _red_segments_by_col(red_mask).items():
        teal_rows = np.flatnonzero(union_mask[:, c])
        assert teal_rows.size > 0, "every red bar column must have blocking foreground above it"
        assert int(teal_rows.max()) < top
        dest_top = int(teal_rows.max()) + 1
        dest_bottom = dest_top + int(height)
        assert dest_bottom <= left.shape[0]
        out[dest_top:dest_bottom, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Merge the two yellow half-masks into one cyan blocker, then slide each red bar upward until it docks under that emergent shape.

    Input structure:
    - The grid contains two equal-width panels separated by one full-height blue divider column.
    - Each panel has black background, yellow mask fragments in the upper area, and red vertical bar segments near the bottom.
    - The yellow fragments align by row and column across the divider; red bars occupy columns whose merged mask contains yellow above.

    Rule:
    - Remove the divider and align the left and right panels into one compact grid.
    - Paint a cell teal wherever either aligned panel contains yellow, leaving black where both are black.
    - Preserve the merged red vertical segments, but move each segment straight upward within its own column until its top cell sits exactly one row below the lowest teal cell in that column; clear the original red positions.

    Key insight:
    - The blocker is not visible in either panel alone: only the yellow union reveals the cyan shape that determines where every red bar stops.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    left, right = _split_panels(g)
    return _solve_from_masks(left, right)


def _sample_union_mask() -> tuple[np.ndarray, list[int]] | None:
    span = random.randint(5, 8)
    start = random.randint(1, PANEL_SIZE - span - 1)
    cols = list(range(start, start + span))

    intervals: list[tuple[int, int]] = []
    first_bottom = random.randint(3, 6)
    first_thickness = random.choice([1, 2])
    intervals.append((first_bottom - first_thickness + 1, first_bottom))

    for _ in cols[1:]:
        prev_top, prev_bottom = intervals[-1]
        candidates = []
        for bottom in range(3, 7):
            for thickness in (1, 2):
                top = bottom - thickness + 1
                if top <= prev_bottom and bottom >= prev_top:
                    candidates.append((top, bottom))
        if not candidates:
            return None
        intervals.append(random.choice(candidates))

    mask = np.zeros((GRID_SIZE, PANEL_SIZE), dtype=bool)
    for c, (top, bottom) in zip(cols, intervals):
        mask[top : bottom + 1, c] = True

    area = int(mask.sum())
    thicknesses = [bottom - top + 1 for top, bottom in intervals]
    if area < 8 or area > 14:
        return None
    if len(set(intervals)) == 1:
        return None
    if max(thicknesses) < 2:
        return None
    if not _connected(mask):
        return None

    n_bars = random.randint(2, min(5, span))
    bar_cols = sorted(random.sample(cols, n_bars))
    thick_bar_cols = [c for c, (top, bottom) in zip(cols, intervals) if bottom - top + 1 >= 2 and c in bar_cols]
    if not thick_bar_cols:
        return None
    return mask, bar_cols


def _split_union_mask(union_mask: np.ndarray, bar_cols: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    if len(bar_cols) < 2:
        return None

    left_only_col, right_only_col = random.sample(bar_cols, 2)
    left_mask = np.zeros_like(union_mask, dtype=bool)
    right_mask = np.zeros_like(union_mask, dtype=bool)

    other_coords = [(int(r), int(c)) for r, c in np.argwhere(union_mask) if c not in {left_only_col, right_only_col}]
    force_overlap = random.random() < 0.6 and len(other_coords) >= 1
    overlap_coord = random.choice(other_coords) if force_overlap else None

    for r, c in np.argwhere(union_mask):
        r = int(r)
        c = int(c)
        if c == left_only_col:
            left_mask[r, c] = True
        elif c == right_only_col:
            right_mask[r, c] = True
        elif overlap_coord is not None and (r, c) == overlap_coord:
            left_mask[r, c] = True
            right_mask[r, c] = True
        else:
            mode = random.choices(("left", "right", "both"), weights=(4, 4, 3), k=1)[0]
            if mode in ("left", "both"):
                left_mask[r, c] = True
            if mode in ("right", "both"):
                right_mask[r, c] = True

    left_count = int(left_mask.sum())
    right_count = int(right_mask.sum())
    if not (5 <= left_count <= 10 and 5 <= right_count <= 10):
        return None
    if np.array_equal(left_mask, union_mask) or np.array_equal(right_mask, union_mask):
        return None
    if np.any(np.logical_and(left_mask, right_mask)) != force_overlap and overlap_coord is not None:
        return None
    if left_mask[:, right_only_col].any():
        return None
    if right_mask[:, left_only_col].any():
        return None
    return left_mask, right_mask


def _assign_bars(bar_cols: list[int], union_mask: np.ndarray) -> tuple[dict[int, int], dict[int, int]] | None:
    bar_heights = {int(c): random.randint(1, 2) for c in bar_cols}
    side_by_col: dict[int, int] = {}
    if len(bar_cols) == 1:
        side_by_col[int(bar_cols[0])] = random.choice((0, 1))
    else:
        shuffled = bar_cols[:]
        random.shuffle(shuffled)
        side_by_col[int(shuffled[0])] = 0
        side_by_col[int(shuffled[1])] = 1
        for c in shuffled[2:]:
            side_by_col[int(c)] = random.choice((0, 1))

    for c, height in bar_heights.items():
        teal_rows = np.flatnonzero(union_mask[:, c])
        if teal_rows.size == 0:
            return None
        dock_top = int(teal_rows.max()) + 1
        if dock_top + height > GRID_SIZE:
            return None
    return bar_heights, side_by_col


def _build_input(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    bar_heights: dict[int, int],
    side_by_col: dict[int, int],
) -> np.ndarray:
    left = np.full((GRID_SIZE, PANEL_SIZE), BLACK, dtype=int)
    right = np.full((GRID_SIZE, PANEL_SIZE), BLACK, dtype=int)
    left[left_mask] = YELLOW
    right[right_mask] = YELLOW

    for c, height in bar_heights.items():
        panel = left if side_by_col[c] == 0 else right
        panel[GRID_SIZE - height :, c] = RED

    grid = np.full((GRID_SIZE, INPUT_WIDTH), BLACK, dtype=int)
    grid[:, :PANEL_SIZE] = left
    grid[:, SEPARATOR_COL] = BLUE
    grid[:, SEPARATOR_COL + 1 :] = right
    return grid


def _semantic_checks(
    grid: np.ndarray,
    union_mask: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    bar_heights: dict[int, int],
) -> bool:
    if grid.shape != (GRID_SIZE, INPUT_WIDTH):
        return False
    if not _connected(union_mask):
        return False
    if np.any(union_mask[7:, :]):
        return False
    if not (5 <= int(left_mask.sum()) <= 10 and 5 <= int(right_mask.sum()) <= 10):
        return False

    expected = _compose_output(union_mask, bar_heights)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    left_panel, right_panel = _split_panels(grid)
    right_red_only = np.where(right_panel == RED, RED, BLACK)
    left_red_only = np.where(left_panel == RED, RED, BLACK)
    try:
        left_only = _solve_from_masks(left_panel, right_red_only)
    except Exception:
        left_only = None
    try:
        right_only = _solve_from_masks(left_red_only, right_panel)
    except Exception:
        right_only = None
    if left_only is not None and np.array_equal(left_only, expected):
        return False
    if right_only is not None and np.array_equal(right_only, expected):
        return False

    for c, height in bar_heights.items():
        teal_rows = np.flatnonzero(union_mask[:, c])
        if teal_rows.size == 0:
            return False
        dest_top = int(teal_rows.max()) + 1
        if expected[dest_top : dest_top + height, c].sum() == 0:
            return False
    if not any(not left_mask[:, c].any() for c in bar_heights):
        return False
    if not any(not right_mask[:, c].any() for c in bar_heights):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    sampled = _sample_union_mask()
    if sampled is None:
        return None
    union_mask, bar_cols = sampled

    split = _split_union_mask(union_mask, bar_cols)
    if split is None:
        return None
    left_mask, right_mask = split

    bars = _assign_bars(bar_cols, union_mask)
    if bars is None:
        return None
    bar_heights, side_by_col = bars

    grid = _build_input(left_mask, right_mask, bar_heights, side_by_col)
    if not _semantic_checks(grid, union_mask, left_mask, right_mask, bar_heights):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-union ARC input")
