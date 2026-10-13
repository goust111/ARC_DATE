import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

PANEL_H = 7
PANEL_W = 6
INPUT_SHAPE = (PANEL_H, PANEL_W * 2 + 1)
GEN_MAX_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    """Yield 4-neighbors using the shared ARC (row, col) convention."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox(mask: np.ndarray):
    """Inclusive bbox of a boolean mask, mirroring arc_utils.bbox."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box):
    """Inclusive crop, mirroring arc_utils.crop_bbox."""
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    out: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        out.append((r, c))
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return out


def _is_single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_component_coords(mask)) == int(mask.sum())


def _normalized_shape(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    assert box is not None
    return crop_bbox(mask.astype(int), box)


def _find_separator_col(grid: np.ndarray) -> int:
    grey_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    for c in grey_cols:
        if c == grid.shape[1] - 1 - c:
            return int(c)
    raise AssertionError("expected a centered grey separator column")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two side panels, paint every union-foreground column teal, and tile the result 2x2.

    Input structure:
    - The grid has a black background with one full-height grey separator column in the middle.
    - The separator splits the canvas into two aligned panels of equal size.
    - The left panel contains a small red 4-connected pattern and the right panel contains a small blue 4-connected pattern.

    Rule:
    - Remove the grey separator and align the two panels cell by cell.
    - Mark a cell as foreground in the merged panel whenever either side has a foreground pixel there, and recolor all merged foreground to red.
    - For every column containing at least one merged foreground pixel, recolor only its black background cells to teal.
    - Leave columns with no merged foreground entirely black, then tile the finished panel into a 2x2 repetition.

    Key insight:
    - The decisive foreground columns come from the union of the two aligned panel masks, not from either panel alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape

    left_fg = left != BLACK
    right_fg = right != BLACK
    union = left_fg | right_fg

    panel = np.zeros_like(left, dtype=int)
    panel[union] = RED

    fg_cols = np.any(union, axis=0)
    for c in np.flatnonzero(fg_cols):
        fill_mask = panel[:, c] == BLACK
        panel[fill_mask, c] = TEAL

    return np.tile(panel, (2, 2))


def _sample_projection_sets() -> tuple[set[int], set[int]]:
    cols = list(range(PANEL_W))
    while True:
        inter_n = random.choice((1, 2))
        left_only_n = random.choice((1, 2))
        right_only_n = random.choice((1, 2))
        if inter_n + left_only_n + right_only_n > PANEL_W:
            continue
        inter = set(random.sample(cols, inter_n))
        remaining = [c for c in cols if c not in inter]
        left_only = set(random.sample(remaining, left_only_n))
        remaining = [c for c in remaining if c not in left_only]
        right_only = set(random.sample(remaining, right_only_n))
        left_cols = inter | left_only
        right_cols = inter | right_only
        if len(left_cols) >= 2 and len(right_cols) >= 2:
            return left_cols, right_cols


def _sample_connected_shape(target_cols: set[int], area: int) -> np.ndarray | None:
    for _ in range(200):
        cells = {(random.randrange(PANEL_H), random.choice(tuple(target_cols)))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < PANEL_H and 0 <= cc < PANEL_W and cc in target_cols and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        used_cols = {c for _, c in cells}
        if used_cols != target_cols:
            continue
        mask = np.zeros((PANEL_H, PANEL_W), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _is_single_component(mask):
            return mask
    return None


def _panel_with_shape(mask: np.ndarray, color: int) -> np.ndarray:
    panel = np.zeros((PANEL_H, PANEL_W), dtype=int)
    panel[mask] = int(color)
    return panel


def _build_input(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    grid = np.zeros(INPUT_SHAPE, dtype=int)
    grid[:, PANEL_W] = GREY
    grid[:, :PANEL_W] = _panel_with_shape(left_mask, RED)
    grid[:, PANEL_W + 1 :] = _panel_with_shape(right_mask, BLUE)
    return grid


def _valid_instance(left_mask: np.ndarray, right_mask: np.ndarray) -> bool:
    if not (3 <= int(left_mask.sum()) <= 6 and 3 <= int(right_mask.sum()) <= 6):
        return False
    if not (_is_single_component(left_mask) and _is_single_component(right_mask)):
        return False

    left_cols = set(np.flatnonzero(np.any(left_mask, axis=0)))
    right_cols = set(np.flatnonzero(np.any(right_mask, axis=0)))
    union_cols = left_cols | right_cols
    if not (left_cols & right_cols):
        return False
    if not (left_cols - right_cols):
        return False
    if not (right_cols - left_cols):
        return False
    if not (len(union_cols) > len(left_cols) and len(union_cols) > len(right_cols)):
        return False

    if np.array_equal(_normalized_shape(left_mask), _normalized_shape(right_mask)):
        return False

    grid = _build_input(left_mask, right_mask)
    out = main(grid)
    if out.shape != (PANEL_H * 2, PANEL_W * 2):
        return False

    merged = out[:PANEL_H, :PANEL_W]
    expected_cols = np.any(left_mask | right_mask, axis=0)
    actual_cols = np.any(merged != BLACK, axis=0)
    if not np.array_equal(expected_cols, actual_cols):
        return False

    for c in np.flatnonzero(expected_cols):
        vals = set(map(int, merged[:, c]))
        if not vals <= {RED, TEAL} or RED not in vals:
            return False
    for c in np.flatnonzero(~expected_cols):
        if np.any(merged[:, c] != BLACK):
            return False

    return (
        np.array_equal(out[:PANEL_H, :PANEL_W], out[:PANEL_H, PANEL_W:])
        and np.array_equal(out[:PANEL_H, :PANEL_W], out[PANEL_H:, :PANEL_W])
        and np.array_equal(out[:PANEL_H, :PANEL_W], out[PANEL_H:, PANEL_W:])
    )


def _gen_attempt() -> np.ndarray | None:
    left_cols, right_cols = _sample_projection_sets()
    left_area = random.randint(max(3, len(left_cols)), 6)
    right_area = random.randint(max(3, len(right_cols)), 6)
    left_mask = _sample_connected_shape(left_cols, left_area)
    right_mask = _sample_connected_shape(right_cols, right_area)
    if left_mask is None or right_mask is None:
        return None
    if not _valid_instance(left_mask, right_mask):
        return None
    return _build_input(left_mask, right_mask)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

