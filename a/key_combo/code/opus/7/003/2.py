from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
PAIR_COLORS = (1, 2, 3)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _meeting_axis(width: int) -> int:
    return (int(width) - 1) // 2


def _pair_rows(grid: np.ndarray) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    height, width = grid.shape
    for r in range(height):
        left = int(grid[r, 0])
        if left == BLACK or left == GREY:
            continue
        if int(grid[r, width - 1]) != left:
            continue
        rows.append((int(r), left))
    return rows


def _complete_orange_mask(visible_mask: np.ndarray, axis: int) -> np.ndarray:
    out = visible_mask.copy()
    box = _bbox(visible_mask)
    if box is None:
        return out

    height, width = visible_mask.shape
    r0, c0, r1, c1 = box
    left = min(c0, 2 * int(axis) - c1)
    right = max(c1, 2 * int(axis) - c0)
    if not (0 <= left <= right < width):
        return out

    for r in range(r0, r1 + 1):
        for c in range(left, right + 1):
            if visible_mask[r, c]:
                continue
            mc = 2 * int(axis) - c
            if 0 <= mc < width and visible_mask[r, mc]:
                out[r, c] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use opposite-border meeting trails to reveal the vertical symmetry axis, then complete the damaged orange figure.

    Input structure:
    - Black background contains several rows whose leftmost and rightmost cells are same-color singleton border pixels.
    - Those border pairs are blue, red, or green and lie on rows disjoint from the orange figure.
    - The remaining orange cells form a partially erased figure whose true vertical symmetry axis is not uniquely recoverable from the figure alone.

    Rule:
    - For every row whose two border cells share the same non-black color, slide the pair inward along that row and leave a color trail.
    - Mark the meeting cell gray; for even widths this is the left of the two central cells.
    - All gray meeting cells share one column A, which is the figure's vertical symmetry axis.
    - Complete the orange figure by restoring every black cell whose reflection across column A is an existing orange cell, and keep the trails and gray meeting marks.

    Key insight:
    - The border-pair motion does not just add decoration: its common meeting column is the only reliable source of the orange figure's symmetry axis.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    height, width = g.shape
    axis = _meeting_axis(width)

    for r, color in _pair_rows(g):
        out[r, :] = int(color)
        out[r, axis] = GREY

    orange_visible = g == ORANGE
    orange_full = _complete_orange_mask(orange_visible, axis)
    out[orange_full] = ORANGE
    for r, color in _pair_rows(g):
        out[r, :] = int(color)
        out[r, axis] = GREY
    return out


def _spaced_rows(candidates: list[int], count: int) -> list[int] | None:
    order = candidates[:]
    random.shuffle(order)
    chosen: list[int] = []
    for r in order:
        if all(abs(r - old) > 1 for old in chosen):
            chosen.append(int(r))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _grow_half_mask(height: int, local_width: int) -> np.ndarray:
    target_low = max(6, height + 1, local_width + 2)
    target_high = min(height * local_width - 1, max(target_low, (2 * height * local_width) // 3))
    target = random.randint(target_low, target_high)

    seed = (random.randrange(height), random.randrange(local_width))
    cells = {seed}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < local_width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((height, local_width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _half_mask_ok(mask: np.ndarray) -> bool:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 3 or len(cols) < 2:
        return False
    if not mask[:, -1].any():
        return False
    if not mask[:, :-1].any():
        return False
    if int(mask.sum()) < 6:
        return False
    return True


def _build_global_masks(
    half_mask: np.ndarray,
    top: int,
    axis: int,
    keep_preferred: str,
) -> tuple[np.ndarray, np.ndarray]:
    local_h, local_w = half_mask.shape
    span = local_w - 1
    full_w = 2 * span + 1

    full_local = np.zeros((local_h, full_w), dtype=bool)
    visible_local = np.zeros((local_h, full_w), dtype=bool)
    for r in range(local_h):
        for c in range(local_w):
            if not half_mask[r, c]:
                continue
            mc = 2 * span - c
            full_local[r, c] = True
            full_local[r, mc] = True
            if c == span:
                visible_local[r, c] = True
                continue

            roll = random.random()
            if keep_preferred == "left":
                if roll < 0.72:
                    visible_local[r, c] = True
                elif roll < 0.88:
                    visible_local[r, mc] = True
                else:
                    visible_local[r, c] = True
                    visible_local[r, mc] = True
            else:
                if roll < 0.72:
                    visible_local[r, mc] = True
                elif roll < 0.88:
                    visible_local[r, c] = True
                else:
                    visible_local[r, c] = True
                    visible_local[r, mc] = True

    global_full = np.zeros((top + local_h, axis + span + 1), dtype=bool)
    global_visible = np.zeros_like(global_full)
    left = axis - span
    for r in range(local_h):
        for c in range(full_w):
            if full_local[r, c]:
                global_full[top + r, left + c] = True
            if visible_local[r, c]:
                global_visible[top + r, left + c] = True
    return global_full, global_visible


def _candidate_other_axes(visible_mask: np.ndarray, actual_axis: int) -> list[int]:
    height, width = visible_mask.shape
    del height
    others: list[int] = []
    if not visible_mask.any():
        return others
    for axis in range(1, width - 1):
        if axis == actual_axis:
            continue
        completed = _complete_orange_mask(visible_mask, axis)
        if np.array_equal(completed, visible_mask):
            continue
        others.append(axis)
    return others


def _compose_input(
    height: int,
    width: int,
    pair_rows: list[int],
    pair_colors: list[int],
    visible_mask: np.ndarray,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[visible_mask] = ORANGE
    for r, color in zip(pair_rows, pair_colors):
        grid[r, 0] = int(color)
        grid[r, width - 1] = int(color)
    return grid


def _expected_output(inp: np.ndarray, full_mask: np.ndarray) -> np.ndarray:
    out = inp.copy()
    axis = _meeting_axis(inp.shape[1])
    for r, color in _pair_rows(inp):
        out[r, :] = int(color)
        out[r, axis] = GREY
    out[full_mask] = ORANGE
    for r, color in _pair_rows(inp):
        out[r, :] = int(color)
        out[r, axis] = GREY
    return out


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(14, 24)
    width = random.randint(13, 29)
    axis = _meeting_axis(width)

    max_span = min(axis - 1, width - axis - 2, 5)
    if max_span < 2:
        return None
    span = random.randint(2, max_span)
    local_w = span + 1

    shape_h = random.randint(4, min(8, height - 4))
    top = random.randint(2, height - shape_h - 2)
    blocked_rows = set(range(max(0, top - 1), min(height, top + shape_h + 1)))
    row_candidates = [r for r in range(1, height - 1) if r not in blocked_rows]
    if len(row_candidates) < 4:
        return None

    pair_count = random.randint(2, min(3, len(PAIR_COLORS), len(row_candidates) // 2))
    pair_rows = _spaced_rows(row_candidates, pair_count)
    if pair_rows is None:
        return None
    pair_colors = random.sample(list(PAIR_COLORS), k=pair_count)

    for _ in range(100):
        half_mask = _grow_half_mask(shape_h, local_w)
        if not _half_mask_ok(half_mask):
            continue

        keep_preferred = random.choice(("left", "right"))
        full_small, visible_small = _build_global_masks(half_mask, top, axis, keep_preferred)
        full_mask = np.zeros((height, width), dtype=bool)
        visible_mask = np.zeros((height, width), dtype=bool)
        full_mask[: full_small.shape[0], : full_small.shape[1]] = full_small
        visible_mask[: visible_small.shape[0], : visible_small.shape[1]] = visible_small

        if not np.all(visible_mask <= full_mask):
            continue
        if int(full_mask.sum()) - int(visible_mask.sum()) < 4:
            continue

        full_box = _bbox(full_mask)
        vis_box = _bbox(visible_mask)
        if full_box is None or vis_box is None:
            continue
        if full_box[0] != vis_box[0] or full_box[2] != vis_box[2]:
            continue
        if full_box[1] == 0 or full_box[3] == width - 1:
            continue
        if np.any(visible_mask[pair_rows, :]):
            continue

        other_axes = _candidate_other_axes(visible_mask, axis)
        if not other_axes:
            continue
        if not any(not np.array_equal(_complete_orange_mask(visible_mask, wrong), full_mask) for wrong in other_axes):
            continue

        inp = _compose_input(height, width, pair_rows, pair_colors, visible_mask)
        expected = _expected_output(inp, full_mask)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue

        if max(inp.shape) > 30 or max(actual.shape) > 30:
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
