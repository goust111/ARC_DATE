from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8

TOGGLE = np.arange(10, dtype=int)
TOGGLE[BLACK] = TEAL
TOGGLE[TEAL] = BLACK
TOGGLE[PINK] = ORANGE
TOGGLE[ORANGE] = PINK

RETRY_LIMIT = 400


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int) -> Iterable[tuple[int, int]]:
    if connectivity == 4:
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        deltas = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    for dr, dc in deltas:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    next_label = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_label += 1
            stack = [(r, c)]
            labels[r, c] = next_label
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors(cr, cc, h, w, connectivity):
                    if mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        stack.append((nr, nc))
    return labels, next_label


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    labels, n = _label_mask(mask, connectivity=connectivity)
    return [np.argwhere(labels == k) for k in range(1, n + 1)]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _compute_channel(path_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, n_paths = _label_mask(path_mask, connectivity=4)
    assert n_paths == 2, "expected exactly two orange path components"
    dilated = [_dilate8(labels == k) for k in range(1, 3)]
    channel = dilated[0] & dilated[1] & ~path_mask
    assert np.any(channel), "expected a non-empty overlap channel"
    return channel, labels


def _selected_from_markers(grid: np.ndarray) -> tuple[str, list[int], np.ndarray]:
    h, w = grid.shape
    row_indices = [r for r in range(1, h - 1) if grid[r, 0] == YELLOW and grid[r, w - 1] == YELLOW]
    col_indices = [c for c in range(1, w - 1) if grid[0, c] == YELLOW and grid[h - 1, c] == YELLOW]
    assert bool(row_indices) != bool(col_indices), "yellow markers must pair on exactly one edge axis"

    expected = set()
    selected = np.zeros((h, w), dtype=bool)
    if row_indices:
        for r in row_indices:
            expected.add((r, 0))
            expected.add((r, w - 1))
            selected[r, 1 : w - 1] = True
        axis = "row"
        indices = row_indices
    else:
        for c in col_indices:
            expected.add((0, c))
            expected.add((h - 1, c))
            selected[1 : h - 1, c] = True
        axis = "col"
        indices = col_indices

    actual = {(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW)}
    assert actual == expected, "all yellow markers must belong to opposite-edge pairs"
    return axis, indices, selected


def _mask_from_cells(shape: tuple[int, int], cells: Iterable[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _left_normal(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _wall_sets(path: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    left: set[tuple[int, int]] = set()
    right: set[tuple[int, int]] = set()
    channel = set(path)
    for i, (r, c) in enumerate(path):
        dirs = []
        if i > 0:
            pr, pc = path[i - 1]
            dirs.append((r - pr, c - pc))
        if i + 1 < len(path):
            nr, nc = path[i + 1]
            dirs.append((nr - r, nc - c))

        for dr, dc in dirs:
            lr, lc = _left_normal(dr, dc)
            if (r + lr, c + lc) not in channel:
                left.add((r + lr, c + lc))
            if (r - lr, c - lc) not in channel:
                right.add((r - lr, c - lc))

        if len(dirs) == 2 and dirs[0] != dirs[1]:
            for side in (1, -1):
                offsets = []
                for dr, dc in dirs:
                    lr, lc = _left_normal(dr, dc)
                    offsets.append((side * lr, side * lc))
                direct = [(r + dr, c + dc) for dr, dc in offsets]
                if all(cell not in channel for cell in direct):
                    er = r + offsets[0][0] + offsets[1][0]
                    ec = c + offsets[0][1] + offsets[1][1]
                    if side == 1:
                        left.add((er, ec))
                    else:
                        right.add((er, ec))
    return left, right


def _is_thin_component(mask: np.ndarray) -> bool:
    comps = _components(mask, connectivity=4)
    if len(comps) != 1:
        return False
    cells = {tuple(map(int, xy)) for xy in comps[0]}
    for r, c in cells:
        degree = 0
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            degree += (nr, nc) in cells
        if degree > 2:
            return False
    return True


def _partition_total(total: int, count: int, low: int, high: int, rng: np.random.Generator) -> list[int] | None:
    if not (count * low <= total <= count * high):
        return None
    parts: list[int] = []
    remaining = total
    for i in range(count - 1):
        rest = count - i - 1
        lo = max(low, remaining - high * rest)
        hi = min(high, remaining - low * rest)
        if lo > hi:
            return None
        val = int(rng.integers(lo, hi + 1))
        parts.append(val)
        remaining -= val
    parts.append(remaining)
    return parts


def _gen_channel_path(h: int, w: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    hdir = int(rng.choice([-1, 1]))
    turns = int(rng.integers(2, 5))
    directions = [(1, 0)] + [((0, hdir) if i % 2 == 0 else (1, 0)) for i in range(turns)]

    v_count = sum(dr != 0 for dr, _ in directions)
    h_count = len(directions) - v_count
    start_r = int(rng.integers(2, 5))
    end_margin = int(rng.integers(2, 5))
    max_total_v = (h - 1 - end_margin) - start_r
    min_total_v = max(2 * v_count, h // 2 - 1)
    if min_total_v > max_total_v:
        return None
    total_v = int(rng.integers(min_total_v, max_total_v + 1))
    v_steps = _partition_total(total_v, v_count, 2, 5, rng)
    if v_steps is None:
        return None

    h_steps = [int(rng.integers(2, 5)) for _ in range(h_count)]
    total_h = sum(h_steps)
    c_min, c_max = 2, w - 3
    if hdir == 1:
        lo, hi = c_min, c_max - total_h
    else:
        lo, hi = c_min + total_h, c_max
    if lo > hi:
        return None
    start_c = int(rng.integers(lo, hi + 1))

    path = [(start_r, start_c)]
    vi = 0
    hi = 0
    for dr, dc in directions:
        steps = v_steps[vi] if dr else h_steps[hi]
        if dr:
            vi += 1
        else:
            hi += 1
        r, c = path[-1]
        for _ in range(steps):
            r += dr
            c += dc
            path.append((r, c))

    if len(path) != len(set(path)):
        return None
    if any(r < 2 or r > h - 3 or c < 2 or c > w - 3 for r, c in path):
        return None
    return path


def _render_paths(shape: tuple[int, int], channel_path: list[tuple[int, int]]) -> np.ndarray | None:
    left, right = _wall_sets(channel_path)
    channel_cells = set(channel_path)
    if left & right or left & channel_cells or right & channel_cells:
        return None
    h, w = shape
    for r, c in left | right:
        if not (0 <= r < h and 0 <= c < w):
            return None

    left_mask = _mask_from_cells(shape, left)
    right_mask = _mask_from_cells(shape, right)
    if not _is_thin_component(left_mask) or not _is_thin_component(right_mask):
        return None

    path_mask = left_mask | right_mask
    channel, _ = _compute_channel(path_mask)
    if not np.all(channel[_mask_from_cells(shape, channel_path)]):
        return None
    if np.any(channel[0, :]) or np.any(channel[-1, :]) or np.any(channel[:, 0]) or np.any(channel[:, -1]):
        return None
    return path_mask


def _selected_mask(shape: tuple[int, int], axis: str, indices: list[int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if axis == "row":
        for r in indices:
            mask[r, 1 : w - 1] = True
    else:
        for c in indices:
            mask[1 : h - 1, c] = True
    return mask


def _choose_marker_indices(
    channel: np.ndarray, path_mask: np.ndarray, axis: str, rng: np.random.Generator
) -> list[int] | None:
    h, w = channel.shape
    if axis == "row":
        universe = list(range(1, h - 1))
        hit_count = channel.sum(axis=1).astype(int)
        path_count = path_mask.sum(axis=1).astype(int)
    else:
        universe = list(range(1, w - 1))
        hit_count = channel.sum(axis=0).astype(int)
        path_count = path_mask.sum(axis=0).astype(int)

    preferred_pair_count = int(rng.choice([2, 3]))
    all_valid: list[tuple[int, ...]] = []
    preferred: list[tuple[int, ...]] = []
    for pair_count in (2, 3):
        for combo in itertools.combinations(universe, pair_count):
            if any(b - a < 2 for a, b in zip(combo, combo[1:])):
                continue
            combo_hits = sum(int(hit_count[i]) > 0 for i in combo)
            combo_misses = sum(int(hit_count[i]) == 0 and int(path_count[i]) == 0 for i in combo)
            target_size = int(sum(int(hit_count[i]) for i in combo))
            if combo_hits >= 1 and combo_misses >= 1 and target_size >= 3:
                all_valid.append(combo)
                if pair_count == preferred_pair_count:
                    preferred.append(combo)

    choices = preferred if preferred else all_valid
    if not choices:
        return None
    chosen = choices[int(rng.integers(len(choices)))]
    return list(chosen)


def _paint_contents(grid: np.ndarray, channel: np.ndarray, selected: np.ndarray, rng: np.random.Generator) -> bool:
    target = channel & selected
    if int(target.sum()) < 3:
        return False

    grid[channel] = BLACK
    target_cells = [tuple(map(int, xy)) for xy in np.argwhere(target)]
    unselected_channel = [
        tuple(map(int, xy))
        for xy in np.argwhere(channel & ~selected)
    ]
    selected_only = [
        tuple(map(int, xy))
        for xy in np.argwhere(selected & ~channel & (grid != ORANGE) & (grid != YELLOW))
    ]
    if not unselected_channel or not selected_only:
        return False

    rng.shuffle(target_cells)
    grid[target_cells[0]] = BLACK
    grid[target_cells[1]] = TEAL
    grid[target_cells[2]] = PINK
    for cell in target_cells[3:]:
        grid[cell] = int(rng.choice([BLACK, BLACK, BLACK, TEAL, PINK]))

    rng.shuffle(unselected_channel)
    keep_colored = max(1, len(unselected_channel) // 4)
    for i, cell in enumerate(unselected_channel):
        grid[cell] = int(rng.choice([TEAL, PINK])) if i < keep_colored else BLACK

    rng.shuffle(selected_only)
    grid[selected_only[0]] = TEAL
    if len(selected_only) > 1:
        grid[selected_only[1]] = PINK

    return True


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        path_mask = grid == ORANGE
        channel, _ = _compute_channel(path_mask)
        axis, indices, selected = _selected_from_markers(grid)
        target = channel & selected
        selected_only = selected & ~channel & (grid != ORANGE) & (grid != YELLOW)
        unselected_channel = channel & ~selected

        if axis == "row":
            hit_sizes = [int(channel[r, :].sum()) for r in indices]
        else:
            hit_sizes = [int(channel[:, c].sum()) for c in indices]

        if not any(size > 0 for size in hit_sizes):
            return False
        if not any(size == 0 for size in hit_sizes):
            return False
        if int(target.sum()) < 3:
            return False
        if not np.any(selected_only) or not np.any(unselected_channel):
            return False
        if not np.any(np.isin(grid[selected_only], [TEAL, PINK])):
            return False
        if not np.any(np.isin(grid[unselected_channel], [TEAL, PINK])):
            return False
        if not {BLACK, TEAL, PINK}.issubset(set(map(int, np.unique(grid[target])))):
            return False

        out = main(grid)
        if not np.array_equal(out != grid, target):
            return False
        if not np.array_equal(out[target], TOGGLE[grid[target]]):
            return False
        if not np.array_equal(out[selected_only], grid[selected_only]):
            return False
        if not np.array_equal(out[unselected_channel], grid[unselected_channel]):
            return False
    except Exception:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    base_h = int(rng.integers(13, 18))
    base_w = int(rng.integers(13, 18))
    channel_path = _gen_channel_path(base_h, base_w, rng)
    if channel_path is None:
        return None

    path_mask = _render_paths((base_h, base_w), channel_path)
    if path_mask is None:
        return None
    channel, _ = _compute_channel(path_mask)

    if rng.random() < 0.5:
        path_mask = path_mask.T.copy()
        channel = channel.T.copy()

    h, w = path_mask.shape
    axis = "row" if rng.random() < 0.5 else "col"
    indices = _choose_marker_indices(channel, path_mask, axis, rng)
    if indices is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[path_mask] = ORANGE
    if axis == "row":
        for r in indices:
            grid[r, 0] = YELLOW
            grid[r, w - 1] = YELLOW
    else:
        for c in indices:
            grid[0, c] = YELLOW
            grid[h - 1, c] = YELLOW

    selected = _selected_mask((h, w), axis, indices)
    if not _paint_contents(grid, channel, selected, rng):
        return None
    if not _semantic_ok(grid):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap colors only where opposite-edge marker segments cross the dilation-overlap channel between two orange paths.

    Input structure:
    - The background is black, with exactly two 4-connected one-pixel-wide orange path components.
    - One pair of opposite grid edges carries 2-3 aligned yellow marker pairs, selecting full rows or full columns between the paired endpoints.
    - The gap around the two paths contains black, teal, and pink cells; only some of them lie in the geometric overlap channel.

    Rule:
    - Dilate each orange path by one cell in the 8-neighborhood, intersect the two dilations, and remove the orange path cells themselves to get the middle channel.
    - Read every yellow opposite-edge pair as one selected line segment: left/right pairs select a row interior, top/bottom pairs select a column interior.
    - Recolor only the intersection of the channel and the union of all selected segments, using black <-> teal and pink <-> orange.
    - Keep the yellow markers, the orange paths, unselected channel cells, and selected non-channel cells unchanged.

    Key insight:
    - The changed pixels are defined by the intersection of two independently inferred structures: the overlap channel from the two path halos and the straight segments chosen by the yellow edge markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    channel, _ = _compute_channel(g == ORANGE)
    _axis, _indices, selected = _selected_from_markers(g)
    target = channel & selected
    target_colors = set(map(int, np.unique(g[target])))
    assert target_colors.issubset({BLACK, PINK, ORANGE, TEAL}), "target cells must use toggle colors"

    out = g.copy()
    out[target] = TOGGLE[g[target]]
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 043/2")
