from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MARKER = 8

INPUT_COLORS = (1, 2, 3, 6, 7, 9)
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
BOX_H_RANGE = (6, 10)
BOX_W_RANGE = (6, 10)
MARKER_COUNT_RANGE = (2, 4)
GEN_RETRIES = 500
MIN_INSIDE_BLACK = 5
MIN_TOTAL_FOREGROUND = 30
MAX_TOTAL_FOREGROUND = 50


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _inside_color_counts(window: np.ndarray) -> Counter[int]:
    return Counter(int(v) for v in window.flat if int(v) not in (BLACK, MARKER))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the mixed foreground inside the sparse marker bounding box by local majority versus non-majority.

    Input structure:
    - The grid is a black background with a mixed foreground made from three non-black,
      non-marker colors plus 2-4 sparse cyan marker pixels.
    - The cyan markers define a bounding rectangle and are not part of the counted
      foreground colors.
    - Foreground pixels also exist outside the marker rectangle, so the counted region
      is only the marker bounding box.

    Rule:
    - Find the bounding box of all cyan marker pixels.
    - Inside that box, count each non-black, non-cyan color and identify the unique
      most frequent color.
    - Recolor all pixels of that majority color inside the box to yellow, recolor every
      other non-black, non-cyan pixel inside the box to grey, and keep black pixels,
      cyan markers, and every pixel outside the box unchanged.

    Key insight:
    - The markers decide where counting applies, and the local color frequencies inside
      that box decide which color gets the majority highlight; neither step can be
      skipped without changing the answer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_mask = g == MARKER
    marker_bbox = _bbox(marker_mask)
    assert marker_bbox is not None, "expected at least one marker"
    r0, c0, r1, c1 = marker_bbox

    window = g[r0 : r1 + 1, c0 : c1 + 1]
    counts = _inside_color_counts(window)
    assert len(counts) >= 2, "expected at least two counted colors in the box"

    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    assert len(winners) == 1, "expected a unique local majority color"
    majority = winners[0]

    out = g.copy()
    box = out[r0 : r1 + 1, c0 : c1 + 1]
    for color in counts:
        if color == majority:
            box[box == color] = YELLOW
        else:
            box[box == color] = GREY
    return out


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _choose_marker_points(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    layouts = [
        [(r0, c0), (r1, c1)],
        [(r0, c1), (r1, c0)],
        [(r0, c0), (r1, c1), (r0, mid_c)],
        [(r0, c1), (r1, c0), (mid_r, c1)],
        [(r0, c0), (r1, c1), (r1, mid_c)],
        [(r0, c1), (r1, c0), (mid_r, c0)],
        [(r0, c0), (r0, c1), (r1, c0), (r1, c1)],
        [(r0, mid_c), (r1, mid_c), (mid_r, c0), (mid_r, c1)],
    ]
    marker_count = _randint(*MARKER_COUNT_RANGE)
    candidates = [layout for layout in layouts if len(layout) == marker_count]
    return candidates[_randint(0, len(candidates) - 1)]


def _sample_total_cells(
    candidates: list[tuple[int, int]], total: int, prefer_cluster: bool
) -> list[tuple[int, int]] | None:
    if total > len(candidates):
        return None
    if not prefer_cluster:
        pick = np.random.choice(len(candidates), size=total, replace=False)
        return [candidates[int(i)] for i in pick]

    allowed = set(candidates)
    chosen = {candidates[_randint(0, len(candidates) - 1)]}
    frontier = {
        cell
        for r, c in chosen
        for cell in _neighbors8(r, c)
        if cell in allowed and cell not in chosen
    }

    while len(chosen) < total:
        pool = list(frontier) if frontier and np.random.random() < 0.75 else list(allowed - chosen)
        if not pool:
            return None
        cell = pool[_randint(0, len(pool) - 1)]
        chosen.add(cell)
        frontier.discard(cell)
        for neighbor in _neighbors8(*cell):
            if neighbor in allowed and neighbor not in chosen:
                frontier.add(neighbor)
    return list(chosen)


def _assign_colors(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    colors: list[int],
    counts: list[int],
) -> None:
    shuffled = cells.copy()
    np.random.shuffle(shuffled)
    start = 0
    for color, count in zip(colors, counts):
        for r, c in shuffled[start : start + count]:
            grid[r, c] = color
        start += count


def _sample_inside_counts(capacity: int) -> list[int] | None:
    if capacity < MIN_INSIDE_BLACK + 9:
        return None
    upper = min(11, capacity - MIN_INSIDE_BLACK - 2)
    if upper < 4:
        return None
    options = list(range(2, upper + 1))
    for _ in range(200):
        vals = np.random.choice(options, size=3, replace=False)
        counts = sorted((int(v) for v in vals), reverse=True)
        total = sum(counts)
        if total > capacity - MIN_INSIDE_BLACK:
            continue
        if total < min(18, capacity - MIN_INSIDE_BLACK):
            continue
        return counts
    return None


def _pick_outside_counts(
    inside_counts: list[int],
    capacity: int,
    require_global_flip: bool,
) -> list[int] | None:
    if capacity < 6:
        return None
    winner_idx = 0
    for _ in range(300):
        if require_global_flip:
            target_idx = _randint(1, 2)
            out = [_randint(1, 4), _randint(1, 4), _randint(1, 4)]
            out[target_idx] += _randint(4, 9)
        else:
            out = [_randint(3, 8), _randint(1, 5), _randint(1, 5)]
        total = sum(out)
        if total > capacity:
            continue
        if total < 8:
            continue
        global_counts = [inside_counts[i] + out[i] for i in range(3)]
        winner_total = max(global_counts)
        winners = [i for i, count in enumerate(global_counts) if count == winner_total]
        if len(winners) != 1:
            continue
        flipped = winners[0] != winner_idx
        if flipped != require_global_flip:
            continue
        if out[winner_idx] <= 0:
            continue
        if out[1] + out[2] <= 0:
            continue
        return out
    return None


def _global_majority(colors: list[int], grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.flat if int(v) not in (BLACK, MARKER))
    return max(colors, key=lambda color: counts[color])


def _validate_example(grid: np.ndarray, expect_flip: bool) -> bool:
    marker_mask = grid == MARKER
    marker_count = int(np.count_nonzero(marker_mask))
    if not MARKER_COUNT_RANGE[0] <= marker_count <= MARKER_COUNT_RANGE[1]:
        return False

    box = _bbox(marker_mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box

    # Markers must be sparse with 4-neighbor clearance from every foreground pixel.
    for r, c in np.argwhere(marker_mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if int(grid[nr, nc]) not in (BLACK, MARKER):
                    return False

    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    counts = _inside_color_counts(window)
    if len(counts) != 3:
        return False
    if len(set(counts.values())) != 3:
        return False
    if int(np.count_nonzero(window == BLACK)) < MIN_INSIDE_BLACK:
        return False

    outside_mask = np.ones(grid.shape, dtype=bool)
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    outside_fg = (grid != BLACK) & (grid != MARKER) & outside_mask
    if int(np.count_nonzero(outside_fg)) == 0:
        return False

    colors_by_count = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    local_colors = [color for color, _count in colors_by_count]
    local_majority = local_colors[0]
    if int(np.count_nonzero((grid == local_majority) & outside_mask)) == 0:
        return False
    if sum(int(np.count_nonzero((grid == color) & outside_mask)) for color in local_colors[1:]) == 0:
        return False

    global_majority = _global_majority(local_colors, grid)
    if (global_majority != local_majority) != expect_flip:
        return False

    out = main(grid)
    if not np.array_equal(out[outside_mask], grid[outside_mask]):
        return False
    out_window = out[r0 : r1 + 1, c0 : c1 + 1]
    if int(np.count_nonzero(out_window == YELLOW)) != counts[local_majority]:
        return False
    if int(np.count_nonzero(out_window == GREY)) != sum(counts[color] for color in local_colors[1:]):
        return False
    for color in local_colors:
        target = YELLOW if color == local_majority else GREY
        if not np.all(out_window[window == color] == target):
            return False

    changed = out != grid
    if int(np.count_nonzero(changed)) != sum(counts.values()):
        return False
    return True


def _build_attempt() -> np.ndarray | None:
    h = _randint(*GRID_H_RANGE)
    w = _randint(*GRID_W_RANGE)

    box_h = _randint(BOX_H_RANGE[0], min(BOX_H_RANGE[1], h - 2))
    box_w = _randint(BOX_W_RANGE[0], min(BOX_W_RANGE[1], w - 2))
    r0 = _randint(1, h - box_h - 1)
    c0 = _randint(1, w - box_w - 1)
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1

    grid = np.zeros((h, w), dtype=int)
    marker_points = _choose_marker_points(r0, c0, r1, c1)
    for r, c in marker_points:
        grid[r, c] = MARKER

    forbidden: set[tuple[int, int]] = set(marker_points)
    for r, c in marker_points:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                forbidden.add((nr, nc))

    inside_candidates = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in forbidden
    ]
    inside_counts = _sample_inside_counts(len(inside_candidates))
    if inside_counts is None:
        return None

    pad_top = _randint(1, 3)
    pad_bottom = _randint(1, 3)
    pad_left = _randint(1, 3)
    pad_right = _randint(1, 3)
    zone_r0 = max(0, r0 - pad_top)
    zone_c0 = max(0, c0 - pad_left)
    zone_r1 = min(h - 1, r1 + pad_bottom)
    zone_c1 = min(w - 1, c1 + pad_right)
    outside_candidates = [
        (r, c)
        for r in range(zone_r0, zone_r1 + 1)
        for c in range(zone_c0, zone_c1 + 1)
        if not (r0 <= r <= r1 and c0 <= c <= c1) and (r, c) not in forbidden
    ]

    expect_flip = bool(_randint(0, 1))
    outside_counts = _pick_outside_counts(inside_counts, len(outside_candidates), expect_flip)
    if outside_counts is None:
        return None

    total_foreground = sum(inside_counts) + sum(outside_counts)
    if not MIN_TOTAL_FOREGROUND <= total_foreground <= MAX_TOTAL_FOREGROUND:
        return None

    inside_cells = _sample_total_cells(
        inside_candidates, sum(inside_counts), prefer_cluster=bool(_randint(0, 1))
    )
    outside_cells = _sample_total_cells(
        outside_candidates, sum(outside_counts), prefer_cluster=bool(_randint(0, 1))
    )
    if inside_cells is None or outside_cells is None:
        return None

    colors = list(np.random.choice(np.array(INPUT_COLORS, dtype=int), size=3, replace=False))
    _assign_colors(grid, inside_cells, colors, inside_counts)
    _assign_colors(grid, outside_cells, colors, outside_counts)

    if not _validate_example(grid, expect_flip):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse-marker majority-highlight example")
