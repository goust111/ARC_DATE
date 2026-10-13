from __future__ import annotations

from collections import deque
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((1, 0), (-1, 0), (0, 1), (0, -1))
    return (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    dirs = _neighbors(connectivity)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _find_filled_rectangle(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _components(grid != bg, connectivity=4):
        area = int(comp.shape[0])
        r0, c0, r1, c1 = _bbox(comp)
        if area != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] != bg):
            candidates.append((area, (r0, c0, r1, c1)))
    assert candidates
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _row_dominant_color(row: np.ndarray) -> int:
    values, counts = np.unique(row, return_counts=True)
    order = np.lexsort((values, -counts))
    return int(values[order[0]])


def _count_true_components(mask: np.ndarray, connectivity: int = 8) -> int:
    if not bool(mask.any()):
        return 0
    return len(_components(mask.astype(bool), connectivity=connectivity))


def _mask_span(mask: np.ndarray) -> int:
    cols = np.flatnonzero(mask.any(axis=0))
    if cols.size == 0:
        return 0
    return int(cols[-1] - cols[0] + 1)


def _adjacency_score(mask: np.ndarray) -> int:
    h, w = mask.shape
    score = 0
    for r in range(h - 1):
        upper = np.flatnonzero(mask[r])
        lower = mask[r + 1]
        for c in upper:
            lo = max(0, int(c) - 1)
            hi = min(w, int(c) + 2)
            if bool(lower[lo:hi].any()):
                score += 1
    return int(score)


def _edge_hits(mask: np.ndarray) -> int:
    if mask.shape[1] == 0:
        return 0
    return int(mask[:, 0].sum() + mask[:, -1].sum())


def _undo_shift(rows: np.ndarray, step: int) -> np.ndarray:
    corrected = np.zeros_like(rows, dtype=bool)
    for k in range(rows.shape[0]):
        corrected[k] = np.roll(rows[k], -k * int(step))
    return corrected


def _infer_shift(rows: np.ndarray) -> int:
    h, w = rows.shape
    if h <= 1 or w <= 1:
        return 0
    best_score: tuple[int, int, int, int, int] | None = None
    best_step = 0
    for step in range(w):
        corrected = _undo_shift(rows, step)
        score = (
            _mask_span(corrected),
            _count_true_components(corrected, connectivity=8),
            _edge_hits(corrected),
            -_adjacency_score(corrected),
            int(step),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_step = int(step)
    return best_step


def _indicator_order(
    grid: np.ndarray,
    bg: int,
    rect_bbox: tuple[int, int, int, int],
    stripe_colors: set[int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = rect_bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    coords = np.argwhere((grid != bg) & outside)
    assert coords.shape[0] == 2
    coords = coords[np.argsort(coords[:, 0])]
    top = tuple(map(int, coords[0]))
    bottom = tuple(map(int, coords[1]))
    assert top[1] == bottom[1] and bottom[0] == top[0] + 1
    top_color = int(grid[top])
    bottom_color = int(grid[bottom])
    assert {top_color, bottom_color} == stripe_colors
    return top_color, bottom_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the alternating stripe rectangle into two de-shifted marker layers and stack them by the corner indicator.

    Input structure:
    - A black background surrounds one filled non-background rectangle made of alternating horizontal stripes.
    - Each stripe has one dominant stripe color, with some cells replaced by a shared marker color.
    - Outside the rectangle, a separate 2x1 vertical indicator uses the two stripe colors to encode output order.

    Rule:
    - Find the filled rectangle, infer the two stripe colors from the row-wise dominant colors, and treat the remaining rectangle color as the marker color.
    - Separate rows by stripe color in top-to-bottom order; for each color, infer the constant per-row cyclic horizontal drift that makes the marker rows most compact when undone.
    - Shift the k-th row of each color layer left by k times that drift, keep only marker cells, and stack the two recovered layers following the indicator's top-to-bottom color order.

    Key insight:
    - The visible rectangle interleaves two hidden images, and each one becomes a compact sprite only after undoing its arithmetic row-to-row horizontal offset.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    rect_bbox = _find_filled_rectangle(g, bg)
    r0, c0, r1, c1 = rect_bbox
    rect = g[r0 : r1 + 1, c0 : c1 + 1]

    row_dominants = [_row_dominant_color(rect[r]) for r in range(rect.shape[0])]
    stripe_colors = list(dict.fromkeys(row_dominants))
    assert len(stripe_colors) == 2
    assert all(row_dominants[r] != row_dominants[r - 1] for r in range(1, len(row_dominants)))

    marker_candidates = set(int(v) for v in np.unique(rect)) - set(stripe_colors)
    assert len(marker_candidates) == 1
    marker = marker_candidates.pop()

    layers: dict[int, np.ndarray] = {}
    for color in stripe_colors:
        rows = np.asarray([rect[r] == marker for r in range(rect.shape[0]) if row_dominants[r] == color], dtype=bool)
        step = _infer_shift(rows)
        layers[int(color)] = _undo_shift(rows, step)

    ordered_colors = _indicator_order(g, bg, rect_bbox, set(stripe_colors))
    parts: list[np.ndarray] = []
    for color in ordered_colors:
        mask = layers[int(color)]
        part = np.full(mask.shape, bg, dtype=int)
        part[mask] = int(marker)
        parts.append(part)
    return np.vstack(parts)


# --------------- generator ---------------

BG = 0
COLOR_POOL = tuple(range(1, 10))

RECT_HEIGHT_RANGE = (6, 12)
RECT_WIDTH_RANGE = (10, 18)
MAX_SHIFT = 4

MARGIN_MIN = 1
MARGIN_MAX = 4
GENERATION_RETRIES = 500


def _sample_interval_shape(height: int, rect_width: int, blob_width: int) -> list[tuple[int, int]]:
    max_len = min(blob_width, max(1, rect_width // 2 - 1))
    for _ in range(200):
        intervals: list[tuple[int, int]] = []
        length = random.randint(1, max_len)
        start = random.randint(0, blob_width - length)
        intervals.append((start, start + length - 1))
        for _row in range(1, height):
            length = random.randint(1, max_len)
            prev_l, prev_r = intervals[-1]
            lo = max(0, prev_l - length)
            hi = min(blob_width - length, prev_r + 1)
            if lo > hi:
                break
            start = random.randint(lo, hi)
            intervals.append((start, start + length - 1))
        if len(intervals) != height:
            continue
        if len(set(intervals)) == 1:
            continue
        span = max(b for _a, b in intervals) - min(a for a, _b in intervals) + 1
        if span < 2:
            continue
        return intervals
    raise RuntimeError("failed to build interval shape")


def _build_aligned_layer(height: int, width: int) -> np.ndarray:
    max_blob_width = min(7, width - 2, max(3, width // 2))
    min_blob_width = min(3, max_blob_width)
    blob_width = random.randint(min_blob_width, max_blob_width)
    intervals = _sample_interval_shape(height, width, blob_width)
    x0 = random.randint(0, width - blob_width)
    mask = np.zeros((height, width), dtype=bool)
    for r, (a, b) in enumerate(intervals):
        mask[r, x0 + a : x0 + b + 1] = True
    return mask


def _count_rows(total_rows: int, start_idx: int, color_idx: int) -> int:
    return sum(1 for r in range(total_rows) if (start_idx + r) % 2 == color_idx)


def _choose_shift(height: int, width: int) -> int:
    if height <= 1:
        return 0
    return random.randint(1, min(MAX_SHIFT, width - 1))


def _choose_margins(corner: str) -> tuple[int, int, int, int]:
    top_min = 2 if corner in ("tl", "tr") else 1
    bottom_min = 2 if corner in ("bl", "br") else 1
    left_min = 2 if corner in ("tl", "bl") else 1
    right_min = 2 if corner in ("tr", "br") else 1
    top = random.randint(top_min, MARGIN_MAX)
    bottom = random.randint(bottom_min, MARGIN_MAX)
    left = random.randint(left_min, MARGIN_MAX)
    right = random.randint(right_min, MARGIN_MAX)
    return top, bottom, left, right


def _place_indicator(
    grid: np.ndarray,
    corner: str,
    top_color: int,
    bottom_color: int,
) -> None:
    h, w = grid.shape
    if corner == "tl":
        cells = ((0, 0), (1, 0))
    elif corner == "tr":
        cells = ((0, w - 1), (1, w - 1))
    elif corner == "bl":
        cells = ((h - 2, 0), (h - 1, 0))
    else:
        cells = ((h - 2, w - 1), (h - 1, w - 1))
    grid[cells[0]] = int(top_color)
    grid[cells[1]] = int(bottom_color)


def _render_input(
    rect_height: int,
    rect_width: int,
    stripe_colors: tuple[int, int],
    marker: int,
    start_idx: int,
    layers: dict[int, np.ndarray],
    shifts: dict[int, int],
) -> np.ndarray:
    rect = np.empty((rect_height, rect_width), dtype=int)
    seen_per_color = {stripe_colors[0]: 0, stripe_colors[1]: 0}
    for r in range(rect_height):
        color = stripe_colors[(start_idx + r) % 2]
        row_idx = seen_per_color[int(color)]
        seen_per_color[int(color)] += 1
        rect[r, :] = int(color)
        shifted = np.roll(layers[int(color)][row_idx], row_idx * shifts[int(color)])
        rect[r, shifted] = int(marker)
    return rect


def _expected_output(
    width: int,
    bg: int,
    marker: int,
    layers: dict[int, np.ndarray],
    ordered_colors: tuple[int, int],
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for color in ordered_colors:
        mask = layers[int(color)]
        part = np.full((mask.shape[0], width), bg, dtype=int)
        part[mask] = int(marker)
        parts.append(part)
    return np.vstack(parts)


def _gen_attempt() -> np.ndarray | None:
    rect_height = random.randint(*RECT_HEIGHT_RANGE)
    rect_width = random.randint(*RECT_WIDTH_RANGE)
    start_idx = random.randint(0, 1)
    stripe_colors = tuple(random.sample(COLOR_POOL, 2))
    marker = random.choice([c for c in COLOR_POOL if c not in stripe_colors])
    corner = random.choice(("tl", "tr", "bl", "br"))
    margins = _choose_margins(corner)
    top_margin, bottom_margin, left_margin, right_margin = margins
    grid_height = rect_height + top_margin + bottom_margin
    grid_width = rect_width + left_margin + right_margin
    if grid_height > 30 or grid_width > 30:
        return None

    layer_heights = {
        stripe_colors[0]: _count_rows(rect_height, start_idx, 0),
        stripe_colors[1]: _count_rows(rect_height, start_idx, 1),
    }
    if min(layer_heights.values()) < 3:
        return None

    layers = {
        color: _build_aligned_layer(layer_heights[color], rect_width)
        for color in stripe_colors
    }
    shifts = {
        color: _choose_shift(layer_heights[color], rect_width)
        for color in stripe_colors
    }
    rect = _render_input(rect_height, rect_width, stripe_colors, marker, start_idx, layers, shifts)

    grid = np.full((grid_height, grid_width), BG, dtype=int)
    r0 = top_margin
    c0 = left_margin
    grid[r0 : r0 + rect_height, c0 : c0 + rect_width] = rect

    ordered_colors = (
        random.choice(stripe_colors),
        stripe_colors[0] if random.choice((True, False)) else stripe_colors[1],
    )
    if ordered_colors[0] == ordered_colors[1]:
        ordered_colors = (ordered_colors[0], stripe_colors[0] if ordered_colors[0] == stripe_colors[1] else stripe_colors[1])
    _place_indicator(grid, corner, ordered_colors[0], ordered_colors[1])

    expected = _expected_output(rect_width, BG, marker, layers, ordered_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if actual.shape != expected.shape:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
