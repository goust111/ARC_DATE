from __future__ import annotations

import random
from collections import Counter

import numpy as np


MAX_GRID_SIZE = 30
MIN_PANEL_SIZE = 4
MAX_GENERATION_ATTEMPTS = 400
COLOR_VALUES = tuple(range(10))


def _spans_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    assert idx.size > 0
    spans: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        spans.append((start, prev))
        start = prev = value
    spans.append((start, prev))
    return spans


def _panel_layout_for_separator(
    grid: np.ndarray, sep: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    row_is_sep = np.all(grid == sep, axis=1)
    col_is_sep = np.all(grid == sep, axis=0)
    if not row_is_sep.any() or not col_is_sep.any():
        return None
    if row_is_sep.all() or col_is_sep.all():
        return None

    row_spans = _spans_from_mask(~row_is_sep)
    col_spans = _spans_from_mask(~col_is_sep)
    if len(col_spans) < 3:
        return None

    row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
    if len(row_sizes) != 1 or len(col_sizes) != 1:
        return None
    return row_spans, col_spans


def _odd_panel_indices(
    grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]
) -> list[int]:
    odd_indices: list[int] = []
    for r0, r1 in row_spans:
        panels = [grid[r0 : r1 + 1, c0 : c1 + 1] for c0, c1 in col_spans]
        signatures = [panel.tobytes() for panel in panels]
        counts = Counter(signatures)
        if len(counts) != 2:
            raise AssertionError("each panel row must contain exactly two panel variants")
        if sorted(counts.values()) != [1, len(panels) - 1]:
            raise AssertionError("each panel row must contain one unique odd panel")
        odd_indices.append(next(i for i, sig in enumerate(signatures) if counts[sig] == 1))
    return odd_indices


def _solve_layout(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    candidates: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], list[int]]] = []
    for sep in map(int, np.unique(grid)):
        layout = _panel_layout_for_separator(grid, sep)
        if layout is None:
            continue
        row_spans, col_spans = layout
        non_sep = grid[grid != sep]
        if non_sep.size == 0:
            continue
        bg = int(Counter(map(int, non_sep)).most_common(1)[0][0])
        try:
            odd_indices = _odd_panel_indices(grid, row_spans, col_spans)
        except AssertionError:
            continue
        candidates.append((sep, bg, row_spans, col_spans, odd_indices))

    assert len(candidates) == 1, "expected a unique separator/layout interpretation"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the odd panel in each panel-row and overlay all odd panels onto one panel-sized canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    _sep, bg, row_spans, col_spans, odd_indices = _solve_layout(g)
    panel_h = row_spans[0][1] - row_spans[0][0] + 1
    panel_w = col_spans[0][1] - col_spans[0][0] + 1
    out = np.full((panel_h, panel_w), bg, dtype=int)

    for (r0, r1), odd_idx in zip(row_spans, odd_indices):
        c0, c1 = col_spans[odd_idx]
        odd_panel = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = odd_panel != bg
        out[mask] = odd_panel[mask]

    return out


def _layout_spans(count: int, size: int, sep_width: int, outer_border: bool) -> tuple[list[tuple[int, int]], int]:
    cursor = sep_width if outer_border else 0
    spans: list[tuple[int, int]] = []
    for index in range(count):
        spans.append((cursor, cursor + size - 1))
        cursor += size
        if index != count - 1 or outer_border:
            cursor += sep_width
    return spans, cursor


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _grow_connected_mask(allowed: np.ndarray, target: int) -> np.ndarray:
    h, w = allowed.shape
    choices = [tuple(map(int, coord)) for coord in np.argwhere(allowed)]
    if not choices:
        return np.zeros_like(allowed, dtype=bool)

    start = random.choice(choices)
    mask = np.zeros_like(allowed, dtype=bool)
    mask[start] = True
    frontier = {cell for cell in _neighbors4(start[0], start[1], h, w) if allowed[cell]}

    while int(mask.sum()) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nr, nc in _neighbors4(cell[0], cell[1], h, w):
            if allowed[nr, nc] and not mask[nr, nc]:
                frontier.add((nr, nc))

    return mask


def _box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _reserved_boxes(
    panel_h: int, panel_w: int, rows: int, overlap_mode: bool
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int]]:
    box_h = max(2, min(3, panel_h // 2))
    box_w = max(2, min(3, panel_w // 2))
    corners = [
        (0, 0, box_h - 1, box_w - 1),
        (0, panel_w - box_w, box_h - 1, panel_w - 1),
        (panel_h - box_h, 0, panel_h - 1, box_w - 1),
        (panel_h - box_h, panel_w - box_w, panel_h - 1, panel_w - 1),
    ]
    if overlap_mode:
        shared_r0 = (panel_h - box_h) // 2
        shared_c0 = (panel_w - box_w) // 2
        shared = (shared_r0, shared_c0, shared_r0 + box_h - 1, shared_c0 + box_w - 1)
        return [shared for _ in range(rows)], shared

    random.shuffle(corners)
    chosen = corners[:rows]
    return chosen, chosen[0]


def _paint_random_objects(
    panel: np.ndarray, bg: int, colors: list[int], forbidden: np.ndarray, max_fill: int
) -> None:
    h, w = panel.shape
    occupied = forbidden.copy()
    filled = 0
    object_count = random.randint(0, 2)
    for _ in range(object_count):
        remaining = max_fill - filled
        if remaining <= 0:
            break
        target = random.randint(1, min(4, remaining))
        allowed = ~occupied
        mask = _grow_connected_mask(allowed, target)
        size = int(mask.sum())
        if size == 0:
            break
        panel[mask] = random.choice(colors)
        occupied |= mask
        filled += size

    if filled == 0 and random.random() < 0.35:
        allowed = ~occupied
        cells = [tuple(map(int, coord)) for coord in np.argwhere(allowed)]
        if cells:
            r, c = random.choice(cells)
            panel[r, c] = random.choice(colors)

    assert Counter(map(int, panel.reshape(-1))).most_common(1)[0][0] == bg


def _build_patch(box: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    box_mask = _box_mask(shape, box)
    box_area = int(box_mask.sum())
    target = random.randint(1, max(1, min(4, box_area)))
    mask = _grow_connected_mask(box_mask, target)
    assert mask.any()
    return mask


def _make_row_panels(
    panel_h: int,
    panel_w: int,
    bg: int,
    motif_colors: list[int],
    reserved_box: tuple[int, int, int, int],
    shared_patch: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    common = np.full((panel_h, panel_w), bg, dtype=int)
    forbidden = _box_mask((panel_h, panel_w), reserved_box)
    max_common_fill = max(1, (panel_h * panel_w) // 4)
    _paint_random_objects(common, bg, motif_colors, forbidden, max_common_fill)

    odd = common.copy()
    patch_mask = shared_patch if shared_patch is not None else _build_patch(reserved_box, (panel_h, panel_w))
    patch_color = random.choice(motif_colors)
    odd[patch_mask] = patch_color
    assert not np.array_equal(common, odd)
    return common, odd


def _expected_overlay(odd_panels: list[np.ndarray], bg: int) -> np.ndarray:
    out = np.full_like(odd_panels[0], bg)
    for panel in odd_panels:
        mask = panel != bg
        out[mask] = panel[mask]
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(2, 4)
    cols = random.randint(3, 5)
    sep_width = random.randint(1, 2)
    outer_border = random.random() < 0.5
    overlap_mode = random.random() < 0.5

    max_panel_h = (MAX_GRID_SIZE - sep_width * ((rows - 1) + (2 if outer_border else 0))) // rows
    max_panel_w = (MAX_GRID_SIZE - sep_width * ((cols - 1) + (2 if outer_border else 0))) // cols
    if max_panel_h < MIN_PANEL_SIZE or max_panel_w < MIN_PANEL_SIZE:
        return None

    panel_h = random.randint(MIN_PANEL_SIZE, min(7, max_panel_h))
    panel_w = random.randint(MIN_PANEL_SIZE, min(7, max_panel_w))

    row_spans, total_h = _layout_spans(rows, panel_h, sep_width, outer_border)
    col_spans, total_w = _layout_spans(cols, panel_w, sep_width, outer_border)
    if total_h > MAX_GRID_SIZE or total_w > MAX_GRID_SIZE:
        return None

    sep, bg, *rest = random.sample(COLOR_VALUES, 6)
    motif_colors = list(rest)

    grid = np.full((total_h, total_w), sep, dtype=int)
    reserved_boxes, shared_box = _reserved_boxes(panel_h, panel_w, rows, overlap_mode)
    shared_patch = _build_patch(shared_box, (panel_h, panel_w)) if overlap_mode else None

    odd_panels: list[np.ndarray] = []
    for row_index, (r0, r1) in enumerate(row_spans):
        common, odd = _make_row_panels(
            panel_h=panel_h,
            panel_w=panel_w,
            bg=bg,
            motif_colors=motif_colors,
            reserved_box=reserved_boxes[row_index],
            shared_patch=shared_patch,
        )
        odd_panels.append(odd)
        odd_col = random.randrange(cols)
        for col_index, (c0, c1) in enumerate(col_spans):
            grid[r0 : r1 + 1, c0 : c1 + 1] = odd if col_index == odd_col else common

    expected = _expected_overlay(odd_panels, bg)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    sig = {grid.shape, expected.shape}
    if len(sig) < 2 and rows == 2 and cols == 3 and sep_width == 1 and not outer_border:
        # Bias away from the smallest/most repetitive layout.
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
