from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
AZURE = 8
LEGEND_INK = 1
COLOR_POOL = (2, 3, 4, 6, 7, 9)

STEP_DIRS = [(-2, 0), (2, 0), (0, -2), (0, 2)]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PATTERN_TO_AXIS = {
    frozenset({(0, 1), (1, 1), (2, 1)}): "vertical",
    frozenset({(1, 0), (1, 1), (1, 2)}): "horizontal",
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}): "both",
}
AXES = ("vertical", "horizontal", "both")

LEGEND_UNIT_W = 3
LEGEND_UNIT_H = 4
GEN_LEGEND_GAPS = (1, 2, 3)
CONTENT_GAP = 1

GRID_HEIGHT_RANGE = (21, 29)
GRID_WIDTH_RANGE = (21, 29)
MARKER_COUNT_RANGE = (1, 4)
SELECTED_MARKER_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _checkerboard_pattern(phase: int) -> np.ndarray:
    arr = np.full((3, 3), BG, dtype=int)
    for r in range(3):
        for c in range(3):
            if (r + c) % 2 == phase:
                arr[r, c] = AZURE
    return arr


def _is_checkerboard(sub: np.ndarray) -> bool:
    if sub.shape != (3, 3):
        return False
    if set(map(int, np.unique(sub))) - {BG, AZURE}:
        return False
    return np.array_equal(sub, _checkerboard_pattern(0)) or np.array_equal(sub, _checkerboard_pattern(1))


def _find_checkerboard(g: np.ndarray) -> tuple[int, int]:
    h, w = g.shape
    candidates = []
    for r in range(h - 2):
        for c in range(w - 2):
            if _is_checkerboard(g[r : r + 3, c : c + 3]):
                candidates.append((r, c))
    assert len(candidates) == 1, "expected exactly one 3x3 azure checkerboard"
    return candidates[0]


def _find_rings(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    rings = []
    for r in range(h - 2):
        for c in range(w - 2):
            sub = g[r : r + 3, c : c + 3]
            if int(sub[1, 1]) != BG:
                continue
            border = [int(sub[rr, cc]) for rr in range(3) for cc in range(3) if (rr, cc) != (1, 1)]
            if border[0] == BG:
                continue
            if len(set(border)) != 1:
                continue
            cells = {(r + rr, c + cc) for rr in range(3) for cc in range(3)}
            ring_cells = cells - {(r + 1, c + 1)}
            rings.append(
                {
                    "top": r,
                    "left": c,
                    "color": border[0],
                    "cells": ring_cells,
                    "bbox": (r, c, r + 2, c + 2),
                }
            )
    seen = {(ring["top"], ring["left"]) for ring in rings}
    assert len(seen) == len(rings), "duplicate ring detection"
    return rings


def _adjacent_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out - cells


def _parse_legend(g: np.ndarray) -> tuple[dict[int, str], np.ndarray]:
    h, w = g.shape
    best_entries: list[tuple[int, int, str]] = []
    best_row = -1
    for r in range(LEGEND_UNIT_H - 1, h):
        entries: list[tuple[int, int, str]] = []
        for c in range(w - LEGEND_UNIT_W + 1):
            swatch = g[r, c : c + LEGEND_UNIT_W]
            color = int(swatch[0])
            if color == BG or not np.all(swatch == color):
                continue
            sym = g[r - 3 : r, c : c + 3]
            mask = frozenset(map(tuple, np.argwhere(sym != BG)))
            vals = {int(v) for v in np.unique(sym) if int(v) != BG}
            axis = PATTERN_TO_AXIS.get(mask)
            if axis is None or len(vals) != 1:
                continue
            entries.append((c, color, axis))
        non_overlapping = []
        last_end = -1
        for c, color, axis in sorted(entries):
            if c <= last_end:
                continue
            non_overlapping.append((c, color, axis))
            last_end = c + LEGEND_UNIT_W - 1
        if len(non_overlapping) > len(best_entries) or (
            len(non_overlapping) == len(best_entries) and r > best_row
        ):
            best_entries = non_overlapping
            best_row = r
    assert len(best_entries) >= 3, "expected a bottom legend with at least three entries"
    mapping: dict[int, str] = {}
    mask = np.zeros_like(g, dtype=bool)
    for c, color, axis in best_entries:
        assert color not in mapping, "legend colors must be unique"
        mapping[color] = axis
        mask[best_row - 3 : best_row + 1, c : c + 3] = True
    return mapping, mask


def _bool_mask(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in cells:
        out[r, c] = True
    return out


def _trace_guide_component(
    g: np.ndarray,
    start: tuple[int, int],
    color: int,
    blocked: np.ndarray,
) -> set[tuple[int, int]]:
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in STEP_DIRS:
            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc):
                continue
            if blocked[nr, nc]:
                continue
            if int(g[nr, nc]) != color:
                continue
            if (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def _degree_in_component(cell: tuple[int, int], comp: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in comp for dr, dc in STEP_DIRS)


def _reflect_targets(
    r: int,
    c: int,
    axis_r: int,
    axis_c: int,
    axis_type: str,
) -> set[tuple[int, int]]:
    if axis_type == "vertical":
        return {(r, c), (r, 2 * axis_c - c)}
    if axis_type == "horizontal":
        return {(r, c), (2 * axis_r - r, c)}
    assert axis_type == "both"
    rows = (r, 2 * axis_r - r)
    cols = (c, 2 * axis_c - c)
    return {(rr, cc) for rr in rows for cc in cols}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the checkerboard's guide path to choose a ring color, look that color
    up in the bottom legend, then mirror-complete only sparse markers of that color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "grid center must lie on a full row and column"

    src_r, src_c = _find_checkerboard(g)
    source_cells = {(src_r + r, src_c + c) for r in range(3) for c in range(3)}
    source_mask = _bool_mask(g.shape, source_cells)

    rings = _find_rings(g)
    assert len(rings) >= 3, "expected multiple colored rings"
    ring_mask = np.zeros_like(g, dtype=bool)
    by_color: dict[int, list[dict]] = {}
    for ring in rings:
        for r, c in ring["cells"]:
            ring_mask[r, c] = True
        by_color.setdefault(int(ring["color"]), []).append(ring)
    assert all(len(items) == 1 for items in by_color.values()), "ring colors must be unique"

    legend, legend_mask = _parse_legend(g)

    start_candidates = []
    for r, c in _adjacent_cells(source_cells, g.shape):
        if int(g[r, c]) != BG:
            start_candidates.append((r, c))
    assert len(start_candidates) == 1, "checkerboard must have exactly one adjacent guide pixel"
    start = start_candidates[0]
    path_color = int(g[start])

    blocked = ring_mask | legend_mask
    guide = _trace_guide_component(g, start, path_color, blocked)
    assert start in guide
    assert max(_degree_in_component(cell, guide) for cell in guide) <= 2, "guide component must be a simple path"

    source_touch = [cell for cell in guide if any((cell[0] + dr, cell[1] + dc) in source_cells for dr, dc in DIR4)]
    assert source_touch == [start], "guide path must touch the checkerboard at exactly one endpoint"

    ring_hits = []
    for cell in guide:
        for ring in by_color.get(path_color, []):
            if any((cell[0] + dr, cell[1] + dc) in ring["cells"] for dr, dc in DIR4):
                ring_hits.append((cell, ring))
    assert len(ring_hits) == 1, "guide path must terminate at exactly one ring"
    _, selected_ring = ring_hits[0]
    selected_color = int(selected_ring["color"])
    assert selected_color == path_color
    assert selected_color in legend, "selected ring color must exist in the legend"

    protected = source_mask | ring_mask | legend_mask
    for r, c in guide:
        protected[r, c] = True

    axis_type = legend[selected_color]
    mid_r = h // 2
    mid_c = w // 2

    out = g.copy()
    marker_coords = np.argwhere((g == selected_color) & ~protected)
    for r0, c0 in marker_coords:
        r = int(r0)
        c = int(c0)
        for rr, cc in _reflect_targets(r, c, mid_r, mid_c, axis_type):
            if not _in_bounds(g.shape, rr, cc):
                continue
            if protected[rr, cc]:
                continue
            existing = int(out[rr, cc])
            assert existing in (BG, selected_color), "mirrored markers must not conflict"
            out[rr, cc] = selected_color

    return out


def _odd_between(lo: int, hi: int) -> int:
    choices = [v for v in range(lo, hi + 1) if v % 2 == 1]
    return random.choice(choices)


def _place_checkerboard(grid: np.ndarray, top: int, left: int) -> None:
    grid[top : top + 3, left : left + 3] = _checkerboard_pattern(0)


def _place_ring(grid: np.ndarray, top: int, left: int, color: int) -> None:
    grid[top : top + 3, left : left + 3] = color
    grid[top + 1, left + 1] = BG


def _legend_pattern(axis_type: str) -> set[tuple[int, int]]:
    for pattern, name in PATTERN_TO_AXIS.items():
        if name == axis_type:
            return set(pattern)
    raise AssertionError("unknown axis type")


def _place_legend(
    grid: np.ndarray,
    color_to_axis: dict[int, str],
) -> np.ndarray:
    h, w = grid.shape
    colors = list(color_to_axis)
    gap = random.choice(GEN_LEGEND_GAPS)
    total_w = len(colors) * LEGEND_UNIT_W + (len(colors) - 1) * gap
    start_c = random.randint(0, w - total_w)
    top = h - LEGEND_UNIT_H
    mask = np.zeros_like(grid, dtype=bool)
    c = start_c
    for color in colors:
        for rr, cc in _legend_pattern(color_to_axis[color]):
            grid[top + rr, c + cc] = LEGEND_INK
            mask[top + rr, c + cc] = True
        grid[top + 3, c : c + 3] = color
        mask[top + 3, c : c + 3] = True
        c += LEGEND_UNIT_W + gap
    return mask


def _expand4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _box_cells(top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r in range(3) for c in range(3)}


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    return (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _sample_boxes(h: int, w: int) -> tuple[tuple[int, int], dict[int, tuple[int, int]]] | None:
    legend_top = h - LEGEND_UNIT_H
    content_last = legend_top - CONTENT_GAP - 1
    max_top = content_last - 2
    if max_top < 2:
        return None
    positions = [(r, c) for r in range(0, max_top + 1, 2) for c in range(0, w - 2, 2)]
    random.shuffle(positions)
    source = None
    rings: dict[int, tuple[int, int]] = {}
    boxes: list[tuple[int, int, int, int]] = []
    for top, left in positions:
        box = (top, left, top + 2, left + 2)
        if source is None:
            source = (top, left)
            boxes.append(box)
            continue
        if not _boxes_separated(boxes[0], box, 4):
            continue
        rings[0] = (top, left)
        boxes.append(box)
        break
    if source is None or 0 not in rings:
        return None
    for idx in (1, 2):
        placed = False
        for top, left in positions:
            box = (top, left, top + 2, left + 2)
            if any(not _boxes_separated(prev, box, 2) for prev in boxes):
                continue
            rings[idx] = (top, left)
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return source, rings


def _attachment_points(top: int, left: int) -> dict[str, tuple[int, int]]:
    return {
        "top": (top - 1, left + 1),
        "bottom": (top + 3, left + 1),
        "left": (top + 1, left - 1),
        "right": (top + 1, left + 3),
    }


def _object_mask(
    shape: tuple[int, int],
    source: tuple[int, int],
    rings: dict[int, tuple[int, int]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for top, left in [source] + list(rings.values()):
        for r, c in _box_cells(top, left):
            mask[r, c] = True
    return mask


def _build_guide_path(
    shape: tuple[int, int],
    source: tuple[int, int],
    target_ring: tuple[int, int],
    other_rings: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = shape
    legend_top = h - LEGEND_UNIT_H
    content_last = legend_top - CONTENT_GAP - 1
    object_cells = _box_cells(*source) | _box_cells(*target_ring)
    for ring in other_rings:
        object_cells |= _box_cells(*ring)
    blocked = _expand4(_bool_mask(shape, object_cells))

    source_points = list(_attachment_points(*source).items())
    target_points = list(_attachment_points(*target_ring).items())
    random.shuffle(source_points)
    random.shuffle(target_points)

    for _, start in source_points:
        if not _in_bounds(shape, *start):
            continue
        if start[0] > content_last:
            continue
        for _, end in target_points:
            if not _in_bounds(shape, *end):
                continue
            if end[0] > content_last:
                continue
            allowed = np.zeros(shape, dtype=bool)
            for r in range(1, content_last + 1, 2):
                for c in range(1, w, 2):
                    if not blocked[r, c]:
                        allowed[r, c] = True
            allowed[start] = True
            allowed[end] = True
            parent = {start: None}
            q = deque([start])
            found = False
            while q:
                cur = q.popleft()
                if cur == end:
                    found = True
                    break
                dirs = STEP_DIRS[:]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr, nc = cur[0] + dr, cur[1] + dc
                    nxt = (nr, nc)
                    if not _in_bounds(shape, nr, nc):
                        continue
                    if nr > content_last or not allowed[nr, nc]:
                        continue
                    if nxt in parent:
                        continue
                    parent[nxt] = cur
                    q.append(nxt)
            if not found:
                continue
            path = []
            cur = end
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            if len(path) >= 3:
                return path
    return None


def _step2_neighbors_of_path(shape: tuple[int, int], path: list[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in path:
        for dr, dc in STEP_DIRS:
            nr, nc = r + dr, c + dc
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out - set(path)


def _too_close(cell: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in others:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return True
    return False


def _candidate_marker_cells(
    shape: tuple[int, int],
    fixed_buffer: np.ndarray,
    forbid_step2: set[tuple[int, int]],
    forbid_axes: bool,
) -> list[tuple[int, int]]:
    h, w = shape
    legend_top = h - LEGEND_UNIT_H
    content_last = legend_top - CONTENT_GAP - 1
    mid_r = h // 2
    mid_c = w // 2
    cells = []
    for r in range(content_last + 1):
        for c in range(w):
            if fixed_buffer[r, c]:
                continue
            if (r, c) in forbid_step2:
                continue
            if forbid_axes and (r == mid_r or c == mid_c):
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _sample_selected_markers(
    shape: tuple[int, int],
    axis_type: str,
    fixed_buffer: np.ndarray,
    guide_path: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = shape
    mid_r = h // 2
    mid_c = w // 2
    count = random.randint(*SELECTED_MARKER_COUNT_RANGE)
    forbid_step2 = _step2_neighbors_of_path(shape, guide_path)
    candidates = _candidate_marker_cells(shape, fixed_buffer, forbid_step2, forbid_axes=True)

    if axis_type == "vertical":
        side = random.choice(("left", "right"))
        candidates = [cell for cell in candidates if (cell[1] < mid_c) == (side == "left")]
    elif axis_type == "horizontal":
        side = random.choice(("top", "bottom"))
        candidates = [cell for cell in candidates if (cell[0] < mid_r) == (side == "top")]
    else:
        quadrant = random.choice(((0, 0), (0, 1), (1, 0), (1, 1)))
        candidates = [
            cell
            for cell in candidates
            if (cell[0] < mid_r) == (quadrant[0] == 0) and (cell[1] < mid_c) == (quadrant[1] == 0)
        ]

    picked: list[tuple[int, int]] = []
    occupied_targets: set[tuple[int, int]] = set()
    for r, c in candidates:
        targets = _reflect_targets(r, c, mid_r, mid_c, axis_type)
        if any(fixed_buffer[rr, cc] for rr, cc in targets):
            continue
        if any(target in occupied_targets for target in targets):
            continue
        if _too_close((r, c), set(picked)):
            continue
        picked.append((r, c))
        occupied_targets |= targets
        if len(picked) == count:
            return picked
    return None


def _sample_other_markers(
    shape: tuple[int, int],
    fixed_buffer: np.ndarray,
    reserved: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    legend_top = h - LEGEND_UNIT_H
    content_last = legend_top - CONTENT_GAP - 1
    cells = [
        (r, c)
        for r in range(content_last + 1)
        for c in range(w)
        if not fixed_buffer[r, c] and (r, c) not in reserved
    ]
    random.shuffle(cells)
    out: list[tuple[int, int]] = []
    used = set(reserved)
    for cell in cells:
        if _too_close(cell, used):
            continue
        out.append(cell)
        used.add(cell)
        if len(out) == count:
            return out
    return None


def _expected_output(
    inp: np.ndarray,
    selected_color: int,
    axis_type: str,
    selected_markers: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    mid_r = inp.shape[0] // 2
    mid_c = inp.shape[1] // 2
    for r, c in selected_markers:
        for rr, cc in _reflect_targets(r, c, mid_r, mid_c, axis_type):
            out[rr, cc] = selected_color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = _odd_between(*GRID_HEIGHT_RANGE)
    w = _odd_between(*GRID_WIDTH_RANGE)

    colors = random.sample(COLOR_POOL, 3)
    axes = list(AXES)
    random.shuffle(axes)
    color_to_axis = {color: axis for color, axis in zip(colors, axes)}
    selected_color = random.choice(colors)

    box_sample = _sample_boxes(h, w)
    if box_sample is None:
        return None
    source, ring_positions = box_sample
    color_to_ring = {color: ring_positions[idx] for idx, color in enumerate(colors)}
    guide_path = _build_guide_path(
        (h, w),
        source,
        color_to_ring[selected_color],
        [pos for color, pos in color_to_ring.items() if color != selected_color],
    )
    if guide_path is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    _place_checkerboard(grid, *source)
    for color in colors:
        _place_ring(grid, *color_to_ring[color], color)
    for r, c in guide_path:
        grid[r, c] = selected_color
    legend_mask = _place_legend(grid, color_to_axis)

    fixed_mask = grid != BG
    fixed_buffer = _expand4(fixed_mask | legend_mask)
    selected_markers = _sample_selected_markers((h, w), color_to_axis[selected_color], fixed_buffer, guide_path)
    if selected_markers is None:
        return None

    selected_targets = set()
    mid_r = h // 2
    mid_c = w // 2
    for cell in selected_markers:
        selected_targets |= _reflect_targets(cell[0], cell[1], mid_r, mid_c, color_to_axis[selected_color])

    reserved = set(selected_targets)
    all_markers: dict[int, list[tuple[int, int]]] = {selected_color: selected_markers}
    for color in colors:
        if color == selected_color:
            continue
        count = random.randint(*MARKER_COUNT_RANGE)
        markers = _sample_other_markers((h, w), fixed_buffer, reserved, count)
        if markers is None:
            return None
        all_markers[color] = markers
        reserved |= set(markers)

    for color, cells in all_markers.items():
        for r, c in cells:
            grid[r, c] = color

    expected = _expected_output(grid, selected_color, color_to_axis[selected_color], selected_markers)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
