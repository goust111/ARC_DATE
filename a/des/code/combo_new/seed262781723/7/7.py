from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
EDGES = ("top", "bottom", "left", "right")
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _is_consecutive(values: list[int]) -> bool:
    return values == list(range(values[0], values[-1] + 1))


def _classify_segment(comp: list[tuple[int, int]], h: int, w: int) -> dict[str, int] | None:
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    if len(rows) == 1 and rows[0] in (0, h - 1) and _is_consecutive(cols):
        return {
            "side": 0 if rows[0] == 0 else 1,
            "start": cols[0],
            "end": cols[-1],
        }
    if len(cols) == 1 and cols[0] in (0, w - 1) and _is_consecutive(rows):
        return {
            "side": 2 if cols[0] == 0 else 3,
            "start": rows[0],
            "end": rows[-1],
        }
    return None


def _distance_to_side(comp: list[tuple[int, int]], side: str, h: int, w: int) -> int:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    if side == "top":
        return min(rows)
    if side == "bottom":
        return h - 1 - max(rows)
    if side == "left":
        return min(cols)
    return w - 1 - max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each interior object onto its same-colored border segment, thickening that segment inward layer by layer.

    Input structure:
    - The background is black.
    - Zero or one straight colored segment lies on each grid edge; each segment is a single non-black color and different segments use different colors.
    - The interior contains several small 4-connected monochrome objects, and each object's color matches exactly one border segment.

    Rule:
    - Keep the border segments and the background fixed, then erase all interior objects.
    - For each interior object, find the border segment with the same color.
    - If that segment is horizontal, use the object's column range; if it is vertical, use the object's row range.
    - Sort same-colored objects by orthogonal distance to their segment from near to far, and paint their projected ranges on the 1st, 2nd, 3rd, ... interior row or column next to that segment.

    Key insight:
    - Color identifies the matching edge segment, while the object's span parallel to that segment becomes a compressed projection and the distance order determines the inward layer number.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())

    segments_by_color: dict[int, dict[str, int | str]] = {}
    objects_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    all_objects: list[list[tuple[int, int]]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            touches_border = any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp)
            if touches_border:
                seg = _classify_segment(comp, h, w)
                if seg is None:
                    continue
                side_idx = int(seg["side"])
                segments_by_color[color] = {
                    "side": EDGES[side_idx],
                    "start": int(seg["start"]),
                    "end": int(seg["end"]),
                }
            else:
                all_objects.append(comp)
                objects_by_color.setdefault(color, []).append(comp)

    out = g.copy()
    for comp in all_objects:
        for r, c in comp:
            out[r, c] = bg

    for color, comps in objects_by_color.items():
        seg = segments_by_color.get(int(color))
        if seg is None:
            continue
        side = str(seg["side"])
        ordered = sorted(
            comps,
            key=lambda comp: (
                _distance_to_side(comp, side, h, w),
                min(r for r, _ in comp),
                min(c for _, c in comp),
            ),
        )
        for layer, comp in enumerate(ordered, start=1):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            if side == "top":
                row = layer
                if 0 < row < h - 1:
                    out[row, min(cols) : max(cols) + 1] = color
            elif side == "bottom":
                row = h - 1 - layer
                if 0 < row < h - 1:
                    out[row, min(cols) : max(cols) + 1] = color
            elif side == "left":
                col = layer
                if 0 < col < w - 1:
                    out[min(rows) : max(rows) + 1, col] = color
            else:
                col = w - 1 - layer
                if 0 < col < w - 1:
                    out[min(rows) : max(rows) + 1, col] = color

    return out


def _box_is_clear(occupied: np.ndarray, r0: int, c0: int, hh: int, ww: int) -> bool:
    h, w = occupied.shape
    r1 = r0 + hh - 1
    c1 = c0 + ww - 1
    rr0 = max(0, r0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w - 1, c1 + 1)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _grow_shape(base: set[tuple[int, int]], hh: int, ww: int, target_area: int) -> set[tuple[int, int]]:
    cells = set(base)
    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < hh and 0 <= nc < ww and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells


def _make_horizontal_shape(length: int, height: int, edge: str) -> set[tuple[int, int]]:
    base_row = 0 if edge == "top" else height - 1
    base = {(base_row, c) for c in range(length)}
    max_area = min(7, length * height)
    target_area = random.randint(length, max_area)
    return _grow_shape(base, height, length, target_area)


def _make_vertical_shape(length: int, width: int, edge: str) -> set[tuple[int, int]]:
    base_col = 0 if edge == "left" else width - 1
    base = {(r, base_col) for r in range(length)}
    max_area = min(7, length * width)
    target_area = random.randint(length, max_area)
    return _grow_shape(base, length, width, target_area)


def _edge_counts(active_edges: list[str]) -> dict[str, int]:
    counts = {edge: 0 for edge in EDGES}
    for edge in active_edges:
        counts[edge] = random.randint(1, 2)
    if active_edges and random.random() < 0.4:
        counts[random.choice(active_edges)] = 3
    return counts


def _sample_layout() -> tuple[int, int, list[str], dict[str, int]] | None:
    active_count = random.choice([1, 2, 2, 3, 3, 4])
    active_edges = random.sample(list(EDGES), active_count)
    counts = _edge_counts(active_edges)

    top_layers = counts["top"]
    bottom_layers = counts["bottom"]
    left_layers = counts["left"]
    right_layers = counts["right"]

    min_h = max(12, top_layers + bottom_layers + 8)
    min_w = max(12, left_layers + right_layers + 8)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, min(26, 30))
    w = random.randint(min_w, min(26, 30))

    horiz_lo = max(2, left_layers + 1)
    horiz_hi = min(w - 3, w - right_layers - 2)
    vert_lo = max(2, top_layers + 1)
    vert_hi = min(h - 3, h - bottom_layers - 2)
    if horiz_hi - horiz_lo + 1 < 3 or vert_hi - vert_lo + 1 < 3:
        return None
    return h, w, active_edges, counts


def _segment_interval(min_start: int, max_end: int, low: int, high: int) -> tuple[int, int]:
    left_room = min_start - low
    right_room = high - max_end
    extra_left = random.randint(0, min(2, max(0, left_room)))
    extra_right = random.randint(0, min(2, max(0, right_room)))
    return min_start - extra_left, max_end + extra_right


def _try_generate() -> np.ndarray | None:
    sampled = _sample_layout()
    if sampled is None:
        return None
    h, w, active_edges, counts = sampled
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    colors = random.sample(PALETTE, len(active_edges))
    color_by_edge = {edge: color for edge, color in zip(active_edges, colors, strict=False)}

    horiz_lo = max(2, counts["left"] + 1)
    horiz_hi = min(w - 3, w - counts["right"] - 2)
    vert_lo = max(2, counts["top"] + 1)
    vert_hi = min(h - 3, h - counts["bottom"] - 2)

    objects: list[dict[str, object]] = []
    spans_by_edge: dict[str, list[tuple[int, int]]] = {edge: [] for edge in active_edges}
    dists_by_edge: dict[str, set[int]] = {edge: set() for edge in active_edges}

    for edge in random.sample(active_edges, len(active_edges)):
        for _ in range(counts[edge]):
            placed = False
            for _attempt in range(300):
                if edge in ("top", "bottom"):
                    max_len = min(5, horiz_hi - horiz_lo + 1)
                    if max_len < 3:
                        return None
                    span_len = random.randint(3, max_len)
                    span_start = random.randint(horiz_lo, horiz_hi - span_len + 1)
                    box_h = random.randint(2, 3)
                    dist_max = h - box_h - 2
                    choices = [d for d in range(2, dist_max + 1) if d not in dists_by_edge[edge]]
                    if not choices:
                        break
                    dist = random.choice(choices)
                    r0 = dist if edge == "top" else h - dist - box_h
                    c0 = span_start
                    if not _box_is_clear(occupied, r0, c0, box_h, span_len):
                        continue
                    shape = _make_horizontal_shape(span_len, box_h, edge)
                else:
                    max_len = min(5, vert_hi - vert_lo + 1)
                    if max_len < 3:
                        return None
                    span_len = random.randint(3, max_len)
                    span_start = random.randint(vert_lo, vert_hi - span_len + 1)
                    box_w = random.randint(2, 3)
                    dist_max = w - box_w - 2
                    choices = [d for d in range(2, dist_max + 1) if d not in dists_by_edge[edge]]
                    if not choices:
                        break
                    dist = random.choice(choices)
                    r0 = span_start
                    c0 = dist if edge == "left" else w - dist - box_w
                    if not _box_is_clear(occupied, r0, c0, span_len, box_w):
                        continue
                    shape = _make_vertical_shape(span_len, box_w, edge)

                cells = [(r0 + rr, c0 + cc) for rr, cc in sorted(shape)]
                for rr, cc in cells:
                    grid[rr, cc] = color_by_edge[edge]
                    occupied[rr, cc] = True

                spans_by_edge[edge].append((span_start, span_start + span_len - 1))
                dists_by_edge[edge].add(dist)
                objects.append(
                    {
                        "edge": edge,
                        "color": color_by_edge[edge],
                        "cells": cells,
                        "span": (span_start, span_start + span_len - 1),
                        "dist": dist,
                    }
                )
                placed = True
                break
            if not placed:
                return None

    segment_by_edge: dict[str, tuple[int, int]] = {}
    lengths: list[int] = []
    for edge in active_edges:
        spans = spans_by_edge[edge]
        start = min(a for a, _ in spans)
        end = max(b for _, b in spans)
        if edge in ("top", "bottom"):
            seg = _segment_interval(start, end, 1, w - 2)
        else:
            seg = _segment_interval(start, end, 1, h - 2)
        segment_by_edge[edge] = seg
        lengths.append(seg[1] - seg[0] + 1)

    if len(lengths) != len(set(lengths)):
        return None

    for edge, color in color_by_edge.items():
        start, end = segment_by_edge[edge]
        if edge == "top":
            grid[0, start : end + 1] = color
        elif edge == "bottom":
            grid[h - 1, start : end + 1] = color
        elif edge == "left":
            grid[start : end + 1, 0] = color
        else:
            grid[start : end + 1, w - 1] = color

    expected = grid.copy()
    for obj in objects:
        for r, c in obj["cells"]:
            expected[r, c] = BG

    target_cells: dict[tuple[int, int], int] = {}
    for edge in active_edges:
        edge_objects = sorted(
            [obj for obj in objects if obj["edge"] == edge],
            key=lambda obj: int(obj["dist"]),
        )
        for layer, obj in enumerate(edge_objects, start=1):
            start, end = obj["span"]
            color = int(obj["color"])
            if edge == "top":
                coords = [(layer, c) for c in range(start, end + 1)]
            elif edge == "bottom":
                coords = [(h - 1 - layer, c) for c in range(start, end + 1)]
            elif edge == "left":
                coords = [(r, layer) for r in range(start, end + 1)]
            else:
                coords = [(r, w - 1 - layer) for r in range(start, end + 1)]
            for coord in coords:
                prev = target_cells.get(coord)
                if prev is not None and prev != color:
                    return None
                target_cells[coord] = color

    for (r, c), color in target_cells.items():
        expected[r, c] = color

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
