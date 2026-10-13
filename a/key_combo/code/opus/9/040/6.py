from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4


_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_GEN_GRID_MIN = 22
_GEN_GRID_MAX = 30
_GEN_BBOX_MIN = 5
_GEN_BBOX_MAX = 6
_GEN_GROUPS = [3, 4]  # total matching groups, including the original one
_GEN_MARKER_COUNTS = [2, 3]
_GEN_BBOX_GAP = 8
_GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in _DIR4:
        yield int(r) + dr, int(c) + dc


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(np.array(comp, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _inside_bbox_cells(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(grid[r, c]) == int(color):
                cells.append((int(r), int(c)))
    return cells


def _longest_span(indices: np.ndarray) -> tuple[int, int]:
    assert indices.size > 0
    best_lo = best_hi = int(indices[0])
    start = prev = int(indices[0])
    best_len = 1
    for idx in map(int, indices[1:]):
        if idx == prev + 1:
            prev = idx
            continue
        cur_len = prev - start + 1
        if cur_len > best_len:
            best_lo, best_hi, best_len = start, prev, cur_len
        start = prev = idx
    cur_len = prev - start + 1
    if cur_len > best_len:
        best_lo, best_hi = start, prev
    return int(best_lo), int(best_hi)


def _side_spans(body_mask: np.ndarray) -> dict[str, tuple[int, int]]:
    top = _longest_span(np.flatnonzero(body_mask[0]))
    bottom = _longest_span(np.flatnonzero(body_mask[-1]))
    left = _longest_span(np.flatnonzero(body_mask[:, 0]))
    right = _longest_span(np.flatnonzero(body_mask[:, -1]))
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _stamp_body(out: np.ndarray, body_mask: np.ndarray, anchor: tuple[int, int]) -> None:
    r0, c0 = anchor
    for rr, cc in np.argwhere(body_mask):
        out[int(r0) + int(rr), int(c0) + int(cc)] = BLUE


def _paint_horizontal(out: np.ndarray, row: int, c0: int, c1: int) -> set[tuple[int, int]]:
    added: set[tuple[int, int]] = set()
    if not 0 <= row < out.shape[0]:
        return added
    for c in range(int(c0), int(c1) + 1):
        if 0 <= c < out.shape[1] and int(out[row, c]) == BLACK:
            out[row, c] = YELLOW
            added.add((int(row), int(c)))
    return added


def _paint_vertical(out: np.ndarray, col: int, r0: int, r1: int) -> set[tuple[int, int]]:
    added: set[tuple[int, int]] = set()
    if not 0 <= col < out.shape[1]:
        return added
    for r in range(int(r0), int(r1) + 1):
        if 0 <= r < out.shape[0] and int(out[r, col]) == BLACK:
            out[r, col] = YELLOW
            added.add((int(r), int(col)))
    return added


def _paint_shrinking_from_body(out: np.ndarray, body_mask: np.ndarray, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    spans = _side_spans(body_mask)
    r0, c0 = anchor
    h, w = body_mask.shape
    added: set[tuple[int, int]] = set()

    top_lo, top_hi = spans["top"]
    for k in range(1, top_hi - top_lo + 1):
        lo, hi = top_lo + k, top_hi - k
        if lo > hi:
            break
        added |= _paint_horizontal(out, int(r0) - k, int(c0) + lo, int(c0) + hi)

    bottom_lo, bottom_hi = spans["bottom"]
    for k in range(1, bottom_hi - bottom_lo + 1):
        lo, hi = bottom_lo + k, bottom_hi - k
        if lo > hi:
            break
        added |= _paint_horizontal(out, int(r0) + h - 1 + k, int(c0) + lo, int(c0) + hi)

    left_lo, left_hi = spans["left"]
    for k in range(1, left_hi - left_lo + 1):
        lo, hi = left_lo + k, left_hi - k
        if lo > hi:
            break
        added |= _paint_vertical(out, int(c0) - k, int(r0) + lo, int(r0) + hi)

    right_lo, right_hi = spans["right"]
    for k in range(1, right_hi - right_lo + 1):
        lo, hi = right_lo + k, right_hi - k
        if lo > hi:
            break
        added |= _paint_vertical(out, int(c0) + w - 1 + k, int(r0) + lo, int(r0) + hi)

    return added


def _candidate_groups(
    red_cells: set[tuple[int, int]],
    offsets: tuple[tuple[int, int], ...],
    shape: tuple[int, int],
) -> list[dict]:
    h, w = shape
    max_r = max(r for r, _ in offsets)
    max_c = max(c for _, c in offsets)
    candidates: dict[frozenset[tuple[int, int]], tuple[int, int]] = {}

    for cell_r, cell_c in red_cells:
        for off_r, off_c in offsets:
            anchor = (int(cell_r) - int(off_r), int(cell_c) - int(off_c))
            ar, ac = anchor
            if ar < 0 or ac < 0 or ar + max_r >= h or ac + max_c >= w:
                continue
            cells = frozenset((int(ar) + dr, int(ac) + dc) for dr, dc in offsets)
            if cells <= red_cells:
                candidates[cells] = (int(ar), int(ac))

    return [{"anchor": anchor, "cells": cells} for cells, anchor in candidates.items()]


def _unique_marker_groups(
    red_cells: set[tuple[int, int]],
    offsets: tuple[tuple[int, int], ...],
    shape: tuple[int, int],
) -> list[dict]:
    candidates = _candidate_groups(red_cells, offsets, shape)
    if not candidates:
        raise AssertionError("expected at least one matching marker group")

    red_list = sorted(red_cells)
    by_cell: dict[tuple[int, int], list[int]] = {cell: [] for cell in red_list}
    for idx, cand in enumerate(candidates):
        for cell in cand["cells"]:
            if cell in by_cell:
                by_cell[cell].append(int(idx))
    assert all(by_cell[cell] for cell in red_list), "every red cell must appear in a candidate group"

    solutions: list[tuple[int, ...]] = []

    def search(covered: set[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if len(covered) == len(red_cells):
            solutions.append(tuple(chosen))
            return
        remaining = [cell for cell in red_list if cell not in covered]
        pivot = min(remaining, key=lambda cell: sum(not (candidates[idx]["cells"] & covered) for idx in by_cell[cell]))
        for idx in by_cell[pivot]:
            cells = candidates[idx]["cells"]
            if cells & covered:
                continue
            search(covered | set(cells), chosen + [int(idx)])

    search(set(), [])
    assert len(solutions) == 1, "marker groups must form a unique exact cover"
    chosen = [candidates[idx] for idx in solutions[0]]
    chosen.sort(key=lambda item: item["anchor"])
    return chosen


def _instances_from_grid(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[tuple[int, int], ...], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    blue_components = _components4(g == BLUE)
    assert len(blue_components) == 1, "expected exactly one blue source sprite"
    bbox = _bbox_from_coords(blue_components[0])
    body_mask = _crop(g == BLUE, bbox)
    r0, c0, _r1, _c1 = bbox

    markers = _inside_bbox_cells(g, bbox, RED)
    assert 2 <= len(markers) <= 3, "expected 2-3 internal red markers"
    marker_offsets = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in markers))

    red_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == RED)}
    groups = _unique_marker_groups(red_cells, marker_offsets, g.shape)
    anchors = [tuple(map(int, group["anchor"])) for group in groups]
    assert (int(r0), int(c0)) in anchors, "source markers must be one of the matching groups"
    return body_mask, (int(r0), int(c0)), marker_offsets, anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the blue source sprite onto every matching red-marker group, then emit yellow shrinking runs whose lengths are set by each stamped shape's edge spans.

    Input structure:
    - The background is black.
    - There is exactly one 4-connected blue sprite body whose bounding box is about 5x5 to 6x6.
    - Inside that sprite bbox are 2-3 red marker pixels, and elsewhere there are 2-3 more red-only marker groups with the same bbox-relative marker offsets.

    Rule:
    - Read the red marker offsets relative to the original blue sprite bbox.
    - Find the unique exact cover of all red pixels by translated copies of that offset pattern.
    - Stamp the blue sprite body at every matched anchor, treating black cells inside the source bbox as transparent.
    - For the original sprite and every stamped copy, take the longest contiguous blue run on the top, bottom, left, and right bbox edges and extend a yellow run outward from that side, shrinking by two cells per layer and staying centered on that side run.
    - Yellow only fills black background cells and never overwrites blue or red.

    Key insight:
    - The red markers only choose where copies go; the copied blue shape then acts as a parameter source, because each edge run length of that shape determines the geometry of the yellow shrinking expansion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    body_mask, _source_anchor, _marker_offsets, anchors = _instances_from_grid(g)

    out = g.copy()
    for anchor in anchors:
        _stamp_body(out, body_mask, anchor)
    for anchor in anchors:
        _paint_shrinking_from_body(out, body_mask, anchor)
    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components4(mask)
    return len(comps) == 1


def _shape_from_corners(h: int, w: int, corners: tuple[int, int, int, int]) -> np.ndarray:
    tl, tr, bl, br = corners
    mask = np.ones((int(h), int(w)), dtype=bool)
    if tl:
        mask[0, 0] = False
    if tr:
        mask[0, -1] = False
    if bl:
        mask[-1, 0] = False
    if br:
        mask[-1, -1] = False
    return mask


def _shape_side_lengths(mask: np.ndarray) -> tuple[int, int, int, int]:
    spans = _side_spans(mask)
    return (
        int(spans["top"][1] - spans["top"][0] + 1),
        int(spans["bottom"][1] - spans["bottom"][0] + 1),
        int(spans["left"][1] - spans["left"][0] + 1),
        int(spans["right"][1] - spans["right"][0] + 1),
    )


def _rect_shortcut_changes_expansion(mask: np.ndarray) -> bool:
    h, w = mask.shape
    pad = 4
    canvas_shape = (h + 2 * pad, w + 2 * pad)
    anchor = (pad, pad)

    correct = np.zeros(canvas_shape, dtype=int)
    _stamp_body(correct, mask, anchor)
    _paint_shrinking_from_body(correct, mask, anchor)

    rect = np.ones_like(mask, dtype=bool)
    shortcut = np.zeros(canvas_shape, dtype=int)
    _stamp_body(shortcut, mask, anchor)
    _paint_shrinking_from_body(shortcut, rect, anchor)
    return not np.array_equal(correct, shortcut)


def _generate_shape() -> np.ndarray:
    corner_options = [(tl, tr, bl, br) for tl in (0, 1) for tr in (0, 1) for bl in (0, 1) for br in (0, 1)]
    for _ in range(_GEN_RETRIES):
        h = random.randint(_GEN_BBOX_MIN, _GEN_BBOX_MAX)
        w = random.randint(_GEN_BBOX_MIN, _GEN_BBOX_MAX)
        random.shuffle(corner_options)
        for corners in corner_options:
            if sum(corners) == 0:
                continue
            mask = _shape_from_corners(h, w, corners)
            side_lengths = _shape_side_lengths(mask)
            if min(side_lengths) < 3:
                continue
            if len(set(side_lengths)) < 2:
                continue
            tl, tr, bl, br = corners
            if tl == tr and bl == br and tl == bl:
                continue
            if not _rect_shortcut_changes_expansion(mask):
                continue
            return mask
    raise RuntimeError("failed to generate a non-degenerate sprite shape")


def _marker_candidates(body_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = body_mask.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if bool(body_mask[r, c]):
                cells.append((int(r), int(c)))
    return cells


def _generate_markers(body_mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    candidates = _marker_candidates(body_mask)
    for _ in range(_GEN_RETRIES):
        count = random.choice(_GEN_MARKER_COUNTS)
        if len(candidates) < count:
            continue
        chosen = tuple(sorted(random.sample(candidates, count)))
        rows = {r for r, _ in chosen}
        cols = {c for _, c in chosen}
        if len(rows) == 1 or len(cols) == 1:
            continue
        blue_only = body_mask.copy()
        for r, c in chosen:
            blue_only[int(r), int(c)] = False
        if not blue_only.any() or not _is_connected(blue_only):
            continue
        return chosen
    raise RuntimeError("failed to place internal markers")


def _visible_body(base_body: np.ndarray, markers: tuple[tuple[int, int], ...]) -> np.ndarray:
    body = np.array(base_body, copy=True, dtype=bool)
    for r, c in markers:
        body[int(r), int(c)] = False
    return body


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(int(br0) - int(ar1) - 1, int(ar0) - int(br1) - 1, 0)
    col_gap = max(int(bc0) - int(ac1) - 1, int(ac0) - int(bc1) - 1, 0)
    return int(row_gap), int(col_gap)


def _far_enough(boxes: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int]) -> bool:
    for box in boxes:
        row_gap, col_gap = _bbox_gap(box, candidate)
        if max(row_gap, col_gap) < _GEN_BBOX_GAP:
            return False
    return True


def _anchor_positions(h: int, w: int, grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    positions = [(r, c) for r in range(grid_h - h + 1) for c in range(grid_w - w + 1)]
    random.shuffle(positions)
    return positions


def _choose_anchors(body_mask: np.ndarray, group_count: int, grid_shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = body_mask.shape
    grid_h, grid_w = grid_shape
    positions = _anchor_positions(h, w, grid_h, grid_w)
    if not positions:
        return None

    anchors: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []

    edge_positions = [
        (r, c)
        for r, c in positions
        if r == 0 or c == 0 or r + h == grid_h or c + w == grid_w
    ]
    if edge_positions and random.random() < 0.6:
        first = random.choice(edge_positions)
        anchors.append((int(first[0]), int(first[1])))
        boxes.append((int(first[0]), int(first[1]), int(first[0]) + h - 1, int(first[1]) + w - 1))

    for r, c in positions:
        if len(anchors) >= group_count:
            break
        box = (int(r), int(c), int(r) + h - 1, int(c) + w - 1)
        if _far_enough(boxes, box):
            anchors.append((int(r), int(c)))
            boxes.append(box)

    if len(anchors) != group_count:
        return None
    random.shuffle(anchors)
    return anchors


def _make_input(body_mask: np.ndarray, markers: tuple[tuple[int, int], ...], anchors: list[tuple[int, int]]) -> np.ndarray:
    h = max(anchor[0] for anchor in anchors) + body_mask.shape[0]
    w = max(anchor[1] for anchor in anchors) + body_mask.shape[1]
    grid = np.zeros((int(h), int(w)), dtype=int)

    source_anchor = anchors[0]
    _stamp_body(grid, body_mask, source_anchor)
    for anchor in anchors:
        for dr, dc in markers:
            grid[int(anchor[0]) + int(dr), int(anchor[1]) + int(dc)] = RED
    return grid


def _expansion_cells(body_mask: np.ndarray, anchor: tuple[int, int], grid_shape: tuple[int, int]) -> set[tuple[int, int]]:
    canvas = np.zeros(grid_shape, dtype=int)
    _stamp_body(canvas, body_mask, anchor)
    return _paint_shrinking_from_body(canvas, body_mask, anchor)


def _validate_semantics(
    grid: np.ndarray,
    body_mask: np.ndarray,
    markers: tuple[tuple[int, int], ...],
    anchors: list[tuple[int, int]],
) -> bool:
    try:
        solved_body, source_anchor, solved_markers, solved_anchors = _instances_from_grid(grid)
    except AssertionError:
        return False

    if not np.array_equal(solved_body, body_mask):
        return False
    if tuple(solved_markers) != tuple(markers):
        return False
    if set(solved_anchors) != set(anchors):
        return False
    if source_anchor != anchors[0]:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if int((out == YELLOW).sum()) == 0:
        return False

    h, w = body_mask.shape
    boxes = [(r, c, r + h - 1, c + w - 1) for r, c in anchors]
    expansion_sets = [_expansion_cells(body_mask, anchor, grid.shape) for anchor in anchors]

    for i, cells in enumerate(expansion_sets):
        if not cells:
            return False
        for j, box in enumerate(boxes):
            if i == j:
                continue
            r0, c0, r1, c1 = box
            for r, c in cells:
                if r0 <= r <= r1 and c0 <= c <= c1:
                    return False
        for j in range(i + 1, len(expansion_sets)):
            if cells & expansion_sets[j]:
                return False

    for anchor in anchors[1:]:
        new_blue = 0
        for rr, cc in np.argwhere(body_mask):
            gr = int(anchor[0]) + int(rr)
            gc = int(anchor[1]) + int(cc)
            if int(grid[gr, gc]) == BLACK and int(out[gr, gc]) == BLUE:
                new_blue += 1
        if new_blue == 0:
            return False

    only_source = grid.copy()
    for anchor in anchors[1:]:
        for dr, dc in markers:
            only_source[int(anchor[0]) + int(dr), int(anchor[1]) + int(dc)] = BLACK
    if np.array_equal(main(only_source), out):
        return False

    rect_out = grid.copy()
    rect_mask = np.ones_like(body_mask, dtype=bool)
    for anchor in anchors:
        _stamp_body(rect_out, body_mask, anchor)
    for anchor in anchors:
        _paint_shrinking_from_body(rect_out, rect_mask, anchor)
    if np.array_equal(rect_out, out):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        base_body = _generate_shape()
        markers = _generate_markers(base_body)
        body_mask = _visible_body(base_body, markers)
        group_count = random.choice(_GEN_GROUPS)
        grid_h = random.randint(_GEN_GRID_MIN, _GEN_GRID_MAX)
        grid_w = random.randint(_GEN_GRID_MIN, _GEN_GRID_MAX)
        anchors = _choose_anchors(body_mask, group_count, (grid_h, grid_w))
        if anchors is None:
            continue
        grid = np.zeros((grid_h, grid_w), dtype=int)
        _stamp_body(grid, body_mask, anchors[0])
        for anchor in anchors:
            for dr, dc in markers:
                grid[int(anchor[0]) + int(dr), int(anchor[1]) + int(dc)] = RED
        if _validate_semantics(grid, body_mask, markers, anchors):
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
