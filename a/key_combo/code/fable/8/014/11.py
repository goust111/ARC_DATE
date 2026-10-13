from __future__ import annotations

import random

import numpy as np


GREY = 5
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _extract_markers(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    values, counts = np.unique(grid, return_counts=True)
    for color, count in zip(values.tolist(), counts.tolist()):
        color = int(color)
        if color in (bg, GREY):
            continue
        assert count == 1, "every marker color must appear exactly once"
        r, c = np.argwhere(grid == color)[0]
        markers.append((color, int(r), int(c)))
    return markers


def _nearest_marker(markers: list[tuple[int, int, int]], r: int, c: int) -> tuple[int | None, int, bool]:
    distances = [_chebyshev((r, c), (mr, mc)) for _color, mr, mc in markers]
    best = min(distances)
    winners = [idx for idx, dist in enumerate(distances) if dist == best]
    if len(winners) != 1:
        return None, best, False
    color = int(markers[winners[0]][0])
    return color, best, True


def main(grid: np.ndarray) -> np.ndarray:
    """Reveal even-distance nearest-marker rings only on the grey objects.

    Input structure:
    - A dominant background contains one or more grey 8-connected objects and several colored singleton markers.
    - Marker colors are all distinct, and markers sit outside the grey objects.
    - Only grey cells may change; background cells and marker pixels stay visible as they are.

    Rule:
    - For each grey cell, compute its Chebyshev distance to every marker.
    - If the nearest marker is unique and that distance is even, recolor the grey cell to that marker's color.
    - Grey cells at odd distance, or tied between two or more nearest markers, remain grey; background and marker cells stay unchanged.

    Key insight:
    - The picture shows only the even rings of a hidden Chebyshev Voronoi field, and only where those invisible rings intersect grey objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    assert bg != GREY, "background must be distinct from grey objects"
    assert np.any(g == GREY), "expected grey object cells"

    markers = _extract_markers(g, bg)
    assert markers, "expected at least one marker"

    out = g.copy()
    for r, c in np.argwhere(g == GREY):
        color, best, unique = _nearest_marker(markers, int(r), int(c))
        if unique and best % 2 == 0:
            out[int(r), int(c)] = int(color)
    return out


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0 - 1:
        return b0 - a1 - 1
    if b1 < a0 - 1:
        return a0 - b1 - 1
    return 0


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = _interval_gap(a[0], a[2], b[0], b[2])
    col_gap = _interval_gap(a[1], a[3], b[1], b[3])
    return max(row_gap, col_gap)


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = patch.shape
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[patch] = int(color)


def _rect_patch(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _l_patch(h: int, w: int) -> np.ndarray:
    t = 1 if min(h, w) <= 4 else random.randint(1, 2)
    patch = np.zeros((h, w), dtype=bool)
    patch[:, :t] = True
    patch[h - t :, :] = True
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if random.random() < 0.5:
        patch = np.flipud(patch)
    return patch


def _plus_patch(h: int, w: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=bool)
    r = random.randint(1, h - 2)
    c = random.randint(1, w - 2)
    patch[r, :] = True
    patch[:, c] = True
    if h >= 5 and random.random() < 0.5:
        patch[max(0, r - 1) : min(h, r + 2), c] = True
    if w >= 5 and random.random() < 0.5:
        patch[r, max(0, c - 1) : min(w, c + 2)] = True
    return patch


def _stair_patch(h: int, w: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=bool)
    band_w = random.randint(2, min(4, w))
    left = random.randint(0, w - band_w)
    for r in range(h):
        if r > 0:
            left = min(max(0, left + random.choice((-1, 0, 1))), w - band_w)
        patch[r, left : left + band_w] = True
    return patch


def _random_object_patch() -> np.ndarray:
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    kind = random.choice(("rect", "l", "plus", "stair"))
    if kind == "rect":
        return _rect_patch(h, w)
    if kind == "l":
        return _l_patch(h, w)
    if kind == "plus":
        return _plus_patch(max(h, 3), max(w, 3))
    return _stair_patch(h, w)


def _bridge_layout(shape: tuple[int, int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int, int, int]]] | None:
    h, w = shape
    horizontal = random.random() < 0.5
    if horizontal:
        band_h = random.randint(3, min(7, h - 4))
        spans = [span for span in (8, 10, 12, 14) if span <= w - 4]
        if not spans:
            return None
        span = random.choice(spans)
        r0 = random.randint(2, h - band_h - 3)
        mid_r = r0 + band_h // 2
        c1 = random.randint(1, w - span - 2)
        c2 = c1 + span
        obj_c0 = c1 + 2
        obj_c1 = c2 - 2
        patch = np.ones((band_h, obj_c1 - obj_c0 + 1), dtype=bool)
        markers = [(mid_r, c1), (mid_r, c2)]
        boxes = [(r0, obj_c0, r0 + band_h - 1, obj_c1)]
        boxes.extend((r, c, r, c) for r, c in markers)
        return patch, [(r0, obj_c0)] + markers, boxes

    band_w = random.randint(3, min(7, w - 4))
    spans = [span for span in (8, 10, 12, 14) if span <= h - 4]
    if not spans:
        return None
    span = random.choice(spans)
    c0 = random.randint(2, w - band_w - 3)
    mid_c = c0 + band_w // 2
    r1 = random.randint(1, h - span - 2)
    r2 = r1 + span
    obj_r0 = r1 + 2
    obj_r1 = r2 - 2
    patch = np.ones((obj_r1 - obj_r0 + 1, band_w), dtype=bool)
    markers = [(r1, mid_c), (r2, mid_c)]
    boxes = [(obj_r0, c0, obj_r1, c0 + band_w - 1)]
    boxes.extend((r, c, r, c) for r, c in markers)
    return patch, [(obj_r0, c0)] + markers, boxes


def _place_single_marker(
    grid: np.ndarray,
    occupied_boxes: list[tuple[int, int, int, int]],
    min_gap: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != _dominant_color(grid):
                continue
            bbox = (r, c, r, c)
            if all(_bbox_gap(bbox, other) >= min_gap for other in occupied_boxes):
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_object_near_marker(
    grid: np.ndarray,
    occupied_boxes: list[tuple[int, int, int, int]],
    markers: list[tuple[int, int, int]],
    focus_idx: int,
) -> tuple[np.ndarray, int, int, tuple[int, int, int, int]] | None:
    focus_r = int(markers[focus_idx][1])
    focus_c = int(markers[focus_idx][2])
    h, w = grid.shape

    for _ in range(18):
        patch = _random_object_patch()
        ph, pw = patch.shape
        positions = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if any(_bbox_gap(bbox, other) < 1 for other in occupied_boxes):
                    continue
                gap = _bbox_gap(bbox, (focus_r, focus_c, focus_r, focus_c))
                if not (1 <= gap <= 4):
                    continue
                center = (r0 + ph // 2, c0 + pw // 2)
                focus_dist = _chebyshev(center, (focus_r, focus_c))
                other_dists = [
                    _chebyshev(center, (int(rm), int(cm)))
                    for idx, (_color, rm, cm) in enumerate(markers)
                    if idx != focus_idx
                ]
                if other_dists and min(other_dists) <= focus_dist:
                    continue
                positions.append((r0, c0, bbox))
        if positions:
            r0, c0, bbox = random.choice(positions)
            return patch, r0, c0, bbox
    return None


def _analysis(grid: np.ndarray) -> tuple[int, int, int, set[int]]:
    bg = _dominant_color(grid)
    markers = _extract_markers(grid, bg)
    tie_count = 0
    odd_unique_count = 0
    recolor_count = 0
    reveal_colors: set[int] = set()
    for r, c in np.argwhere(grid == GREY):
        color, best, unique = _nearest_marker(markers, int(r), int(c))
        if not unique:
            tie_count += 1
        elif best % 2 == 1:
            odd_unique_count += 1
        else:
            recolor_count += 1
            reveal_colors.add(int(color))
    return tie_count, odd_unique_count, recolor_count, reveal_colors


def _valid_example(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    if np.array_equal(out, grid):
        return False
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return False

    tie_count, odd_unique_count, recolor_count, reveal_colors = _analysis(grid)
    if tie_count < 2 or odd_unique_count < 2 or recolor_count < 8:
        return False
    if len(reveal_colors) < 2:
        return False
    if np.count_nonzero(grid == GREY) < 20:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    bg = random.choice([color for color in PALETTE if color != GREY])
    marker_count = random.randint(2, 4)
    marker_colors = random.sample([color for color in PALETTE if color not in (bg, GREY)], marker_count)

    grid = np.full((h, w), bg, dtype=int)

    bridge = _bridge_layout((h, w))
    if bridge is None:
        return None
    bridge_patch, placements, occupied_boxes = bridge
    bridge_r0, bridge_c0 = placements[0]
    _paint_patch(grid, bridge_patch, bridge_r0, bridge_c0, GREY)

    markers: list[tuple[int, int, int]] = []
    for color, (r, c) in zip(marker_colors[:2], placements[1:]):
        grid[r, c] = int(color)
        markers.append((int(color), int(r), int(c)))

    for color in marker_colors[2:]:
        pos = _place_single_marker(grid, occupied_boxes, min_gap=3)
        if pos is None:
            return None
        r, c = pos
        grid[r, c] = int(color)
        markers.append((int(color), int(r), int(c)))
        occupied_boxes.append((r, c, r, c))

    bonus_objects = random.randint(1, 2)
    focus_order = list(range(len(markers)))
    random.shuffle(focus_order)
    for focus_idx in focus_order[2:]:
        placed = _place_object_near_marker(grid, occupied_boxes, markers, focus_idx)
        if placed is None:
            return None
        patch, r0, c0, bbox = placed
        _paint_patch(grid, patch, r0, c0, GREY)
        occupied_boxes.append(bbox)

    for _ in range(bonus_objects):
        focus_idx = random.randrange(len(markers))
        placed = _place_object_near_marker(grid, occupied_boxes, markers, focus_idx)
        if placed is None:
            return None
        patch, r0, c0, bbox = placed
        _paint_patch(grid, patch, r0, c0, GREY)
        occupied_boxes.append(bbox)

    if not _valid_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-ring example")
