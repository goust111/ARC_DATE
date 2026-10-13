from __future__ import annotations

import random

import numpy as np


BLACK = 0
TEAL = 8
OBJECT_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
GENERATION_ATTEMPTS = 300


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _touches_edge(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _extract_legends(grid: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    legend_pairs: list[tuple[int, int]] = []
    legend_mask = np.zeros_like(grid, dtype=bool)

    for coords in _find_components(grid != BLACK):
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        bbox = min(rows), min(cols), max(rows), max(cols)
        colors = sorted({int(grid[r, c]) for r, c in coords})
        is_legend = (
            len(coords) == 2
            and bbox[0] == bbox[2]
            and bbox[3] - bbox[1] == 1
            and len(colors) == 2
            and _touches_edge(bbox, grid.shape)
        )
        if not is_legend:
            continue
        pair = (colors[0], colors[1])
        if pair not in legend_pairs:
            legend_pairs.append(pair)
        for r, c in coords:
            legend_mask[r, c] = True

    assert legend_pairs, "expected at least one edge legend"
    return legend_pairs, legend_mask


def _extract_objects(
    grid: np.ndarray, legend_mask: np.ndarray
) -> dict[int, tuple[int, int, int, int]]:
    objects: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(set(map(int, np.unique(grid))) - {BLACK, TEAL}):
        mask = (grid == color) & ~legend_mask
        if not np.any(mask):
            continue
        bbox = _bbox_from_mask(mask)
        r0, c0, r1, c1 = bbox
        patch = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert int(mask.sum()) == patch.size and np.all(patch), "each object must be one solid rectangle"
        assert color not in objects, "object colors must be unique"
        objects[color] = bbox
    assert objects, "expected at least one non-legend object"
    return objects


def _bridge_bbox(
    box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0a, c0a, r1a, c1a = box1
    r0b, c0b, r1b, c1b = box2

    overlap_r0 = max(r0a, r0b)
    overlap_r1 = min(r1a, r1b)
    if overlap_r0 <= overlap_r1:
        if c1a + 1 <= c0b - 1:
            return overlap_r0, c1a + 1, overlap_r1, c0b - 1
        if c1b + 1 <= c0a - 1:
            return overlap_r0, c1b + 1, overlap_r1, c0a - 1

    overlap_c0 = max(c0a, c0b)
    overlap_c1 = min(c1a, c1b)
    if overlap_c0 <= overlap_c1:
        if r1a + 1 <= r0b - 1:
            return r1a + 1, overlap_c0, r0b - 1, overlap_c1
        if r1b + 1 <= r0a - 1:
            return r1b + 1, overlap_c0, r0a - 1, overlap_c1

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the edge color-pair legend to bridge only the listed rectangle pairs with teal.

    Input structure:
    - A black background contains several differently colored solid rectangles.
    - One edge also contains several separated 1x2 legend blocks, each made from two object colors.
    - Teal is absent from the input and is reserved for the bridges.

    Rule:
    - Read each edge legend block as an unordered pair of object colors.
    - Find the two solid rectangles of those colors and fill the black rectangular gap between them inside their overlapping row band or overlapping column band.
    - Leave unlisted aligned pairs, the objects themselves, and the legend blocks unchanged.

    Key insight:
    - The small two-color edge pieces are not ordinary objects; they are an edge list telling you exactly which aligned rectangle pairs should be connected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert not np.any(g == TEAL), "teal must be absent from the input"

    legend_pairs, legend_mask = _extract_legends(g)
    objects = _extract_objects(g, legend_mask)

    out = g.copy()
    for color_a, color_b in legend_pairs:
        assert color_a in objects and color_b in objects, "legend colors must refer to object colors"
        bridge = _bridge_bbox(objects[color_a], objects[color_b])
        assert bridge is not None, "listed colors must identify separated aligned rectangles"
        r0, c0, r1, c1 = bridge
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == BLACK), "bridge gap must be pure background"
        out[r0 : r1 + 1, c0 : c1 + 1] = TEAL
    return out


def _sample_horizontal_track(
    row_top: int,
    row_bottom: int,
    col_left: int,
    col_right: int,
    colors: tuple[int, int, int],
) -> list[tuple[int, tuple[int, int, int, int]]] | None:
    if row_bottom - row_top + 1 < 4:
        return None

    overlap_h = random.randint(1, 2)
    overlap_r0 = random.randint(row_top, row_bottom - overlap_h + 1)
    overlap_r1 = overlap_r0 + overlap_h - 1
    top_room = min(1, overlap_r0 - row_top)
    bottom_room = min(1, row_bottom - overlap_r1)

    row_spans: list[tuple[int, int]] = []
    widths: list[int] = []
    for _ in range(3):
        top_extra = random.randint(0, top_room)
        bottom_extra = random.randint(0, bottom_room)
        row_spans.append((overlap_r0 - top_extra, overlap_r1 + bottom_extra))
        widths.append(random.randint(2, 4))

    gaps = [random.randint(1, 3), random.randint(1, 3)]
    total_w = sum(widths) + sum(gaps)
    if total_w > col_right - col_left + 1:
        return None

    start_col = random.randint(col_left, col_right - total_w + 1)
    cursor = start_col
    track: list[tuple[int, tuple[int, int, int, int]]] = []
    for idx, color in enumerate(colors):
        r0, r1 = row_spans[idx]
        w = widths[idx]
        track.append((color, (r0, cursor, r1, cursor + w - 1)))
        cursor += w
        if idx < 2:
            cursor += gaps[idx]
    return track


def _sample_vertical_track(
    row_top: int,
    row_bottom: int,
    col_left: int,
    col_right: int,
    colors: tuple[int, int, int],
) -> list[tuple[int, tuple[int, int, int, int]]] | None:
    if row_bottom - row_top + 1 < 10:
        return None

    overlap_w = random.randint(1, 2)
    overlap_c0 = random.randint(col_left, col_right - overlap_w + 1)
    overlap_c1 = overlap_c0 + overlap_w - 1
    left_room = min(1, overlap_c0 - col_left)
    right_room = min(1, col_right - overlap_c1)

    col_spans: list[tuple[int, int]] = []
    heights: list[int] = []
    for _ in range(3):
        left_extra = random.randint(0, left_room)
        right_extra = random.randint(0, right_room)
        col_spans.append((overlap_c0 - left_extra, overlap_c1 + right_extra))
        heights.append(random.randint(2, 4))

    gaps = [random.randint(1, 3), random.randint(1, 3)]
    total_h = sum(heights) + sum(gaps)
    if total_h > row_bottom - row_top + 1:
        return None

    start_row = random.randint(row_top, row_bottom - total_h + 1)
    cursor = start_row
    track: list[tuple[int, tuple[int, int, int, int]]] = []
    for idx, color in enumerate(colors):
        c0, c1 = col_spans[idx]
        h = heights[idx]
        track.append((color, (cursor, c0, cursor + h - 1, c1)))
        cursor += h
        if idx < 2:
            cursor += gaps[idx]
    return track


def _draw_rectangles(
    grid: np.ndarray, objects: list[tuple[int, tuple[int, int, int, int]]]
) -> None:
    for color, (r0, c0, r1, c1) in objects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _place_legends(grid: np.ndarray, edge: str, pairs: list[tuple[int, int]]) -> None:
    pairs = pairs[:]
    random.shuffle(pairs)

    if edge in {"top", "bottom"}:
        gaps = [random.randint(1, 3) for _ in range(max(0, len(pairs) - 1))]
        total = 2 * len(pairs) + sum(gaps)
        start = random.randint(0, grid.shape[1] - total)
        row = 0 if edge == "top" else grid.shape[0] - 1
        cursor = start
        for idx, pair in enumerate(pairs):
            colors = [pair[0], pair[1]]
            random.shuffle(colors)
            grid[row, cursor] = colors[0]
            grid[row, cursor + 1] = colors[1]
            cursor += 2
            if idx < len(gaps):
                cursor += gaps[idx]
        return

    gaps = [random.randint(1, 3) for _ in range(max(0, len(pairs) - 1))]
    total = len(pairs) + sum(gaps)
    start = random.randint(0, grid.shape[0] - total)
    c0 = 0 if edge == "left" else grid.shape[1] - 2
    cursor = start
    for idx, pair in enumerate(pairs):
        colors = [pair[0], pair[1]]
        random.shuffle(colors)
        grid[cursor, c0] = colors[0]
        grid[cursor, c0 + 1] = colors[1]
        cursor += 1
        if idx < len(gaps):
            cursor += gaps[idx]


def _validate_instance(
    grid: np.ndarray,
    object_boxes: dict[int, tuple[int, int, int, int]],
    candidates: list[tuple[int, int]],
    selected: list[tuple[int, int]],
) -> bool:
    selected_set = set(selected)
    if len(selected_set) != len(selected):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False

    for pair in candidates:
        bridge = _bridge_bbox(object_boxes[pair[0]], object_boxes[pair[1]])
        if bridge is None:
            return False
        r0, c0, r1, c1 = bridge
        has_teal = np.any(out[r0 : r1 + 1, c0 : c1 + 1] == TEAL)
        if pair in selected_set:
            if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == TEAL):
                return False
        elif has_teal:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(21, 29)
    w = random.randint(22, 30)
    edge = random.choice(("top", "bottom", "left", "right"))

    obj_top = 2 if edge == "top" else 1
    obj_bottom = h - 3 if edge == "bottom" else h - 2
    obj_left = 3 if edge == "left" else 1
    obj_right = w - 4 if edge == "right" else w - 2

    h_zone_top = obj_top + random.randint(0, 1)
    h_zone_bottom = h_zone_top + 4
    v_zone_top = h_zone_bottom + random.randint(2, 3)
    v_zone_bottom = obj_bottom
    if h_zone_bottom > obj_bottom or v_zone_top > v_zone_bottom:
        return None

    colors = random.sample(list(OBJECT_COLORS), 6)
    horizontal = _sample_horizontal_track(
        h_zone_top,
        h_zone_bottom,
        obj_left,
        obj_right,
        tuple(colors[:3]),
    )
    vertical = _sample_vertical_track(
        v_zone_top,
        v_zone_bottom,
        obj_left,
        obj_right,
        tuple(colors[3:]),
    )
    if horizontal is None or vertical is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    objects = horizontal + vertical
    _draw_rectangles(grid, objects)

    candidates = [
        tuple(sorted((horizontal[0][0], horizontal[1][0]))),
        tuple(sorted((horizontal[1][0], horizontal[2][0]))),
        tuple(sorted((vertical[0][0], vertical[1][0]))),
        tuple(sorted((vertical[1][0], vertical[2][0]))),
    ]
    selected = [
        random.choice(candidates[:2]),
        random.choice(candidates[2:]),
    ]
    remaining = [pair for pair in candidates if pair not in selected]
    if remaining and random.random() < 0.6:
        selected.append(random.choice(remaining))
    selected = list(dict.fromkeys(selected))
    if len(selected) >= len(candidates):
        return None

    _place_legends(grid, edge, selected)

    object_boxes = {color: bbox for color, bbox in objects}
    if not _validate_instance(grid, object_boxes, candidates, selected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
