from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
LEGEND_COLS = 2
PALETTE = (1, 2, 3, 4, 7, 8, 9)
PAIR_COUNT_CHOICES = (2, 3, 4)
SIZE_BY_PAIR_COUNT = {
    2: ((16, 20), (16, 20)),
    3: ((18, 22), (18, 22)),
    4: ((20, 24), (20, 24)),
}
BOX_HEIGHT_RANGE = (3, 5)
BOX_WIDTH_RANGE = (4, 6)
FORCED_OVERLAP_MIN = 2
MAX_GENERATE_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.asarray(coords, dtype=int)
    if pts.size == 0:
        raise ValueError("bbox requires at least one coordinate")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _expand_bbox(
    box: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, box)
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _intersection(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _legend_pairs(grid: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        source = int(grid[r, 0])
        fill = int(grid[r, 1])
        if source == BACKGROUND:
            break
        assert fill != BACKGROUND, "legend fill colors must be non-black"
        pairs.append((source, fill))
    assert 2 <= len(pairs) <= 4, "legend must have 2-4 rows"
    assert len({source for source, _fill in pairs}) == len(pairs), "source colors must be unique"
    assert len({fill for _source, fill in pairs}) == len(pairs), "fill colors must be unique"
    return pairs


def _legend_mask(shape: tuple[int, int], rows: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:rows, :LEGEND_COLS] = True
    return mask


def _complete_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    r0, c0, _r1, c1 = _bbox_coords(pts)
    axis_sum = int(c0 + c1)
    completed = {tuple(map(int, rc)) for rc in pts.tolist()}
    h, w = map(int, shape)
    for r, c in list(completed):
        mc = axis_sum - int(c)
        if 0 <= int(r) < h and 0 <= int(mc) < w and (int(r), int(mc)) not in completed:
            completed.add((int(r), int(mc)))
    ordered = np.array(sorted(completed), dtype=int)
    assert ordered.ndim == 2 and ordered.shape[1] == 2
    return ordered


def _render_output(
    grid: np.ndarray, *, complete: bool, reverse_priority: bool
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= LEGEND_COLS + 1, "grid must leave room for dots to the right of the legend"

    pairs = _legend_pairs(g)
    legend_mask = _legend_mask(g.shape, len(pairs))
    dot_mask = (g != BACKGROUND) & ~legend_mask
    assert set(map(int, g[dot_mask])).issubset({source for source, _fill in pairs})

    coords_by_source: dict[int, np.ndarray] = {}
    source_mask = np.zeros(g.shape, dtype=bool)
    out = np.full_like(g, BACKGROUND)

    for source, _fill in pairs:
        visible = np.argwhere((g == source) & ~legend_mask)
        assert visible.size > 0, "each source color must appear outside the legend"
        coords = _complete_coords(visible, g.shape) if complete else np.asarray(visible, dtype=int)
        coords_by_source[source] = coords
        assert not source_mask[coords[:, 0], coords[:, 1]].any(), "source objects must stay disjoint"
        source_mask[coords[:, 0], coords[:, 1]] = True

    order = pairs if reverse_priority else list(reversed(pairs))
    for source, fill in order:
        r0, c0, r1, c1 = _expand_bbox(_bbox_coords(coords_by_source[source]), (h, w))
        paint = np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        paint &= ~source_mask[r0 : r1 + 1, c0 : c1 + 1]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[paint] = int(fill)

    for source, _fill in pairs:
        coords = coords_by_source[source]
        out[coords[:, 0], coords[:, 1]] = int(source)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each sparse symmetric dot set, then expand and stack legend-colored fill rectangles.

    Input structure:
    - The grid has black background, a two-column legend in the upper-left corner, and sparse source-color dots
      in columns 2 and beyond.
    - Each legend row stores one `(source color, fill color)` pair; source colors are unique and the legend ends
      at the first row whose left legend cell is black.
    - Outside the legend, each source color forms a sparse point set that is intended to be left-right symmetric
      inside its own bounding box, except that one or two mirrored points are missing.

    Rule:
    - Read the legend rows from top to bottom.
    - For each source color, collect its non-legend dots, compute their bounding box, and use that box's vertical
      center line as the mirror axis.
    - For every visible dot, add its mirrored partner if that mirrored cell is in bounds and currently black.
    - Recompute the source-color bounding box after completion, expand it by one cell in every direction, and
      treat that expanded box as the fill region for the legend-paired fill color.
    - Paint expanded boxes from the last legend row back to the first so earlier legend rows are in front; fill
      may overwrite lower-priority fills but never overwrites any source-color dot. Clear the legend itself to black.

    Key insight:
    - The sparse dots are not the final object: you must first restore the mirrored omissions, then use the
      completed dots to define rectangular fill layers whose overlaps are resolved only by legend order.
    """
    return _render_output(np.asarray(grid, dtype=int), complete=True, reverse_priority=False)


def _ranges_overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _rect_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], min_gap: int
) -> bool:
    return (
        a[2] + min_gap < b[0]
        or b[2] + min_gap < a[0]
        or a[3] + min_gap < b[1]
        or b[3] + min_gap < a[1]
    )


def _sample_forced_pair(
    shape: tuple[int, int],
    size_a: tuple[int, int],
    size_b: tuple[int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    h, w = map(int, shape)
    ha, wa = map(int, size_a)
    hb, wb = map(int, size_b)

    for _ in range(300):
        if random.random() < 0.5:
            total_w = wa + wb + 1
            if total_w > w - 3:
                continue
            c0 = random.randint(3, w - total_w)
            a_c0 = c0
            b_c0 = c0 + wa + 1
            a_r0 = random.randint(0, h - ha)
            b_r0 = random.randint(0, h - hb)
            overlap = _ranges_overlap_len(a_r0, a_r0 + ha - 1, b_r0, b_r0 + hb - 1)
            if overlap < FORCED_OVERLAP_MIN:
                continue
            a = (a_r0, a_c0, a_r0 + ha - 1, a_c0 + wa - 1)
            b = (b_r0, b_c0, b_r0 + hb - 1, b_c0 + wb - 1)
        else:
            total_h = ha + hb + 1
            if total_h > h:
                continue
            a_r0 = random.randint(0, h - total_h)
            b_r0 = a_r0 + ha + 1
            a_c0 = random.randint(3, w - wa)
            b_c0 = random.randint(3, w - wb)
            overlap = _ranges_overlap_len(a_c0, a_c0 + wa - 1, b_c0, b_c0 + wb - 1)
            if overlap < FORCED_OVERLAP_MIN:
                continue
            a = (a_r0, a_c0, a_r0 + ha - 1, a_c0 + wa - 1)
            b = (b_r0, b_c0, b_r0 + hb - 1, b_c0 + wb - 1)

        expanded_a = _expand_bbox(a, shape)
        expanded_b = _expand_bbox(b, shape)
        inter = _intersection(expanded_a, expanded_b)
        if inter is None:
            continue
        if (inter[2] - inter[0] + 1) * (inter[3] - inter[1] + 1) < 2:
            continue
        return a, b
    return None


def _sample_additional_box(
    shape: tuple[int, int],
    size: tuple[int, int],
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = map(int, shape)
    box_h, box_w = map(int, size)
    candidates: list[tuple[int, int, int, int]] = []
    for r0 in range(0, h - box_h + 1):
        for c0 in range(3, w - box_w + 1):
            box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if all(_rect_separated(box, other, min_gap=2) for other in existing):
                candidates.append(box)
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_boxes(shape: tuple[int, int], n_pairs: int) -> list[tuple[int, int, int, int]] | None:
    sizes = [
        (
            random.randint(*BOX_HEIGHT_RANGE),
            random.randint(*BOX_WIDTH_RANGE),
        )
        for _ in range(n_pairs)
    ]
    forced = _sample_forced_pair(shape, sizes[0], sizes[1])
    if forced is None:
        return None
    boxes = [forced[0], forced[1]]
    for size in sizes[2:]:
        box = _sample_additional_box(shape, size, boxes)
        if box is None:
            return None
        boxes.append(box)
    random.shuffle(boxes)
    return boxes


def _sample_point_sets(
    box: tuple[int, int, int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    r0, c0, r1, c1 = map(int, box)
    axis_sum = c0 + c1

    inner_cols = [c for c in range(c0 + 1, c1 + 1) if c < axis_sum - c]
    if not inner_cols:
        raise ValueError("box is too narrow for recoverable missing points")

    points = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

    candidate_slots = [(r, c) for r in range(r0, r1 + 1) for c in inner_cols]
    preferred = [(r, c) for r, c in candidate_slots if r not in (r0, r1)]
    slot_pool = preferred if preferred else candidate_slots
    extra_count = 1 if len(slot_pool) == 1 else random.choice((1, 2))
    slots = random.sample(slot_pool, extra_count)

    mirrored_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in slots:
        mate = (r, axis_sum - c)
        points.add((r, c))
        points.add(mate)
        mirrored_pairs.append(((r, c), mate))

    visible = set(points)
    remove_count = 1 if len(mirrored_pairs) == 1 else random.choice((1, 2))
    for left, right in random.sample(mirrored_pairs, remove_count):
        visible.remove(random.choice((left, right)))

    if not (3 <= len(visible) <= 7):
        raise ValueError("visible point count drifted outside the intended range")
    return points, visible


def _choose_fill_colors(sources: list[int]) -> list[int]:
    remaining = [color for color in PALETTE if color not in sources]
    if len(remaining) >= len(sources) and random.random() < 0.5:
        fills = random.sample(remaining, len(sources))
        return [int(color) for color in fills]

    fills = sources[:]
    while True:
        random.shuffle(fills)
        if all(src != fill for src, fill in zip(sources, fills)):
            return [int(color) for color in fills]


def _build_input(
    shape: tuple[int, int],
    pairs: list[tuple[int, int]],
    visible_by_source: dict[int, set[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    grid[: len(pairs), :LEGEND_COLS] = np.asarray(pairs, dtype=int)
    for source, _fill in pairs:
        coords = np.array(sorted(visible_by_source[source]), dtype=int)
        grid[coords[:, 0], coords[:, 1]] = int(source)
    return grid


def _validate_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    pairs = _legend_pairs(g)
    legend_mask = _legend_mask(g.shape, len(pairs))

    boxes: list[tuple[int, int, int, int]] = []
    missing_found = False
    for source, _fill in pairs:
        visible = np.argwhere((g == source) & ~legend_mask)
        if visible.size == 0 or not (3 <= len(visible) <= 7):
            return False
        completed = _complete_coords(visible, g.shape)
        if len(completed) <= len(visible):
            return False
        if _bbox_coords(visible) != _bbox_coords(completed):
            return False
        missing_found = True
        boxes.append(_expand_bbox(_bbox_coords(completed), g.shape))
    if not missing_found:
        return False

    overlap_seen = False
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            inter = _intersection(boxes[i], boxes[j])
            if inter is None:
                continue
            r0, c0, r1, c1 = inter
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if np.any(patch == BACKGROUND):
                overlap_seen = True
    if not overlap_seen:
        return False

    solved = main(g)
    no_completion = _render_output(g, complete=False, reverse_priority=False)
    reverse_priority = _render_output(g, complete=True, reverse_priority=True)
    if not np.any(solved != no_completion):
        return False
    if not np.any(solved != reverse_priority):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        n_pairs = random.choice(PAIR_COUNT_CHOICES)
        (h_lo, h_hi), (w_lo, w_hi) = SIZE_BY_PAIR_COUNT[n_pairs]
        shape = (random.randint(h_lo, h_hi), random.randint(w_lo, w_hi))

        boxes = _sample_boxes(shape, n_pairs)
        if boxes is None:
            continue

        sources = random.sample(PALETTE, n_pairs)
        fills = _choose_fill_colors(sources)
        pairs = list(zip(sources, fills))

        visible_by_source: dict[int, set[tuple[int, int]]] = {}
        ok = True
        for source, box in zip(sources, boxes):
            try:
                _full, visible = _sample_point_sets(box)
            except ValueError:
                ok = False
                break
            visible_by_source[int(source)] = visible
        if not ok:
            continue

        grid = _build_input(shape, pairs, visible_by_source)
        if _validate_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 6")
