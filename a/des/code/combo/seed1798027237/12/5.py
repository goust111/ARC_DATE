from __future__ import annotations

from collections import Counter
import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a non-empty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "background must be uniquely dominant"
    return int(ranked[0][0])


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    """Return non-separator half-open spans for one axis."""
    seps = sorted(set(int(x) for x in separators))
    assert seps, "expected at least one separator index"

    spans: list[tuple[int, int]] = []
    start = 0
    i = 0
    while i < len(seps):
        sep_start = seps[i]
        assert start < sep_start, "separator lines must leave a non-empty region"
        spans.append((start, sep_start))
        while i + 1 < len(seps) and seps[i + 1] == seps[i] + 1:
            i += 1
        start = seps[i] + 1
        i += 1

    assert start < length, "separator lines must leave a final non-empty region"
    spans.append((start, length))
    return spans


def _separator_structure(
    grid: np.ndarray, bg: int
) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = map(int, grid.shape)
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if not rows or not cols:
            continue
        try:
            row_spans = _spans(height, rows)
            col_spans = _spans(width, cols)
        except AssertionError:
            continue
        if len(row_spans) == 2 and len(col_spans) == 2:
            candidates.append((color, rows, cols, row_spans, col_spans))

    assert len(candidates) == 1, "expected one separator color that forms a 2x2 quadrant split"
    return candidates[0]


def _components_in_rect(
    grid: np.ndarray, r0: int, r1: int, c0: int, c1: int, ignored: set[int]
) -> list[dict]:
    sub_h = int(r1 - r0)
    sub_w = int(c1 - c0)
    seen = np.zeros((sub_h, sub_w), dtype=bool)
    comps: list[dict] = []

    for lr in range(sub_h):
        for lc in range(sub_w):
            if seen[lr, lc]:
                continue
            color = int(grid[r0 + lr, c0 + lc])
            if color in ignored:
                seen[lr, lc] = True
                continue

            seen[lr, lc] = True
            stack = [(lr, lc)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((r0 + cr, c0 + cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < sub_h and 0 <= nc < sub_w) or seen[nr, nc]:
                        continue
                    if int(grid[r0 + nr, c0 + nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            min_r = int(arr[:, 0].min())
            min_c = int(arr[:, 1].min())
            max_r = int(arr[:, 0].max())
            max_c = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (min_r, min_c, max_r, max_c),
                    "center2": (min_r + max_r, min_c + max_c),
                }
            )

    return comps


def _legend_axis(components: list[dict]) -> str | None:
    if len(components) < 2:
        return None
    rows = {int(comp["center2"][0]) for comp in components}
    cols = {int(comp["center2"][1]) for comp in components}
    if len(rows) == 1 and len(cols) > 1:
        return "row"
    if len(cols) == 1 and len(rows) > 1:
        return "col"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder quadrant objects according to the aligned color legend in the fourth quadrant.

    Input structure:
    - The background is the most common color.
    - One full separator row and one full separator column split the grid into four quadrants.
    - Exactly one quadrant is a legend: its monochrome objects are aligned on one row or one column.
    - Each other quadrant contains distinct-colored monochrome 4-connected objects whose colors come from the legend.

    Rule:
    - Read the legend colors from left-to-right or top-to-bottom.
    - In each non-legend quadrant, sort occupied positions by object center from top-to-bottom then left-to-right.
    - Sort the quadrant's objects by the legend color priority.
    - Move the i-th color-ranked object to the i-th position-ranked center, preserving its shape.

    Key insight:
    The legend provides a color order, while each target quadrant provides the destination centers.
    The output is produced by rebinding color-ranked objects to position-ranked slots.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    sep_color, _sep_rows, _sep_cols, row_spans, col_spans = _separator_structure(g, bg)

    quadrants = []
    for row_i, (r0, r1) in enumerate(row_spans):
        for col_i, (c0, c1) in enumerate(col_spans):
            comps = _components_in_rect(g, r0, r1, c0, c1, {int(bg), int(sep_color)})
            assert comps, "every quadrant must contain at least one foreground object"
            quadrants.append(
                {
                    "index": (row_i, col_i),
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "components": comps,
                }
            )

    legend_candidates = []
    for quadrant in quadrants:
        comps = quadrant["components"]
        axis = _legend_axis(comps)
        colors = [int(comp["color"]) for comp in comps]
        if axis is not None and len(colors) == len(set(colors)):
            legend_candidates.append((quadrant, axis))

    assert len(legend_candidates) == 1, "expected exactly one aligned legend quadrant"
    legend_quadrant, legend_axis = legend_candidates[0]

    legend_components = legend_quadrant["components"]
    if legend_axis == "row":
        legend_components = sorted(
            legend_components, key=lambda comp: (int(comp["center2"][1]), int(comp["center2"][0]))
        )
    else:
        legend_components = sorted(
            legend_components, key=lambda comp: (int(comp["center2"][0]), int(comp["center2"][1]))
        )
    priority = [int(comp["color"]) for comp in legend_components]
    priority_rank = {color: idx for idx, color in enumerate(priority)}
    assert len(priority_rank) == len(priority), "legend colors must be distinct"

    out = g.copy()
    for quadrant in quadrants:
        if quadrant["index"] == legend_quadrant["index"]:
            continue

        comps = quadrant["components"]
        colors = [int(comp["color"]) for comp in comps]
        assert len(colors) == len(set(colors)), "quadrant objects must have pairwise distinct colors"
        assert set(colors).issubset(priority_rank), "quadrant colors must come from the legend"

        by_position = sorted(comps, key=lambda comp: (int(comp["center2"][0]), int(comp["center2"][1])))
        by_color = sorted(comps, key=lambda comp: priority_rank[int(comp["color"])])

        out[quadrant["r0"] : quadrant["r1"], quadrant["c0"] : quadrant["c1"]] = int(bg)
        for src, dst in zip(by_color, by_position):
            dr2 = int(dst["center2"][0]) - int(src["center2"][0])
            dc2 = int(dst["center2"][1]) - int(src["center2"][1])
            assert dr2 % 2 == 0 and dc2 % 2 == 0, "center translation must land on grid cells"
            dr = dr2 // 2
            dc = dc2 // 2
            for r, c in src["coords"]:
                rr = int(r) + int(dr)
                cc = int(c) + int(dc)
                assert quadrant["r0"] <= rr < quadrant["r1"], "translated object must stay within its quadrant"
                assert quadrant["c0"] <= cc < quadrant["c1"], "translated object must stay within its quadrant"
                assert int(out[rr, cc]) == int(bg), "translated objects must not overlap"
                out[rr, cc] = int(src["color"])

    return out


# --------------- generator (o5x) ---------------

# Palette and layout.
ALL_COLORS = tuple(range(10))
SEPARATOR_COLOR = 5
QUAD_SIZE_RANGE = (10, 14)
LEGEND_BLOCK_SIZES = (1, 2, 3)
LEGEND_GAPS = (1, 2)
OBJECT_COUNT_RANGE = (3, 5)
BOX_SIZE_CHOICES = (3, 5)
OBJECT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400
QUADRANT_PLACEMENT_ATTEMPTS = 60


def _is_unique_mode(grid: np.ndarray, color: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    top = counts.most_common(2)
    return bool(top) and int(top[0][0]) == int(color) and (len(top) == 1 or top[0][1] > top[1][1])


def _quadrant_bounds(
    top_h: int, bottom_h: int, left_w: int, right_w: int
) -> dict[int, tuple[int, int, int, int]]:
    height = int(top_h + 1 + bottom_h)
    width = int(left_w + 1 + right_w)
    return {
        0: (0, top_h, 0, left_w),
        1: (0, top_h, left_w + 1, width),
        2: (top_h + 1, height, 0, left_w),
        3: (top_h + 1, height, left_w + 1, width),
    }


def _place_legend(
    height: int, width: int, colors: list[int], axis: str
) -> list[tuple[int, int, int, int]] | None:
    choices = []
    for block in LEGEND_BLOCK_SIZES:
        for gap in LEGEND_GAPS:
            span = len(colors) * block + (len(colors) - 1) * gap
            if axis == "row":
                if block <= height and span <= width:
                    choices.append((block, gap, span))
            else:
                if span <= height and block <= width:
                    choices.append((block, gap, span))

    if not choices:
        return None

    block, gap, span = random.choice(choices)
    placements = []
    if axis == "row":
        r0 = random.randint(0, height - block)
        c0 = random.randint(0, width - span)
        for i, color in enumerate(colors):
            placements.append((r0, c0 + i * (block + gap), block, int(color)))
    else:
        r0 = random.randint(0, height - span)
        c0 = random.randint(0, width - block)
        for i, color in enumerate(colors):
            placements.append((r0 + i * (block + gap), c0, block, int(color)))
    return placements


def _mark_box(occupied: np.ndarray, top: int, left: int, box: int) -> None:
    h, w = map(int, occupied.shape)
    r0 = max(0, int(top - OBJECT_PADDING))
    r1 = min(h, int(top + box + OBJECT_PADDING))
    c0 = max(0, int(left - OBJECT_PADDING))
    c1 = min(w, int(left + box + OBJECT_PADDING))
    occupied[r0:r1, c0:c1] = True


def _box_candidates(occupied: np.ndarray, box: int) -> list[tuple[int, int]]:
    h, w = map(int, occupied.shape)
    candidates = []
    for top in range(h - box + 1):
        for left in range(w - box + 1):
            r0 = max(0, int(top - OBJECT_PADDING))
            r1 = min(h, int(top + box + OBJECT_PADDING))
            c0 = max(0, int(left - OBJECT_PADDING))
            c1 = min(w, int(left + box + OBJECT_PADDING))
            if not occupied[r0:r1, c0:c1].any():
                candidates.append((top, left))
    return candidates


def _sample_box_positions(height: int, width: int, box: int, count: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    tops: list[tuple[int, int]] = []
    for _ in range(count):
        candidates = _box_candidates(occupied, box)
        if not candidates:
            return None
        top, left = random.choice(candidates)
        tops.append((int(top), int(left)))
        _mark_box(occupied, int(top), int(left), int(box))

    centers = [(top + box // 2, left + box // 2) for top, left in tops]
    if len({r for r, _c in centers}) == 1 or len({c for _r, c in centers}) == 1:
        return None
    return tops


def _manhattan_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    out = [(int(r), int(c))]
    if random.random() < 0.5:
        while c != end[1]:
            c += 1 if end[1] > c else -1
            out.append((int(r), int(c)))
        while r != end[0]:
            r += 1 if end[0] > r else -1
            out.append((int(r), int(c)))
    else:
        while r != end[0]:
            r += 1 if end[0] > r else -1
            out.append((int(r), int(c)))
        while c != end[1]:
            c += 1 if end[1] > c else -1
            out.append((int(r), int(c)))
    return out


def _frontier(cells: set[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    frontier = set()
    for r, c in cells:
        for dr, dc in NEIGHBORS4:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < size and 0 <= cc < size and (rr, cc) not in cells:
                frontier.add((rr, cc))
    return list(frontier)


def _random_shape(box: int) -> np.ndarray:
    center = (box // 2, box // 2)
    anchors = [
        (0, random.randint(0, box - 1)),
        (box - 1, random.randint(0, box - 1)),
        (random.randint(0, box - 1), 0),
        (random.randint(0, box - 1), box - 1),
    ]
    cells = {center}
    for anchor in anchors:
        cells.update(_manhattan_path(center, anchor))

    min_target = len(cells)
    max_target = box * box
    desired = random.randint(min_target, min(max_target, min_target + box + 3))
    while len(cells) < desired:
        choices = _frontier(cells, box)
        if not choices:
            break
        cells.add(random.choice(choices))

    return np.array(sorted(cells), dtype=int)


def _paint_coords(grid: np.ndarray, top: int, left: int, coords: np.ndarray, color: int) -> None:
    for r, c in coords:
        grid[int(top + r), int(left + c)] = int(color)


def _paint_square(grid: np.ndarray, top: int, left: int, size: int, color: int) -> None:
    grid[int(top) : int(top + size), int(left) : int(left + size)] = int(color)


def _render_quadrant_input(
    grid: np.ndarray, bounds: tuple[int, int, int, int], objects: list[dict], box: int
) -> None:
    r0, _r1, c0, _c1 = bounds
    radius = box // 2
    for obj in objects:
        center_r, center_c = obj["center"]
        top = r0 + center_r - radius
        left = c0 + center_c - radius
        _paint_coords(grid, int(top), int(left), obj["shape"], int(obj["color"]))


def _render_quadrant_output(
    grid: np.ndarray, bounds: tuple[int, int, int, int], objects: list[dict], priority_rank: dict[int, int], bg: int
) -> None:
    r0, r1, c0, c1 = bounds
    grid[r0:r1, c0:c1] = int(bg)
    ordered_positions = sorted((tuple(obj["center"]) for obj in objects), key=lambda p: (int(p[0]), int(p[1])))
    ordered_objects = sorted(objects, key=lambda obj: priority_rank[int(obj["color"])])

    for src, dst_center in zip(ordered_objects, ordered_positions):
        box = int(src["box"])
        radius = box // 2
        top = r0 + int(dst_center[0]) - radius
        left = c0 + int(dst_center[1]) - radius
        _paint_coords(grid, int(top), int(left), src["shape"], int(src["color"]))


def _sample_quadrant_objects(height: int, width: int, colors: list[int]) -> tuple[int, list[dict]] | None:
    box_choices = [box for box in BOX_SIZE_CHOICES if box <= height and box <= width]
    random.shuffle(box_choices)
    for box in box_choices:
        for _ in range(QUADRANT_PLACEMENT_ATTEMPTS):
            tops = _sample_box_positions(height, width, box, len(colors))
            if tops is None:
                continue
            objects = []
            shuffled_colors = random.sample(colors, len(colors))
            for (top, left), color in zip(tops, shuffled_colors):
                objects.append(
                    {
                        "center": (int(top + box // 2), int(left + box // 2)),
                        "color": int(color),
                        "shape": _random_shape(box),
                        "box": int(box),
                    }
                )
            return int(box), objects
    return None


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    bg_choices = [color for color in ALL_COLORS if color != SEPARATOR_COLOR]
    bg = int(random.choice(bg_choices))

    count = random.randint(*OBJECT_COUNT_RANGE)
    object_colors = random.sample([c for c in ALL_COLORS if c not in {bg, SEPARATOR_COLOR}], count)
    legend_quadrant = random.randrange(4)
    legend_axis = random.choice(("row", "col"))

    size_min, size_max = QUAD_SIZE_RANGE
    top_h = random.randint(size_min, size_max)
    bottom_h = random.randint(size_min, size_max)
    left_w = random.randint(size_min, size_max)
    right_w = random.randint(size_min, size_max)
    bounds = _quadrant_bounds(top_h, bottom_h, left_w, right_w)
    height = top_h + 1 + bottom_h
    width = left_w + 1 + right_w

    grid = np.full((height, width), bg, dtype=int)
    grid[top_h, :] = int(SEPARATOR_COLOR)
    grid[:, left_w] = int(SEPARATOR_COLOR)

    lr0, lr1, lc0, lc1 = bounds[legend_quadrant]
    legend_placements = _place_legend(lr1 - lr0, lc1 - lc0, object_colors, legend_axis)
    if legend_placements is None:
        return None
    for local_r, local_c, size, color in legend_placements:
        _paint_square(grid, lr0 + local_r, lc0 + local_c, size, color)

    quadrant_specs: dict[int, dict] = {}
    for quadrant in range(4):
        if quadrant == legend_quadrant:
            continue
        r0, r1, c0, c1 = bounds[quadrant]
        sampled = _sample_quadrant_objects(r1 - r0, c1 - c0, object_colors)
        if sampled is None:
            return None
        box, objects = sampled
        quadrant_specs[quadrant] = {"box": box, "objects": objects}
        _render_quadrant_input(grid, bounds[quadrant], objects, box)

    priority_rank = {color: idx for idx, color in enumerate(object_colors)}
    expected = grid.copy()
    for quadrant, spec in quadrant_specs.items():
        _render_quadrant_output(expected, bounds[quadrant], spec["objects"], priority_rank, bg)

    if np.array_equal(grid, expected):
        return None
    if not _is_unique_mode(grid, bg):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_example()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input for puzzle 5")
