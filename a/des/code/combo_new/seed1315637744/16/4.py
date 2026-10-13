from __future__ import annotations

import random

import numpy as np


GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
COLOR_COUNT_RANGE = (2, 5)
COUNT_OPTIONS = (2, 3, 4, 5, 6)
MAX_BOX_SIDE = 11
OVERLAP_PROBABILITY = 0.75
MAX_BOX_TRIES = 80
MAX_POINT_TRIES = 30
MAX_GENERATION_ATTEMPTS = 600


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if not _boxes_overlap(a, b):
        return 0
    row_overlap = min(a[2], b[2]) - max(a[0], b[0]) + 1
    col_overlap = min(a[3], b[3]) - max(a[1], b[1]) + 1
    return int(row_overlap * col_overlap)


def _box_area(box: tuple[int, int, int, int]) -> int:
    return int((box[2] - box[0] + 1) * (box[3] - box[1] + 1))


def _candidate_box_sizes(height: int, width: int, count: int) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    max_h = min(MAX_BOX_SIDE, height)
    max_w = min(MAX_BOX_SIDE, width)
    for box_h in range(3, max_h + 1, 2):
        for box_w in range(3, max_w + 1, 2):
            capacity = ((box_h + 1) // 2) * ((box_w + 1) // 2)
            if capacity >= count and box_h * box_w >= count + 4:
                sizes.append((box_h, box_w))
    return sizes


def _render_layers(
    shape: tuple[int, int],
    bg: int,
    groups: list[dict[str, object]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for group in sorted(groups, key=lambda item: int(item["count"])):
        r0, c0, r1, c1 = group["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(group["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each color's bounding rectangle, layering colors from fewer seed pixels to more seed pixels.

    Input structure:
    - A dominant background color fills most of the grid.
    - Several other colors appear only as isolated single-pixel seeds scattered on the background.
    - Each non-background color occurs a different number of times, and same-color seeds do not touch.

    Rule:
    - Group all non-background pixels by color and count the seeds in each group.
    - For each color, compute the smallest axis-aligned bounding box covering all of its seeds.
    - Starting from a background copy, fill each color's full bounding box in ascending order of seed count.
    - Later fills overwrite earlier ones, so colors with more seeds sit on top wherever rectangles overlap.

    Key insight:
    - The output rectangles come from per-color bounding boxes, but their overlap order is determined by visible seed multiplicity rather than by position, size, or color identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one non-background color"

    groups: list[dict[str, object]] = []
    seen_counts: set[int] = set()
    for color in colors:
        points = np.argwhere(g == color)
        count = int(points.shape[0])
        assert count not in seen_counts, "non-background colors must have distinct seed counts"
        seen_counts.add(count)

        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0)
        groups.append(
            {
                "color": color,
                "count": count,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )

    return _render_layers(g.shape, bg, groups)


def _place_boxes(
    rng: random.Random,
    height: int,
    width: int,
    counts: list[int],
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    partial_overlap_found = False

    for index, count in enumerate(counts):
        size_options = _candidate_box_sizes(height, width, count)
        if not size_options:
            return None
        rng.shuffle(size_options)

        placed = None
        for box_h, box_w in size_options[:]:
            require_overlap = bool(boxes) and (
                index == len(counts) - 1 and not partial_overlap_found
            )
            prefer_overlap = bool(boxes) and (
                require_overlap or rng.random() < OVERLAP_PROBABILITY
            )

            for _ in range(MAX_BOX_TRIES):
                if prefer_overlap:
                    ref = boxes[rng.randrange(len(boxes))]
                    row_lo = max(0, ref[0] - box_h + 1)
                    row_hi = min(height - box_h, ref[2])
                    col_lo = max(0, ref[1] - box_w + 1)
                    col_hi = min(width - box_w, ref[3])
                    if row_lo > row_hi or col_lo > col_hi:
                        continue
                    r0 = rng.randint(row_lo, row_hi)
                    c0 = rng.randint(col_lo, col_hi)
                else:
                    r0 = rng.randint(0, height - box_h)
                    c0 = rng.randint(0, width - box_w)

                candidate = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
                if candidate in boxes:
                    continue

                partials = 0
                for other in boxes:
                    area = _overlap_area(candidate, other)
                    if area == 0:
                        continue
                    if area < min(_box_area(candidate), _box_area(other)):
                        partials += 1

                if require_overlap and partials == 0:
                    continue
                if prefer_overlap and boxes and partials == 0:
                    continue

                placed = candidate
                partial_overlap_found = partial_overlap_found or partials > 0
                break

            if placed is not None:
                break

        if placed is None:
            return None
        boxes.append(placed)

    if not partial_overlap_found:
        return None
    return boxes


def _point_conflicts(
    point: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    if point in occupied:
        return True
    r, c = point
    return (
        (r - 1, c) in occupied
        or (r + 1, c) in occupied
        or (r, c - 1) in occupied
        or (r, c + 1) in occupied
    )


def _sample_points_for_box(
    rng: random.Random,
    box: tuple[int, int, int, int],
    count: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    candidates = [(r, c) for r in range(r0, r1 + 1, 2) for c in range(c0, c1 + 1, 2)]
    diagonals = [
        [(r0, c0), (r1, c1)],
        [(r0, c1), (r1, c0)],
    ]
    rng.shuffle(diagonals)

    for anchors in diagonals:
        if any(_point_conflicts(anchor, occupied) for anchor in anchors):
            continue

        remaining = [pt for pt in candidates if pt not in anchors and not _point_conflicts(pt, occupied)]
        need = count - len(anchors)
        if need < 0 or len(remaining) < need:
            continue

        for _ in range(MAX_POINT_TRIES):
            chosen = anchors + rng.sample(remaining, need)
            return chosen

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    color_count = rng.randint(*COLOR_COUNT_RANGE)

    bg = rng.randrange(10)
    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    colors = palette[:color_count]

    counts = list(rng.sample(COUNT_OPTIONS, color_count))
    boxes = _place_boxes(rng, height, width, counts)
    if boxes is None:
        return None

    groups: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()

    order = list(range(color_count))
    rng.shuffle(order)
    for index in order:
        points = _sample_points_for_box(rng, boxes[index], counts[index], occupied)
        if points is None:
            return None
        occupied.update(points)
        groups.append(
            {
                "color": int(colors[index]),
                "count": int(counts[index]),
                "bbox": boxes[index],
                "points": points,
            }
        )

    inp = np.full((height, width), bg, dtype=int)
    for group in groups:
        for r, c in group["points"]:
            inp[r, c] = int(group["color"])

    expected = _render_layers(inp.shape, bg, groups)
    visible_colors = {int(color) for color in np.unique(expected) if int(color) != bg}
    if visible_colors != {int(group["color"]) for group in groups}:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid input")
