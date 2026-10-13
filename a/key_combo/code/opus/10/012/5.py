from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
GREY = 5
OBJECT_COLORS = tuple(color for color in range(1, 10) if color != GREY)

GRID_MIN = 13
GRID_MAX = 18
OBJECT_COUNT_CHOICES = (3, 4)
EXPANDED_SIDE_MIN = 4
EXPANDED_SIDE_MAX = 8
DOT_COUNT_MIN = 2
DOT_COUNT_MAX = 4
WINNER_GREY_MAX = 2
OTHER_GREY_MAX = 6
BOX_GAP = 1
PLACEMENT_TRIES = 120
DOT_TRIES = 80
GENERATION_TRIES = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _expand_bbox(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _filled_crop(grid: np.ndarray, color: int, box: tuple[int, int, int, int]) -> np.ndarray:
    patch = _crop(grid, box)
    patch[patch == BACKGROUND] = int(color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each sparse color-dot group, then crop the expanded rectangle with the fewest grey markers.

    Input structure:
    - The grid has a black background, several non-black non-grey dot colors, and scattered grey pixels.
    - Each non-grey foreground color appears as one sparse dot group whose full color-wise bbox defines an object.
    - Grey pixels do not belong to any object and may lie inside or outside the later expanded rectangles.

    Rule:
    - For each non-grey foreground color, compute the bbox of all pixels of that color and expand it by one cell in each direction, clamped to the grid.
    - Inside that expanded rectangle, fill every black pixel with the object's color while preserving all existing non-black pixels, including grey markers.
    - Count how many grey pixels each expanded rectangle contains and select the unique rectangle with the smallest count.
    - Return the tight crop of that selected expanded rectangle after the black cells inside it have been filled.

    Key insight:
    - Grey pixels matter only after the color groups are expanded: they neither define the objects nor change their bboxes, but they determine which expanded rectangle is kept.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = sorted(set(int(value) for value in np.unique(g)) - {BACKGROUND, GREY})
    assert colors, "expected at least one non-grey foreground color"

    candidates: list[tuple[int, np.ndarray]] = []
    for color in colors:
        coords = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        assert coords, "each selected color must appear at least once"
        expanded = _expand_bbox(_bbox(coords), g.shape)
        grey_count = int(np.count_nonzero(_crop(g, expanded) == GREY))
        candidates.append((grey_count, _filled_crop(g, color, expanded)))

    min_grey = min(count for count, _ in candidates)
    winners = [patch for count, patch in candidates if count == min_grey]
    assert len(winners) == 1, "expected a unique rectangle with the fewest grey markers"
    return winners[0]


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = BOX_GAP
) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _contract_expanded_box(
    box: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return (
        0 if r0 == 0 else r0 + 1,
        0 if c0 == 0 else c0 + 1,
        h - 1 if r1 == h - 1 else r1 - 1,
        w - 1 if c1 == w - 1 else c1 - 1,
    )


def _sample_sparse_dots(
    box: tuple[int, int, int, int], count: int
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    for _ in range(DOT_TRIES):
        dots = random.sample(cells, count)
        if _bbox(dots) == box:
            return dots
    return None


def _place_expanded_boxes(
    height: int, width: int, dims: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int] | None] = [None] * len(dims)
    order = sorted(range(len(dims)), key=lambda i: dims[i][0] * dims[i][1], reverse=True)
    force_border = random.random() < 0.45
    forced_idx = random.choice(order) if force_border else None

    for idx in order:
        box_h, box_w = dims[idx]
        placed = False
        for _ in range(PLACEMENT_TRIES):
            if idx == forced_idx:
                side = random.choice(("top", "bottom", "left", "right"))
                if side == "top":
                    r0, c0 = 0, random.randint(0, width - box_w)
                elif side == "bottom":
                    r0, c0 = height - box_h, random.randint(0, width - box_w)
                elif side == "left":
                    r0, c0 = random.randint(0, height - box_h), 0
                else:
                    r0, c0 = random.randint(0, height - box_h), width - box_w
            else:
                r0 = random.randint(0, height - box_h)
                c0 = random.randint(0, width - box_w)

            candidate = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if all(
                other is None or not _boxes_too_close(candidate, other)
                for other in boxes
            ):
                boxes[idx] = candidate
                placed = True
                break
        if not placed:
            return None

    return [box for box in boxes if box is not None]


def _choose_grey_counts(count: int) -> tuple[list[int], int]:
    winner_idx = random.randrange(count)
    winner_count = random.randint(0, WINNER_GREY_MAX)
    larger_counts = random.sample(range(winner_count + 1, OTHER_GREY_MAX + 1), count - 1)
    grey_counts = [winner_count] * count
    cursor = 0
    for idx in range(count):
        if idx == winner_idx:
            continue
        grey_counts[idx] = larger_counts[cursor]
        cursor += 1
    return grey_counts, winner_idx


def _sample_positions(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    return random.sample(cells, count)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    object_count = random.choice(OBJECT_COUNT_CHOICES)
    colors = random.sample(OBJECT_COLORS, object_count)
    expanded_dims = [
        (
            random.randint(EXPANDED_SIDE_MIN, EXPANDED_SIDE_MAX),
            random.randint(EXPANDED_SIDE_MIN, EXPANDED_SIDE_MAX),
        )
        for _ in range(object_count)
    ]
    expanded_boxes = _place_expanded_boxes(height, width, expanded_dims)
    if expanded_boxes is None:
        return None

    grey_counts, winner_idx = _choose_grey_counts(object_count)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    object_dots: list[list[tuple[int, int]]] = []

    for box_h, box_w, expanded_box in zip(
        (dim[0] for dim in expanded_dims),
        (dim[1] for dim in expanded_dims),
        expanded_boxes,
    ):
        dot_box = _contract_expanded_box(expanded_box, grid.shape)
        dot_area = (dot_box[2] - dot_box[0] + 1) * (dot_box[3] - dot_box[1] + 1)
        max_dots = min(DOT_COUNT_MAX, dot_area - 1)
        if max_dots < DOT_COUNT_MIN:
            return None
        dot_count = random.randint(DOT_COUNT_MIN, max_dots)
        dots = _sample_sparse_dots(dot_box, dot_count)
        if dots is None:
            return None
        object_dots.append(dots)

    for color, dots in zip(colors, object_dots):
        for r, c in dots:
            grid[r, c] = color

    occupied = {cell for dots in object_dots for cell in dots}
    for expanded_box, grey_count in zip(expanded_boxes, grey_counts):
        r0, c0, r1, c1 = expanded_box
        cells = [
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if (r, c) not in occupied
        ]
        if len(cells) < grey_count:
            return None
        for r, c in _sample_positions(cells, grey_count):
            grid[r, c] = GREY
            occupied.add((r, c))

    outside_cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r, c] == BACKGROUND
        and all(not (box[0] <= r <= box[2] and box[1] <= c <= box[3]) for box in expanded_boxes)
    ]
    outside_grey_count = min(
        len(outside_cells),
        random.randint(0, 4) + (1 if outside_cells and random.random() < 0.6 else 0),
    )
    for r, c in _sample_positions(outside_cells, outside_grey_count):
        grid[r, c] = GREY

    expected = _filled_crop(grid, colors[winner_idx], expanded_boxes[winner_idx])
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
