from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
BASE_SHAPE = (30, 30)
PAIR_COUNT_RANGE = (2, 5)
EVEN_DISTANCES = (2, 4, 6, 8, 10)
ODD_DISTANCES = (1, 3, 5, 7, 9)
OUTER_PADDING_MAX = 2
GENERATION_ATTEMPTS = 400


def _nonzero_colors(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != BACKGROUND]


def _input_cells(spec: dict) -> list[tuple[int, int]]:
    axis = spec["axis"]
    distance = int(spec["distance"])
    r, c = spec["anchor"]

    if distance % 2 == 0:
        arm = distance // 2
        if axis == "h":
            return [(int(r), int(c - arm)), (int(r), int(c + arm))]
        return [(int(r - arm), int(c)), (int(r + arm), int(c))]

    span = distance // 2
    if axis == "h":
        return [(int(r), int(c - span)), (int(r), int(c + span + 1))]
    return [(int(r - span), int(c)), (int(r + span + 1), int(c))]


def _output_cells(spec: dict) -> list[tuple[int, int]]:
    axis = spec["axis"]
    distance = int(spec["distance"])
    r, c = spec["anchor"]

    if distance % 2 == 0:
        arm = distance // 2
        cells = {(int(r + delta), int(c)) for delta in range(-arm, arm + 1)}
        cells.update((int(r), int(c + delta)) for delta in range(-arm, arm + 1))
        return sorted(cells)

    if axis == "h":
        return [(int(r), int(c)), (int(r), int(c + 1))]
    return [(int(r), int(c)), (int(r + 1), int(c))]


def _reserve_bbox(spec: dict) -> tuple[int, int, int, int]:
    axis = spec["axis"]
    distance = int(spec["distance"])
    r, c = spec["anchor"]

    if distance % 2 == 0:
        arm = distance // 2
        return int(r - arm - 1), int(c - arm - 1), int(r + arm + 1), int(c + arm + 1)

    span = distance // 2
    if axis == "h":
        return int(r - 1), int(c - span - 1), int(r + 1), int(c + span + 2)
    return int(r - span - 1), int(c - 1), int(r + span + 2), int(c + 1)


def _reserve_area_hint(spec: dict) -> int:
    distance = int(spec["distance"])
    if distance % 2 == 0:
        side = distance + 3
        return side * side
    return 3 * (distance + 3)


def _anchor_candidates(spec: dict, shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    axis = spec["axis"]
    distance = int(spec["distance"])

    if distance % 2 == 0:
        arm = distance // 2
        rows = range(arm + 1, height - arm - 1)
        cols = range(arm + 1, width - arm - 1)
    elif axis == "h":
        span = distance // 2
        rows = range(1, height - 1)
        cols = range(span + 1, width - span - 2)
    else:
        span = distance // 2
        rows = range(span + 1, height - span - 2)
        cols = range(1, width - 1)

    anchors = [(int(r), int(c)) for r in rows for c in cols]
    random.shuffle(anchors)
    return anchors


def _box_is_free(occupied: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return not bool(occupied[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_box(occupied: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_specs(specs: list[dict]) -> list[dict] | None:
    occupied = np.zeros(BASE_SHAPE, dtype=bool)
    ordered = [dict(spec) for spec in specs]
    random.shuffle(ordered)
    ordered.sort(key=_reserve_area_hint, reverse=True)

    placed: list[dict] = []
    for spec in ordered:
        chosen = None
        for anchor in _anchor_candidates(spec, BASE_SHAPE):
            trial = dict(spec)
            trial["anchor"] = anchor
            if _box_is_free(occupied, _reserve_bbox(trial)):
                chosen = trial
                break
        if chosen is None:
            return None
        _mark_box(occupied, _reserve_bbox(chosen))
        placed.append(chosen)
    return placed


def _crop_box(specs: list[dict]) -> tuple[int, int, int, int]:
    boxes = [_reserve_bbox(spec) for spec in specs]
    r0 = min(box[0] for box in boxes)
    c0 = min(box[1] for box in boxes)
    r1 = max(box[2] for box in boxes)
    c1 = max(box[3] for box in boxes)

    extra_top = random.randint(0, min(OUTER_PADDING_MAX, r0))
    extra_left = random.randint(0, min(OUTER_PADDING_MAX, c0))
    extra_bottom = random.randint(0, min(OUTER_PADDING_MAX, BASE_SHAPE[0] - 1 - r1))
    extra_right = random.randint(0, min(OUTER_PADDING_MAX, BASE_SHAPE[1] - 1 - c1))
    return r0 - extra_top, c0 - extra_left, r1 + extra_bottom, c1 + extra_right


def _render_from_specs(specs: list[dict], crop_box: tuple[int, int, int, int], which: str) -> np.ndarray:
    r0, c0, r1, c1 = crop_box
    grid = np.full((r1 - r0 + 1, c1 - c0 + 1), BACKGROUND, dtype=int)

    for spec in specs:
        cells = _input_cells(spec) if which == "input" else _output_cells(spec)
        for r, c in cells:
            rr = int(r - r0)
            cc = int(c - c0)
            assert 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]
            assert grid[rr, cc] == BACKGROUND
            grid[rr, cc] = int(spec["color"])
    return grid


def _sample_specs() -> list[dict]:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    even_count = random.randint(1, pair_count - 1)
    odd_count = pair_count - even_count

    distances = random.sample(EVEN_DISTANCES, even_count) + random.sample(ODD_DISTANCES, odd_count)
    colors = random.sample(PALETTE, pair_count)
    random.shuffle(distances)

    specs: list[dict] = []
    for color, distance in zip(colors, distances):
        specs.append(
            {
                "color": int(color),
                "axis": random.choice(("h", "v")),
                "distance": int(distance),
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    placed = _place_specs(specs)
    if placed is None:
        return None

    crop_box = _crop_box(placed)
    inp = _render_from_specs(placed, crop_box, "input")
    expected = _render_from_specs(placed, crop_box, "output")

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each aligned same-color pixel pair toward the middle, turning even gaps into crosses and odd gaps into touching midpoint pairs.

    Input structure:
    - The grid has a black background and several foreground colors.
    - Each foreground color appears exactly twice, and the two cells of a color lie on one row or one column.
    - Different pairs are spaced so their transformed results do not overlap.

    Rule:
    - For each color pair, measure the row or column gap d between its two cells.
    - If d is even, erase the pair and draw a same-color plus sign centered at the midpoint, with arm length d/2.
    - If d is odd, erase the pair and slide both cells inward to the two central adjacent cells on their original row or column.

    Key insight:
    - Every pair follows the same "move toward the midpoint" instruction; parity alone decides whether the middle is a cell that can bloom into a cross or a split center that leaves two cells face-to-face.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    out = np.full_like(g, BACKGROUND)
    height, width = g.shape

    for color in _nonzero_colors(g):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"

        (r0, c0), (r1, c1) = sorted((tuple(map(int, coord)) for coord in coords))
        same_row = r0 == r1
        same_col = c0 == c1
        assert same_row ^ same_col, "each same-color pair must be aligned horizontally or vertically"

        cells: list[tuple[int, int]]
        if same_row:
            left, right = sorted((c0, c1))
            distance = right - left
            assert distance > 0
            if distance % 2 == 0:
                arm = distance // 2
                center_r, center_c = r0, left + arm
                plus = {(center_r + delta, center_c) for delta in range(-arm, arm + 1)}
                plus.update((center_r, center_c + delta) for delta in range(-arm, arm + 1))
                cells = sorted((int(r), int(c)) for r, c in plus)
            else:
                span = distance // 2
                cells = [(int(r0), int(left + span)), (int(r0), int(left + span + 1))]
        else:
            top, bottom = sorted((r0, r1))
            distance = bottom - top
            assert distance > 0
            if distance % 2 == 0:
                arm = distance // 2
                center_r, center_c = top + arm, c0
                plus = {(center_r + delta, center_c) for delta in range(-arm, arm + 1)}
                plus.update((center_r, center_c + delta) for delta in range(-arm, arm + 1))
                cells = sorted((int(r), int(c)) for r, c in plus)
            else:
                span = distance // 2
                cells = [(int(top + span), int(c0)), (int(top + span + 1), int(c0))]

        for r, c in cells:
            assert 0 <= r < height and 0 <= c < width, "transformed cells must stay in bounds"
            assert out[r, c] == BACKGROUND, "transformed pair outputs must not overlap"
            out[r, c] = int(color)

    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_ATTEMPTS} attempts")
