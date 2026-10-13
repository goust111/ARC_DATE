from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7

NOTCH_COLORS = (BLUE, RED, GREEN, ORANGE, MAGENTA)
DIRECTIONS = ("up", "right", "down", "left")

# --------------- generator config ---------------
GRID_H_RANGE = (9, 22)
GRID_W_RANGE = (9, 22)
FRAME_H_RANGE = (7, 18)
FRAME_W_RANGE = (7, 18)
OUTER_MARGIN_MAX = 2
POINT_COUNT_WEIGHTS = {1: 2, 2: 4, 3: 3, 4: 2}
DIR_PRESENT_PROB = 0.55
DISTRACTOR_NOTCH_MAX = 5
FORCED_OVERLAP_PROB = 0.45
GEN_RETRIES = 400


def _mode_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).reshape(-1)
    counts = Counter(map(int, flat))
    best_color, best_count = max(counts.items(), key=lambda item: item[1])
    assert sum(int(count == best_count) for count in counts.values()) == 1
    return int(best_color)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _infer_structure(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], int, int, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_color(g)
    bbox = _bbox_from_mask(g != bg)
    r0, c0, r1, c1 = bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2

    border_mask = _frame_mask(g.shape, bbox)
    border_color = _mode_color(g[border_mask])

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    marker_candidates = sorted(
        {
            int(v)
            for v in inner.reshape(-1)
            if int(v) != int(bg) and int(v) != int(border_color)
        }
    )
    assert len(marker_candidates) == 1
    marker_color = int(marker_candidates[0])

    points = sorted(
        (int(r), int(c))
        for r, c in np.argwhere(g == marker_color)
        if r0 < int(r) < r1 and c0 < int(c) < c1
    )
    assert points
    return int(bg), bbox, int(border_color), int(marker_color), points


def _aligned_notches(
    grid: np.ndarray, bbox: tuple[int, int, int, int], border_color: int
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = bbox

    top = {c: int(g[r0, c]) for c in range(c0 + 1, c1) if int(g[r0, c]) != int(border_color)}
    right = {r: int(g[r, c1]) for r in range(r0 + 1, r1) if int(g[r, c1]) != int(border_color)}
    bottom = {c: int(g[r1, c]) for c in range(c0 + 1, c1) if int(g[r1, c]) != int(border_color)}
    left = {r: int(g[r, c0]) for r in range(r0 + 1, r1) if int(g[r, c0]) != int(border_color)}
    return top, right, bottom, left


def _draw_arms(
    out: np.ndarray,
    points: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    marker_color: int,
    top: dict[int, int],
    right: dict[int, int],
    bottom: dict[int, int],
    left: dict[int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox

    for r, c in points:
        color = top.get(c)
        if color is None:
            continue
        for rr in range(r - 1, r0, -1):
            out[rr, c] = int(color)

    for r, c in points:
        color = right.get(r)
        if color is None:
            continue
        for cc in range(c + 1, c1):
            out[r, cc] = int(color)

    for r, c in points:
        color = bottom.get(c)
        if color is None:
            continue
        for rr in range(r + 1, r1):
            out[rr, c] = int(color)

    for r, c in points:
        color = left.get(r)
        if color is None:
            continue
        for cc in range(c - 1, c0, -1):
            out[r, cc] = int(color)

    for r, c in points:
        out[r, c] = int(marker_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw colored cross arms from yellow intersection points to aligned border notches.

    The frame is a 1-pixel border whose default color is replaced at selected edge
    cells by colored notch markers. Each interior yellow point inherits up/right/down/left
    arm colors from same-column / same-row notches, and each arm extends exactly across
    the black gap between the point and the frame. Overlaps are resolved by drawing in the
    order up, right, down, left.
    """
    g = np.asarray(grid, dtype=int)
    _, bbox, border_color, marker_color, points = _infer_structure(g)
    top, right, bottom, left = _aligned_notches(g, bbox, border_color)
    return _draw_arms(g.copy(), points, bbox, marker_color, top, right, bottom, left)


def _render_expected(
    inp: np.ndarray,
    bbox: tuple[int, int, int, int],
    points: list[tuple[int, int]],
    arms_by_point: list[dict[str, int]],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.asarray(inp, dtype=int).copy()

    for (r, c), arms in zip(points, arms_by_point):
        if "up" in arms:
            for rr in range(r - 1, r0, -1):
                out[rr, c] = int(arms["up"])

    for (r, c), arms in zip(points, arms_by_point):
        if "right" in arms:
            for cc in range(c + 1, c1):
                out[r, cc] = int(arms["right"])

    for (r, c), arms in zip(points, arms_by_point):
        if "down" in arms:
            for rr in range(r + 1, r1):
                out[rr, c] = int(arms["down"])

    for (r, c), arms in zip(points, arms_by_point):
        if "left" in arms:
            for cc in range(c - 1, c0, -1):
                out[r, cc] = int(arms["left"])

    for r, c in points:
        out[r, c] = YELLOW
    return out


def _pick_frame_size() -> tuple[int, int, int, int, int, int]:
    for _ in range(100):
        frame_h = random.randint(*FRAME_H_RANGE)
        frame_w = random.randint(*FRAME_W_RANGE)
        top_margin = random.randint(0, OUTER_MARGIN_MAX)
        bottom_margin = random.randint(0, OUTER_MARGIN_MAX)
        left_margin = random.randint(0, OUTER_MARGIN_MAX)
        right_margin = random.randint(0, OUTER_MARGIN_MAX)
        grid_h = frame_h + top_margin + bottom_margin
        grid_w = frame_w + left_margin + right_margin
        if GRID_H_RANGE[0] <= grid_h <= GRID_H_RANGE[1] and GRID_W_RANGE[0] <= grid_w <= GRID_W_RANGE[1]:
            return grid_h, grid_w, top_margin, left_margin, frame_h, frame_w
    raise RuntimeError("failed to pick a valid frame size")


def _pick_point_count(max_points: int) -> int:
    options = [n for n in sorted(POINT_COUNT_WEIGHTS) if n <= max_points]
    weights = [POINT_COUNT_WEIGHTS[n] for n in options]
    assert options
    return int(random.choices(options, weights=weights, k=1)[0])


def _pick_directions() -> dict[str, int]:
    arms: dict[str, int] = {}
    for direction in DIRECTIONS:
        if random.random() < DIR_PRESENT_PROB:
            arms[direction] = random.choice(NOTCH_COLORS)
    if not arms:
        direction = random.choice(DIRECTIONS)
        arms[direction] = random.choice(NOTCH_COLORS)
    return arms


def _force_overlap(points: list[tuple[int, int]], arms_by_point: list[dict[str, int]]) -> None:
    if len(points) < 2:
        return
    (r_a, c_a), (r_b, c_b) = points[0], points[1]
    horizontal = "right" if c_b > c_a else "left"
    vertical = "up" if r_a < r_b else "down"
    arms_by_point[0].setdefault(horizontal, random.choice(NOTCH_COLORS))
    arms_by_point[1].setdefault(vertical, random.choice(NOTCH_COLORS))


def _candidate_edge_slots(
    bbox: tuple[int, int, int, int], point_rows: set[int], point_cols: set[int]
) -> list[tuple[str, int]]:
    r0, c0, r1, c1 = bbox
    slots: list[tuple[str, int]] = []
    for c in range(c0 + 1, c1):
        if c not in point_cols:
            slots.append(("top", c))
            slots.append(("bottom", c))
    for r in range(r0 + 1, r1):
        if r not in point_rows:
            slots.append(("left", r))
            slots.append(("right", r))
    return slots


def _place_distractor_notches(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    point_rows: set[int],
    point_cols: set[int],
) -> None:
    slots = _candidate_edge_slots(bbox, point_rows, point_cols)
    if not slots:
        return

    count = random.randint(0, min(DISTRACTOR_NOTCH_MAX, len(slots)))
    for side, pos in random.sample(slots, count):
        color = random.choice(NOTCH_COLORS)
        r0, c0, r1, c1 = bbox
        if side == "top":
            grid[r0, pos] = color
        elif side == "bottom":
            grid[r1, pos] = color
        elif side == "left":
            grid[pos, c0] = color
        else:
            grid[pos, c1] = color


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    grid_h, grid_w, r0, c0, frame_h, frame_w = _pick_frame_size()
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    bbox = (r0, c0, r1, c1)

    row_choices = list(range(r0 + 2, r1 - 1))
    col_choices = list(range(c0 + 2, c1 - 1))
    max_points = min(4, len(row_choices), len(col_choices))
    assert max_points >= 1
    point_count = _pick_point_count(max_points)

    point_rows = random.sample(row_choices, point_count)
    point_cols = random.sample(col_choices, point_count)
    random.shuffle(point_cols)
    points = sorted((int(r), int(c)) for r, c in zip(point_rows, point_cols))
    assert len({r for r, _ in points}) == point_count
    assert len({c for _, c in points}) == point_count

    arms_by_point = [_pick_directions() for _ in points]
    if len(points) >= 2 and random.random() < FORCED_OVERLAP_PROB:
        _force_overlap(points, arms_by_point)

    inp = np.full((grid_h, grid_w), BLACK, dtype=int)
    inp[r0, c0 : c1 + 1] = GRAY
    inp[r1, c0 : c1 + 1] = GRAY
    inp[r0 : r1 + 1, c0] = GRAY
    inp[r0 : r1 + 1, c1] = GRAY

    for (r, c), arms in zip(points, arms_by_point):
        inp[r, c] = YELLOW
        if "up" in arms:
            inp[r0, c] = int(arms["up"])
        if "right" in arms:
            inp[r, c1] = int(arms["right"])
        if "down" in arms:
            inp[r1, c] = int(arms["down"])
        if "left" in arms:
            inp[r, c0] = int(arms["left"])

    _place_distractor_notches(inp, bbox, {r for r, _ in points}, {c for _, c in points})

    border_values = np.concatenate(
        [
            inp[r0, c0 : c1 + 1],
            inp[r1, c0 : c1 + 1],
            inp[r0 + 1 : r1, c0],
            inp[r0 + 1 : r1, c1],
        ]
    )
    assert _mode_color(border_values) == GRAY

    expected = _render_expected(inp, bbox, points, arms_by_point)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp, expected = _generate_case()
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
