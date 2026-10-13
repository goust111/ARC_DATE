from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))

GRID_SIZE_RANGE = (16, 26)
RECT_COUNT_RANGE = (3, 6)
SIDE_LEN_RANGE = (3, 10)
GEN_RETRIES = 400
PAIR_RETRIES = 120
EXTRA_RETRIES = 160


def _pair_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangles_from_grid(grid: np.ndarray) -> list[dict]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BACKGROUND]
    rects = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        r0, c0, r1, c1 = _pair_bbox(coords)
        assert r0 < r1 and c0 < c1, "each color pair must span a non-degenerate rectangle"
        rects.append({"color": color, "bbox": (r0, c0, r1, c1)})

    left_edges = [rect["bbox"][1] for rect in rects]
    assert len(left_edges) == len(set(left_edges)), "rectangle left edges must be unique"
    assert len(rects) >= 2, "expected at least two hidden rectangles"
    return rects


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return (
        max(a[0], b[0]) <= min(a[2], b[2])
        and max(a[1], b[1]) <= min(a[3], b[3])
    )


def _intersection_counts(rects: list[dict]) -> list[int]:
    counts = [0] * len(rects)
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _intersects(rects[i]["bbox"], rects[j]["bbox"]):
                counts[i] += 1
                counts[j] += 1
    return counts


def _bars_from_rectangles(rects: list[dict]) -> np.ndarray:
    ordered = sorted(rects, key=lambda rect: rect["bbox"][1])
    counts = _intersection_counts(ordered)
    height = max(counts)
    assert height > 0, "puzzle instances must contain at least one intersecting pair"

    out = np.zeros((height, len(ordered)), dtype=int)
    for col, (rect, count) in enumerate(zip(ordered, counts)):
        if count > 0:
            out[height - count :, col] = int(rect["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Report each hidden rectangle's intersection count as a same-color vertical bar.

    Input structure:
    - The grid has a black background and several non-black colors.
    - Each foreground color appears exactly twice as isolated marker pixels.
    - The two same-color pixels are opposite corners of one hidden axis-aligned rectangle.

    Rule:
    - For each color, take the inclusive bounding box of its two pixels as that color's hidden rectangle.
    - Count how many other hidden rectangles share at least one cell with it; full containment and single-corner contact both count.
    - Sort rectangles by left boundary from left to right, then output one column per rectangle, filled bottom-up with that color for exactly its intersection count cells.

    Key insight:
    - The visible pixels are only diagonal generators; the real objects are the unseen rectangles implied by those pairs, and only their pairwise overlap counts survive in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    rects = _rectangles_from_grid(g)
    return _bars_from_rectangles(rects)


def _marker_cells(rect: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    if rect["diag"] == "main":
        return (r0, c0), (r1, c1)
    return (r0, c1), (r1, c0)


def _reserve_marker_neighborhoods(markers: tuple[tuple[int, int], tuple[int, int]], reserved: set[tuple[int, int]]) -> None:
    for r, c in markers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                reserved.add((r + dr, c + dc))


def _accept_rect(
    rect: dict,
    height: int,
    width: int,
    used_lefts: set[int],
    reserved: set[tuple[int, int]],
) -> bool:
    r0, c0, r1, c1 = rect["bbox"]
    if not (0 <= r0 < r1 < height and 0 <= c0 < c1 < width):
        return False
    if c0 in used_lefts:
        return False
    markers = _marker_cells(rect)
    if markers[0] == markers[1]:
        return False
    if any(marker in reserved for marker in markers):
        return False
    return True


def _commit_rect(rect: dict, rects: list[dict], used_lefts: set[int], reserved: set[tuple[int, int]]) -> None:
    rects.append(rect)
    used_lefts.add(rect["bbox"][1])
    _reserve_marker_neighborhoods(_marker_cells(rect), reserved)


def _make_rect(color: int, bbox: tuple[int, int, int, int], diag: str) -> dict:
    return {"color": int(color), "bbox": tuple(map(int, bbox)), "diag": diag}


def _random_diag() -> str:
    return random.choice(("main", "anti"))


def _random_span(limit: int) -> int:
    lo, hi = SIDE_LEN_RANGE
    return random.randint(lo, min(hi, limit))


def _sample_overlap_pair(height: int, width: int, colors: list[int]) -> list[dict] | None:
    for _ in range(PAIR_RETRIES):
        h1 = _random_span(height - 1)
        w1 = _random_span(width - 1)
        if width - w1 < 4:
            continue
        r0a = random.randint(0, height - h1)
        c0a = random.randint(0, width - w1 - 1)
        a = (r0a, c0a, r0a + h1 - 1, c0a + w1 - 1)

        c0b_lo = c0a + 1
        c0b_hi = min(a[3] - 1, width - 3)
        if c0b_lo > c0b_hi:
            continue
        h2 = _random_span(height - 1)
        min_r0b = max(0, a[0] - h2 + 1)
        max_r0b = min(height - h2, a[2] - 1)
        if min_r0b > max_r0b:
            continue
        c0b = random.randint(c0b_lo, c0b_hi)
        w2 = _random_span(width - c0b)
        r0b = random.randint(min_r0b, max_r0b)
        b = (r0b, c0b, r0b + h2 - 1, c0b + w2 - 1)

        rects: list[dict] = []
        used_lefts: set[int] = set()
        reserved: set[tuple[int, int]] = set()
        first = _make_rect(colors[0], a, _random_diag())
        second = _make_rect(colors[1], b, _random_diag())
        if not _accept_rect(first, height, width, used_lefts, reserved):
            continue
        _commit_rect(first, rects, used_lefts, reserved)
        if not _accept_rect(second, height, width, used_lefts, reserved):
            continue
        if not _intersects(a, b):
            continue
        _commit_rect(second, rects, used_lefts, reserved)
        return rects
    return None


def _sample_touch_pair(height: int, width: int, colors: list[int]) -> list[dict] | None:
    for _ in range(PAIR_RETRIES):
        h1 = _random_span(height - 4)
        w1 = _random_span(width - 4)
        h2 = _random_span(height - h1 + 1)
        w2 = _random_span(width - w1 + 1)
        total_h = h1 + h2 - 1
        total_w = w1 + w2 - 1
        if total_h > height or total_w > width:
            continue

        r0a = random.randint(0, height - total_h)
        c0a = random.randint(0, width - total_w)
        a = (r0a, c0a, r0a + h1 - 1, c0a + w1 - 1)
        b = (a[2], a[3], a[2] + h2 - 1, a[3] + w2 - 1)

        rects: list[dict] = []
        used_lefts: set[int] = set()
        reserved: set[tuple[int, int]] = set()
        first = _make_rect(colors[0], a, "anti")
        second = _make_rect(colors[1], b, "anti")
        if not _accept_rect(first, height, width, used_lefts, reserved):
            continue
        _commit_rect(first, rects, used_lefts, reserved)
        if not _accept_rect(second, height, width, used_lefts, reserved):
            continue
        if not _intersects(a, b):
            continue
        _commit_rect(second, rects, used_lefts, reserved)
        return rects
    return None


def _sample_containment_pair(height: int, width: int, colors: list[int]) -> list[dict] | None:
    for _ in range(PAIR_RETRIES):
        outer_h = random.randint(7, min(12, height))
        outer_w = random.randint(7, min(12, width))
        if outer_h > height or outer_w > width:
            continue
        r0a = random.randint(0, height - outer_h)
        c0a = random.randint(0, width - outer_w)
        a = (r0a, c0a, r0a + outer_h - 1, c0a + outer_w - 1)

        inner_h = random.randint(3, outer_h - 2)
        inner_w = random.randint(3, outer_w - 2)
        r0b = random.randint(r0a + 1, a[2] - inner_h)
        c0b = random.randint(c0a + 1, a[3] - inner_w)
        b = (r0b, c0b, r0b + inner_h - 1, c0b + inner_w - 1)

        rects: list[dict] = []
        used_lefts: set[int] = set()
        reserved: set[tuple[int, int]] = set()
        first = _make_rect(colors[0], a, _random_diag())
        second = _make_rect(colors[1], b, _random_diag())
        if not _accept_rect(first, height, width, used_lefts, reserved):
            continue
        _commit_rect(first, rects, used_lefts, reserved)
        if not _accept_rect(second, height, width, used_lefts, reserved):
            continue
        if not _intersects(a, b):
            continue
        _commit_rect(second, rects, used_lefts, reserved)
        return rects
    return None


def _add_random_rect(
    rects: list[dict],
    used_lefts: set[int],
    reserved: set[tuple[int, int]],
    height: int,
    width: int,
    color: int,
) -> bool:
    for _ in range(EXTRA_RETRIES):
        h = _random_span(height - 1)
        w = _random_span(width - 1)
        r0 = random.randint(0, height - h)
        c0 = random.randint(0, width - w)
        rect = _make_rect(color, (r0, c0, r0 + h - 1, c0 + w - 1), _random_diag())
        if not _accept_rect(rect, height, width, used_lefts, reserved):
            continue
        _commit_rect(rect, rects, used_lefts, reserved)
        return True
    return False


def _render_input(height: int, width: int, rects: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for rect in rects:
        for r, c in _marker_cells(rect):
            grid[r, c] = int(rect["color"])
    return grid


def _expected_from_rects(rects: list[dict]) -> np.ndarray:
    simplified = [{"color": rect["color"], "bbox": rect["bbox"]} for rect in rects]
    return _bars_from_rectangles(simplified)


def _sample_base_pair(height: int, width: int, colors: list[int]) -> list[dict] | None:
    kind = random.choice(("overlap", "touch", "contain"))
    if kind == "overlap":
        return _sample_overlap_pair(height, width, colors)
    if kind == "touch":
        return _sample_touch_pair(height, width, colors)
    return _sample_containment_pair(height, width, colors)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    n_rects = random.randint(*RECT_COUNT_RANGE)
    colors = random.sample(PALETTE, n_rects)

    rects = _sample_base_pair(height, width, colors[:2])
    if rects is None:
        return None

    used_lefts = {rect["bbox"][1] for rect in rects}
    reserved: set[tuple[int, int]] = set()
    for rect in rects:
        _reserve_marker_neighborhoods(_marker_cells(rect), reserved)

    for color in colors[2:]:
        if not _add_random_rect(rects, used_lefts, reserved, height, width, color):
            return None

    inp = _render_input(height, width, rects)
    expected = _expected_from_rects(rects)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-rectangle input")
