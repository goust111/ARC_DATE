import numpy as np


_DIR_VECTORS = {
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "up": (-1, 0),
}

_ARROW_OFFSETS = {
    "right": [(0, -1), (0, 0), (0, 1), (-1, 1), (1, 1)],
    "down": [(-1, 0), (0, 0), (1, 0), (1, -1), (1, 1)],
    "left": [(0, -1), (0, 0), (0, 1), (-1, -1), (1, -1)],
    "up": [(-1, 0), (0, 0), (1, 0), (-1, -1), (-1, 1)],
}

_ARROW_TEMPLATES = {
    frozenset({(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)}): ("right", (1, 2)),
    frozenset({(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)}): ("left", (1, 0)),
    frozenset({(0, 1), (1, 1), (2, 1), (0, 0), (0, 2)}): ("up", (0, 1)),
    frozenset({(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)}): ("down", (2, 1)),
}

_RNG = np.random.default_rng()

BG_COLOR = 0
COLOR_POOL = np.arange(1, 10)
MIN_EDGE_GAP = 4
MAX_EXTRA_TOTAL = 5
MAX_EXTRA_PER_EDGE = 2
SPAN_SLACK_MAX = 8
MARGIN_LOW = 1
MARGIN_HIGH_EXCLUSIVE = 4
MAX_GENERATION_ATTEMPTS = 300

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _is_four_connected(coords: np.ndarray) -> bool:
    cells = {tuple(map(int, cell)) for cell in coords}
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in _NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(cells)


def _extract_arrows(grid: np.ndarray) -> tuple[list[dict[str, object]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    arrows: list[dict[str, object]] = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 5, "each arrow must have exactly five cells"
        assert _is_four_connected(coords), "each arrow must be a single 4-connected object"

        min_r, min_c = coords.min(axis=0)
        norm = frozenset((int(r - min_r), int(c - min_c)) for r, c in coords)
        template = _ARROW_TEMPLATES.get(norm)
        assert template is not None, "foreground objects must match one of the four arrow templates"

        direction, tip_rel = template
        cells = [(int(r), int(c)) for r, c in coords]
        arrows.append(
            {
                "color": color,
                "direction": direction,
                "delta": _DIR_VECTORS[direction],
                "tip": (int(min_r) + tip_rel[0], int(min_c) + tip_rel[1]),
                "cells": cells,
            }
        )

    assert len(arrows) >= 2, "expected at least two arrows"
    return arrows, bg


def _find_next_arrow(arrows: list[dict[str, object]], idx: int) -> int:
    tip_r, tip_c = arrows[idx]["tip"]
    dr, dc = arrows[idx]["delta"]
    best_dist = None
    best_idx = None

    for j, other in enumerate(arrows):
        if j == idx:
            continue
        cells = other["cells"]
        if dr == 0:
            candidates = [c for r, c in cells if r == tip_r and ((c > tip_c) if dc > 0 else (c < tip_c))]
            if not candidates:
                continue
            frontier = min(candidates) if dc > 0 else max(candidates)
            dist = abs(frontier - tip_c)
        else:
            candidates = [r for r, c in cells if c == tip_c and ((r > tip_r) if dr > 0 else (r < tip_r))]
            if not candidates:
                continue
            frontier = min(candidates) if dr > 0 else max(candidates)
            dist = abs(frontier - tip_r)

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = j

    assert best_idx is not None, "every arrow must point to another arrow"
    return int(best_idx)


def _assert_single_cycle(mapping: list[int]) -> None:
    n = len(mapping)
    assert sorted(mapping) == list(range(n)), "the pointer graph must be a permutation"
    seen = set()
    cur = 0
    while cur not in seen:
        seen.add(cur)
        cur = mapping[cur]
    assert cur == 0 and len(seen) == n, "all arrows must belong to one directed cycle"


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate arrow colors forward along the unique pointer cycle defined by arrow directions."""
    g = np.asarray(grid, dtype=int)
    arrows, _bg = _extract_arrows(g)
    mapping = [_find_next_arrow(arrows, i) for i in range(len(arrows))]
    _assert_single_cycle(mapping)

    out = g.copy()
    next_colors = [int(arrows[j]["color"]) for j in mapping]
    for arrow, color in zip(arrows, next_colors):
        for r, c in arrow["cells"]:
            out[r, c] = color
    return out


def _arrow_cells(center: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0 = center
    return [(r0 + dr, c0 + dc) for dr, dc in _ARROW_OFFSETS[direction]]


def _sample_extras() -> np.ndarray:
    extras = np.zeros(4, dtype=int)
    total = int(_RNG.integers(0, MAX_EXTRA_TOTAL + 1))
    for _ in range(total):
        choices = np.flatnonzero(extras < MAX_EXTRA_PER_EDGE)
        edge = int(_RNG.choice(choices))
        extras[edge] += 1
    return extras


def _sample_axis_points(start: int, span: int, count: int) -> list[int]:
    intervals = count - 1
    min_span = MIN_EDGE_GAP * intervals
    assert span >= min_span
    extra = span - min_span
    if intervals == 1:
        extras = np.array([extra], dtype=int)
    elif extra == 0:
        extras = np.zeros(intervals, dtype=int)
    else:
        extras = _RNG.multinomial(extra, np.full(intervals, 1.0 / intervals))

    steps = extras + MIN_EDGE_GAP
    points = [start]
    cur = start
    for step in steps:
        cur += int(step)
        points.append(cur)
    return points


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    ar, ac = a
    br, bc = b
    if ar == br and bc > ac:
        return "right"
    if ar == br and bc < ac:
        return "left"
    if ac == bc and br > ar:
        return "down"
    if ac == bc and br < ar:
        return "up"
    raise AssertionError("consecutive cycle vertices must share a row or column")


def _build_cycle_centers() -> list[tuple[int, int]]:
    extras = _sample_extras()

    width_span = MIN_EDGE_GAP * max(int(extras[0]) + 1, int(extras[2]) + 1)
    height_span = MIN_EDGE_GAP * max(int(extras[1]) + 1, int(extras[3]) + 1)
    width_span += int(_RNG.integers(0, SPAN_SLACK_MAX + 1))
    height_span += int(_RNG.integers(0, SPAN_SLACK_MAX + 1))

    r0 = int(_RNG.integers(MARGIN_LOW, MARGIN_HIGH_EXCLUSIVE))
    c0 = int(_RNG.integers(MARGIN_LOW, MARGIN_HIGH_EXCLUSIVE))
    r1 = r0 + height_span
    c1 = c0 + width_span

    top_cols = _sample_axis_points(c0, width_span, int(extras[0]) + 2)
    right_rows = _sample_axis_points(r0, height_span, int(extras[1]) + 2)
    bottom_cols = _sample_axis_points(c0, width_span, int(extras[2]) + 2)
    left_rows = _sample_axis_points(r0, height_span, int(extras[3]) + 2)

    top = [(r0, c) for c in top_cols]
    right = [(r, c1) for r in right_rows]
    bottom = [(r1, c) for c in bottom_cols[::-1]]
    left = [(r, c0) for r in left_rows[::-1]]

    centers = top + right[1:] + bottom[1:] + left[1:-1]
    if int(_RNG.integers(0, 2)):
        centers = centers[::-1]
    shift = int(_RNG.integers(0, len(centers)))
    return centers[shift:] + centers[:shift]


def _render_cycle(centers: list[tuple[int, int]], colors: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    max_r = max(r for r, _ in centers) + int(_RNG.integers(MARGIN_LOW, MARGIN_HIGH_EXCLUSIVE)) + 1
    max_c = max(c for _, c in centers) + int(_RNG.integers(MARGIN_LOW, MARGIN_HIGH_EXCLUSIVE)) + 1
    grid = np.full((max_r + 1, max_c + 1), BG_COLOR, dtype=int)
    expected = np.full_like(grid, BG_COLOR)
    occupied = set()
    arrows = []

    for i, center in enumerate(centers):
        nxt = centers[(i + 1) % len(centers)]
        direction = _direction_between(center, nxt)
        cells = _arrow_cells(center, direction)
        for r, c in cells:
            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                return None
            if (r, c) in occupied:
                return None
        occupied.update(cells)
        arrows.append({"cells": cells, "color": int(colors[i])})

    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[r, c] = arrow["color"]

    for i, arrow in enumerate(arrows):
        next_color = int(arrows[(i + 1) % len(arrows)]["color"])
        for r, c in arrow["cells"]:
            expected[r, c] = next_color

    return grid, expected


def _random_transform_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = int(_RNG.integers(0, 4))
    inp = np.rot90(inp, k)
    out = np.rot90(out, k)
    if int(_RNG.integers(0, 2)):
        inp = np.fliplr(inp)
        out = np.fliplr(out)
    if int(_RNG.integers(0, 2)):
        inp = np.flipud(inp)
        out = np.flipud(out)
    return inp.copy(), out.copy()


def _gen_attempt() -> np.ndarray | None:
    centers = _build_cycle_centers()
    if len(centers) > len(COLOR_POOL):
        return None
    colors = _RNG.choice(COLOR_POOL, size=len(centers), replace=False)

    rendered = _render_cycle(centers, colors)
    if rendered is None:
        return None
    inp, expected = rendered
    inp, expected = _random_transform_pair(inp, expected)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-cycle input")
