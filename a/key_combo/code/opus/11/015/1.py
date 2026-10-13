from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

TOP_COLOR = RED
BOTTOM_COLOR = BLUE
LEFT_COLOR = GREEN
RIGHT_COLOR = YELLOW
SIDE_COLORS = (TOP_COLOR, BOTTOM_COLOR, LEFT_COLOR, RIGHT_COLOR)
HORIZONTAL_COLORS = (LEFT_COLOR, RIGHT_COLOR)
VERTICAL_COLORS = (TOP_COLOR, BOTTOM_COLOR)

MIN_SIZE = 11
MAX_SIZE = 13
MAX_PER_COLOR = 3
GEN_ATTEMPTS = 300
CROSS_PROBABILITY = 0.65


def _solid_color(values: np.ndarray) -> int:
    colors = {int(v) for v in values.tolist()}
    assert len(colors) == 1
    return colors.pop()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _directions(grid: np.ndarray) -> dict[int, tuple[int, int, str]]:
    top = _solid_color(grid[0, 1:-1])
    bottom = _solid_color(grid[-1, 1:-1])
    left = _solid_color(grid[1:-1, 0])
    right = _solid_color(grid[1:-1, -1])
    corners = {
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    }
    assert corners == {BLACK}
    assert len({BLACK, top, bottom, left, right}) == 5
    return {
        top: (-1, 0, "vertical"),
        bottom: (1, 0, "vertical"),
        left: (0, -1, "horizontal"),
        right: (0, 1, "horizontal"),
    }


def _ray_canvas(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    directions = _directions(g)
    out = np.zeros_like(g)

    for wanted_orientation in ("horizontal", "vertical"):
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                color = int(g[r, c])
                spec = directions.get(color)
                if spec is None or spec[2] != wanted_orientation:
                    continue
                dr, dc, _orientation = spec
                rr, cc = r, c
                while 1 <= rr < h - 1 and 1 <= cc < w - 1:
                    out[rr, cc] = color
                    rr += dr
                    cc += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each interior pixel as a ray toward its matching frame side, let vertical rays win at crossings, and crop to the ray footprint.

    Input structure:
    - The grid has a one-cell-thick rectangular frame whose corners are black and whose four sides are solid, pairwise distinct colors.
    - The interior is mostly black and contains 4-7 sparse non-black pixels.
    - Each interior non-black pixel matches exactly one frame-side color, which determines its ray direction.

    Rule:
    - Read the four side colors from the frame: top means up, bottom means down, left means left, and right means right.
    - For every interior non-black pixel, fill a one-cell-thick ray of the same color starting at that pixel and continuing straight toward the matching side until the inner edge of the frame.
    - Draw horizontal rays first and vertical rays second, so any horizontal/vertical crossing keeps the vertical ray color.
    - Crop the result to the tight bounding box of all ray pixels; background cells inside that crop remain black.

    Key insight:
    - The frame sides are direction labels, not obstacles: each matching interior pixel expands into a full ray, and the only ambiguity is resolved by orientation, with vertical rays overwriting horizontal ones at intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert min(g.shape) >= 5

    rays = _ray_canvas(g)
    box = _bbox(rays != BLACK)
    assert box is not None
    r0, c0, r1, c1 = box
    return rays[r0 : r1 + 1, c0 : c1 + 1]


def _segments(h: int, w: int, sources: list[tuple[int, int, int]]) -> list[tuple[str, int, int, int, int]]:
    segments: list[tuple[str, int, int, int, int]] = []
    for r, c, color in sources:
        if color == LEFT_COLOR:
            segments.append(("horizontal", r, 1, c, color))
        elif color == RIGHT_COLOR:
            segments.append(("horizontal", r, c, w - 2, color))
        elif color == TOP_COLOR:
            segments.append(("vertical", 1, r, c, color))
        elif color == BOTTOM_COLOR:
            segments.append(("vertical", r, h - 2, c, color))
        else:
            raise AssertionError(f"unexpected color {color}")
    return segments


def _crossing_count(h: int, w: int, sources: list[tuple[int, int, int]]) -> int:
    horizontal = []
    vertical = []
    for orient, a, b, c, _color in _segments(h, w, sources):
        if orient == "horizontal":
            horizontal.append((a, b, c))
        else:
            vertical.append((a, b, c))

    total = 0
    for row, c0, c1 in horizontal:
        for r0, r1, col in vertical:
            if r0 <= row <= r1 and c0 <= col <= c1:
                total += 1
    return total


def _sample_counts(total_pixels: int) -> dict[int, int]:
    counts = {color: 1 for color in SIDE_COLORS}
    for _ in range(total_pixels - len(SIDE_COLORS)):
        choices = [color for color in SIDE_COLORS if counts[color] < MAX_PER_COLOR]
        counts[random.choice(choices)] += 1
    return counts


def _empty_framed_grid(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[0, 1:-1] = TOP_COLOR
    grid[-1, 1:-1] = BOTTOM_COLOR
    grid[1:-1, 0] = LEFT_COLOR
    grid[1:-1, -1] = RIGHT_COLOR
    return grid


def _shuffle(values: list[int]) -> list[int]:
    out = list(values)
    random.shuffle(out)
    return out


def _build_no_cross_sources(h: int, w: int, counts: dict[int, int]) -> list[tuple[int, int, int]]:
    top_count = counts[TOP_COLOR]
    bottom_count = counts[BOTTOM_COLOR]
    left_count = counts[LEFT_COLOR]
    right_count = counts[RIGHT_COLOR]

    top_rows = list(range(1, 1 + top_count))
    bottom_rows = list(range(h - 1 - bottom_count, h - 1))
    middle_rows = list(range(1 + top_count, h - 1 - bottom_count))
    left_cols = list(range(1, 1 + left_count))
    right_cols = list(range(w - 1 - right_count, w - 1))
    middle_cols = list(range(1 + left_count, w - 1 - right_count))

    num_h = left_count + right_count
    num_v = top_count + bottom_count
    assert len(middle_rows) >= num_h
    assert len(middle_cols) >= num_v

    horizontal_rows = random.sample(middle_rows, num_h)
    vertical_cols = random.sample(middle_cols, num_v)
    random.shuffle(horizontal_rows)
    random.shuffle(vertical_cols)

    sources: list[tuple[int, int, int]] = []
    top_cols = vertical_cols[:top_count]
    bottom_cols = vertical_cols[top_count:]
    left_rows = horizontal_rows[:left_count]
    right_rows = horizontal_rows[left_count:]

    for r, c in zip(_shuffle(top_rows), _shuffle(top_cols)):
        sources.append((r, c, TOP_COLOR))
    for r, c in zip(_shuffle(bottom_rows), _shuffle(bottom_cols)):
        sources.append((r, c, BOTTOM_COLOR))
    for r, c in zip(_shuffle(left_rows), _shuffle(left_cols)):
        sources.append((r, c, LEFT_COLOR))
    for r, c in zip(_shuffle(right_rows), _shuffle(right_cols)):
        sources.append((r, c, RIGHT_COLOR))
    return sources


def _build_cross_sources(h: int, w: int, counts: dict[int, int]) -> list[tuple[int, int, int]]:
    cross_h_color = random.choice([color for color in HORIZONTAL_COLORS if counts[color] > 0])
    cross_v_color = random.choice([color for color in VERTICAL_COLORS if counts[color] > 0])

    normal = dict(counts)
    normal[cross_h_color] -= 1
    normal[cross_v_color] -= 1

    top_count = normal[TOP_COLOR]
    bottom_count = normal[BOTTOM_COLOR]
    left_count = normal[LEFT_COLOR]
    right_count = normal[RIGHT_COLOR]

    top_rows = list(range(1, 1 + top_count))
    bottom_rows = list(range(h - 1 - bottom_count, h - 1))
    middle_rows = list(range(1 + top_count, h - 1 - bottom_count))
    left_cols = list(range(1, 1 + left_count))
    right_cols = list(range(w - 1 - right_count, w - 1))
    middle_cols = list(range(1 + left_count, w - 1 - right_count))

    num_other_h = left_count + right_count
    num_other_v = top_count + bottom_count
    assert len(middle_rows) >= num_other_h + 2
    assert len(middle_cols) >= num_other_v + 2

    row_pair = sorted(random.sample(middle_rows, 2))
    if cross_v_color == TOP_COLOR:
        cross_row, vertical_row = row_pair[0], row_pair[1]
    else:
        vertical_row, cross_row = row_pair[0], row_pair[1]
    remaining_rows = [row for row in middle_rows if row not in {cross_row, vertical_row}]
    other_horizontal_rows = random.sample(remaining_rows, num_other_h)
    random.shuffle(other_horizontal_rows)

    col_pair = sorted(random.sample(middle_cols, 2))
    if cross_h_color == LEFT_COLOR:
        cross_col, horizontal_col = col_pair[0], col_pair[1]
    else:
        horizontal_col, cross_col = col_pair[0], col_pair[1]
    remaining_cols = [col for col in middle_cols if col not in {cross_col, horizontal_col}]
    other_vertical_cols = random.sample(remaining_cols, num_other_v)
    random.shuffle(other_vertical_cols)

    sources: list[tuple[int, int, int]] = []
    top_cols = other_vertical_cols[:top_count]
    bottom_cols = other_vertical_cols[top_count:]
    left_rows = other_horizontal_rows[:left_count]
    right_rows = other_horizontal_rows[left_count:]

    for r, c in zip(_shuffle(top_rows), _shuffle(top_cols)):
        sources.append((r, c, TOP_COLOR))
    for r, c in zip(_shuffle(bottom_rows), _shuffle(bottom_cols)):
        sources.append((r, c, BOTTOM_COLOR))
    for r, c in zip(_shuffle(left_rows), _shuffle(left_cols)):
        sources.append((r, c, LEFT_COLOR))
    for r, c in zip(_shuffle(right_rows), _shuffle(right_cols)):
        sources.append((r, c, RIGHT_COLOR))
    sources.append((vertical_row, cross_col, cross_v_color))
    sources.append((cross_row, horizontal_col, cross_h_color))
    return sources


def _valid_sources(h: int, w: int, sources: list[tuple[int, int, int]], want_cross: bool) -> bool:
    rows = [r for r, _c, _color in sources]
    cols = [c for _r, c, _color in sources]
    if len(rows) != len(set(rows)) or len(cols) != len(set(cols)):
        return False
    if len(sources) < 4 or len(sources) > 7:
        return False
    if {color for _r, _c, color in sources} != set(SIDE_COLORS):
        return False
    crossings = _crossing_count(h, w, sources)
    return crossings == 1 if want_cross else crossings == 0


def _render_input(h: int, w: int, sources: list[tuple[int, int, int]]) -> np.ndarray:
    grid = _empty_framed_grid(h, w)
    for r, c, color in sources:
        grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    total_pixels = random.randint(4, 7)
    counts = _sample_counts(total_pixels)
    want_cross = random.random() < CROSS_PROBABILITY

    sources = (
        _build_cross_sources(h, w, counts)
        if want_cross
        else _build_no_cross_sources(h, w, counts)
    )
    if not _valid_sources(h, w, sources, want_cross):
        return None

    grid = _render_input(h, w, sources)
    try:
        out = main(grid)
    except Exception:
        return None

    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
