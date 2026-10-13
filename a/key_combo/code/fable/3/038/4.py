from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
CUP_COLOR = 5
STREAM_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
OPENINGS = ("up", "down", "left", "right")
MAX_GENERATION_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _detect_opening(cup_mask: np.ndarray, box: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "cup must enclose a non-empty interior"

    openings = {
        "up": not np.any(cup_mask[r0, c0 + 1 : c1]),
        "down": not np.any(cup_mask[r1, c0 + 1 : c1]),
        "left": not np.any(cup_mask[r0 + 1 : r1, c0]),
        "right": not np.any(cup_mask[r0 + 1 : r1, c1]),
    }
    found = [name for name, is_open in openings.items() if is_open]
    assert len(found) == 1, "expected exactly one open side"
    return found[0]


def _read_stream(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    opening: str,
    bg: int,
) -> list[int]:
    r0, c0, r1, c1 = box
    stream_coords = np.argwhere((grid != bg) & (grid != CUP_COLOR))
    assert stream_coords.size > 0, "expected at least one stream pixel"

    if opening in {"up", "down"}:
        cols = {int(c) for _, c in stream_coords}
        assert len(cols) == 1, "vertical pours must align on one axis column"
        axis_c = next(iter(cols))
        assert c0 < axis_c < c1, "stream axis must pass through the cup opening"
        coords = [(int(r), int(c)) for r, c in stream_coords]
        if opening == "up":
            assert all(r < r0 for r, _ in coords), "stream must stay outside the opening"
            coords.sort(key=lambda rc: rc[0], reverse=True)
        else:
            assert all(r > r1 for r, _ in coords), "stream must stay outside the opening"
            coords.sort(key=lambda rc: rc[0])
    else:
        rows = {int(r) for r, _ in stream_coords}
        assert len(rows) == 1, "horizontal pours must align on one axis row"
        axis_r = next(iter(rows))
        assert r0 < axis_r < r1, "stream axis must pass through the cup opening"
        coords = [(int(r), int(c)) for r, c in stream_coords]
        if opening == "left":
            assert all(c < c0 for _, c in coords), "stream must stay outside the opening"
            coords.sort(key=lambda rc: rc[1], reverse=True)
        else:
            assert all(c > c1 for _, c in coords), "stream must stay outside the opening"
            coords.sort(key=lambda rc: rc[1])

    return [int(grid[r, c]) for r, c in coords]


def _paint_layers(
    out: np.ndarray,
    box: tuple[int, int, int, int],
    opening: str,
    stream_colors: list[int],
) -> None:
    assert stream_colors, "stream must contain at least one color"
    r0, c0, r1, c1 = box
    inside = out[r0 + 1 : r1, c0 + 1 : c1]

    if opening in {"up", "down"}:
        depth = inside.shape[0]
        for layer in range(depth):
            color = stream_colors[layer % len(stream_colors)]
            target_r = depth - 1 - layer if opening == "up" else layer
            inside[target_r, :] = color
    else:
        depth = inside.shape[1]
        for layer in range(depth):
            color = stream_colors[layer % len(stream_colors)]
            target_c = depth - 1 - layer if opening == "left" else layer
            inside[:, target_c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour the outside color stream into the open cup so the closest color settles deepest and the layer colors cycle until the cup is full.

    Input structure:
    - The background is black and contains one gray rectangular cup outline with exactly one missing side segment as its opening.
    - The cup may face up, down, left, or right, and its interior is the black rectangle inside the outline.
    - Outside the opening, a one-pixel-wide stream of colored cells lies on the opening axis with possible gaps between cells.

    Rule:
    - Detect the opening direction from the missing side of the cup outline.
    - Read the stream colors from nearest to farthest from the opening.
    - Partition the cup interior into whole rows or columns parallel to the cup bottom, fill the deepest layer with the nearest stream color, continue toward the opening, and cycle the stream order if there are more layers than stream cells.
    - Erase all outside stream pixels and keep the cup wall unchanged.

    Key insight:
    - The external stream is ordered by pouring time, so the closest color enters first and ends up deepest; the cup orientation only changes which interior rows or columns count as layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    cup_mask = g == CUP_COLOR
    assert np.any(cup_mask), "expected one gray cup"
    box = _bbox(cup_mask)
    opening = _detect_opening(cup_mask, box)
    stream_colors = _read_stream(g, box, opening, bg)

    out = g.copy()
    out[(out != bg) & (out != CUP_COLOR)] = bg
    _paint_layers(out, box, opening, stream_colors)
    return out


def _odd_choice(lo: int, hi: int) -> int:
    return random.choice([value for value in range(lo, hi + 1) if value % 2 == 1])


def _choose_dimensions(opening: str) -> tuple[int, int]:
    if opening in {"up", "down"}:
        return random.randint(3, 6), _odd_choice(3, 7)
    return _odd_choice(3, 7), random.randint(3, 6)


def _choose_stream_len(depth: int) -> int:
    lengths = list(range(1, depth + 1))
    weights = [depth - length + 2 for length in lengths]
    return random.choices(lengths, weights=weights, k=1)[0]


def _choose_distances(count: int) -> list[int]:
    distances = []
    current = random.randint(1, 2)
    for _ in range(count):
        distances.append(current)
        current += random.randint(1, 3)
    return distances


def _paint_cup(
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    opening: str,
) -> tuple[int, int, int, int]:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1

    grid[r0 : r1 + 1, c0] = CUP_COLOR
    grid[r0 : r1 + 1, c1] = CUP_COLOR
    grid[r0, c0 : c1 + 1] = CUP_COLOR
    grid[r1, c0 : c1 + 1] = CUP_COLOR

    if opening == "up":
        grid[r0, c0 + 1 : c1] = BACKGROUND
    elif opening == "down":
        grid[r1, c0 + 1 : c1] = BACKGROUND
    elif opening == "left":
        grid[r0 + 1 : r1, c0] = BACKGROUND
    else:
        grid[r0 + 1 : r1, c1] = BACKGROUND

    return r0, c0, r1, c1


def _stream_positions(
    box: tuple[int, int, int, int],
    opening: str,
    distances: list[int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    axis_r = r0 + 1 + (r1 - r0 - 2) // 2
    axis_c = c0 + 1 + (c1 - c0 - 2) // 2

    if opening == "up":
        return [(r0 - distance, axis_c) for distance in distances]
    if opening == "down":
        return [(r1 + distance, axis_c) for distance in distances]
    if opening == "left":
        return [(axis_r, c0 - distance) for distance in distances]
    return [(axis_r, c1 + distance) for distance in distances]


def _expected_output(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    opening: str,
    stream_positions: list[tuple[int, int]],
    stream_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for r, c in stream_positions:
        out[r, c] = BACKGROUND
    _paint_layers(out, box, opening, stream_colors)
    return out


def _gen_attempt() -> np.ndarray | None:
    opening = random.choice(OPENINGS)
    inner_h, inner_w = _choose_dimensions(opening)
    depth = inner_h if opening in {"up", "down"} else inner_w
    stream_len = _choose_stream_len(depth)
    stream_colors = random.sample(STREAM_PALETTE, k=stream_len)
    distances = _choose_distances(stream_len)
    farthest = distances[-1]

    side_margin_a = random.randint(1, 3)
    side_margin_b = random.randint(1, 3)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if opening == "up":
        top_clear = random.randint(1, 2)
        bottom_clear = random.randint(1, 3)
        height = top_clear + farthest + outer_h + bottom_clear
        width = side_margin_a + outer_w + side_margin_b
        r0 = top_clear + farthest
        c0 = side_margin_a
    elif opening == "down":
        top_clear = random.randint(1, 3)
        bottom_clear = random.randint(1, 2)
        height = top_clear + outer_h + farthest + bottom_clear
        width = side_margin_a + outer_w + side_margin_b
        r0 = top_clear
        c0 = side_margin_a
    elif opening == "left":
        left_clear = random.randint(1, 2)
        right_clear = random.randint(1, 3)
        height = side_margin_a + outer_h + side_margin_b
        width = left_clear + farthest + outer_w + right_clear
        r0 = side_margin_a
        c0 = left_clear + farthest
    else:
        left_clear = random.randint(1, 3)
        right_clear = random.randint(1, 2)
        height = side_margin_a + outer_h + side_margin_b
        width = left_clear + outer_w + farthest + right_clear
        r0 = side_margin_a
        c0 = left_clear

    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    box = _paint_cup(grid, r0, c0, inner_h, inner_w, opening)
    positions = _stream_positions(box, opening, distances)
    for (r, c), color in zip(positions, stream_colors):
        grid[r, c] = color

    expected = _expected_output(grid, box, opening, positions, stream_colors)
    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 4")
