from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
DEFAULT_FRAME_COLOR = 5
ALT_FRAME_COLOR = 9
CONTROL_COLOR_POOL = (1, 2, 3, 4)
ENDPOINT_POOL_DEFAULT = (6, 7, 8, 9)
ENDPOINT_POOL_WITH_GREY = (5, 6, 7, 8)

GRID_SIZE_RANGE = (12, 18)
INTERIOR_SIZE_RANGE = (5, 8)
PAIR_COUNT_CHOICES = (2, 3)
GREY_BRANCH_PROB = 0.45
CROSS_BRANCH_PROB = 0.55
MAX_RETRIES = 400


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _sample_spaced_pair(values: list[int], rng: random.Random) -> tuple[int, int] | None:
    values = sorted({int(v) for v in values})
    for min_gap in (2, 1):
        candidates = [
            (a, b)
            for i, a in enumerate(values)
            for b in values[i + 1 :]
            if b - a >= min_gap
        ]
        if candidates:
            return tuple(map(int, rng.choice(candidates)))
    return None


def _draw_axis_segment(grid: np.ndarray, a: Coord, b: Coord, color: int) -> None:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = int(color)
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = int(color)
        return
    raise AssertionError("endpoint pair must be horizontal or vertical")


def _extreme_control_positions(grid: np.ndarray, bg: int) -> tuple[Coord, Coord, Coord, Coord, tuple[int, int, int, int]]:
    fg = np.argwhere(grid != int(bg))
    assert fg.size > 0, "expected foreground pixels"

    r_top = int(fg[:, 0].min())
    r_bot = int(fg[:, 0].max())
    c_left = int(fg[:, 1].min())
    c_right = int(fg[:, 1].max())

    top_coords = [tuple(map(int, rc)) for rc in fg if int(rc[0]) == r_top]
    bot_coords = [tuple(map(int, rc)) for rc in fg if int(rc[0]) == r_bot]
    left_coords = [tuple(map(int, rc)) for rc in fg if int(rc[1]) == c_left]
    right_coords = [tuple(map(int, rc)) for rc in fg if int(rc[1]) == c_right]

    assert len(top_coords) == 1, "top extreme must be unique"
    assert len(bot_coords) == 1, "bottom extreme must be unique"
    assert len(left_coords) == 1, "left extreme must be unique"
    assert len(right_coords) == 1, "right extreme must be unique"

    top = top_coords[0]
    bottom = bot_coords[0]
    left = left_coords[0]
    right = right_coords[0]
    assert len({top, bottom, left, right}) == 4, "control pixels must be distinct"
    return top, bottom, left, right, (r_top, c_left, r_bot, c_right)


def _endpoint_jobs(
    grid: np.ndarray,
    bg: int,
    control_positions: set[Coord],
    bounds: tuple[int, int, int, int],
) -> list[tuple[Coord, int, Coord, Coord]]:
    r_top, c_left, r_bot, c_right = map(int, bounds)
    counts = Counter(map(int, grid[grid != int(bg)].ravel()))

    control_colors = {int(grid[r, c]) for r, c in control_positions}
    for color in control_colors:
        assert counts[color] == 1, "each control color must appear once"

    endpoint_colors = [int(color) for color, count in counts.items() if color not in control_colors and count == 2]
    assert 2 <= len(endpoint_colors) <= 3, "expected 2 or 3 endpoint colors"
    assert len(control_colors) + len(endpoint_colors) == len(counts), "unexpected extra colors"

    jobs: list[tuple[Coord, int, Coord, Coord]] = []
    for color in endpoint_colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))]
        assert len(coords) == 2
        a, b = coords
        for r, c in coords:
            assert r_top < r < r_bot and c_left < c < c_right, "endpoints must lie strictly inside the frame"
        assert a[0] == b[0] or a[1] == b[1], "endpoint pair must be horizontal or vertical"
        first = min(a, b)
        jobs.append((first, int(color), a, b))

    jobs.sort(key=lambda item: item[0])
    return jobs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the control-defined frame, then connect each same-colored endpoint pair inside it.

    Input structure:
    - A mostly black grid contains exactly four distinct single-pixel control markers and 2 or 3 endpoint pairs.
    - The control markers uniquely occupy the global top, bottom, left, and right foreground extremes.
    - Every endpoint color appears exactly twice, lies strictly inside the rectangle implied by those extremes, and the two endpoints share one row or one column.

    Rule:
    - Use the control markers' extreme row and column coordinates to define a rectangle.
    - Draw that rectangle in grey, unless an endpoint pair already uses grey, in which case draw the frame in maroon.
    - Restore the four control markers on top of the frame, then connect each endpoint pair with a horizontal or vertical segment in the pair's color.
    - If segments cross, later colors overwrite earlier ones according to the endpoint colors' first appearance positions in row-major input order.

    Key insight:
    - The same four extreme control pixels both locate the frame and define the only valid drawing region for the endpoint connections inside it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    top, bottom, left, right, bounds = _extreme_control_positions(g, bg)
    control_positions = {top, bottom, left, right}
    jobs = _endpoint_jobs(g, bg, control_positions, bounds)

    frame_color = ALT_FRAME_COLOR if any(color == DEFAULT_FRAME_COLOR for _first, color, _a, _b in jobs) else DEFAULT_FRAME_COLOR
    r_top, c_left, r_bot, c_right = map(int, bounds)

    out = g.copy()
    out[r_top, c_left : c_right + 1] = frame_color
    out[r_bot, c_left : c_right + 1] = frame_color
    out[r_top : r_bot + 1, c_left] = frame_color
    out[r_top : r_bot + 1, c_right] = frame_color

    for r, c in control_positions:
        out[r, c] = int(g[r, c])

    for _first, color, a, b in jobs:
        _draw_axis_segment(out, a, b, color)

    return out


def _sample_bounds(rng: random.Random, height: int, width: int) -> tuple[int, int, int, int]:
    max_inner_h = min(INTERIOR_SIZE_RANGE[1], height - 4)
    max_inner_w = min(INTERIOR_SIZE_RANGE[1], width - 4)
    inner_h = rng.randint(INTERIOR_SIZE_RANGE[0], max_inner_h)
    inner_w = rng.randint(INTERIOR_SIZE_RANGE[0], max_inner_w)

    r_top = rng.randint(1, height - inner_h - 3)
    c_left = rng.randint(1, width - inner_w - 3)
    r_bot = r_top + inner_h + 1
    c_right = c_left + inner_w + 1
    return int(r_top), int(c_left), int(r_bot), int(c_right)


def _sample_controls(rng: random.Random, bounds: tuple[int, int, int, int]) -> dict[Coord, int]:
    r_top, c_left, r_bot, c_right = map(int, bounds)
    colors = list(CONTROL_COLOR_POOL)
    rng.shuffle(colors)

    top = (r_top, rng.randint(c_left + 1, c_right - 1))
    bottom = (r_bot, rng.randint(c_left + 1, c_right - 1))
    left = (rng.randint(r_top + 1, r_bot - 1), c_left)
    right = (rng.randint(r_top + 1, r_bot - 1), c_right)
    return {
        top: colors[0],
        bottom: colors[1],
        left: colors[2],
        right: colors[3],
    }


def _parallel_pairs(
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    colors: list[int],
    orientation: str,
) -> list[dict[str, object]] | None:
    r_top, c_left, r_bot, c_right = map(int, bounds)
    pairs: list[dict[str, object]] = []

    if orientation == "H":
        rows = rng.sample(list(range(r_top + 1, r_bot)), len(colors))
        for row, color in zip(rows, colors, strict=False):
            cols = _sample_spaced_pair(list(range(c_left + 1, c_right)), rng)
            if cols is None:
                return None
            a = (int(row), int(cols[0]))
            b = (int(row), int(cols[1]))
            pairs.append({"color": int(color), "a": a, "b": b})
        return pairs

    cols = rng.sample(list(range(c_left + 1, c_right)), len(colors))
    for col, color in zip(cols, colors, strict=False):
        rows = _sample_spaced_pair(list(range(r_top + 1, r_bot)), rng)
        if rows is None:
            return None
        a = (int(rows[0]), int(col))
        b = (int(rows[1]), int(col))
        pairs.append({"color": int(color), "a": a, "b": b})
    return pairs


def _crossing_pairs(
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    colors: list[int],
) -> list[dict[str, object]] | None:
    r_top, c_left, r_bot, c_right = map(int, bounds)
    cross_row = rng.randint(r_top + 2, r_bot - 2)
    cross_col = rng.randint(c_left + 2, c_right - 2)

    left_cols = list(range(c_left + 1, cross_col))
    right_cols = list(range(cross_col + 1, c_right))
    upper_rows = list(range(r_top + 1, cross_row))
    lower_rows = list(range(cross_row + 1, r_bot))
    if not left_cols or not right_cols or not upper_rows or not lower_rows:
        return None

    pairs: list[dict[str, object]] = [
        {"color": int(colors[0]), "a": (cross_row, int(rng.choice(left_cols))), "b": (cross_row, int(rng.choice(right_cols)))},
        {"color": int(colors[1]), "a": (int(rng.choice(upper_rows)), cross_col), "b": (int(rng.choice(lower_rows)), cross_col)},
    ]

    if len(colors) == 2:
        return pairs

    extra_color = int(colors[2])
    extra_horizontal = rng.random() < 0.5
    if extra_horizontal:
        candidate_rows = [r for r in range(r_top + 1, r_bot) if r != cross_row]
        if not candidate_rows:
            return None
        side_choices = []
        if len(range(c_left + 1, cross_col)) >= 2:
            side_choices.append(list(range(c_left + 1, cross_col)))
        if len(range(cross_col + 1, c_right)) >= 2:
            side_choices.append(list(range(cross_col + 1, c_right)))
        if not side_choices:
            return None
        cols = _sample_spaced_pair(rng.choice(side_choices), rng)
        if cols is None:
            return None
        row = int(rng.choice(candidate_rows))
        pairs.append({"color": extra_color, "a": (row, int(cols[0])), "b": (row, int(cols[1]))})
        return pairs

    candidate_cols = [c for c in range(c_left + 1, c_right) if c != cross_col]
    if not candidate_cols:
        return None
    side_choices = []
    if len(range(r_top + 1, cross_row)) >= 2:
        side_choices.append(list(range(r_top + 1, cross_row)))
    if len(range(cross_row + 1, r_bot)) >= 2:
        side_choices.append(list(range(cross_row + 1, r_bot)))
    if not side_choices:
        return None
    rows = _sample_spaced_pair(rng.choice(side_choices), rng)
    if rows is None:
        return None
    col = int(rng.choice(candidate_cols))
    pairs.append({"color": extra_color, "a": (int(rows[0]), col), "b": (int(rows[1]), col)})
    return pairs


def _build_pair_specs(
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    pair_count: int,
    use_grey_endpoint: bool,
    force_cross: bool,
) -> list[dict[str, object]] | None:
    if use_grey_endpoint:
        colors = [DEFAULT_FRAME_COLOR]
        colors.extend(rng.sample([c for c in ENDPOINT_POOL_WITH_GREY if c != DEFAULT_FRAME_COLOR], pair_count - 1))
    else:
        colors = rng.sample(list(ENDPOINT_POOL_DEFAULT), pair_count)
    rng.shuffle(colors)

    if force_cross:
        return _crossing_pairs(rng, bounds, colors)
    orientation = rng.choice(("H", "V"))
    return _parallel_pairs(rng, bounds, colors, orientation)


def _render_input(
    shape: tuple[int, int],
    controls: dict[Coord, int],
    pairs: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for coord, color in controls.items():
        grid[coord] = int(color)
    for pair in pairs:
        color = int(pair["color"])
        a = pair["a"]
        b = pair["b"]
        grid[a] = color
        grid[b] = color
    return grid


def _render_output(
    inp: np.ndarray,
    controls: dict[Coord, int],
    pairs: list[dict[str, object]],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    out = inp.copy()
    frame_color = ALT_FRAME_COLOR if any(int(pair["color"]) == DEFAULT_FRAME_COLOR for pair in pairs) else DEFAULT_FRAME_COLOR
    r_top, c_left, r_bot, c_right = map(int, bounds)

    out[r_top, c_left : c_right + 1] = frame_color
    out[r_bot, c_left : c_right + 1] = frame_color
    out[r_top : r_bot + 1, c_left] = frame_color
    out[r_top : r_bot + 1, c_right] = frame_color

    for coord, color in controls.items():
        out[coord] = int(color)

    ordered = sorted(pairs, key=lambda pair: min(pair["a"], pair["b"]))
    for pair in ordered:
        _draw_axis_segment(out, pair["a"], pair["b"], int(pair["color"]))

    return out


def _valid_extremes(grid: np.ndarray) -> bool:
    fg = np.argwhere(grid != BACKGROUND)
    if fg.size == 0:
        return False
    return (
        int(np.sum(fg[:, 0] == int(fg[:, 0].min()))) == 1
        and int(np.sum(fg[:, 0] == int(fg[:, 0].max()))) == 1
        and int(np.sum(fg[:, 1] == int(fg[:, 1].min()))) == 1
        and int(np.sum(fg[:, 1] == int(fg[:, 1].max()))) == 1
    )


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(*GRID_SIZE_RANGE)
    width = rng.randint(*GRID_SIZE_RANGE)
    bounds = _sample_bounds(rng, height, width)
    controls = _sample_controls(rng, bounds)

    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    use_grey_endpoint = rng.random() < GREY_BRANCH_PROB
    r_top, c_left, r_bot, c_right = map(int, bounds)
    inner_h = r_bot - r_top - 1
    inner_w = c_right - c_left - 1
    can_cross = pair_count >= 2 and inner_h >= 4 and inner_w >= 4
    force_cross = can_cross and rng.random() < CROSS_BRANCH_PROB

    pairs = _build_pair_specs(rng, bounds, pair_count, use_grey_endpoint, force_cross)
    if pairs is None:
        return None

    inp = _render_input((height, width), controls, pairs)
    if not _valid_extremes(inp):
        return None

    expected = _render_output(inp, controls, pairs, bounds)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
