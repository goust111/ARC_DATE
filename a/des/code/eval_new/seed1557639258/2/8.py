import numpy as np


GREY = 5
MAX_RETRIES = 300
GROUP_COUNT_MIN = 2
GROUP_COUNT_MAX = 4
SPAN_MIN = 1
SPAN_MAX = 5
MAX_MARGIN_SUM = 8


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    arr = [int(v) for v in values]
    out: list[tuple[int, int, int]] = []
    start = 0
    for idx in range(1, len(arr) + 1):
        if idx == len(arr) or arr[idx] != arr[start]:
            out.append((start, idx, arr[start]))
            start = idx
    return out


def _conflict_color(top: int, left: int, bottom: int, right: int) -> int:
    if top == left:
        return int(top)
    candidates: list[int] = []
    for color in (int(bottom), int(right)):
        if color not in (int(top), int(left)) and color not in candidates:
            candidates.append(color)
    assert len(candidates) == 1, "conflicting headers must expose a unique third color"
    return candidates[0]


def _grey_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == GREY)
    assert pts.size > 0, "expected a grey interior region"
    r0, c0 = np.min(pts, axis=0)
    r1, c1 = np.max(pts, axis=0)
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey region must be rectangular"
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a grey rectangular chamber with colored domino-like tiles indexed by four edge templates.

    Input structure:
    - bg is the surrounding background. One rectangle is enclosed by a 1-cell border, and its full interior is grey (5).
    - A 1-cell template row sits just outside the top and bottom border edges; a 1-cell template column sits just outside the left and right border edges.
    - The top and left templates are partitioned into same-color runs, whose run lengths define column groups and row groups inside the grey rectangle.

    Rule:
    - Locate the grey rectangle and read the top-template runs as column spans and the left-template runs as row spans.
    - Each row-group x column-group block becomes one filled tile.
    - If a tile's top color and left color match, fill the whole tile with that shared color; otherwise use the unique third color visible on the corresponding bottom or right template segment.
    - Remove the border and all templates by returning only the filled interior rectangle.

    Key insight:
    - The four edge strips act like a 2D lookup table: top/left choose the tile coordinates, while bottom/right disambiguate mismatched header pairs by exposing the missing third color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _grey_bbox(g)
    assert r0 >= 2 and c0 >= 2, "templates must exist above and left of the chamber"
    assert r1 + 2 < g.shape[0] and c1 + 2 < g.shape[1], "templates must exist below and right of the chamber"

    top = g[r0 - 2, c0 : c1 + 1]
    bottom = g[r1 + 2, c0 : c1 + 1]
    left = g[r0 : r1 + 1, c0 - 2]
    right = g[r0 : r1 + 1, c1 + 2]

    row_runs = _runs(left)
    col_runs = _runs(top)
    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)

    for rs, re, left_color in row_runs:
        right_color = int(right[rs])
        assert np.all(right[rs:re] == right_color), "right template must align to left template runs"
        for cs, ce, top_color in col_runs:
            bottom_color = int(bottom[cs])
            assert np.all(bottom[cs:ce] == bottom_color), "bottom template must align to top template runs"
            fill = _conflict_color(int(top_color), int(left_color), bottom_color, right_color)
            out[rs:re, cs:ce] = fill

    return out


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _sample_lengths(rng: np.random.Generator, n_groups: int) -> list[int]:
    return [_randint(rng, SPAN_MIN, SPAN_MAX) for _ in range(n_groups)]


def _sample_run_colors(rng: np.random.Generator, n_groups: int, palette: list[int]) -> list[int]:
    colors: list[int] = []
    while len(colors) < n_groups:
        color = int(rng.choice(palette))
        if not colors or color != colors[-1]:
            colors.append(color)
    return colors


def _next_map(cycle: list[int]) -> dict[int, int]:
    return {int(cycle[i]): int(cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))}


def _branch_flags(top_colors: list[int], left_colors: list[int], cycle: list[int]) -> tuple[bool, bool, bool]:
    idx = {int(color): i for i, color in enumerate(cycle)}
    same = False
    bottom_resolves = False
    right_resolves = False
    for top in top_colors:
        ti = idx[int(top)]
        prev_color = int(cycle[(ti - 1) % 3])
        next_color = int(cycle[(ti + 1) % 3])
        for left in left_colors:
            if int(top) == int(left):
                same = True
            elif int(left) == prev_color:
                bottom_resolves = True
            elif int(left) == next_color:
                right_resolves = True
    return same, bottom_resolves, right_resolves


def _expected_output(
    row_lengths: list[int],
    col_lengths: list[int],
    left_colors: list[int],
    top_colors: list[int],
    cycle: list[int],
) -> np.ndarray:
    next_color = _next_map(cycle)
    height = sum(row_lengths)
    width = sum(col_lengths)
    out = np.empty((height, width), dtype=int)

    r = 0
    for row_span, left in zip(row_lengths, left_colors):
        c = 0
        right = int(next_color[int(left)])
        for col_span, top in zip(col_lengths, top_colors):
            bottom = int(next_color[int(top)])
            fill = _conflict_color(int(top), int(left), bottom, right)
            out[r : r + row_span, c : c + col_span] = fill
            c += col_span
        r += row_span

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    palette = [c for c in range(10) if c != GREY]
    picked = [int(v) for v in rng.choice(palette, size=5, replace=False)]
    bg, border = picked[0], picked[1]
    cycle = picked[2:]

    n_row_groups = _randint(rng, GROUP_COUNT_MIN, GROUP_COUNT_MAX)
    n_col_groups = _randint(rng, GROUP_COUNT_MIN, GROUP_COUNT_MAX)
    row_lengths = _sample_lengths(rng, n_row_groups)
    col_lengths = _sample_lengths(rng, n_col_groups)

    for _ in range(100):
        left_colors = _sample_run_colors(rng, n_row_groups, cycle)
        top_colors = _sample_run_colors(rng, n_col_groups, cycle)
        same, bottom_resolves, right_resolves = _branch_flags(top_colors, left_colors, cycle)
        if same and bottom_resolves and right_resolves:
            break
    else:
        return None

    expected = _expected_output(row_lengths, col_lengths, left_colors, top_colors, cycle)
    if np.unique(expected).size < 3:
        return None

    inner_h, inner_w = expected.shape
    occupied_h = inner_h + 4
    occupied_w = inner_w + 4
    if occupied_h + 2 > 30 or occupied_w + 2 > 30:
        return None

    extra_h = _randint(rng, 2, min(MAX_MARGIN_SUM, 30 - occupied_h))
    extra_w = _randint(rng, 2, min(MAX_MARGIN_SUM, 30 - occupied_w))
    top_pad = _randint(rng, 1, extra_h - 1)
    left_pad = _randint(rng, 1, extra_w - 1)
    bottom_pad = extra_h - top_pad
    right_pad = extra_w - left_pad

    total_h = occupied_h + top_pad + bottom_pad
    total_w = occupied_w + left_pad + right_pad
    assert total_h == occupied_h + extra_h
    assert total_w == occupied_w + extra_w

    grid = np.full((total_h, total_w), bg, dtype=int)

    inner_r0 = top_pad + 2
    inner_c0 = left_pad + 2
    inner_r1 = inner_r0 + inner_h - 1
    inner_c1 = inner_c0 + inner_w - 1

    grid[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1] = GREY
    grid[inner_r0 - 1, inner_c0 - 1 : inner_c1 + 2] = border
    grid[inner_r1 + 1, inner_c0 - 1 : inner_c1 + 2] = border
    grid[inner_r0 - 1 : inner_r1 + 2, inner_c0 - 1] = border
    grid[inner_r0 - 1 : inner_r1 + 2, inner_c1 + 1] = border

    next_color = _next_map(cycle)
    top_template = np.concatenate([np.full(span, color, dtype=int) for span, color in zip(col_lengths, top_colors)])
    bottom_template = np.concatenate(
        [np.full(span, next_color[int(color)], dtype=int) for span, color in zip(col_lengths, top_colors)]
    )
    left_template = np.concatenate([np.full(span, color, dtype=int) for span, color in zip(row_lengths, left_colors)])
    right_template = np.concatenate(
        [np.full(span, next_color[int(color)], dtype=int) for span, color in zip(row_lengths, left_colors)]
    )

    grid[inner_r0 - 2, inner_c0 : inner_c1 + 1] = top_template
    grid[inner_r1 + 2, inner_c0 : inner_c1 + 1] = bottom_template
    grid[inner_r0 : inner_r1 + 1, inner_c0 - 2] = left_template
    grid[inner_r0 : inner_r1 + 1, inner_c1 + 2] = right_template

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
