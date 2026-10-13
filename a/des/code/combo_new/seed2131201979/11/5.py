import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
MIN_H = 15
MAX_H = 30
MIN_W = 14
MAX_W = 30
MIN_LINE_COUNT = 2
MAX_H_LINES = 4
MAX_V_LINES = 4
SAMPLE_MIN_SEP = 2
LINE_MIN_SEP = 3


def _place_positions(count: int, lo: int, hi: int, min_sep: int, rng: random.Random) -> list[int] | None:
    span = hi - lo + 1
    needed = count + (count - 1) * (min_sep - 1)
    if count <= 0 or span < needed:
        return None

    compressed = span - (count - 1) * (min_sep - 1)
    picks = sorted(rng.sample(range(compressed), count))
    return [lo + pick + i * (min_sep - 1) for i, pick in enumerate(picks)]


def _scan_visible_color(g: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    h, w = g.shape
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        color = int(g[rr, cc])
        if color == GREY:
            rr += dr
            cc += dc
            continue
        assert color != BG, "crossing arm must reveal a visible line color"
        return color
    raise AssertionError("crossing arm must stay inside the grid")


def _priority_order(g: np.ndarray) -> list[int]:
    order = [int(g[r, 0]) for r in range(g.shape[0]) if int(g[r, 0]) != BG]
    assert order, "expected a non-empty left-edge priority column"
    assert len(order) == len(set(order)), "priority colors must be unique"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve each masked line crossing by restoring the color of the higher-priority line.

    Input structure:
    - The background is black.
    - The left edge contains a vertical priority list of single colored sample
      pixels, read from top to bottom.
    - Elsewhere, the grid contains one-cell-thick horizontal and vertical
      colored line segments, and every crossing pixel is replaced by grey.

    Rule:
    - Keep the left-edge sample column unchanged.
    - For each grey crossing, extend left/right to recover the horizontal line
      color and extend up/down to recover the vertical line color.
    - Compare those two colors by the left-edge priority list and recolor the
      grey crossing with whichever color appears higher in that list.

    Key insight:
    - Grey hides only the identity of the top layer, not the two participating
      lines; their colors can be recovered from the visible arms and ranked by
      the sample column.
    """
    g = np.asarray(grid, dtype=int)
    order = _priority_order(g)
    rank = {color: idx for idx, color in enumerate(order)}

    out = g.copy()
    for r, c in np.argwhere(g == GREY):
        left = _scan_visible_color(g, int(r), int(c), 0, -1)
        right = _scan_visible_color(g, int(r), int(c), 0, 1)
        up = _scan_visible_color(g, int(r), int(c), -1, 0)
        down = _scan_visible_color(g, int(r), int(c), 1, 0)

        assert left == right, "horizontal crossing arms must share one color"
        assert up == down, "vertical crossing arms must share one color"
        assert left != up, "a crossing must involve two different colors"
        assert left in rank and up in rank, "every line color must appear in the priority list"

        out[int(r), int(c)] = left if rank[left] < rank[up] else up

    return out


def _interleave_priority(horizontal_colors: list[int], vertical_colors: list[int], rng: random.Random) -> list[int]:
    hs = horizontal_colors[:]
    vs = vertical_colors[:]
    rng.shuffle(hs)
    rng.shuffle(vs)

    use_horizontal = bool(rng.randrange(2))
    priority: list[int] = []
    while hs or vs:
        if use_horizontal and hs:
            priority.append(hs.pop())
        elif (not use_horizontal) and vs:
            priority.append(vs.pop())
        elif hs:
            priority.append(hs.pop())
        else:
            priority.append(vs.pop())
        use_horizontal = not use_horizontal

    return priority


def _build_expected(
    inp: np.ndarray,
    h_rows: list[int],
    v_cols: list[int],
    h_colors: list[int],
    v_colors: list[int],
    rank: dict[int, int],
) -> np.ndarray:
    out = inp.copy()
    for i, row in enumerate(h_rows):
        for j, col in enumerate(v_cols):
            h_color = h_colors[i]
            v_color = v_colors[j]
            out[row, col] = h_color if rank[h_color] < rank[v_color] else v_color
    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_h = rng.randint(MIN_LINE_COUNT, MAX_H_LINES)
    n_v = rng.randint(MIN_LINE_COUNT, min(MAX_V_LINES, len(COLOR_POOL) - n_h))
    total_colors = n_h + n_v

    min_h = max(MIN_H, total_colors * 2 - 1)
    h = rng.randint(min_h, MAX_H)
    w = rng.randint(MIN_W, MAX_W)

    sample_rows = _place_positions(total_colors, 0, h - 1, SAMPLE_MIN_SEP, rng)
    h_rows = _place_positions(n_h, 1, h - 2, LINE_MIN_SEP, rng)
    v_cols = _place_positions(n_v, 3, w - 2, LINE_MIN_SEP, rng)
    if sample_rows is None or h_rows is None or v_cols is None:
        return None

    colors = list(rng.sample(COLOR_POOL, total_colors))
    rng.shuffle(colors)
    h_colors = colors[:n_h]
    v_colors = colors[n_h:]
    priority = _interleave_priority(h_colors, v_colors, rng)
    rank = {color: idx for idx, color in enumerate(priority)}

    has_horizontal_top = any(rank[hc] < rank[vc] for hc in h_colors for vc in v_colors)
    has_vertical_top = any(rank[vc] < rank[hc] for hc in h_colors for vc in v_colors)
    assert has_horizontal_top and has_vertical_top

    grid = np.full((h, w), BG, dtype=int)
    for row, color in zip(sample_rows, priority):
        grid[row, 0] = color

    min_col = min(v_cols)
    max_col = max(v_cols)
    min_row = min(h_rows)
    max_row = max(h_rows)

    for row, color in zip(h_rows, h_colors):
        left_extra = rng.randint(1, min_col - 2)
        right_extra = rng.randint(1, w - 1 - max_col)
        c0 = min_col - left_extra
        c1 = max_col + right_extra
        grid[row, c0 : c1 + 1] = color

    for col, color in zip(v_cols, v_colors):
        top_extra = rng.randint(1, min_row)
        bottom_extra = rng.randint(1, h - 1 - max_row)
        r0 = min_row - top_extra
        r1 = max_row + bottom_extra
        grid[r0 : r1 + 1, col] = color

    for row in h_rows:
        for col in v_cols:
            grid[row, col] = GREY

    expected = _build_expected(grid, h_rows, v_cols, h_colors, v_colors, rank)
    visible_colors = {int(v) for v in np.unique(grid)} - {BG, GREY}
    if visible_colors != set(priority):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-priority puzzle")
