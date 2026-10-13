import random

import numpy as np


ORANGE = 7
BLUE = 1
RED = 2
GREY = 5

MAX_GENERATION_ATTEMPTS = 400


def _separator_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == GREY))]


def _separator_cols(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == GREY))]


def _count_blue(block: np.ndarray) -> int:
    return int(np.count_nonzero(block == BLUE))


def _paint_from_start(line: np.ndarray, indexes: list[int], d: int) -> None:
    for idx in indexes[:d]:
        line[idx] = RED


def _paint_from_end(line: np.ndarray, indexes: list[int], d: int) -> None:
    for idx in indexes[len(indexes) - d :]:
        line[idx] = RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn separator-line count imbalances into red bars.

    A full grey separator row compares the blue counts in the horizontal bands
    immediately above and below it. A full grey separator column compares the
    blue counts in the vertical bands immediately left and right of it.

    To resolve the prompt's endpoint ambiguity consistently:
    - separator row: upper side wins -> paint from the left end, lower side wins
      -> paint from the right end.
    - separator column: left side wins -> paint from the top end, right side
      wins -> paint from the bottom end.

    When row and column separators coexist, intersections stay grey and do not
    count toward either line's drawable length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = map(int, g.shape)
    sep_rows = _separator_rows(g)
    sep_cols = _separator_cols(g)
    assert sep_rows or sep_cols, "expected at least one full grey separator line"

    out = g.copy()

    row_bounds = [-1] + sep_rows + [h]
    drawable_cols = [c for c in range(w) if c not in set(sep_cols)]
    for i, r in enumerate(sep_rows, start=1):
        top0 = row_bounds[i - 1] + 1
        top1 = r
        bot0 = r + 1
        bot1 = row_bounds[i + 1]
        assert top0 < top1 and bot0 < bot1, "separator rows must separate non-empty bands"

        top_count = _count_blue(g[top0:top1, :])
        bottom_count = _count_blue(g[bot0:bot1, :])
        d = abs(top_count - bottom_count)
        assert d <= len(drawable_cols), "difference must fit on the separator row"

        if top_count > bottom_count:
            _paint_from_start(out[r, :], drawable_cols, d)
        elif bottom_count > top_count:
            _paint_from_end(out[r, :], drawable_cols, d)

    col_bounds = [-1] + sep_cols + [w]
    drawable_rows = [r for r in range(h) if r not in set(sep_rows)]
    for i, c in enumerate(sep_cols, start=1):
        left0 = col_bounds[i - 1] + 1
        left1 = c
        right0 = c + 1
        right1 = col_bounds[i + 1]
        assert left0 < left1 and right0 < right1, "separator columns must separate non-empty bands"

        left_count = _count_blue(g[:, left0:left1])
        right_count = _count_blue(g[:, right0:right1])
        d = abs(left_count - right_count)
        assert d <= len(drawable_rows), "difference must fit on the separator column"

        if left_count > right_count:
            _paint_from_start(out[:, c], drawable_rows, d)
        elif right_count > left_count:
            _paint_from_end(out[:, c], drawable_rows, d)

    return out


def _positive_parts(parts: int, lo: int, hi: int) -> list[int]:
    return [random.randint(lo, hi) for _ in range(parts)]


def _relation_pattern(n_sep: int) -> list[int]:
    choices = [-1, 0, 1]
    if n_sep == 1:
        return [random.choice(choices)]

    base = choices * ((n_sep + 2) // 3)
    random.shuffle(base)
    rels = base[:n_sep]
    if len(set(rels)) == 1:
        rels[0] = 0
        rels[-1] = 1 if rels[0] != 1 else -1
    random.shuffle(rels)
    return rels


def _realize_counts(relations: list[int], max_count: int) -> list[int] | None:
    step_max = min(5, max_count - 1)
    if step_max <= 0:
        return None

    need_smaller = any(rel > 0 for rel in relations)
    need_larger = any(rel < 0 for rel in relations)
    for _ in range(300):
        lo = 1 + step_max if need_smaller and max_count >= 1 + step_max else 1
        hi = max_count - step_max if need_larger and max_count >= 1 + step_max else max_count
        if lo > hi:
            lo, hi = 1, max_count
        counts = [random.randint(lo, hi)]
        ok = True
        for rel in relations:
            prev = counts[-1]
            if rel == 0:
                counts.append(prev)
                continue
            if rel > 0:
                nxt_lo = max(1, prev - step_max)
                nxt_hi = prev - 1
            else:
                nxt_lo = prev + 1
                nxt_hi = min(max_count, prev + step_max)
            if nxt_lo > nxt_hi:
                ok = False
                break
            counts.append(random.randint(nxt_lo, nxt_hi))
        if ok:
            return counts
    return None


def _random_positions(height: int, width: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    return random.sample(cells, count)


def _fill_horizontal_bands(grid: np.ndarray, heights: list[int], counts: list[int]) -> None:
    row = 0
    for band_h, blue_count in zip(heights, counts):
        positions = _random_positions(band_h, grid.shape[1], blue_count)
        for rr, cc in positions:
            grid[row + rr, cc] = BLUE
        row += band_h
        if row < grid.shape[0]:
            grid[row, :] = GREY
            row += 1


def _fill_vertical_bands(grid: np.ndarray, widths: list[int], counts: list[int]) -> None:
    col = 0
    for band_w, blue_count in zip(widths, counts):
        positions = _random_positions(grid.shape[0], band_w, blue_count)
        for rr, cc in positions:
            grid[rr, col + cc] = BLUE
        col += band_w
        if col < grid.shape[1]:
            grid[:, col] = GREY
            col += 1


def _gen_horizontal() -> np.ndarray | None:
    n_bands = random.randint(2, 5)
    heights = _positive_parts(n_bands, 2, 5)
    width = random.randint(8, 16)
    grid = np.full((sum(heights) + n_bands - 1, width), ORANGE, dtype=int)

    max_count = min(width - 1, min(h * width - 1 for h in heights))
    relations = _relation_pattern(n_bands - 1)
    counts = _realize_counts(relations, max_count)
    if counts is None:
        return None

    _fill_horizontal_bands(grid, heights, counts)
    return grid


def _gen_vertical() -> np.ndarray | None:
    n_bands = random.randint(2, 5)
    widths = _positive_parts(n_bands, 2, 5)
    height = random.randint(8, 16)
    grid = np.full((height, sum(widths) + n_bands - 1), ORANGE, dtype=int)

    max_count = min(height - 1, min(height * w - 1 for w in widths))
    relations = _relation_pattern(n_bands - 1)
    counts = _realize_counts(relations, max_count)
    if counts is None:
        return None

    _fill_vertical_bands(grid, widths, counts)
    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if not np.any(grid == BLUE):
        return False
    if not (_separator_rows(grid) or _separator_cols(grid)):
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return bool(np.any(out == RED) or np.any(main(grid) == GREY))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        if random.random() < 0.5:
            grid = _gen_horizontal()
        else:
            grid = _gen_vertical()
        if grid is not None and _valid_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
