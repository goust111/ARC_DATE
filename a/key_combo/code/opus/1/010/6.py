import random
from itertools import product

import numpy as np


BACKGROUND = 0
FOREGROUND_COLORS = (1, 2, 3, 4, 6, 8, 9)
M_CHOICES = (4, 5)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _connected(mask: np.ndarray, connectivity: int) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False

    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    seen = {tuple(map(int, points[0]))}
    stack = [tuple(map(int, points[0]))]
    point_set = {tuple(map(int, pt)) for pt in points}
    while stack:
        r, c = stack.pop()
        for dr, dc in dirs:
            nxt = (r + dr, c + dc)
            if nxt in point_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(point_set)


def _direction_changes(seq: list[int]) -> int:
    signs = []
    for a, b in zip(seq, seq[1:]):
        delta = b - a
        if delta > 0:
            sign = 1
        elif delta < 0:
            sign = -1
        else:
            sign = 0
        if sign != 0 and (not signs or signs[-1] != sign):
            signs.append(sign)
    return max(0, len(signs) - 1)


def _profile_family(intervals: tuple[tuple[int, int], ...]) -> str:
    widths = [right - left + 1 for left, right in intervals]
    diffs = [b - a for a, b in zip(widths, widths[1:])]
    if all(delta >= 0 for delta in diffs) and any(delta > 0 for delta in diffs):
        return "expand"
    if all(delta <= 0 for delta in diffs) and any(delta < 0 for delta in diffs):
        return "contract"
    if any(delta > 0 for delta in diffs) and any(delta < 0 for delta in diffs):
        return "waist"
    return "flat"


def _intervals_to_template(intervals: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    m = len(intervals)
    template = np.full((m, m), BACKGROUND, dtype=int)
    for r, (left, right) in enumerate(intervals):
        template[r, left : right + 1] = color
    return template


def _intervals_to_outline(intervals: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    m = len(intervals)
    outline = np.full((m, m), BACKGROUND, dtype=int)
    for r, (left, right) in enumerate(intervals):
        outline[r, left] = color
        outline[r, right] = color
    return outline


def _enumerate_interval_templates(m: int) -> dict[str, list[tuple[tuple[int, int], ...]]]:
    families: dict[str, list[tuple[tuple[int, int], ...]]] = {}
    row_pairs = [(left, right) for left in range(m) for right in range(left + 1, m)]
    area_threshold = (3 * m * m + 9) // 10

    for intervals in product(row_pairs, repeat=m):
        lefts = [left for left, _ in intervals]
        rights = [right for _, right in intervals]
        widths = [right - left + 1 for left, right in intervals]

        if min(lefts) != 0 or max(rights) != m - 1:
            continue
        if any(
            abs(lefts[r + 1] - lefts[r]) > 1 or abs(rights[r + 1] - rights[r]) > 1
            for r in range(m - 1)
        ):
            continue
        if sum(widths) < area_threshold:
            continue
        if max(widths) <= 2:
            continue
        if all(left == 0 for left in lefts) and all(right == m - 1 for right in rights):
            continue
        if _direction_changes(lefts) > 1 or _direction_changes(rights) > 1:
            continue
        if _direction_changes(widths) > 2:
            continue

        template_mask = np.zeros((m, m), dtype=bool)
        outline_mask = np.zeros((m, m), dtype=bool)
        for r, (left, right) in enumerate(intervals):
            template_mask[r, left : right + 1] = True
            outline_mask[r, left] = True
            outline_mask[r, right] = True

        if not _connected(template_mask, connectivity=4):
            continue
        if not _connected(outline_mask, connectivity=8):
            continue

        family = _profile_family(intervals)
        families.setdefault(family, []).append(intervals)

    assert families, "template library must not be empty"
    return families


_TEMPLATE_LIBRARY = {m: _enumerate_interval_templates(m) for m in M_CHOICES}


def _fill_outline_window(window: np.ndarray, fg: int, bg: int) -> np.ndarray:
    m = window.shape[0]
    template = np.full((m, m), bg, dtype=int)
    for r in range(m):
        cols = np.flatnonzero(window[r] == fg)
        assert cols.size in (0, 2), "each template row must contain either 0 or 2 outline pixels"
        if cols.size == 2:
            template[r, cols[0] : cols[1] + 1] = fg
    return template


def _stamp_self_similar(template: np.ndarray, fg: int, bg: int) -> np.ndarray:
    m = template.shape[0]
    out = np.full((m * m, m * m), bg, dtype=int)
    mask = template == fg
    for r in range(m):
        for c in range(m):
            if mask[r, c]:
                out[r * m : (r + 1) * m, c * m : (c + 1) * m] = template
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn a sparse top-left outline into a filled template, then expand it as a self-similar fractal stamp.

    Input structure:
    - The grid is a square black-dominant canvas whose side length is M^2.
    - A single non-background color appears only inside the top-left MxM window.
    - Inside that window, each occupied row contains exactly two foreground pixels marking the ends of a horizontal span, and the foreground bbox is exactly MxM.

    Rule:
    - Determine M from the foreground bbox and extract the top-left MxM outline window.
    - For each occupied row, fill every cell between the leftmost and rightmost outline pixel to recover the solid MxM template.
    - Partition an M^2 x M^2 output into MxM blocks of size MxM, and copy the full template into every block whose corresponding template cell is foreground.
    - Leave blocks corresponding to template background cells black.

    Key insight:
    - The visible input is only a sparse row-boundary encoding of the real template; the fractal is defined by the filled spans, not by the sparse outline pixels themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    bg = _background_color(g)
    fg_colors = [int(color) for color in np.unique(g) if color != bg]
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = fg_colors[0]

    coords = np.argwhere(g == fg)
    assert coords.size > 0, "expected a foreground outline"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r0) == 0 and int(c0) == 0, "the outline must start in the top-left corner"
    m = int(r1 - r0 + 1)
    assert m == int(c1 - c0 + 1), "foreground bbox must be square"
    assert g.shape == (m * m, m * m), "grid side length must equal M^2"
    assert np.all(g[m:, :] == bg) and np.all(g[:, m:] == bg), "only the top-left MxM window may contain foreground"

    outline_window = g[:m, :m]
    template = _fill_outline_window(outline_window, fg=fg, bg=bg)
    return _stamp_self_similar(template, fg=fg, bg=bg)


def generate_input() -> np.ndarray:
    rng = random.Random()

    m = rng.choice(M_CHOICES)
    families = _TEMPLATE_LIBRARY[m]
    family = rng.choice(tuple(families))
    intervals = rng.choice(families[family])
    color = rng.choice(FOREGROUND_COLORS)

    grid = np.full((m * m, m * m), BACKGROUND, dtype=int)
    outline = _intervals_to_outline(intervals, color)
    grid[:m, :m] = outline
    return grid
