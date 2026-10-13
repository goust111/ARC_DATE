from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components  # noqa: E402


Grid = np.ndarray
Rect = tuple[int, int, int, int]


def _find_cross_axes(grid: Grid) -> tuple[int, int, int]:
    rows = []
    for r in range(grid.shape[0]):
        value = int(grid[r, 0])
        if value != Color.BLACK and np.all(grid[r] == value):
            rows.append((r, value))
    cols = []
    for c in range(grid.shape[1]):
        value = int(grid[0, c])
        if value != Color.BLACK and np.all(grid[:, c] == value):
            cols.append((c, value))
    assert len(rows) == 1, f"expected one full row bar, got {rows}"
    assert len(cols) == 1, f"expected one full column bar, got {cols}"
    axis_r, row_color = rows[0]
    axis_c, col_color = cols[0]
    assert row_color == col_color, "row and column bars must share a color"
    return int(axis_r), int(axis_c), int(row_color)


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rect_mask(shape: tuple[int, int], rect: Rect) -> Grid:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _contains(outer: Rect, inner: Rect) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _overlap(a: Rect, b: Rect) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _enumerate_color_rectangles(mask: Grid) -> list[Rect]:
    h, w = mask.shape
    run = np.zeros((h, w), dtype=int)
    for r in range(h):
        span = 0
        for c in range(w - 1, -1, -1):
            if mask[r, c]:
                span += 1
            else:
                span = 0
            run[r, c] = span

    rects: list[Rect] = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            if run[r0, c0] < 2:
                continue
            min_w = run[r0, c0]
            for r1 in range(r0 + 1, h):
                if run[r1, c0] == 0:
                    break
                min_w = min(min_w, int(run[r1, c0]))
                if min_w < 2:
                    break
                rects.append((r0, c0, r1, c0 + min_w - 1))

    rects = sorted(set(rects), key=lambda rect: (_rect_area(rect), rect))
    maximal: list[Rect] = []
    for rect in rects:
        if any(other != rect and _contains(other, rect) for other in rects):
            continue
        maximal.append(rect)
    return maximal


def _detect_occluders(grid: Grid, axis_r: int, axis_c: int, bar_color: int) -> list[Rect]:
    colors = [int(color) for color in np.unique(grid) if color not in (Color.BLACK, bar_color)]
    candidates: list[tuple[int, Rect]] = []
    for color in colors:
        for rect in _enumerate_color_rectangles(grid == color):
            r0, c0, r1, c1 = rect
            if r0 <= axis_r <= r1 or c0 <= axis_c <= c1:
                continue
            candidates.append((color, rect))

    candidates.sort(key=lambda item: (_rect_area(item[1]), item[1]), reverse=True)
    chosen: list[Rect] = []
    for _color, rect in candidates:
        if any(_contains(other, rect) or _overlap(other, rect) for other in chosen):
            continue
        chosen.append(rect)
    chosen.sort()
    return chosen


def _canonical_anchor_bounds(shape: tuple[int, int], axis_r: int, axis_c: int) -> tuple[int, int, int, int]:
    h, w = shape
    qh = min(axis_r, h - 1 - axis_r)
    qw = min(axis_c, w - 1 - axis_c)
    return axis_r - qh, axis_r, axis_c - qw, axis_c


def _orbit(anchor_r: int, anchor_c: int, axis_r: int, axis_c: int) -> tuple[tuple[int, int], ...]:
    return (
        (anchor_r, anchor_c),
        (anchor_r, 2 * axis_c - anchor_c),
        (2 * axis_r - anchor_r, anchor_c),
        (2 * axis_r - anchor_r, 2 * axis_c - anchor_c),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase solid rectangular occluders, then restore the cross-symmetric pattern using occluder colors on repaired cells.

    Input structure:
    - A black background contains one full row bar and one full column bar of the same color, with an off-center crossing.
    - A monochrome pattern occupies cells around the cross and is symmetric under reflection across both bars.
    - One or more disjoint solid rectangles of size at least 2x2 cover part of that pattern; rectangles may share the pattern color or use other colors, but never cover the bars.
    - The pattern's own connected components are not solid filled rectangles.

    Rule:
    - Detect every solid rectangular block of size at least 2x2 and erase it to black.
    - Use the full row bar as the horizontal mirror axis and the full column bar as the vertical mirror axis.
    - Whenever one cell of a four-way symmetry orbit remains as visible pattern, restore the missing orbit cells.
    - Paint each restored cell with the color of the rectangle that originally covered it, while keeping the bars and already visible pattern unchanged.

    Key insight:
    - Rectangles are identified by the geometric predicate "filled rectangle" rather than by color, and symmetry recovery only trusts orbits that still contain visible pattern after those rectangles are removed.
    """
    g = np.asarray(grid, dtype=int)
    axis_r, axis_c, bar_color = _find_cross_axes(g)

    occluders = _detect_occluders(g, axis_r, axis_c, bar_color)
    cleaned = g.copy()
    for rect in occluders:
        r0, c0, r1, c1 = rect
        cleaned[r0 : r1 + 1, c0 : c1 + 1] = Color.BLACK

    remaining_colors = [int(color) for color in np.unique(cleaned) if color not in (Color.BLACK, bar_color)]
    assert len(remaining_colors) == 1, f"expected one visible pattern color after erasing rectangles, got {remaining_colors}"
    pattern_color = remaining_colors[0]

    out = cleaned.copy()
    r_start, r_stop, c_start, c_stop = _canonical_anchor_bounds(g.shape, axis_r, axis_c)
    for anchor_r in range(r_start, r_stop):
        for anchor_c in range(c_start, c_stop):
            orbit = _orbit(anchor_r, anchor_c, axis_r, axis_c)
            if any(cleaned[r, c] == pattern_color for r, c in orbit):
                for r, c in orbit:
                    if out[r, c] == Color.BLACK and g[r, c] != Color.BLACK:
                        out[r, c] = g[r, c]
    return out


GRID_RANGE = (14, 24)
BAR_MARGIN = 3
TARGET_ORBITS = (5, 11)
RECT_SIDE_RANGE = (2, 4)
RECT_COUNT_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 500


def _has_2x2(mask: Grid) -> bool:
    return bool(np.any(mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]))


def _component_is_solid_rectangle(mask: Grid) -> bool:
    comps = find_components(mask.astype(int), bg=0, connectivity=4, monochromatic=True)
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        area = int(comp["area"])
        if area == (r1 - r0 + 1) * (c1 - c0 + 1) and (r1 > r0) and (c1 > c0):
            return True
    return False


def _quadrant_shape(qh: int, qw: int) -> Grid:
    target = random.randint(TARGET_ORBITS[0], min(TARGET_ORBITS[1], qh * qw - 1))
    max_components = random.randint(1, 3)
    mask = np.zeros((qh, qw), dtype=bool)

    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        cells = np.argwhere(mask)
        if cells.size:
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < qh and 0 <= nc < qw and not mask[nr, nc]:
                        frontier.add((nr, nc))

        if not frontier or (random.random() < 0.22 and len(find_components(mask.astype(int), bg=0, connectivity=4, monochromatic=True)) < max_components):
            frontier = {(r, c) for r in range(qh) for c in range(qw) if not mask[r, c]}

        choices = list(frontier)
        random.shuffle(choices)
        placed = False
        for r, c in choices:
            trial = mask.copy()
            trial[r, c] = True
            if _has_2x2(trial):
                continue
            mask = trial
            placed = True
            break
        if not placed:
            break
    return mask


def _full_support(shape: tuple[int, int], axis_r: int, axis_c: int, qmask: Grid) -> Grid:
    support = np.zeros(shape, dtype=bool)
    qh, qw = qmask.shape
    r_start = axis_r - qh
    c_start = axis_c - qw
    for qr, qc in np.argwhere(qmask):
        anchor_r = r_start + int(qr)
        anchor_c = c_start + int(qc)
        for r, c in _orbit(anchor_r, anchor_c, axis_r, axis_c):
            support[r, c] = True
    return support


def _valid_bar_position(size: int) -> int:
    return random.randint(BAR_MARGIN, size - 1 - BAR_MARGIN)


def _candidate_rectangles(support: Grid, axis_r: int, axis_c: int) -> list[Rect]:
    h, w = support.shape
    candidates: list[Rect] = []
    for rh in range(RECT_SIDE_RANGE[0], RECT_SIDE_RANGE[1] + 1):
        for rw in range(RECT_SIDE_RANGE[0], RECT_SIDE_RANGE[1] + 1):
            for r0 in range(0, h - rh + 1):
                r1 = r0 + rh - 1
                if r0 <= axis_r <= r1:
                    continue
                for c0 in range(0, w - rw + 1):
                    c1 = c0 + rw - 1
                    if c0 <= axis_c <= c1:
                        continue
                    sub = support[r0 : r1 + 1, c0 : c1 + 1]
                    support_cells = int(sub.sum())
                    bg_cells = int(sub.size - support_cells)
                    if support_cells == 0 or bg_cells == 0:
                        continue
                    if support_cells == sub.size:
                        continue
                    candidates.append((r0, c0, r1, c1))
    random.shuffle(candidates)
    return candidates


def _orbit_coverage_ok(support: Grid, cover_mask: Grid, axis_r: int, axis_c: int) -> bool:
    r_start, r_stop, c_start, c_stop = _canonical_anchor_bounds(support.shape, axis_r, axis_c)
    covered_any = False
    for anchor_r in range(r_start, r_stop):
        for anchor_c in range(c_start, c_stop):
            orbit = _orbit(anchor_r, anchor_c, axis_r, axis_c)
            if not support[anchor_r, anchor_c]:
                continue
            covered = sum(int(cover_mask[r, c]) for r, c in orbit)
            if covered:
                covered_any = True
            if covered == 4:
                return False
    return covered_any


def _build_expected_output(
    shape: tuple[int, int],
    support: Grid,
    axis_r: int,
    axis_c: int,
    bar_color: int,
    pattern_color: int,
    rectangles: list[tuple[Rect, int]],
) -> Grid:
    out = np.zeros(shape, dtype=int)
    out[axis_r, :] = bar_color
    out[:, axis_c] = bar_color
    out[support] = pattern_color
    for rect, color in rectangles:
        rect_cells = _rect_mask(shape, rect) & support
        out[rect_cells] = color
    return out


def _build_input(output: Grid, rectangles: list[tuple[Rect, int]]) -> Grid:
    inp = output.copy()
    for rect, color in rectangles:
        r0, c0, r1, c1 = rect
        inp[r0 : r1 + 1, c0 : c1 + 1] = color
    return inp


def _sample_rectangle_colors(count: int, pattern_color: int, bar_color: int) -> list[int]:
    palette = [color for color in range(1, 10) if color != bar_color]
    non_pattern = [color for color in palette if color != pattern_color]
    if count == 1:
        return [pattern_color if random.random() < 0.35 else random.choice(non_pattern)]

    colors = [random.choice(non_pattern) for _ in range(count)]
    if random.random() < 0.55:
        colors[random.randrange(count)] = pattern_color
    return colors


def _unique_full_bars(grid: Grid, bar_color: int) -> bool:
    full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == bar_color)]
    full_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == bar_color)]
    return len(full_rows) == 1 and len(full_cols) == 1


def _gen_attempt() -> Grid | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    if min(h, w) < 2 * BAR_MARGIN + 1:
        return None

    axis_r = _valid_bar_position(h)
    axis_c = _valid_bar_position(w)
    if h % 2 == 1 and w % 2 == 1 and axis_r == h // 2 and axis_c == w // 2:
        return None

    qh = min(axis_r, h - 1 - axis_r)
    qw = min(axis_c, w - 1 - axis_c)
    if min(qh, qw) < 2:
        return None

    qmask = _quadrant_shape(qh, qw)
    if int(qmask.sum()) < TARGET_ORBITS[0]:
        return None

    support = _full_support((h, w), axis_r, axis_c, qmask)
    if _has_2x2(support) or _component_is_solid_rectangle(support):
        return None

    bar_color, pattern_color = random.sample([color for color in range(1, 10)], 2)
    candidates = _candidate_rectangles(support, axis_r, axis_c)
    rect_count = random.randint(*RECT_COUNT_RANGE)
    colors = _sample_rectangle_colors(rect_count, pattern_color, bar_color)

    selected: list[Rect] = []
    cover_mask = np.zeros((h, w), dtype=bool)
    for color in colors:
        _ = color
        placed = False
        for rect in candidates:
            if rect in selected or any(_overlap(rect, other) for other in selected):
                continue
            rect_mask = _rect_mask((h, w), rect)
            trial_cover = cover_mask | (rect_mask & support)
            if not _orbit_coverage_ok(support, trial_cover, axis_r, axis_c):
                continue
            selected.append(rect)
            cover_mask = trial_cover
            placed = True
            break
        if not placed:
            return None

    rectangles = list(zip(selected, colors))
    out = _build_expected_output((h, w), support, axis_r, axis_c, bar_color, pattern_color, rectangles)
    inp = _build_input(out, rectangles)

    if np.array_equal(inp, out):
        return None
    if not _unique_full_bars(inp, bar_color):
        return None

    detected = _detect_occluders(inp, axis_r, axis_c, bar_color)
    if sorted(detected) != sorted(selected):
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    changed = np.argwhere(inp != out)
    if changed.size == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
