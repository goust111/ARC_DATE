from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg, in_bounds, neighbors4, paste


BG = 0
BLUE = 1
YELLOW = 4
GREY = 5

AREA_TO_MOVE = {
    3: (1, 0, 2),   # down 2
    4: (0, 1, 3),   # right 3
    5: (0, -1, 2),  # left 2
    6: (1, 0, 3),   # down 3
    7: (0, 1, 4),   # right 4
}

GLYPH_TEMPLATES = {
    3: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
    4: np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
    5: np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=int),
}

ARM_COLORS = [2, 3, 6, 7, 8, 9]
NOISE_COLORS = [1, 2, 3, 6, 7, 8, 9]

HEIGHT_RANGE = (18, 24)
LEFT_WIDTH_RANGE = (10, 13)
RIGHT_WIDTH_RANGE = (11, 13)
MARKER_COUNT_RANGE = (4, 6)
GLYPH_COUNT_CHOICES = [3, 4, 5]
GLYPH_COUNT_WEIGHTS = [2, 3, 3]
MAX_ATTEMPTS = 400


def _find_divider_col(grid: np.ndarray) -> int:
    for c in range(int(grid.shape[1])):
        if bool((grid[:, c] == GREY).all()):
            return int(c)
    raise AssertionError("expected one full-height grey divider column")


def _find_plus_markers(left: np.ndarray, *, bg: int) -> list[dict[str, int]]:
    markers: list[dict[str, int]] = []
    h, w = left.shape
    for r, c in np.argwhere(left == YELLOW):
        r = int(r)
        c = int(c)
        if not (0 < r < h - 1 and 0 < c < w - 1):
            continue
        arms = [int(left[nr, nc]) for nr, nc in neighbors4(r, c)]
        if len(set(arms)) != 1:
            continue
        arm_color = arms[0]
        if arm_color in (int(bg), YELLOW):
            continue
        markers.append({"r": r, "c": c, "color": int(arm_color)})
    markers.sort(key=lambda m: (int(m["r"]), int(m["c"])))
    return markers


def _rank_arm_colors(markers: list[dict[str, int]]) -> list[int]:
    counts: Counter[int] = Counter()
    first_row: dict[int, int] = {}
    for marker in markers:
        color = int(marker["color"])
        row = int(marker["r"])
        counts[color] += 1
        first_row[color] = min(first_row.get(color, row), row)

    ranked = sorted(counts, key=lambda color: (-counts[color], first_row[color]))
    seen_keys = set()
    for color in ranked:
        key = (int(counts[color]), int(first_row[color]))
        assert key not in seen_keys, "ambiguous color ranking"
        seen_keys.add(key)
    return [int(color) for color in ranked]


def _find_glyphs(right: np.ndarray) -> list[dict[str, int]]:
    glyphs: list[dict[str, int]] = []
    comps = find_components(
        right,
        bg=BG,
        colors=[BLUE],
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    for comp in comps:
        bbox = comp["bbox"]
        area = int(comp["area"])
        if bbox_h(bbox) == 3 and bbox_w(bbox) == 3 and area in AREA_TO_MOVE:
            r0, c0, _r1, _c1 = map(int, bbox)
            glyphs.append({"r0": r0, "c0": c0, "area": area})
    glyphs.sort(key=lambda item: (int(item["c0"]), int(item["r0"])))
    return glyphs


def _trace_segments(start: tuple[int, int], areas: list[int]) -> list[list[tuple[int, int]]]:
    r, c = map(int, start)
    segments: list[list[tuple[int, int]]] = []
    for area in areas:
        dr, dc, steps = AREA_TO_MOVE[int(area)]
        cells = [(r, c)]
        rr, cc = r, c
        for _ in range(int(steps)):
            rr += dr
            cc += dc
            cells.append((int(rr), int(cc)))
        segments.append(cells)
        r, c = rr, cc
    return segments


def _draw_polyline(
    out: np.ndarray,
    *,
    divider_col: int,
    start: tuple[int, int],
    areas: list[int],
    colors: list[int],
) -> np.ndarray:
    assert len(areas) == len(colors)
    right_w = int(out.shape[1] - divider_col - 1)
    segments = _trace_segments(start, areas)
    for cells, color in zip(segments, colors):
        for r, c in cells:
            assert in_bounds(out.shape, r, divider_col + 1 + c)
            assert 0 <= c < right_w
            out[int(r), int(divider_col + 1 + c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a color-ranked polyline in the right panel using left-panel yellow plus counts as the palette key.

    Input structure:
    - The background is black and one full-height grey divider splits the grid into left and right panels.
    - Left panel: several legal plus markers made of a yellow center with four equal-colored orthogonal arms, plus irrelevant noise pixels.
    - Right panel: one grey start pixel and several blue 3x3 glyph components whose pixel areas encode segment directions and lengths.

    Rule:
    - Detect every legal yellow-centered plus in the left panel and count how often each arm color occurs.
    - Rank arm colors by descending frequency; break ties by the smallest marker row for that color.
    - Read the blue glyphs in the right panel in column-major order (left to right, top to bottom), decode each area into one segment, and chain the segments from the grey start pixel.
    - Recolor segment k with the k-th ranked arm color, keeping the left panel and divider unchanged while drawing on the original right panel.

    Key insight:
    - The left panel does not encode geometry; it encodes the ordered palette that must be transferred onto the right-panel polyline instructions.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    divider_col = _find_divider_col(g)
    left = g[:, :divider_col]
    right = g[:, divider_col + 1 :]

    markers = _find_plus_markers(left, bg=bg)
    assert markers, "expected at least one legal plus marker"
    ranked_colors = _rank_arm_colors(markers)

    glyphs = _find_glyphs(right)
    assert len(glyphs) == len(ranked_colors), "glyph count must equal arm-color variety count"
    areas = [int(glyph["area"]) for glyph in glyphs]

    start_cells = np.argwhere(right == GREY)
    assert len(start_cells) == 1, "expected one start pixel in the right panel"
    start = tuple(map(int, start_cells[0]))

    out = g.copy()
    return _draw_polyline(out, divider_col=divider_col, start=start, areas=areas, colors=ranked_colors)


def _count_signatures(total: int, parts: int, max_part: int | None = None) -> list[list[int]]:
    if parts == 1:
        if max_part is not None and total > max_part:
            return []
        return [[int(total)]]
    upper = total - parts + 1
    if max_part is not None:
        upper = min(upper, max_part)
    out: list[list[int]] = []
    for first in range(upper, 0, -1):
        for rest in _count_signatures(total - first, parts - 1, first):
            out.append([int(first), *rest])
    return out


def _sample_count_signature(k: int, marker_count: int) -> list[int]:
    signatures = _count_signatures(marker_count, k)
    with_ties = [sig for sig in signatures if len(set(sig)) < len(sig)]
    without_ties = [sig for sig in signatures if len(set(sig)) == len(sig)]
    want_tie = bool(with_ties) and random.random() < 0.5
    if want_tie:
        return list(random.choice(with_ties))
    if without_ties:
        return list(random.choice(without_ties))
    return list(random.choice(signatures))


def _sample_spaced_rows(h: int, n: int, min_gap: int = 3) -> list[int] | None:
    choices = list(range(1, h - 1))
    for _ in range(200):
        rows = sorted(random.sample(choices, n))
        if all(rows[i + 1] - rows[i] >= min_gap for i in range(n - 1)):
            return [int(r) for r in rows]
    return None


def _build_marker_plan(h: int, left_w: int, k: int, marker_count: int) -> tuple[list[dict[str, int]], list[int]] | None:
    rows = _sample_spaced_rows(h, marker_count, min_gap=3)
    if rows is None:
        return None

    count_signature = _sample_count_signature(k, marker_count)
    colors = random.sample(ARM_COLORS, k)
    ranked_colors: list[int] = []
    idx = 0
    while idx < k:
        group_end = idx + 1
        while group_end < k and count_signature[group_end] == count_signature[idx]:
            group_end += 1
        tied_colors = colors[idx:group_end]
        random.shuffle(tied_colors)
        ranked_colors.extend(int(color) for color in tied_colors)
        idx = group_end

    counts_by_color = {int(color): int(count) for color, count in zip(ranked_colors, count_signature)}
    row_colors = list(ranked_colors)
    leftovers: list[int] = []
    for color in ranked_colors:
        leftovers.extend([int(color)] * (counts_by_color[int(color)] - 1))
    random.shuffle(leftovers)
    row_colors.extend(leftovers)
    assert len(row_colors) == marker_count

    markers: list[dict[str, int]] = []
    for row, color in zip(rows, row_colors):
        markers.append({"r": int(row), "c": -1, "color": int(color)})

    for idx, marker in enumerate(markers):
        row = int(marker["r"])
        allowed = []
        for col in range(1, left_w - 1):
            ok = True
            for prev in markers[:idx]:
                row_gap = abs(row - int(prev["r"]))
                if row_gap == 3 and abs(col - int(prev["c"])) <= 1:
                    ok = False
                    break
            if ok:
                allowed.append(int(col))
        if not allowed:
            return None
        marker["c"] = int(random.choice(allowed))
    return markers, ranked_colors


def _paint_plus(grid: np.ndarray, r: int, c: int, arm_color: int) -> None:
    grid[r, c] = YELLOW
    for nr, nc in neighbors4(r, c):
        grid[nr, nc] = int(arm_color)


def _place_noise(grid: np.ndarray, left_w: int, occupied: set[tuple[int, int]]) -> None:
    h = int(grid.shape[0])
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(left_w):
            cell = (int(r), int(c))
            if cell in occupied:
                continue
            close = False
            for rr, cc in occupied:
                if max(abs(r - rr), abs(c - cc)) <= 1:
                    close = True
                    break
            if not close:
                candidates.append(cell)
    random.shuffle(candidates)
    if not candidates:
        noise_count = 0
    else:
        upper = min(8, len(candidates))
        lower = min(3, upper)
        noise_count = random.randint(lower, upper)
    for r, c in candidates[:noise_count]:
        grid[r, c] = random.choice(NOISE_COLORS)


def _sample_areas(k: int) -> list[int]:
    vertical = [3, 6]
    horizontal = [4, 5, 7]
    for _ in range(100):
        areas = random.sample(list(AREA_TO_MOVE), k)
        if any(area in vertical for area in areas) and any(area in horizontal for area in areas):
            random.shuffle(areas)
            return [int(area) for area in areas]
    areas = random.sample(list(AREA_TO_MOVE), k)
    random.shuffle(areas)
    return [int(area) for area in areas]


def _glyph_slots(right_w: int) -> list[tuple[int, int]]:
    row_shift = random.randint(0, 1)
    col_shift = random.randint(0, right_w - 11)
    slots = [
        (row_shift + 0, col_shift + 0),
        (row_shift + 4, col_shift + 0),
        (row_shift + 0, col_shift + 4),
        (row_shift + 4, col_shift + 4),
        (row_shift + 0, col_shift + 8),
    ]
    return [(int(r), int(c)) for r, c in slots]


def _path_col_range(areas: list[int]) -> tuple[int, int]:
    pos = 0
    min_pos = 0
    max_pos = 0
    for area in areas:
        dr, dc, steps = AREA_TO_MOVE[int(area)]
        del dr
        new_pos = pos + dc * steps
        min_pos = min(min_pos, pos, new_pos)
        max_pos = max(max_pos, pos, new_pos)
        pos = new_pos
    return int(min_pos), int(max_pos)


def _render_expected(
    inp: np.ndarray,
    *,
    divider_col: int,
    start: tuple[int, int],
    areas: list[int],
    ranked_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    return _draw_polyline(out, divider_col=divider_col, start=start, areas=areas, colors=ranked_colors)


def _validate_case(
    grid: np.ndarray,
    *,
    divider_col: int,
    markers: list[dict[str, int]],
    ranked_colors: list[int],
    areas: list[int],
    glyph_positions: list[tuple[int, int]],
    start: tuple[int, int],
    expected: np.ndarray,
) -> bool:
    left = grid[:, :divider_col]
    right = grid[:, divider_col + 1 :]

    detected_markers = _find_plus_markers(left, bg=BG)
    expected_markers = sorted((int(m["r"]), int(m["c"]), int(m["color"])) for m in markers)
    got_markers = sorted((int(m["r"]), int(m["c"]), int(m["color"])) for m in detected_markers)
    if got_markers != expected_markers:
        return False

    if _rank_arm_colors(detected_markers) != [int(color) for color in ranked_colors]:
        return False

    glyphs = _find_glyphs(right)
    if [int(glyph["area"]) for glyph in glyphs] != [int(area) for area in areas]:
        return False
    if [(int(glyph["r0"]), int(glyph["c0"])) for glyph in glyphs] != list(glyph_positions):
        return False

    if tuple(map(int, np.argwhere(right == GREY)[0])) != tuple(start):
        return False

    if not np.array_equal(main(grid), expected):
        return False

    if np.array_equal(grid, expected):
        return False

    if not np.array_equal(grid[:, : divider_col + 1], expected[:, : divider_col + 1]):
        return False

    if len(set(ranked_colors)) != len(ranked_colors):
        return False
    return True


def _attempt_case() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    left_w = random.randint(*LEFT_WIDTH_RANGE)
    right_w = random.randint(*RIGHT_WIDTH_RANGE)
    w = left_w + 1 + right_w

    k = random.choices(GLYPH_COUNT_CHOICES, weights=GLYPH_COUNT_WEIGHTS, k=1)[0]
    marker_count = random.randint(max(MARKER_COUNT_RANGE[0], k), MARKER_COUNT_RANGE[1])

    marker_plan = _build_marker_plan(h, left_w, k, marker_count)
    if marker_plan is None:
        return None
    markers, ranked_colors = marker_plan

    areas = _sample_areas(k)
    slots = _glyph_slots(right_w)
    glyph_positions = slots[:k]
    glyph_bottom = max(r0 + 2 for r0, _c0 in glyph_positions)

    down_total = sum(AREA_TO_MOVE[area][2] for area in areas if AREA_TO_MOVE[area][0] != 0)
    start_row_min = glyph_bottom + 2
    start_row_max = h - 1 - down_total
    if start_row_min > start_row_max:
        return None

    min_dx, max_dx = _path_col_range(areas)
    start_col_min = -min_dx
    start_col_max = right_w - 1 - max_dx
    if start_col_min > start_col_max:
        return None

    start = (int(random.randint(start_row_min, start_row_max)), int(random.randint(start_col_min, start_col_max)))

    inp = np.full((h, w), BG, dtype=int)
    inp[:, left_w] = GREY

    occupied: set[tuple[int, int]] = set()
    for marker in markers:
        r = int(marker["r"])
        c = int(marker["c"])
        arm_color = int(marker["color"])
        _paint_plus(inp, r, c, arm_color)
        occupied.add((r, c))
        for nr, nc in neighbors4(r, c):
            occupied.add((int(nr), int(nc)))

    _place_noise(inp, left_w, occupied)

    for (r0, c0), area in zip(glyph_positions, areas):
        patch = GLYPH_TEMPLATES[int(area)] * BLUE
        paste(inp[:, left_w + 1 :], patch, r0, c0)

    sr, sc = start
    inp[sr, left_w + 1 + sc] = GREY

    expected = _render_expected(inp, divider_col=left_w, start=start, areas=areas, ranked_colors=ranked_colors)
    if not _validate_case(
        inp,
        divider_col=left_w,
        markers=markers,
        ranked_colors=ranked_colors,
        areas=areas,
        glyph_positions=glyph_positions,
        start=start,
        expected=expected,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_case()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
