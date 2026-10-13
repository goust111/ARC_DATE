from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GRAY = 5
ORANGE = 7
FILL_COLORS = (1, 2, 3, 4, 6, 8, 9)
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in coords]
    cs = [c for _r, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _most_common_excluding(grid: np.ndarray, excluded: set[int]) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    candidates = [
        (int(count), int(color))
        for color, count in zip(colors, counts)
        if int(color) not in excluded
    ]
    assert candidates, "expected at least one remaining color"
    return max(candidates, key=lambda item: item[0])[1]


def _detect_wall_bg_pillar(grid: np.ndarray) -> tuple[int, int, int]:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    wall = Counter(corners).most_common(1)[0][0]
    bg = _most_common_excluding(grid, {wall})
    pillar = _most_common_excluding(grid, {wall, bg})
    return wall, bg, pillar


def _normalized_pattern(pillars: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    assert pillars, "pattern must contain at least one pillar"
    r0 = min(r for r, _c in pillars)
    c0 = min(c for _r, c in pillars)
    return tuple(sorted((r - r0, c - c0) for r, c in pillars))


def _find_label_color(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    wall: int,
    bg: int,
    pillar: int,
) -> int | None:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    found: dict[tuple[int, int], int] = {}

    def record(rr: int, cc: int) -> None:
        color = int(grid[rr, cc])
        if color not in {wall, bg, pillar}:
            found[(rr, cc)] = color

    for c in range(c0, c1 + 1):
        if r0 >= 2 and int(grid[r0 - 1, c]) == wall:
            record(r0 - 2, c)
        if r1 + 2 < h and int(grid[r1 + 1, c]) == wall:
            record(r1 + 2, c)
    for r in range(r0, r1 + 1):
        if c0 >= 2 and int(grid[r, c0 - 1]) == wall:
            record(r, c0 - 2)
        if c1 + 2 < w and int(grid[r, c1 + 1]) == wall:
            record(r, c1 + 2)

    if len(found) == 1:
        return next(iter(found.values()))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each room's pillar arrangement to a legend pattern, then color that room's floor.

    Input structure:
    - Black floor/background regions are enclosed by one-pixel-thick orange walls, forming rectangular rooms and small legend boxes.
    - Each room contains several gray single-pixel pillars whose relative arrangement identifies the room.
    - On the right side or bottom side, each legend box contains one gray pillar arrangement plus a nearby colored single-pixel label outside the box.

    Rule:
    - Extract the gray pillar coordinates inside every enclosed region and normalize each arrangement by translating its top-left pillar to the origin.
    - Treat any region with an external adjacent color label as a legend entry, mapping its normalized pillar arrangement to that label color.
    - Treat the remaining gray-containing regions as rooms; for each room, find the matching legend arrangement.
    - Fill all black floor cells in that room with the mapped color, while keeping gray pillars, orange walls, and the legend area unchanged.

    Key insight:
    - The number of pillars is not enough; the deciding feature is the normalized spatial pattern of gray pillars, which acts as a key into the color legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    wall, bg, pillar = _detect_wall_bg_pillar(g)
    room_mask = (g == bg) | (g == pillar)
    regions = _components(room_mask)

    legends: dict[tuple[tuple[int, int], ...], int] = {}
    rooms: list[tuple[list[tuple[int, int]], tuple[tuple[int, int], ...]]] = []

    for coords in regions:
        pillars = [(r, c) for r, c in coords if int(g[r, c]) == pillar]
        if not pillars:
            continue
        pattern = _normalized_pattern(pillars)
        label = _find_label_color(g, _bbox(coords), wall, bg, pillar)
        if label is None:
            rooms.append((coords, pattern))
        else:
            assert pattern not in legends or legends[pattern] == label, "legend pattern must map uniquely"
            legends[pattern] = label

    assert legends and rooms, "expected both legend entries and rooms"
    out = g.copy()
    for coords, pattern in rooms:
        assert pattern in legends, "every room pattern must appear in the legend"
        fill = legends[pattern]
        for r, c in coords:
            if int(g[r, c]) == bg:
                out[r, c] = fill
    return out


def _draw_rect_border(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _room_block_size(row_heights: list[int], col_widths: list[int]) -> tuple[int, int]:
    return sum(row_heights) + len(row_heights) + 1, sum(col_widths) + len(col_widths) + 1


def _draw_room_block(
    grid: np.ndarray,
    top: int,
    left: int,
    row_heights: list[int],
    col_widths: list[int],
    wall: int,
) -> list[tuple[int, int, int, int]]:
    block_h, block_w = _room_block_size(row_heights, col_widths)
    r = top
    grid[r, left : left + block_w] = wall
    for room_h in row_heights:
        r += room_h + 1
        grid[r, left : left + block_w] = wall
    c = left
    grid[top : top + block_h, c] = wall
    for room_w in col_widths:
        c += room_w + 1
        grid[top : top + block_h, c] = wall

    rooms: list[tuple[int, int, int, int]] = []
    r = top + 1
    for room_h in row_heights:
        c = left + 1
        for room_w in col_widths:
            rooms.append((r, c, room_h, room_w))
            c += room_w + 1
        r += room_h + 1
    return rooms


def _pattern_extent(pattern: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in pattern) + 1, max(c for _r, c in pattern) + 1


def _place_pattern(
    grid: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
    pattern: tuple[tuple[int, int], ...],
    color: int,
) -> None:
    ph, pw = _pattern_extent(pattern)
    dr = random.randint(0, h - ph)
    dc = random.randint(0, w - pw)
    for r, c in pattern:
        grid[top + dr + r, left + dc + c] = color


def _random_pattern(max_h: int, max_w: int, count: int) -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        ph = random.randint(2, min(4, max_h))
        pw = random.randint(2, min(4, max_w))
        if count > ph * pw:
            continue
        cells = [(r, c) for r in range(ph) for c in range(pw)]
        pts = random.sample(cells, count)
        rows = {r for r, _c in pts}
        cols = {c for _r, c in pts}
        if rows != {0, ph - 1} and (0 not in rows or ph - 1 not in rows):
            continue
        if cols != {0, pw - 1} and (0 not in cols or pw - 1 not in cols):
            continue
        return tuple(sorted(pts))
    raise RuntimeError("failed to generate a nondegenerate pattern")


def _make_patterns(n_patterns: int, max_h: int, max_w: int) -> list[tuple[tuple[int, int], ...]]:
    max_count = min(4, max_h * max_w)
    assert max_h >= 2 and max_w >= 2 and max_count >= 2
    counts = [random.randint(2, max_count) for _ in range(n_patterns)]
    if n_patterns >= 2 and random.random() < 0.7:
        shared = random.randint(2, max_count)
        i, j = random.sample(range(n_patterns), 2)
        counts[i] = shared
        counts[j] = shared

    patterns: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for count in counts:
        for _ in range(200):
            pattern = _random_pattern(max_h, max_w, count)
            if pattern not in seen:
                seen.add(pattern)
                patterns.append(pattern)
                break
        else:
            raise RuntimeError("failed to generate unique patterns")
    return patterns


def _draw_vertical_legend(
    grid: np.ndarray,
    top: int,
    left: int,
    box_side: int,
    patterns: list[tuple[tuple[int, int], ...]],
    colors: list[int],
    gaps: list[int],
    wall: int,
    pillar: int,
) -> None:
    row = top
    for i, (pattern, color) in enumerate(zip(patterns, colors)):
        _draw_rect_border(grid, row, left, box_side + 2, box_side + 2, wall)
        _place_pattern(grid, row + 1, left + 1, box_side, box_side, pattern, pillar)
        label_row = row + 1 + random.randint(0, box_side - 1)
        grid[label_row, left + box_side + 2] = color
        row += box_side + 2
        if i != len(patterns) - 1:
            row += gaps[i]


def _draw_horizontal_legend(
    grid: np.ndarray,
    top: int,
    left: int,
    box_side: int,
    patterns: list[tuple[tuple[int, int], ...]],
    colors: list[int],
    gaps: list[int],
    wall: int,
    pillar: int,
) -> None:
    col = left
    for i, (pattern, color) in enumerate(zip(patterns, colors)):
        _draw_rect_border(grid, top, col, box_side + 2, box_side + 2, wall)
        _place_pattern(grid, top + 1, col + 1, box_side, box_side, pattern, pillar)
        label_col = col + 1 + random.randint(0, box_side - 1)
        grid[top + box_side + 2, label_col] = color
        col += box_side + 2
        if i != len(patterns) - 1:
            col += gaps[i]


def _legend_span(box_side: int, gaps: list[int]) -> int:
    return (len(gaps) + 1) * (box_side + 2) + sum(gaps)


def _assignment(n_rooms: int, n_patterns: int) -> list[int]:
    ids = list(range(n_patterns))
    while len(ids) < n_rooms:
        ids.append(random.randrange(n_patterns))
    random.shuffle(ids)
    return ids


def _expected_output(
    inp: np.ndarray,
    rooms: list[tuple[int, int, int, int]],
    room_ids: list[int],
    fill_colors: list[int],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for (r0, c0, h, w), idx in zip(rooms, room_ids):
        fill = fill_colors[idx]
        patch = out[r0 : r0 + h, c0 : c0 + w]
        patch[patch == bg] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(2, 3)
    room_cols = random.randint(2, 3)
    row_heights = [random.randint(4, 6) for _ in range(room_rows)]
    col_widths = [random.randint(4, 6) for _ in range(room_cols)]
    n_rooms = room_rows * room_cols
    n_patterns = random.randint(2, min(4, n_rooms))
    box_side = random.randint(3, min(5, min(min(row_heights), min(col_widths))))
    patterns = _make_patterns(n_patterns, min(min(row_heights), box_side), min(min(col_widths), box_side))
    fills = random.sample(FILL_COLORS, n_patterns)
    room_ids = _assignment(n_rooms, n_patterns)
    legend_gaps = [random.randint(1, 2) for _ in range(n_patterns - 1)]

    main_h, main_w = _room_block_size(row_heights, col_widths)
    gap = random.randint(2, 4)
    orientation = random.choice(("right", "bottom"))
    if orientation == "right":
        legend_h = _legend_span(box_side, legend_gaps)
        legend_w = box_side + 3
        min_h = max(main_h, legend_h) + 2
        min_w = main_w + gap + legend_w + 2
    else:
        legend_h = box_side + 3
        legend_w = _legend_span(box_side, legend_gaps)
        min_h = main_h + gap + legend_h + 2
        min_w = max(main_w, legend_w) + 2
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = ORANGE
    grid[-1, :] = ORANGE
    grid[:, 0] = ORANGE
    grid[:, -1] = ORANGE

    if orientation == "right":
        main_left_slack = w - 2 - (main_w + gap + legend_w)
        main_top_slack = h - 2 - main_h
        legend_top_slack = h - 2 - legend_h
        main_top = 1 + random.randint(0, main_top_slack)
        main_left = 1 + random.randint(0, main_left_slack)
        legend_top = 1 + random.randint(0, legend_top_slack)
        legend_left = main_left + main_w + gap
    else:
        main_top_slack = h - 2 - (main_h + gap + legend_h)
        main_left_slack = w - 2 - main_w
        legend_left_slack = w - 2 - legend_w
        main_top = 1 + random.randint(0, main_top_slack)
        main_left = 1 + random.randint(0, main_left_slack)
        legend_top = main_top + main_h + gap
        legend_left = 1 + random.randint(0, legend_left_slack)

    rooms = _draw_room_block(grid, main_top, main_left, row_heights, col_widths, ORANGE)
    for (r0, c0, rh, rw), idx in zip(rooms, room_ids):
        _place_pattern(grid, r0, c0, rh, rw, patterns[idx], GRAY)

    if orientation == "right":
        _draw_vertical_legend(grid, legend_top, legend_left, box_side, patterns, fills, legend_gaps, ORANGE, GRAY)
    else:
        _draw_horizontal_legend(grid, legend_top, legend_left, box_side, patterns, fills, legend_gaps, ORANGE, GRAY)

    expected = _expected_output(grid, rooms, room_ids, fills, BLACK)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
