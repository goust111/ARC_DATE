from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SOURCE = 1
GREY = 5
SHELF_COLORS = (2, 3, 4, 7)
DIR_LEFT = "L"
DIR_RIGHT = "R"
DIR_BOTH = "B"
ALL_DIRS = (DIR_LEFT, DIR_RIGHT, DIR_BOTH)
MAX_RETRIES = 300


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _parse_legend(grid: np.ndarray) -> dict[int, str]:
    color_row = np.asarray(grid[-2], dtype=int)
    hint_row = np.asarray(grid[-1], dtype=int)
    mapping: dict[int, str] = {}
    for c in range(color_row.shape[0]):
        color = int(color_row[c])
        if color not in SHELF_COLORS:
            continue
        assert color not in mapping, "legend colors must be unique"
        hint_cols = [
            nc
            for nc in (c - 1, c, c + 1)
            if 0 <= nc < hint_row.shape[0] and int(hint_row[nc]) == GREY
        ]
        assert len(hint_cols) == 1, "each legend color needs one grey indicator"
        delta = int(hint_cols[0] - c)
        if delta == -1:
            mapping[color] = DIR_LEFT
        elif delta == 0:
            mapping[color] = DIR_BOTH
        elif delta == 1:
            mapping[color] = DIR_RIGHT
        else:
            raise AssertionError("legend offset must be -1, 0, or +1")
    assert mapping, "legend mapping must be non-empty"
    return mapping


def _shelf_segments(body: np.ndarray, shelf_colors: set[int]) -> list[list[tuple[int, int, int]]]:
    h, w = body.shape
    segments: list[list[tuple[int, int, int]]] = [[] for _ in range(h)]
    for r in range(h):
        c = 0
        while c < w:
            color = int(body[r, c])
            if color not in shelf_colors:
                c += 1
                continue
            c0 = c
            while c + 1 < w and int(body[r, c + 1]) == color:
                c += 1
            segments[r].append((c0, c, color))
            c += 1
    return segments


def _find_segment(
    segments: list[list[tuple[int, int, int]]],
    row: int,
    col: int,
) -> tuple[int, int, int] | None:
    for a, b, color in segments[row]:
        if a <= col <= b:
            return (a, b, color)
    return None


def _source_starts(body: np.ndarray) -> list[tuple[int, int]]:
    h, w = body.shape
    visited = np.zeros((h, w), dtype=bool)
    starts: set[tuple[int, int]] = set()
    for c0 in range(w):
        if int(body[0, c0]) != SOURCE or visited[0, c0]:
            continue
        q = deque([(0, c0)])
        visited[0, c0] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in _neighbors8(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if visited[nr, nc] or int(body[nr, nc]) != SOURCE:
                    continue
                visited[nr, nc] = True
                q.append((nr, nc))
        bottom_by_col: dict[int, int] = {}
        for r, c in coords:
            bottom_by_col[c] = max(bottom_by_col.get(c, -1), r)
        starts.update((r, c) for c, r in bottom_by_col.items())
    return sorted(starts)


def _simulate(
    body: np.ndarray,
    mapping: dict[int, str],
    *,
    recolor: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    body = np.asarray(body, dtype=int)
    h, w = body.shape
    shelf_colors = set(mapping)
    segments = _shelf_segments(body, shelf_colors)
    out = body.copy()
    starts = _source_starts(body)
    q = deque((r, c, SOURCE) for r, c in starts)
    seen: set[tuple[int, int, int]] = set()
    shelf_hits: list[tuple[int, int, int, str, int]] = []
    painted_by_color = {SOURCE: 0, **{color: 0 for color in mapping}}
    conflict = False

    def paint(r: int, c: int, color: int) -> None:
        nonlocal conflict
        if int(body[r, c]) != BG:
            return
        prev = int(out[r, c])
        if prev == BG:
            out[r, c] = int(color)
            painted_by_color[int(color)] = painted_by_color.get(int(color), 0) + 1
        elif prev != int(color):
            conflict = True

    while q:
        r, c, color = q.popleft()
        if not (0 <= r < h and 0 <= c < w):
            continue
        state = (int(r), int(c), int(color))
        if state in seen:
            continue
        seen.add(state)

        rr, cc = int(r), int(c)
        paint(rr, cc, int(color))

        while True:
            nr = rr + 1
            if nr >= h:
                break
            below = int(body[nr, cc])
            if below in shelf_colors:
                seg = _find_segment(segments, nr, cc)
                assert seg is not None and int(seg[2]) == below
                a, b, shelf_color = seg
                for x in range(a, b + 1):
                    paint(rr, x, int(color))
                direction = mapping[int(shelf_color)]
                next_color = int(shelf_color) if recolor else int(color)
                shelf_hits.append((nr, cc, int(shelf_color), direction, int(color)))
                if direction in (DIR_LEFT, DIR_BOTH) and a - 1 >= 0:
                    q.append((nr, a - 1, next_color))
                if direction in (DIR_RIGHT, DIR_BOTH) and b + 1 < w:
                    q.append((nr, b + 1, next_color))
                break
            rr = nr
            paint(rr, cc, int(color))

    meta: dict[str, object] = {
        "conflict": conflict,
        "starts": starts,
        "shelf_hits": shelf_hits,
        "painted_by_color": painted_by_color,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate legend-controlled shelf overflows where each spill recolors the falling liquid.

    Input structure:
    - The background is black, with blue top-edge source blocks in the top 1-2 rows.
    - The main body contains horizontal 1-pixel-tall colored shelves using red, green, yellow, or orange.
    - The last two rows form a legend: shelf colors on the penultimate row and grey markers on the last row, where left/right/center offset encodes overflow side.

    Rule:
    - Remove the bottom two legend rows in the output.
    - From each source column, liquid falls downward through black cells in its current color.
    - When liquid reaches a shelf, it fills the row above that shelf across the full shelf span, then spills from the legend-selected side: left only, right only, or both sides.
    - Every spilled branch changes color to the shelf color before continuing downward, and the same process repeats on later shelves.
    - Shelves and sources remain unchanged; only black cells on the liquid paths are painted.

    Key insight:
    - Each shelf is both a routing node and a color transformer, so solving the task requires reading the legend and propagating the shelf-induced color changes through the full cascade.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3
    mapping = _parse_legend(g)
    body = g[:-2, :]
    out, _meta = _simulate(body, mapping, recolor=True)
    return out


def _place_source(body: np.ndarray, col: int, *, height: int, width: int) -> None:
    for dc in range(width):
        body[:height, col + dc] = SOURCE


def _place_shelf(body: np.ndarray, row: int, a: int, b: int, color: int) -> None:
    assert 0 <= row < body.shape[0]
    assert 0 <= a <= b < body.shape[1]
    assert np.all(body[row, a : b + 1] == BG), "shelf placements must not overlap"
    body[row, a : b + 1] = int(color)


def _legend_order(mapping: dict[int, str]) -> list[int]:
    left = [color for color, direction in mapping.items() if direction == DIR_LEFT]
    both = [color for color, direction in mapping.items() if direction == DIR_BOTH]
    right = [color for color, direction in mapping.items() if direction == DIR_RIGHT]
    random.shuffle(left)
    random.shuffle(both)
    random.shuffle(right)
    return left + both + right


def _make_legend(width: int, mapping: dict[int, str]) -> tuple[np.ndarray, np.ndarray]:
    color_row = np.zeros(width, dtype=int)
    hint_row = np.zeros(width, dtype=int)
    order = _legend_order(mapping)
    n = len(order)
    start_min = 1
    start_max = width - 2 * n
    assert start_max >= start_min
    start = random.randint(start_min, start_max)
    for i, color in enumerate(order):
        c = start + 2 * i
        color_row[c] = int(color)
        direction = mapping[int(color)]
        if direction == DIR_LEFT:
            hint_row[c - 1] = GREY
        elif direction == DIR_RIGHT:
            hint_row[c + 1] = GREY
        else:
            hint_row[c] = GREY
    return color_row, hint_row


def _mutate_direction(direction: str) -> str:
    choices = [candidate for candidate in ALL_DIRS if candidate != direction]
    return random.choice(choices)


def _build_complex_layout() -> tuple[np.ndarray, dict[int, str], int]:
    has_second_source = random.random() < 0.45
    width = random.randint(24, 30) if has_second_source else random.randint(15, 23)

    root_color, left_color, right_color = random.sample(SHELF_COLORS, 3)
    mapping = {
        int(root_color): DIR_BOTH,
        int(left_color): DIR_LEFT,
        int(right_color): DIR_RIGHT,
    }

    ext_l = random.randint(0, 1)
    ext_r = random.randint(0, 1)
    len2 = random.randint(3, 5)
    len3 = random.randint(3, 5)
    len4 = random.randint(3, 4)
    len5 = random.randint(3, 4)

    center_min = 6 + ext_l
    center_max = width - 7 - ext_r
    center = random.randint(center_min, center_max)

    r1 = random.randint(3, 5)
    r2 = r1 + random.randint(3, 4)
    r3 = r1 + random.randint(3, 4)
    r4 = r2 + random.randint(3, 4)
    r5 = r3 + random.randint(3, 4)
    height_min = max(r4, r5) + 4
    height_max = min(24, height_min + 4)
    height = random.randint(height_min, height_max)

    body = np.zeros((height, width), dtype=int)
    source_height = random.randint(1, 2)
    _place_source(body, center, height=source_height, width=1)

    a1 = center - 1 - ext_l
    b1 = center + 1 + ext_r
    c_left = a1 - 1
    c_right = b1 + 1

    b2 = c_left
    a2 = b2 - len2 + 1
    c_left_2 = b2 + 1

    a3 = c_right
    b3 = a3 + len3 - 1
    c_right_2 = a3 - 1

    b4 = c_left_2
    a4 = b4 - len4 + 1

    a5 = c_right_2
    b5 = a5 + len5 - 1

    leftmost = min(a1, a2, a4, center)
    rightmost = max(b1, b3, b5, center)
    assert leftmost >= 0 and rightmost < width

    _place_shelf(body, r1, a1, b1, root_color)
    _place_shelf(body, r2, a2, b2, right_color)
    _place_shelf(body, r3, a3, b3, left_color)
    _place_shelf(body, r4, a4, b4, left_color)
    _place_shelf(body, r5, a5, b5, right_color)

    if has_second_source:
        gap = 3
        complex_left = leftmost
        complex_right = rightmost
        side = random.choice(["left", "right"])
        source_width = 2 if random.random() < 0.4 else 1
        source_height_2 = random.randint(1, 2)
        if side == "left":
            lane_end = complex_left - gap - source_width
            if lane_end >= 0:
                col = random.randint(0, lane_end)
                _place_source(body, col, height=source_height_2, width=source_width)
        else:
            lane_start = complex_right + gap + 1
            if lane_start + source_width - 1 < width:
                col = random.randint(lane_start, width - source_width)
                _place_source(body, col, height=source_height_2, width=source_width)

    return body, mapping, root_color


def _validate_candidate(body: np.ndarray, mapping: dict[int, str], root_color: int) -> bool:
    out, meta = _simulate(body, mapping, recolor=True)
    if bool(meta["conflict"]):
        return False

    hits = meta["shelf_hits"]
    if len(hits) < 5:
        return False
    hit_colors = {int(color) for _r, _c, color, _d, _in_color in hits}
    if hit_colors != set(mapping):
        return False

    for color in mapping:
        if int(np.count_nonzero(out == color)) <= int(np.count_nonzero(body == color)):
            return False

    if np.array_equal(out, body):
        return False

    no_recolor, _ = _simulate(body, mapping, recolor=False)
    if np.array_equal(no_recolor, out):
        return False

    mutated = dict(mapping)
    mutated[root_color] = _mutate_direction(mutated[root_color])
    changed, _ = _simulate(body, mutated, recolor=True)
    if np.array_equal(changed, out):
        return False

    legend_rows = _make_legend(body.shape[1], mapping)
    inp = np.vstack([body, legend_rows[0][None, :], legend_rows[1][None, :]])
    solver_out = main(inp)
    if not np.array_equal(solver_out, out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    body, mapping, root_color = _build_complex_layout()
    if not _validate_candidate(body, mapping, root_color):
        return None
    color_row, hint_row = _make_legend(body.shape[1], mapping)
    return np.vstack([body, color_row[None, :], hint_row[None, :]])


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
