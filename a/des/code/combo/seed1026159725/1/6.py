from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(10))

GEN_ATTEMPTS = 300
ROW_COUNT_OPTIONS = (2, 3)
PAIR_COUNT_OPTIONS = (1, 2)
SEP_THICKNESS_OPTIONS = (1, 2)
MAX_GRID_SIZE = 30
PANEL_MIN_SIZE = 6
PANEL_MAX_SIZE = 9
SMALL_AREA_RANGE = (3, 7)
BIG_EXTRA_RANGE = (2, 6)
OBJECT_SAME_COLOR_PROB = 0.35


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.flat)
    return int(counts.most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _separator_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    counts: Counter[int] = Counter()
    for r in range(h):
        if bool(np.all(grid[r] == grid[r, 0])):
            counts[int(grid[r, 0])] += 1
    for c in range(w):
        if bool(np.all(grid[:, c] == grid[0, c])):
            counts[int(grid[0, c])] += 1
    counts.pop(int(bg), None)
    assert counts, "expected at least one separator line"
    return int(counts.most_common(1)[0][0])


def _non_separator_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    n = int(mask.shape[0])
    while i < n:
        while i < n and bool(mask[i]):
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not bool(mask[j]):
            j += 1
        spans.append((int(i), int(j)))
        i = j
    return spans


def _parse_layout(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    bg = int(get_bg(grid))
    sep = int(_separator_color(grid, bg))
    h, w = grid.shape
    row_sep = np.array([bool(np.all(grid[r] == sep)) for r in range(h)], dtype=bool)
    col_sep = np.array([bool(np.all(grid[:, c] == sep)) for c in range(w)], dtype=bool)
    row_spans = _non_separator_spans(row_sep)
    col_spans = _non_separator_spans(col_sep)
    assert len(row_spans) >= 2, "expected multiple panel rows"
    assert len(col_spans) >= 2, "expected multiple panel columns"
    return bg, sep, row_sep, col_sep, row_spans, col_spans


def _parse_panel(panel: np.ndarray, bg: int, sep: int) -> dict:
    colors = [int(c) for c in np.unique(panel) if int(c) not in {int(bg), int(sep)}]
    assert len(colors) == 2, "each panel must contain one object color and one marker color"

    marker_colors = [color for color in colors if int(np.count_nonzero(panel == color)) == 1]
    assert len(marker_colors) == 1, "panel must contain exactly one singleton marker color"
    marker_color = int(marker_colors[0])
    marker = tuple(int(v) for v in np.argwhere(panel == marker_color)[0])

    object_color = int(next(color for color in colors if color != marker_color))
    object_mask = panel == object_color
    object_coords = [tuple(int(v) for v in rc) for rc in np.argwhere(object_mask)]
    assert len(object_coords) >= 2, "object area must be at least 2"
    assert len(_components(object_mask)) == 1, "object must be 4-connected"

    return {
        "marker_color": marker_color,
        "marker": marker,
        "object_color": object_color,
        "object_coords": object_coords,
        "area": len(object_coords),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair panels by marker color within each panel row, transfer the smaller shape onto the larger panel,
    erase the smaller object, and connect the pair through the separator columns.
    """
    g = np.asarray(grid, dtype=int)
    bg, sep, _row_sep, col_sep, row_spans, col_spans = _parse_layout(g)
    out = g.copy()

    panel_rows: list[list[dict]] = []
    for row_idx, (r0, r1) in enumerate(row_spans):
        panels: list[dict] = []
        for col_idx, (c0, c1) in enumerate(col_spans):
            info = _parse_panel(g[r0:r1, c0:c1], bg, sep)
            info.update(
                {
                    "panel_row": row_idx,
                    "panel_col": col_idx,
                    "bbox": (r0, r1, c0, c1),
                    "global_marker": (r0 + info["marker"][0], c0 + info["marker"][1]),
                }
            )
            panels.append(info)
        panel_rows.append(panels)

    for panels in panel_rows:
        by_marker: dict[int, list[dict]] = defaultdict(list)
        for info in panels:
            by_marker[int(info["marker_color"])].append(info)

        for marker_color, pair in by_marker.items():
            assert len(pair) == 2, "each marker color must appear in exactly two panels within a row"
            first, second = pair
            assert first["global_marker"][0] == second["global_marker"][0], "paired markers must share one row"

            if int(first["area"]) < int(second["area"]):
                small, big = first, second
            elif int(second["area"]) < int(first["area"]):
                small, big = second, first
            else:
                raise AssertionError("paired objects must have different areas")

            sr0, sr1, sc0, sc1 = small["bbox"]
            for rr, cc in small["object_coords"]:
                out[sr0 + rr, sc0 + cc] = bg

            br0, br1, bc0, bc1 = big["bbox"]
            bmarker_r, bmarker_c = big["marker"]
            smarker_r, smarker_c = small["marker"]
            big_h = int(br1 - br0)
            big_w = int(bc1 - bc0)
            for rr, cc in small["object_coords"]:
                tr = int(bmarker_r + (rr - smarker_r))
                tc = int(bmarker_c + (cc - smarker_c))
                assert 0 <= tr < big_h and 0 <= tc < big_w, "translated shape must remain inside the large panel"
                out[br0 + tr, bc0 + tc] = int(marker_color)

            left, right = sorted(pair, key=lambda item: int(item["bbox"][2]))
            line_r = int(left["global_marker"][0])
            for c in range(int(left["bbox"][3]), int(right["bbox"][2])):
                if bool(col_sep[c]):
                    out[line_r, c] = int(marker_color)

    return out


def _grow_connected_shape(
    allowed_cells: set[tuple[int, int]],
    target_size: int,
    initial: set[tuple[int, int]] | None = None,
    seed_candidates: list[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    if initial is None:
        starts = seed_candidates if seed_candidates else list(allowed_cells)
        if not starts:
            return None
        current = {random.choice(starts)}
    else:
        current = set(initial)
        if not current or not current.issubset(allowed_cells):
            return None

    while len(current) < target_size:
        frontier: list[tuple[int, int]] = []
        for r, c in current:
            for nr, nc in neighbors4(r, c):
                cell = (int(nr), int(nc))
                if cell in allowed_cells and cell not in current and cell not in frontier:
                    frontier.append(cell)
        if not frontier:
            return None
        current.add(random.choice(frontier))
    return current


def _make_pair_shapes(panel_h: int, panel_w: int, marker: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    allowed = {(r, c) for r in range(panel_h) for c in range(panel_w) if (r, c) != marker}
    near_marker = [
        (r, c)
        for r, c in allowed
        if 1 <= abs(r - marker[0]) + abs(c - marker[1]) <= 3
    ]
    panel_cells = int(panel_h * panel_w - 1)

    small_max = min(int(SMALL_AREA_RANGE[1]), max(int(SMALL_AREA_RANGE[0]), panel_cells // 4))
    small_area = random.randint(int(SMALL_AREA_RANGE[0]), int(small_max))

    big_min = small_area + int(BIG_EXTRA_RANGE[0])
    big_max = min(small_area + int(BIG_EXTRA_RANGE[1]), panel_cells // 2, panel_cells - 1)
    if big_max < big_min:
        big_min = small_area + 1
    if big_max < big_min:
        return None

    for _ in range(80):
        small = _grow_connected_shape(allowed, small_area, seed_candidates=near_marker)
        if small is None:
            continue
        big_area = random.randint(int(big_min), int(big_max))
        big = _grow_connected_shape(allowed, big_area, initial=small)
        if big is None:
            continue
        return small, big
    return None


def _paint_panel(
    grid: np.ndarray,
    row_span: tuple[int, int],
    col_span: tuple[int, int],
    shape: set[tuple[int, int]],
    marker: tuple[int, int],
    object_color: int,
    marker_color: int,
    bg: int,
) -> None:
    r0, r1 = row_span
    c0, c1 = col_span
    grid[r0:r1, c0:c1] = bg
    for rr, cc in shape:
        grid[r0 + rr, c0 + cc] = int(object_color)
    grid[r0 + marker[0], c0 + marker[1]] = int(marker_color)


def _grid_spans(count: int, panel_size: int, sep: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for idx in range(count):
        start = cursor
        stop = start + panel_size
        spans.append((int(start), int(stop)))
        cursor = stop + (sep if idx + 1 < count else 0)
    return spans


def _gen_attempt() -> np.ndarray | None:
    panel_rows = random.choice(ROW_COUNT_OPTIONS)
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    panel_cols = 2 * pair_count
    sep = random.choice(SEP_THICKNESS_OPTIONS)

    max_panel_h = min(int(PANEL_MAX_SIZE), (MAX_GRID_SIZE - (panel_rows - 1) * sep) // panel_rows)
    max_panel_w = min(int(PANEL_MAX_SIZE), (MAX_GRID_SIZE - (panel_cols - 1) * sep) // panel_cols)
    if max_panel_h < PANEL_MIN_SIZE or max_panel_w < PANEL_MIN_SIZE:
        return None

    panel_h = random.randint(PANEL_MIN_SIZE, int(max_panel_h))
    panel_w = random.randint(PANEL_MIN_SIZE, int(max_panel_w))

    height = panel_rows * panel_h + (panel_rows - 1) * sep
    width = panel_cols * panel_w + (panel_cols - 1) * sep
    if height > MAX_GRID_SIZE or width > MAX_GRID_SIZE:
        return None

    bg, sep_color = random.sample(PALETTE, 2)
    grid = np.full((height, width), int(sep_color), dtype=int)
    row_spans = _grid_spans(panel_rows, panel_h, sep)
    col_spans = _grid_spans(panel_cols, panel_w, sep)

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            grid[r0:r1, c0:c1] = int(bg)

    for row_idx, row_span in enumerate(row_spans):
        marker_palette = [color for color in PALETTE if color not in {int(bg), int(sep_color)}]
        marker_colors = random.sample(marker_palette, pair_count)
        for pair_idx in range(pair_count):
            marker = (random.randint(1, panel_h - 2), random.randint(1, panel_w - 2))
            shapes = _make_pair_shapes(panel_h, panel_w, marker)
            if shapes is None:
                return None
            small_shape, big_shape = shapes
            big_on_left = bool(random.getrandbits(1))

            marker_color = int(marker_colors[pair_idx])
            object_palette = [color for color in PALETTE if color not in {int(bg), int(sep_color), marker_color}]
            if random.random() < OBJECT_SAME_COLOR_PROB:
                left_color = right_color = int(random.choice(object_palette))
            else:
                left_color, right_color = [int(v) for v in random.sample(object_palette, 2)]

            left_col = 2 * pair_idx
            right_col = left_col + 1
            left_shape = big_shape if big_on_left else small_shape
            right_shape = small_shape if big_on_left else big_shape

            _paint_panel(
                grid,
                row_span,
                col_spans[left_col],
                left_shape,
                marker,
                left_color,
                marker_color,
                int(bg),
            )
            _paint_panel(
                grid,
                row_span,
                col_spans[right_col],
                right_shape,
                marker,
                right_color,
                marker_color,
                int(bg),
            )

    if int(get_bg(grid)) != int(bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
