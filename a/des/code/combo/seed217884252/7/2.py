from __future__ import annotations

import random

import numpy as np


BG = 0
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _trace_corner_run(
    g: np.ndarray,
    start_r: int,
    step_r: int,
    col: int,
    inner_col: int,
) -> tuple[list[int], list[int]] | None:
    h = g.shape[0]
    rows: list[int] = []
    colors: list[int] = []
    r = start_r
    while 0 <= r < h and int(g[r, col]) != BG:
        if int(g[r, inner_col]) != BG:
            return None
        rows.append(int(r))
        colors.append(int(g[r, col]))
        r += step_r
    if len(rows) < 2 or len(set(colors)) != len(colors):
        return None
    rows.sort()
    colors = [int(g[r, col]) for r in rows]
    return rows, colors


def _find_legend(g: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = g.shape
    assert h >= 2 and w >= 2
    candidates = []
    for start_r, step_r, col in (
        (0, 1, 0),
        (h - 1, -1, 0),
        (0, 1, w - 1),
        (h - 1, -1, w - 1),
    ):
        inner_col = 1 if col == 0 else w - 2
        traced = _trace_corner_run(g, start_r, step_r, col, inner_col)
        if traced is None:
            continue
        rows, colors = traced
        candidates.append((len(rows), rows, col, colors))
    assert candidates, "expected one corner legend"
    candidates.sort(reverse=True)
    length, rows, col, colors = candidates[0]
    assert length >= 2

    mask = np.zeros_like(g, dtype=bool)
    for r in rows:
        mask[r, col] = True
    length_by_color = {int(color): i + 1 for i, color in enumerate(colors)}
    return length_by_color, mask


def _extract_paths_for_color(
    g: np.ndarray,
    color: int,
    length_by_color: dict[int, int],
    legend_mask: np.ndarray,
) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
    h, w = g.shape
    comps = _connected_components(g == int(color))
    if not comps:
        return []

    paths: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
    legend_colors = set(length_by_color)
    for comp in comps:
        if len(comp) < 2:
            return []
        coord_set = set(comp)
        neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
        endpoints: list[tuple[int, int]] = []
        for cell in comp:
            nbs = [nb for nb in _neighbors4(cell[0], cell[1], h, w) if nb in coord_set]
            neighbor_map[cell] = nbs
            if len(nbs) == 1:
                endpoints.append(cell)
            elif len(nbs) != 2:
                return []
        if len(endpoints) != 2:
            return []

        endpoint_markers: list[list[tuple[int, int]]] = []
        for endpoint in endpoints:
            markers = []
            for nr, nc in _neighbors4(endpoint[0], endpoint[1], h, w):
                if (nr, nc) in coord_set or legend_mask[nr, nc]:
                    continue
                val = int(g[nr, nc])
                if val in legend_colors:
                    markers.append((nr, nc))
            if len(markers) > 1:
                return []
            endpoint_markers.append(markers)

        if bool(endpoint_markers[0]) == bool(endpoint_markers[1]):
            return []

        marked_idx = 0 if endpoint_markers[0] else 1
        bare_idx = 1 - marked_idx
        marker_cell = endpoint_markers[marked_idx][0]

        order = [endpoints[bare_idx]]
        prev = None
        current = endpoints[bare_idx]
        while True:
            next_cells = [nb for nb in neighbor_map[current] if nb != prev]
            if not next_cells:
                break
            prev, current = current, next_cells[0]
            order.append(current)
        if len(order) != len(comp):
            return []
        paths.append((order, marker_cell))
    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Truncate branchless paths using a corner legend and a marker attached to the opposite endpoint.

    Input structure
    - Background is black.
    - A corner legend is a vertical color list; its row index encodes lengths 1..N.
    - Several single-pixel-wide 4-connected paths share one color and each has exactly two endpoints.
    - Exactly one endpoint of each path is adjacent to a marker pixel colored by the legend.

    Rule
    - Read the marker's color in the legend to get k.
    - Keep the first k path pixels starting from the endpoint without a marker.
    - Erase the remaining path pixels and erase the marker pixel itself.
    - Leave the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    length_by_color, legend_mask = _find_legend(g)

    candidates = [
        int(color)
        for color in np.unique(g)
        if int(color) != BG and int(color) not in length_by_color
    ]
    valid: list[tuple[int, list[tuple[list[tuple[int, int]], tuple[int, int]]]]] = []
    for color in candidates:
        paths = _extract_paths_for_color(g, color, length_by_color, legend_mask)
        if paths:
            valid.append((color, paths))
    assert len(valid) == 1, "expected exactly one path color"

    _path_color, paths = valid[0]
    out = g.copy()
    for order, marker_cell in paths:
        marker_color = int(g[marker_cell])
        k = int(length_by_color[marker_color])
        keep = set(order[:k])
        for cell in order:
            if cell not in keep:
                out[cell] = BG
        out[marker_cell] = BG
    return out


# --------------- generator ---------------

PATH_COLOR = 5
GRID_HEIGHT_RANGE = (12, 22)
GRID_WIDTH_RANGE = (12, 22)
LEGEND_LEN_RANGE = (2, 5)
PATH_COUNT_RANGE = (1, 4)
EXTRA_COLORS = tuple(c for c in range(1, 10) if c != PATH_COLOR)
STYLE_OPTIONS = ("straight", "elbow", "zigzag")
GEN_ATTEMPTS = 400


def _perpendiculars(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    return ((dc, dr), (-dc, -dr))


def _make_relative_path(style: str, keep_len: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        d1 = random.choice(DIR4)
        if style == "straight":
            dirs = [d1]
            steps = [random.randint(max(1, keep_len), keep_len + 5)]
        elif style == "elbow":
            dirs = [d1, random.choice(_perpendiculars(d1))]
            steps = [random.randint(1, 4), random.randint(1, 4)]
        else:
            dirs = [d1, random.choice(_perpendiculars(d1)), d1]
            steps = [random.randint(1, 4), random.randint(1, 4), random.randint(1, 4)]
        if 1 + sum(steps) <= keep_len:
            continue

        order = [(0, 0)]
        seen = {(0, 0)}
        current = (0, 0)
        ok = True
        for direction, step_count in zip(dirs, steps, strict=False):
            for _step in range(step_count):
                nr = current[0] + direction[0]
                nc = current[1] + direction[1]
                nxt = (nr, nc)
                if nxt in seen:
                    ok = False
                    break
                for adj in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                    if adj in seen and adj != current:
                        ok = False
                        break
                if not ok:
                    break
                order.append(nxt)
                seen.add(nxt)
                current = nxt
            if not ok:
                break
        if not ok:
            continue

        min_r = min(r for r, _ in order)
        min_c = min(c for _, c in order)
        normalized = [(r - min_r, c - min_c) for r, c in order]
        return normalized
    return None


def _stamp_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        blocked[r, c] = True
        for dr, dc in DIR8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True


def _legend_rows(h: int, length: int, anchor_top: bool) -> list[int]:
    if anchor_top:
        return list(range(length))
    return list(range(h - length, h))


def _place_legend(
    grid: np.ndarray,
    blocked: np.ndarray,
    colors: list[int],
    corner: str,
) -> np.ndarray:
    h, w = grid.shape
    legend_mask = np.zeros_like(grid, dtype=bool)
    col = 0 if "left" in corner else w - 1
    rows = _legend_rows(h, len(colors), anchor_top=("top" in corner))
    for row, color in zip(rows, colors, strict=False):
        grid[row, col] = int(color)
        legend_mask[row, col] = True
    _stamp_blocked(blocked, [(row, col) for row in rows])
    return legend_mask


def _marker_options(
    path_cells: list[tuple[int, int]],
    blocked: np.ndarray,
) -> list[tuple[int, tuple[int, int]]]:
    h, w = blocked.shape
    path_set = set(path_cells)
    options: list[tuple[int, tuple[int, int]]] = []
    endpoint_indices = (0, len(path_cells) - 1)
    for idx in endpoint_indices:
        endpoint = path_cells[idx]
        for nr, nc in _neighbors4(endpoint[0], endpoint[1], h, w):
            if nr in (0, h - 1) or nc in (0, w - 1):
                continue
            if blocked[nr, nc] or (nr, nc) in path_set:
                continue
            adjacent_path_neighbors = [
                cell for cell in _neighbors4(nr, nc, h, w) if cell in path_set
            ]
            if adjacent_path_neighbors == [endpoint]:
                options.append((idx, (nr, nc)))
    return options


def _place_one_path(
    grid: np.ndarray,
    blocked: np.ndarray,
    keep_len: int,
    marker_color: int,
) -> tuple[list[tuple[int, int]], int, tuple[int, int]] | None:
    h, w = grid.shape
    styles = list(STYLE_OPTIONS)
    random.shuffle(styles)
    for style in styles:
        for _ in range(100):
            rel = _make_relative_path(style, keep_len)
            if rel is None:
                continue
            max_r = max(r for r, _ in rel)
            max_c = max(c for _, c in rel)
            if h - 2 - max_r < 1 or w - 2 - max_c < 1:
                continue

            row_choices = list(range(1, h - 1 - max_r))
            col_choices = list(range(1, w - 1 - max_c))
            random.shuffle(row_choices)
            random.shuffle(col_choices)
            for r0 in row_choices:
                for c0 in col_choices:
                    cells = [(r0 + r, c0 + c) for r, c in rel]
                    if any(blocked[r, c] for r, c in cells):
                        continue
                    marker_choices = _marker_options(cells, blocked)
                    if not marker_choices:
                        continue
                    marker_end, marker_cell = random.choice(marker_choices)
                    for r, c in cells:
                        grid[r, c] = PATH_COLOR
                    grid[marker_cell] = int(marker_color)
                    _stamp_blocked(blocked, cells + [marker_cell])
                    return cells, marker_end, marker_cell
    return None


def _expected_output(
    grid: np.ndarray,
    specs: list[tuple[list[tuple[int, int]], int, tuple[int, int], int]],
) -> np.ndarray:
    out = grid.copy()
    for cells, marker_end, marker_cell, keep_len in specs:
        if marker_end == 0:
            keep = set(cells[-keep_len:])
        else:
            keep = set(cells[:keep_len])
        for cell in cells:
            if cell not in keep:
                out[cell] = BG
        out[marker_cell] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    legend_len = random.randint(*LEGEND_LEN_RANGE)
    if h < legend_len + 3 or w < 6:
        return None

    legend_colors = random.sample(EXTRA_COLORS, legend_len)
    corner = random.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
    path_count = random.randint(*PATH_COUNT_RANGE)

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    _place_legend(grid, blocked, legend_colors, corner)

    specs: list[tuple[list[tuple[int, int]], int, tuple[int, int], int]] = []
    keep_options = list(range(1, legend_len + 1))
    for _ in range(path_count):
        keep_len = random.choice(keep_options)
        marker_color = legend_colors[keep_len - 1]
        placed = _place_one_path(grid, blocked, keep_len, marker_color)
        if placed is None:
            return None
        cells, marker_end, marker_cell = placed
        specs.append((cells, marker_end, marker_cell, keep_len))

    expected = _expected_output(grid, specs)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
