from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SHAPES_BY_AREA = {
    9: [(3, 3)],
    15: [(3, 5), (5, 3)],
    25: [(5, 5)],
}
SMALL_AREAS = (9, 15)
LARGE_AREAS_BY_SMALL = {
    9: (15, 25),
    15: (25,),
}
GEN_RETRIES = 400
PLACE_RETRIES = 300


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _sign(x: int) -> int:
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


def _parse_marker_pairs(row: np.ndarray, bg: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    cells = [int(v) for v in np.asarray(row, dtype=int)]
    c = 0
    while c < len(cells):
        if cells[c] == bg:
            c += 1
            continue
        start = c
        while c < len(cells) and cells[c] != bg:
            c += 1
        run = cells[start:c]
        assert len(run) == 2, "each marker run must contain exactly two colors"
        pairs.append((int(run[0]), int(run[1])))
    assert pairs, "expected at least one marker pair"
    return pairs


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == int(color)
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rs = [rr for rr, _ in coords]
            cs = [cc for _, cc in coords]
            bbox = (min(rs), min(cs), max(rs), max(cs))
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            comps.append(
                {
                    "coords": coords,
                    "area": len(coords),
                    "bbox": bbox,
                    "center": center,
                }
            )
    return comps


def _line_cells(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, p0)
    r1, c1 = map(int, p1)
    dr = r1 - r0
    dc = c1 - c0
    assert r0 == r1 or c0 == c1 or abs(dr) == abs(dc), "centers must align horizontally, vertically, or diagonally"
    step_r = _sign(dr)
    step_c = _sign(dc)
    length = max(abs(dr), abs(dc)) + 1
    return [(r0 + step_r * k, c0 + step_c * k) for k in range(length)]


def _draw_line(grid: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: int) -> None:
    for r, c in _line_cells(p0, p1):
        grid[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect object centers according to the bottom marker pairs.

    For each marker pair (A, B), sort the two A objects and the two B objects by
    area. Draw an A-colored line from smaller-A center to smaller-B center, then
    draw a B-colored line from larger-A center to larger-B center. Finally clear
    the marker row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 2 and w >= 3

    bg = _mode_color(g)
    marker_pairs = _parse_marker_pairs(g[-1], bg)
    marker_colors = {int(color) for pair in marker_pairs for color in pair}

    upper = g[:-1, :]
    upper_colors = {int(v) for v in np.unique(upper) if int(v) != int(bg)}
    assert upper_colors == marker_colors, "upper-grid colors must match marker colors exactly"

    comps_by_color: dict[int, list[dict]] = {}
    for color in marker_colors:
        comps = _components_of_color(upper, color)
        assert len(comps) == 2, "each marker color must have exactly two objects"
        comps.sort(
            key=lambda comp: (
                int(comp["area"]),
                int(comp["bbox"][0]),
                int(comp["bbox"][1]),
                int(comp["center"][0]),
                int(comp["center"][1]),
            )
        )
        assert int(comps[0]["area"]) != int(comps[1]["area"]), "object sizes must be unambiguous"
        comps_by_color[int(color)] = comps

    out = g.copy()
    for color_a, color_b in marker_pairs:
        small_a, large_a = comps_by_color[int(color_a)]
        small_b, large_b = comps_by_color[int(color_b)]
        _draw_line(out, small_a["center"], small_b["center"], int(color_a))
        _draw_line(out, large_a["center"], large_b["center"], int(color_b))
    out[-1, :] = int(bg)
    return out


def _rect_cells(center: tuple[int, int], dims: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, center)
    h, w = map(int, dims)
    hr = h // 2
    hc = w // 2
    return [
        (rr, cc)
        for rr in range(r - hr, r + hr + 1)
        for cc in range(c - hc, c + hc + 1)
    ]


def _mark_forbidden(mask: np.ndarray, cells: set[tuple[int, int]], pad: int = 1) -> None:
    h, w = mask.shape
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True


def _choose_dims_for_color() -> tuple[tuple[int, int], tuple[int, int]]:
    small_area = random.choice(SMALL_AREAS)
    large_area = random.choice(LARGE_AREAS_BY_SMALL[small_area])
    small_dims = random.choice(SHAPES_BY_AREA[small_area])
    large_dims = random.choice(SHAPES_BY_AREA[large_area])
    return tuple(map(int, small_dims)), tuple(map(int, large_dims))


def _place_relation(
    forbidden: np.ndarray,
    dims_a: tuple[int, int],
    dims_b: tuple[int, int],
    orientation: str,
) -> dict | None:
    h, w = forbidden.shape
    usable_bottom = h - 3
    ha, wa = map(int, dims_a)
    hb, wb = map(int, dims_b)
    hra, hca = ha // 2, wa // 2
    hrb, hcb = hb // 2, wb // 2

    if usable_bottom < max(hra, hrb):
        return None

    for _ in range(PLACE_RETRIES):
        if orientation == "h":
            row_lo = max(hra, hrb)
            row_hi = usable_bottom - max(hra, hrb)
            if row_lo > row_hi:
                return None
            r1 = random.randint(row_lo, row_hi)
            c1 = random.randint(hca, w - 1 - hca)
            sign_c = random.choice((-1, 1))
            min_delta = hca + hcb + 3
            max_delta = (w - 1 - hcb) - c1 if sign_c > 0 else c1 - hcb
            if max_delta < min_delta:
                continue
            delta = random.randint(min_delta, max_delta)
            p1 = (r1, c1)
            p2 = (r1, c1 + sign_c * delta)
        elif orientation == "v":
            col_lo = max(hca, hcb)
            col_hi = w - 1 - max(hca, hcb)
            if col_lo > col_hi:
                return None
            c1 = random.randint(col_lo, col_hi)
            r1 = random.randint(hra, usable_bottom - hra)
            sign_r = random.choice((-1, 1))
            min_delta = hra + hrb + 3
            max_delta = (usable_bottom - hrb) - r1 if sign_r > 0 else r1 - hrb
            if max_delta < min_delta:
                continue
            delta = random.randint(min_delta, max_delta)
            p1 = (r1, c1)
            p2 = (r1 + sign_r * delta, c1)
        else:
            r1 = random.randint(hra, usable_bottom - hra)
            c1 = random.randint(hca, w - 1 - hca)
            sign_r = random.choice((-1, 1))
            sign_c = random.choice((-1, 1))
            min_delta = max(hra + hrb + 2, hca + hcb + 2, 3)
            max_delta_r = (usable_bottom - hrb) - r1 if sign_r > 0 else r1 - hrb
            max_delta_c = (w - 1 - hcb) - c1 if sign_c > 0 else c1 - hcb
            max_delta = min(max_delta_r, max_delta_c)
            if max_delta < min_delta:
                continue
            delta = random.randint(min_delta, max_delta)
            p1 = (r1, c1)
            p2 = (r1 + sign_r * delta, c1 + sign_c * delta)

        cells_a = _rect_cells(p1, dims_a)
        cells_b = _rect_cells(p2, dims_b)
        own = set(cells_a) | set(cells_b)
        if len(own) != len(cells_a) + len(cells_b):
            continue
        if any(forbidden[r, c] for r, c in own):
            continue

        path = _line_cells(p1, p2)
        path_only = [cell for cell in path if cell not in own]
        if any(forbidden[r, c] for r, c in path_only):
            continue

        return {
            "orientation": orientation,
            "center_a": p1,
            "center_b": p2,
            "cells_a": cells_a,
            "cells_b": cells_b,
            "path": path,
        }
    return None


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _build_expected(inp: np.ndarray, pair_records: list[dict]) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    for rec in pair_records:
        _draw_line(out, rec["small_a_center"], rec["small_b_center"], int(rec["a"]))
        _draw_line(out, rec["large_a_center"], rec["large_b_center"], int(rec["b"]))
    out[-1, :] = BG
    return out


def _valid_input_structure(grid: np.ndarray, marker_pairs: list[tuple[int, int]]) -> bool:
    if grid.ndim != 2 or grid.shape[0] < 2:
        return False
    if _parse_marker_pairs(grid[-1], BG) != marker_pairs:
        return False

    marker_colors = {int(v) for pair in marker_pairs for v in pair}
    upper = grid[:-1, :]
    upper_colors = {int(v) for v in np.unique(upper) if int(v) != BG}
    if upper_colors != marker_colors:
        return False

    for color in marker_colors:
        comps = _components_of_color(upper, color)
        if len(comps) != 2:
            return False
        areas = sorted(int(comp["area"]) for comp in comps)
        if areas[0] == areas[1]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(1, 3)
    marker_width = 3 * pair_count - 1

    h_min = max(14, 16 + 2 * (pair_count - 1))
    h_max = min(30, 22 + 3 * (pair_count - 1))
    w_min = max(15, marker_width + 4, 17 + 2 * (pair_count - 1))
    w_max = min(30, 24 + 2 * (pair_count - 1))
    if h_min > h_max or w_min > w_max:
        return None

    h = random.randint(h_min, h_max)
    w = random.randint(w_min, w_max)
    if marker_width > w:
        return None

    colors = random.sample(PALETTE, 2 * pair_count)
    forbidden = np.zeros((h, w), dtype=bool)
    forbidden[h - 2 :, :] = True
    pair_records: list[dict] = []
    objects_by_color: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)

    for pair_idx in range(pair_count):
        color_a = int(colors[2 * pair_idx])
        color_b = int(colors[2 * pair_idx + 1])
        dims_a_small, dims_a_large = _choose_dims_for_color()
        dims_b_small, dims_b_large = _choose_dims_for_color()

        local_forbidden = forbidden.copy()
        rel_small = _place_relation(local_forbidden, dims_a_small, dims_b_small, random.choice(("h", "v", "d")))
        if rel_small is None:
            return None
        small_footprint = set(rel_small["cells_a"]) | set(rel_small["cells_b"]) | set(rel_small["path"])
        _mark_forbidden(local_forbidden, small_footprint, pad=1)

        rel_large = _place_relation(local_forbidden, dims_a_large, dims_b_large, random.choice(("h", "v", "d")))
        if rel_large is None:
            return None
        large_footprint = set(rel_large["cells_a"]) | set(rel_large["cells_b"]) | set(rel_large["path"])
        _mark_forbidden(local_forbidden, large_footprint, pad=1)

        forbidden = local_forbidden
        objects_by_color[color_a].append(rel_small["cells_a"])
        objects_by_color[color_a].append(rel_large["cells_a"])
        objects_by_color[color_b].append(rel_small["cells_b"])
        objects_by_color[color_b].append(rel_large["cells_b"])
        pair_records.append(
            {
                "a": color_a,
                "b": color_b,
                "small_a_center": rel_small["center_a"],
                "small_b_center": rel_small["center_b"],
                "large_a_center": rel_large["center_a"],
                "large_b_center": rel_large["center_b"],
            }
        )

    random.shuffle(pair_records)
    marker_pairs = [(int(rec["a"]), int(rec["b"])) for rec in pair_records]

    inp = np.full((h, w), BG, dtype=int)
    for color, objects in objects_by_color.items():
        for cells in objects:
            _paint_cells(inp, cells, int(color))

    start = random.randint(0, w - marker_width)
    col = start
    for idx, (color_a, color_b) in enumerate(marker_pairs):
        inp[h - 1, col] = int(color_a)
        inp[h - 1, col + 1] = int(color_b)
        col += 2
        if idx != len(marker_pairs) - 1:
            col += 1

    if not _valid_input_structure(inp, marker_pairs):
        return None

    expected = _build_expected(inp, pair_records)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
