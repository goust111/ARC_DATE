import random
from collections import defaultdict, deque

import numpy as np


BG = 8
YELLOW = 4
ORANGE = 7
RED = 2

OBJECT_COLORS = (0, 1, 3, 5, 6, 9)
GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)
GAP_VALUES = tuple(range(1, 9))
MAX_GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 80


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(pts, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _row_map(coords: np.ndarray) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = defaultdict(list)
    for r, c in coords:
        rows[int(r)].append(int(c))
    for cols in rows.values():
        cols.sort()
    return dict(rows)


def _col_map(coords: np.ndarray) -> dict[int, list[int]]:
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in coords:
        cols[int(c)].append(int(r))
    for rows in cols.values():
        rows.sort()
    return dict(cols)


def _horizontal_segment(grid: np.ndarray, a: np.ndarray, b: np.ndarray, bg: int) -> list[tuple[int, int]] | None:
    if _bbox(a)[3] < _bbox(b)[1]:
        left, right = a, b
    elif _bbox(b)[3] < _bbox(a)[1]:
        left, right = b, a
    else:
        return None

    rows_left = _row_map(left)
    rows_right = _row_map(right)
    segments: list[tuple[int, int, int]] = []
    for row in sorted(set(rows_left) & set(rows_right)):
        left_edge = max(rows_left[row])
        right_edge = min(rows_right[row])
        if right_edge - left_edge <= 1:
            continue
        if np.all(grid[row, left_edge + 1 : right_edge] == bg):
            segments.append((row, left_edge + 1, right_edge - 1))
    if not segments:
        return None

    row, c0, c1 = segments[len(segments) // 2]
    return [(row, c) for c in range(c0, c1 + 1)]


def _vertical_segment(grid: np.ndarray, a: np.ndarray, b: np.ndarray, bg: int) -> list[tuple[int, int]] | None:
    if _bbox(a)[2] < _bbox(b)[0]:
        top, bottom = a, b
    elif _bbox(b)[2] < _bbox(a)[0]:
        top, bottom = b, a
    else:
        return None

    cols_top = _col_map(top)
    cols_bottom = _col_map(bottom)
    segments: list[tuple[int, int, int]] = []
    for col in sorted(set(cols_top) & set(cols_bottom)):
        top_edge = max(cols_top[col])
        bottom_edge = min(cols_bottom[col])
        if bottom_edge - top_edge <= 1:
            continue
        if np.all(grid[top_edge + 1 : bottom_edge, col] == bg):
            segments.append((col, top_edge + 1, bottom_edge - 1))
    if not segments:
        return None

    col, r0, r1 = segments[len(segments) // 2]
    return [(r, col) for r in range(r0, r1 + 1)]


def _middle_segment(grid: np.ndarray, comp_a: np.ndarray, comp_b: np.ndarray, bg: int) -> list[tuple[int, int]]:
    horizontal = _horizontal_segment(grid, comp_a, comp_b, bg)
    vertical = _vertical_segment(grid, comp_a, comp_b, bg)
    assert (horizontal is None) != (vertical is None)
    segment = horizontal if horizontal is not None else vertical
    assert segment is not None and segment
    return segment


def _paint_segment(out: np.ndarray, segment: list[tuple[int, int]]) -> None:
    n = len(segment)
    half = n // 2
    for idx, (r, c) in enumerate(segment):
        if n % 2 == 1 and idx == half:
            out[r, c] = RED
        elif idx < half:
            out[r, c] = YELLOW
        else:
            out[r, c] = ORANGE


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    out = g.copy()
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        comps = _components_of_mask(g == color)
        assert len(comps) == 2
        segment = _middle_segment(g, comps[0], comps[1], bg)
        _paint_segment(out, segment)
    return out


def _shape_info(coords: list[tuple[int, int]]) -> dict[str, object]:
    pts = sorted(set(coords))
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for dr, dc in pts:
        by_row[dr].append(dc)
        by_col[dc].append(dr)
    for dcs in by_row.values():
        dcs.sort()
    for drs in by_col.values():
        drs.sort()
    assert 0 in by_row and 0 in by_col
    return {
        "coords": tuple(pts),
        "by_row": {dr: tuple(dcs) for dr, dcs in by_row.items()},
        "by_col": {dc: tuple(drs) for dc, drs in by_col.items()},
        "rmin": min(dr for dr, _ in pts),
        "rmax": max(dr for dr, _ in pts),
        "cmin": min(dc for _, dc in pts),
        "cmax": max(dc for _, dc in pts),
        "row_left": min(by_row[0]),
        "row_right": max(by_row[0]),
        "col_top": min(by_col[0]),
        "col_bottom": max(by_col[0]),
    }


def _sample_shape(rng: random.Random) -> dict[str, object]:
    family = rng.choice(("rect", "plus", "diamond", "ring"))
    if family == "rect":
        h = rng.choice((1, 3, 5))
        w = rng.choice((1, 3, 5))
        coords = [(dr, dc) for dr in range(-(h // 2), h // 2 + 1) for dc in range(-(w // 2), w // 2 + 1)]
    elif family == "plus":
        arm = rng.choice((1, 2))
        coords = [(0, dc) for dc in range(-arm, arm + 1)] + [(dr, 0) for dr in range(-arm, arm + 1)]
    elif family == "diamond":
        radius = rng.choice((1, 2))
        coords = [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) <= radius]
    else:
        h = rng.choice((3, 5))
        w = rng.choice((3, 5))
        coords = []
        for dr in range(-(h // 2), h // 2 + 1):
            for dc in range(-(w // 2), w // 2 + 1):
                if abs(dr) == h // 2 or abs(dc) == w // 2:
                    coords.append((dr, dc))
    return _shape_info(coords)


def _translate(coords: tuple[tuple[int, int], ...], center_r: int, center_c: int) -> set[tuple[int, int]]:
    return {(center_r + dr, center_c + dc) for dr, dc in coords}


def _horizontal_corridor(shape: dict[str, object], row: int, left_c: int, right_c: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for dr, dcs in shape["by_row"].items():
        left_edge = left_c + max(dcs)
        right_edge = right_c + min(dcs)
        for c in range(left_edge + 1, right_edge):
            cells.add((row + dr, c))
    return cells


def _vertical_corridor(shape: dict[str, object], top_r: int, bottom_r: int, col: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for dc, drs in shape["by_col"].items():
        top_edge = top_r + max(drs)
        bottom_edge = bottom_r + min(drs)
        for r in range(top_edge + 1, bottom_edge):
            cells.add((r, col + dc))
    return cells


def _expanded(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _placement_candidates(
    shape: dict[str, object],
    orientation: str,
    gap: int,
    forbidden: np.ndarray,
) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]]:
    h, w = forbidden.shape
    coords = shape["coords"]
    candidates: list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]] = []

    if orientation == "H":
        delta = gap + 1 + int(shape["row_right"]) - int(shape["row_left"])
        row_min = -int(shape["rmin"])
        row_max = h - 1 - int(shape["rmax"])
        left_min = -int(shape["cmin"])
        left_max = w - 1 - int(shape["cmax"]) - delta
        for row in range(row_min, row_max + 1):
            for left_c in range(left_min, left_max + 1):
                right_c = left_c + delta
                obj_a = _translate(coords, row, left_c)
                obj_b = _translate(coords, row, right_c)
                corridor = _horizontal_corridor(shape, row, left_c, right_c)
                reserve = obj_a | obj_b | corridor
                if not corridor:
                    continue
                padded = _expanded(reserve, h, w)
                if all(not forbidden[r, c] for r, c in padded):
                    candidates.append((obj_a, obj_b, padded))
    else:
        delta = gap + 1 + int(shape["col_bottom"]) - int(shape["col_top"])
        col_min = -int(shape["cmin"])
        col_max = w - 1 - int(shape["cmax"])
        top_min = -int(shape["rmin"])
        top_max = h - 1 - int(shape["rmax"]) - delta
        for col in range(col_min, col_max + 1):
            for top_r in range(top_min, top_max + 1):
                bottom_r = top_r + delta
                obj_a = _translate(coords, top_r, col)
                obj_b = _translate(coords, bottom_r, col)
                corridor = _vertical_corridor(shape, top_r, bottom_r, col)
                reserve = obj_a | obj_b | corridor
                if not corridor:
                    continue
                padded = _expanded(reserve, h, w)
                if all(not forbidden[r, c] for r, c in padded):
                    candidates.append((obj_a, obj_b, padded))
    return candidates


def _paint_object(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _max_pairs_for_area(area: int) -> int:
    if area < 140:
        return 1
    if area < 240:
        return 2
    return 3


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    area = h * w
    max_pairs = _max_pairs_for_area(area)
    pair_count = rng.randint(1, max_pairs)
    colors = list(OBJECT_COLORS)
    rng.shuffle(colors)

    grid = np.full((h, w), BG, dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    for pair_idx in range(pair_count):
        color = colors[pair_idx]
        placed = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            orientation = rng.choice(("H", "V"))
            parity = rng.choice((0, 1))
            gap_choices = [gap for gap in GAP_VALUES if gap % 2 == parity]
            gap = rng.choice(gap_choices)
            shape = _sample_shape(rng)
            candidates = _placement_candidates(shape, orientation, gap, forbidden)
            if not candidates:
                continue
            obj_a, obj_b, padded = rng.choice(candidates)
            _paint_object(grid, obj_a, color)
            _paint_object(grid, obj_b, color)
            for r, c in padded:
                forbidden[r, c] = True
            placed = True
            break
        if not placed:
            return None

    if _most_common_color(grid) != BG:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 4")
