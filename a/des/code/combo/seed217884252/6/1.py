from __future__ import annotations

import random

import numpy as np


QUADRANTS = (0, 1, 2, 3)  # TL, TR, BR, BL
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
BG = 0
DIVIDER = 5
GEN_RETRIES = 300
SHAPE_RETRIES = 80


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_dividers(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    h, w = grid.shape

    row_candidates: list[tuple[int, int, int, int]] = []
    for r in range(h):
        vals, counts = np.unique(grid[r], return_counts=True)
        if len(vals) != 2 or int(counts.max()) != w - 1 or int(counts.min()) != 1:
            continue
        div = int(vals[np.argmax(counts)])
        mark = int(vals[np.argmin(counts)])
        if div == bg or mark in {bg, div}:
            continue
        marker_cols = np.where(grid[r] != div)[0]
        row_candidates.append((r, int(marker_cols[0]), div, mark))

    col_candidates: list[tuple[int, int, int, int]] = []
    for c in range(w):
        vals, counts = np.unique(grid[:, c], return_counts=True)
        if len(vals) != 2 or int(counts.max()) != h - 1 or int(counts.min()) != 1:
            continue
        div = int(vals[np.argmax(counts)])
        mark = int(vals[np.argmin(counts)])
        if div == bg or mark in {bg, div}:
            continue
        marker_rows = np.where(grid[:, c] != div)[0]
        col_candidates.append((int(marker_rows[0]), c, div, mark))

    matches = [
        (r, c, div, mark)
        for r, c, div, mark in row_candidates
        for rr, cc, cdiv, cmark in col_candidates
        if r == rr and c == cc and div == cdiv and mark == cmark
    ]
    assert len(matches) == 1, "expected one divider row/col pair"
    return matches[0]


def _quadrant_bounds(shape: tuple[int, int], mid_r: int, mid_c: int) -> dict[int, tuple[int, int, int, int]]:
    h, w = shape
    return {
        0: (0, mid_r, 0, mid_c),
        1: (0, mid_r, mid_c + 1, w),
        2: (mid_r + 1, h, mid_c + 1, w),
        3: (mid_r + 1, h, 0, mid_c),
    }


def _quadrant_of(r: int, c: int, mid_r: int, mid_c: int) -> int:
    assert r != mid_r and c != mid_c
    if r < mid_r and c < mid_c:
        return 0
    if r < mid_r and c > mid_c:
        return 1
    if r > mid_r and c > mid_c:
        return 2
    return 3


def _extract_objects(
    grid: np.ndarray, bg: int, divider: int, mid_r: int, mid_c: int
) -> dict[int, dict[str, object] | None]:
    objects: dict[int, dict[str, object] | None] = {}
    for q, (r0, r1, c0, c1) in _quadrant_bounds(grid.shape, mid_r, mid_c).items():
        sub = grid[r0:r1, c0:c1]
        mask = (sub != bg) & (sub != divider)
        if not np.any(mask):
            objects[q] = None
            continue
        colors = np.unique(sub[mask])
        assert len(colors) == 1, "each non-empty quadrant must contain one monochrome object"
        coords = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}
        objects[q] = {"color": int(colors[0]), "coords": coords}
    return objects


def _move_coords(coords: set[tuple[int, int]], source_q: int, mid_r: int, mid_c: int) -> set[tuple[int, int]]:
    moved = set()
    if source_q in {0, 2}:
        for r, c in coords:
            moved.add((r, 2 * mid_c - c))
    else:
        for r, c in coords:
            moved.add((2 * mid_r - r, c))
    return moved


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return True
    return False


def _render_scene(
    shape: tuple[int, int],
    mid_r: int,
    mid_c: int,
    divider: int,
    marker: int,
    objects: dict[int, dict[str, object] | None],
    anchor_q: int,
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[mid_r, :] = divider
    out[:, mid_c] = divider
    out[mid_r, mid_c] = marker

    placed_by_quadrant: dict[int, list[set[tuple[int, int]]]] = {q: [] for q in QUADRANTS}

    for q in QUADRANTS:
        obj = objects[q]
        if obj is None:
            continue
        color = int(obj["color"])
        coords = set(obj["coords"])
        if q == anchor_q:
            target_q = q
            moved = coords
        else:
            target_q = (q + 1) % 4
            moved = _move_coords(coords, q, mid_r, mid_c)

        for r, c in moved:
            assert 0 <= r < shape[0] and 0 <= c < shape[1], "moved object left the grid"
            assert r != mid_r and c != mid_c, "moved object cannot land on a divider"
            assert _quadrant_of(r, c, mid_r, mid_c) == target_q, "object landed in the wrong quadrant"

        for other in placed_by_quadrant[target_q]:
            assert not _touches(moved, other), "output objects in one quadrant must stay visually separate"

        for r, c in moved:
            assert out[r, c] == bg, "output objects cannot overlap"
            out[r, c] = color
        placed_by_quadrant[target_q].append(moved)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the marker color at the divider crossing to choose the anchored object.

    Objects in the other non-empty quadrants move one quadrant clockwise. A move that crosses the
    vertical divider mirrors left-right; a move that crosses the horizontal divider mirrors up-down.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _background_color(g)
    mid_r, mid_c, divider, marker = _find_dividers(g, bg)
    objects = _extract_objects(g, bg, divider, mid_r, mid_c)

    anchor_candidates = [q for q, obj in objects.items() if obj is not None and int(obj["color"]) == marker]
    assert len(anchor_candidates) == 1, "marker color must identify exactly one anchor object"
    anchor_q = anchor_candidates[0]

    return _render_scene(g.shape, mid_r, mid_c, divider, marker, objects, anchor_q, bg)


def _normalize(coords: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    return {(r - r0, c - c0) for r, c in coords}, h, w


def _is_hflip_symmetric(coords: set[tuple[int, int]]) -> bool:
    norm, _h, w = _normalize(coords)
    return {(r, w - 1 - c) for r, c in norm} == norm


def _is_vflip_symmetric(coords: set[tuple[int, int]]) -> bool:
    norm, h, _w = _normalize(coords)
    return {(h - 1 - r, c) for r, c in norm} == norm


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _random_patch(max_h: int, max_w: int, need_axis: str | None) -> set[tuple[int, int]] | None:
    h_hi = min(max_h, 4)
    w_hi = min(max_w, 4)
    if h_hi < 2 or w_hi < 2:
        return None

    for _ in range(SHAPE_RETRIES):
        h = random.randint(2, h_hi)
        w = random.randint(2, w_hi)
        area_hi = min(h * w - 1, 7)
        area_lo = min(area_hi, 3)
        if area_lo > area_hi:
            continue
        area = random.randint(area_lo, area_hi)

        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        frontier = set(_neighbors4(seed[0], seed[1], h, w))
        while len(cells) < area:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(cell[0], cell[1], h, w):
                if nb not in cells:
                    frontier.add(nb)
        if len(cells) != area:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if rows != set(range(h)) or cols != set(range(w)):
            continue
        if need_axis == "h" and _is_hflip_symmetric(cells):
            continue
        if need_axis == "v" and _is_vflip_symmetric(cells):
            continue
        return cells

    return None


def _place_in_quadrant(local_coords: set[tuple[int, int]], q: int, qh: int, qw: int) -> set[tuple[int, int]]:
    rows = [r for r, _ in local_coords]
    cols = [c for _, c in local_coords]
    ph = max(rows) + 1
    pw = max(cols) + 1
    off_r = random.randint(0, qh - ph)
    off_c = random.randint(0, qw - pw)

    if q == 0:
        base_r, base_c = 0, 0
    elif q == 1:
        base_r, base_c = 0, qw + 1
    elif q == 2:
        base_r, base_c = qh + 1, qw + 1
    else:
        base_r, base_c = qh + 1, 0

    return {(base_r + off_r + r, base_c + off_c + c) for r, c in local_coords}


def _build_input(
    qh: int,
    qw: int,
    objects: dict[int, dict[str, object] | None],
    anchor_q: int,
) -> np.ndarray:
    h, w = 2 * qh + 1, 2 * qw + 1
    grid = np.full((h, w), BG, dtype=int)
    grid[qh, :] = DIVIDER
    grid[:, qw] = DIVIDER
    grid[qh, qw] = int(objects[anchor_q]["color"])
    for obj in objects.values():
        if obj is None:
            continue
        color = int(obj["color"])
        for r, c in obj["coords"]:
            grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(4, 8)
    qw = random.randint(4, 8)
    occupied_count = random.choice((2, 3, 4))
    occupied = random.sample(QUADRANTS, occupied_count)
    anchor_q = random.choice(occupied)

    color_map = {
        q: color for q, color in zip(occupied, random.sample(PALETTE, occupied_count), strict=False)
    }
    objects: dict[int, dict[str, object] | None] = {q: None for q in QUADRANTS}

    for q in occupied:
        need_axis = None
        if q != anchor_q:
            need_axis = "h" if q in {0, 2} else "v"
        patch = _random_patch(qh, qw, need_axis)
        if patch is None:
            return None
        placed = _place_in_quadrant(patch, q, qh, qw)
        objects[q] = {"color": color_map[q], "coords": placed}

    inp = _build_input(qh, qw, objects, anchor_q)
    try:
        expected = _render_scene(inp.shape, qh, qw, DIVIDER, int(objects[anchor_q]["color"]), objects, anchor_q, BG)
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
