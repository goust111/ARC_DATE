from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
MARKER = 1
OBSTACLE = 2
WALL = 4
GEN_RETRIES = 500

Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _adjacent_to_color(grid: np.ndarray, pos: Coord, color: int) -> bool:
    return any(
        _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == color
        for nr, nc in _neighbors4(*pos)
    )


def _find_unique_marker(grid: np.ndarray) -> Coord:
    coords = np.argwhere(grid == MARKER)
    assert coords.shape == (1, 2), "expected exactly one blue marker"
    return tuple(map(int, coords[0]))


def _find_sightline_to_obstacle(grid: np.ndarray, start: Coord) -> tuple[Coord, Coord]:
    hits: list[tuple[Coord, Coord]] = []
    sr, sc = start
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = sr + dr, sc + dc
        previous_bg: Coord | None = None
        while _in_bounds(grid.shape, r, c):
            value = int(grid[r, c])
            if value == OBSTACLE:
                assert previous_bg is not None, "marker must not touch the obstacle"
                hits.append(((dr, dc), previous_bg))
                break
            assert value == BG, "sightline must cross only background before the obstacle"
            previous_bg = (r, c)
            r += dr
            c += dc
    assert len(hits) == 1, "expected exactly one obstacle sightline"
    return hits[0]


def _walk_contour(grid: np.ndarray, start: Coord, direction: Coord) -> tuple[Coord, int]:
    dr, dc = direction
    if dr:
        turns = [(dr, -1), (dr, 1)]
    else:
        turns = [(-1, dc), (1, dc)]

    options: list[tuple[int, Coord]] = []
    for tr, tc in turns:
        r, c = start
        length = 0
        while True:
            nr, nc = r + tr, c + tc
            if not _in_bounds(grid.shape, nr, nc):
                break
            if int(grid[nr, nc]) != BG:
                break
            if not _adjacent_to_color(grid, (nr, nc), OBSTACLE):
                break
            r, c = nr, nc
            length += 1
        options.append((length, (r, c)))

    best_len = max(length for length, _ in options)
    endpoints = {coord for length, coord in options if length == best_len}
    assert len(endpoints) == 1, "contour walk must have a unique longest branch"
    return endpoints.pop(), best_len


def _slide_destination(grid: np.ndarray) -> tuple[Coord, Coord, Coord, int]:
    marker_pos = _find_unique_marker(grid)
    direction, contour_start = _find_sightline_to_obstacle(grid, marker_pos)
    assert _adjacent_to_color(grid, contour_start, OBSTACLE), "contour start must touch obstacle"
    dest, path_len = _walk_contour(grid, contour_start, direction)
    assert path_len > 0, "marker must actually slide"
    return marker_pos, contour_start, dest, path_len


def _row_fill_interval(wall_cols: list[int], ref_col: int, width: int) -> tuple[int, int]:
    left = [c for c in wall_cols if c < ref_col]
    right = [c for c in wall_cols if c >= ref_col]
    assert left or right, "wall row must constrain the reference column"
    if left and right:
        start = max(left) + 1
        stop = min(right) - 1
    elif left:
        start = max(left) + 1
        stop = width - 1
    else:
        start = 0
        stop = min(right) - 1
    assert 0 <= start <= stop < width, "wall row must define a non-empty span"
    return start, stop


def _wall_intervals(grid: np.ndarray, ref_col: int) -> tuple[dict[int, tuple[int, int]], int, int]:
    h, w = grid.shape
    intervals: dict[int, tuple[int, int]] = {}
    for r in range(h):
        wall_cols = [int(c) for c in np.flatnonzero(grid[r] == WALL)]
        if wall_cols:
            assert len(wall_cols) <= 2, "each row may contain at most two wall pixels"
            intervals[r] = _row_fill_interval(wall_cols, ref_col, w)
    assert intervals, "expected wall rows"
    top = min(intervals)
    bottom = max(intervals)
    return intervals, top, bottom


def _filled_with_reference(grid: np.ndarray, ref_col: int, erase_marker: bool = True) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    marker_pos = _find_unique_marker(out)
    if erase_marker:
        out[marker_pos] = BG

    intervals, top, bottom = _wall_intervals(out, ref_col)
    h, w = out.shape
    for r in range(h):
        if r in intervals:
            start, stop = intervals[r]
        elif top < r < bottom:
            start, stop = 0, w - 1
        elif r < top:
            start, stop = intervals[top]
        else:
            start, stop = intervals[bottom]
        segment = out[r, start : stop + 1]
        segment[segment == BG] = MARKER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the marker along the obstacle contour and use its endpoint column as the scanline fill reference.

    Input structure:
    - A black background contains one blue singleton marker, one red 4-connected
      irregular obstacle, and several yellow wall pixels spread across multiple rows.
    - The blue marker has exactly one horizontal or vertical sightline to the red
      obstacle through background cells and is not 4-adjacent to the obstacle.
    - Wall rows contain one or two yellow pixels, while some rows inside the wall
      band are empty and some rows outside the band have no walls at all.

    Rule:
    - Follow the unique marker-to-obstacle sightline and take the last background
      cell before the obstacle as the contour start.
    - From that start, try the two diagonal contour-following turns and move the
      marker to the endpoint of the uniquely longest 4-adjacent background path.
    - Erase the original marker, then use the endpoint column as the scanline
      reference: on wall rows fill between the nearest left wall and the nearest
      wall at or right of the reference, with one-sided rows filling to the edge;
      empty rows inside the wall band fill across the whole row; empty rows outside
      the band reuse the nearest endpoint wall-row interval.

    Key insight:
    - The wall fill does not use the marker's initial column: the obstacle shape
      first determines the contour-slide endpoint, and only that endpoint column
      determines every later fill span.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert _get_bg(g) == BG, "background must be black"
    assert set(map(int, np.unique(g))).issubset({BG, MARKER, OBSTACLE, WALL}), "unexpected colors present"

    obstacle_components = _components(g == OBSTACLE)
    assert len(obstacle_components) == 1, "expected one connected red obstacle"

    marker_pos, _contour_start, dest, _path_len = _slide_destination(g)
    out = _filled_with_reference(g, dest[1], erase_marker=True)
    out[dest] = MARKER
    return out


def _gen_boundary(h: int, start_row: int, walk_len: int, start_left: int) -> list[int]:
    lefts = [0] * h
    lefts[start_row] = start_left

    for r in range(start_row - 1, -1, -1):
        lefts[r] = max(0, lefts[r + 1] - random.randint(0, 1))

    for step in range(1, walk_len + 1):
        lefts[start_row + step] = start_left + step

    for r in range(start_row + walk_len + 1, h):
        lefts[r] = lefts[r - 1] + random.randint(0, 1)

    return lefts


def _gen_slide_patch() -> np.ndarray | None:
    h = random.randint(5, 7)
    walk_len = random.randint(2, min(4, h - 2))
    max_start_row = h - walk_len - 1
    if max_start_row < 1:
        return None
    start_row = random.randint(1, max_start_row)
    gap = random.randint(2, 4)
    start_left = random.randint(1, min(2, gap))
    lefts = _gen_boundary(h, start_row, walk_len, start_left)
    max_left = max(lefts)

    width_choices = [w for w in range(max_left + 2, max_left + 5)]
    random.shuffle(width_choices)
    for obstacle_w in width_choices:
        area = sum(obstacle_w - left for left in lefts)
        if not (8 <= area <= 15):
            continue
        if area == h * (obstacle_w - min(lefts)):
            continue
        c0 = gap - start_left + 1
        patch = np.full((h, c0 + obstacle_w), BG, dtype=int)
        for r, left in enumerate(lefts):
            patch[r, c0 + left : c0 + obstacle_w] = OBSTACLE
        patch[start_row, 0] = MARKER
        if len(_components(patch == OBSTACLE)) != 1:
            continue
        bbox_area = (np.max(np.argwhere(patch == OBSTACLE)[:, 0]) - np.min(np.argwhere(patch == OBSTACLE)[:, 0]) + 1) * (
            np.max(np.argwhere(patch == OBSTACLE)[:, 1]) - np.min(np.argwhere(patch == OBSTACLE)[:, 1]) + 1
        )
        if int(np.count_nonzero(patch == OBSTACLE)) == bbox_area:
            continue
        patch = np.rot90(patch, random.randrange(4))
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        try:
            marker_pos, _contour_start, dest, path_len = _slide_destination(patch)
        except AssertionError:
            continue
        if path_len < 2 or marker_pos[1] == dest[1]:
            continue
        return patch
    return None


def _center_empty_rows(top: int, band_len: int, empty_count: int) -> set[int]:
    bottom = top + band_len - 1
    if empty_count == 1:
        if band_len % 2 == 1:
            return {(top + bottom) // 2}
        return {random.choice([(top + bottom) // 2, (top + bottom) // 2 + 1])}
    mid = (top + bottom) // 2
    if band_len % 2 == 1:
        return set(random.choice([(mid - 1, mid), (mid, mid + 1)]))
    return {mid, mid + 1}


def _safe_cols(grid: np.ndarray, row: int) -> set[int]:
    return {int(c) for c in range(grid.shape[1]) if int(grid[row, c]) == BG}


def _pick_wall_pair(grid: np.ndarray, row: int, ref_col: int) -> tuple[int, int] | None:
    safe = _safe_cols(grid, row)
    lefts = [c for c in safe if c < ref_col]
    rights = [c for c in safe if c >= ref_col]
    pairs = [(l, r) for l in lefts for r in rights if r - l >= 2]
    if not pairs:
        return None
    return random.choice(pairs)


def _pick_one_sided_wall(grid: np.ndarray, row: int, ref_col: int, side: str) -> int | None:
    safe = _safe_cols(grid, row)
    if side == "left":
        candidates = [c for c in safe if c < ref_col and c < grid.shape[1] - 1]
    else:
        candidates = [c for c in safe if c >= ref_col and c > 0]
    if not candidates:
        return None
    return random.choice(candidates)


def _fill_row_signature(width: int, wall_cols: list[int], ref_col: int) -> tuple[int, int]:
    return _row_fill_interval(wall_cols, ref_col, width)


def _build_wall_rows(
    grid: np.ndarray,
    marker_col: int,
    ref_col: int,
) -> tuple[dict[int, list[int]], int, int] | None:
    h, w = grid.shape
    band_len = random.randint(5, 7)
    if h < band_len + 4:
        return None
    top = random.randint(2, h - band_len - 2)
    bottom = top + band_len - 1
    empty_count = random.randint(1, min(2, band_len - 3))
    empty_rows = _center_empty_rows(top, band_len, empty_count)
    wall_rows = [r for r in range(top, bottom + 1) if r not in empty_rows]
    if len(wall_rows) < 3:
        return None

    kinds = {r: "both" for r in wall_rows}
    if random.random() < 0.8:
        one_sided_count = 1 if len(wall_rows) <= 4 or random.random() < 0.6 else 2
        for r in random.sample(wall_rows, one_sided_count):
            kinds[r] = random.choice(["left", "right"])
    if all(kind != "both" for kind in kinds.values()):
        kinds[random.choice(wall_rows)] = "both"

    rows: dict[int, list[int]] = {}
    both_rows = 0
    for r in wall_rows:
        kind = kinds[r]
        if kind == "both":
            pair = _pick_wall_pair(grid, r, ref_col)
            if pair is None:
                return None
            rows[r] = [pair[0], pair[1]]
            both_rows += 1
        else:
            col = _pick_one_sided_wall(grid, r, ref_col, kind)
            if col is None:
                return None
            rows[r] = [col]

    if both_rows == 0:
        return None

    top_interval = _fill_row_signature(w, rows[min(rows)], ref_col)
    bottom_interval = _fill_row_signature(w, rows[max(rows)], ref_col)
    if top_interval == (0, w - 1) and bottom_interval == (0, w - 1):
        return None

    differs_from_start = False
    for cols in rows.values():
        ref_sig = _fill_row_signature(w, cols, ref_col)
        try:
            marker_sig = _fill_row_signature(w, cols, marker_col)
        except AssertionError:
            differs_from_start = True
        else:
            if ref_sig != marker_sig:
                differs_from_start = True
        if differs_from_start:
            break
    if not differs_from_start:
        return None

    return rows, top, bottom


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        assert grid.ndim == 2 and max(grid.shape) <= 30
        assert _get_bg(grid) == BG
        marker_pos = _find_unique_marker(grid)
        obstacle_comps = _components(grid == OBSTACLE)
        assert len(obstacle_comps) == 1
        obstacle_area = int(obstacle_comps[0].shape[0])
        assert 8 <= obstacle_area <= 15
        direction, contour_start = _find_sightline_to_obstacle(grid, marker_pos)
        assert _adjacent_to_color(grid, contour_start, OBSTACLE)
        dest, path_len = _walk_contour(grid, contour_start, direction)
        assert path_len >= 2
        assert marker_pos[1] != dest[1]

        wall_rows = {r: [int(c) for c in np.flatnonzero(grid[r] == WALL)] for r in range(grid.shape[0]) if np.any(grid[r] == WALL)}
        assert wall_rows
        assert all(1 <= len(cols) <= 2 for cols in wall_rows.values())
        intervals, top, bottom = _wall_intervals(grid, dest[1])
        assert top > 0 and bottom < grid.shape[0] - 1
        assert any(r not in intervals for r in range(top + 1, bottom))
        assert any(len(cols) == 2 for cols in wall_rows.values())

        expected = main(grid)
        wrong_ref = _filled_with_reference(grid, marker_pos[1], erase_marker=True)
        wrong_ref[dest] = MARKER
        assert not np.array_equal(expected, wrong_ref)

        slide_only = grid.copy()
        slide_only[marker_pos] = BG
        slide_only[dest] = MARKER
        assert not np.array_equal(expected, slide_only)

        extra_blue = np.count_nonzero(expected == MARKER) - 1
        assert extra_blue > 0
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    patch = _gen_slide_patch()
    if patch is None:
        return None

    marker_pos, _contour_start, dest, _path_len = _slide_destination(patch)
    ph, pw = patch.shape

    for _ in range(60):
        h = random.randint(max(15, ph + 6), min(20, ph + 11))
        w = random.randint(max(15, pw + 6), min(20, pw + 11))
        if h > 30 or w > 30:
            continue

        c_lo = max(0, 2 - dest[1])
        c_hi = min(w - pw, w - 3 - dest[1])
        if c_lo > c_hi:
            continue
        r_lo, r_hi = 0, h - ph
        if r_lo > r_hi:
            continue

        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)

        grid = np.full((h, w), BG, dtype=int)
        for rr in range(ph):
            for cc in range(pw):
                if int(patch[rr, cc]) != BG:
                    grid[r0 + rr, c0 + cc] = int(patch[rr, cc])

        marker_final = (marker_pos[0] + r0, marker_pos[1] + c0)
        dest_final = (dest[0] + r0, dest[1] + c0)
        wall_info = _build_wall_rows(grid, marker_final[1], dest_final[1])
        if wall_info is None:
            continue
        wall_rows, _top, _bottom = wall_info
        for r, cols in wall_rows.items():
            for c in cols:
                if int(grid[r, c]) != BG:
                    break
                grid[r, c] = WALL
            else:
                continue
            break
        else:
            if _validate_instance(grid):
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
