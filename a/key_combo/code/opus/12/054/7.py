from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME = 5
BLOCKER = 9
SIDE_COLORS = (1, 2, 3, 4)
DIAG_COLORS = (1, 2, 3, 4, 6, 7)
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _diamond_info(grid: np.ndarray) -> tuple[tuple[int, int], int, tuple[int, int, int, int]]:
    coords = np.argwhere(grid == FRAME)
    assert coords.size > 0, "expected a grey diamond frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 == c1 - c0, "diamond bbox must be square"
    side = int(r1 - r0 + 1)
    assert side % 2 == 1 and side >= 9, "diamond side must be odd and at least 9"

    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    radius = side // 2

    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if abs(r - center[0]) + abs(c - center[1]) == radius
    }
    observed = {tuple(map(int, rc)) for rc in coords}
    assert observed == expected, "frame must be a complete one-pixel diamond"
    return center, radius, (r0, c0, r1, c1)


def _inside(center: tuple[int, int], radius: int, r: int, c: int) -> bool:
    return abs(int(r) - center[0]) + abs(int(c) - center[1]) < radius


def _read_diagonal_palette(
    grid: np.ndarray, top_vertex: tuple[int, int]
) -> tuple[list[int], set[tuple[int, int]]]:
    r, c = top_vertex[0] - 1, top_vertex[1] - 1
    assert _in_bounds(grid.shape, r, c), "expected diagonal marker beside top vertex"
    colors_rev: list[int] = []
    coords: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, r, c) and int(grid[r, c]) in DIAG_COLORS:
        colors_rev.append(int(grid[r, c]))
        coords.append((int(r), int(c)))
        r -= 1
        c -= 1
    colors = list(reversed(colors_rev))
    diag_coords = set(coords)
    assert 2 <= len(colors) <= 4, "diagonal marker count must be 2..4"
    assert len(set(colors)) == len(colors), "diagonal marker colors must be distinct"
    return colors, diag_coords


def _ordered_side_markers(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    diag_coords: set[tuple[int, int]],
) -> list[tuple[int, int, int, int, int]]:
    r0, c0, r1, c1 = bbox
    markers: list[tuple[int, int, int, int, int]] = []

    top_row = r0 - 1
    if top_row >= 0:
        top = [
            (top_row, c, int(grid[top_row, c]), 1, 0)
            for c in range(c0, c1 + 1)
            if (top_row, c) not in diag_coords and int(grid[top_row, c]) in SIDE_COLORS
        ]
        markers.extend(top)

    right_col = c1 + 1
    if right_col < grid.shape[1]:
        right = [
            (r, right_col, int(grid[r, right_col]), 0, -1)
            for r in range(r0, r1 + 1)
            if int(grid[r, right_col]) in SIDE_COLORS
        ]
        markers.extend(right)

    bottom_row = r1 + 1
    if bottom_row < grid.shape[0]:
        bottom = [
            (bottom_row, c, int(grid[bottom_row, c]), -1, 0)
            for c in range(c1, c0 - 1, -1)
            if int(grid[bottom_row, c]) in SIDE_COLORS
        ]
        markers.extend(bottom)

    left_col = c0 - 1
    if left_col >= 0:
        left = [
            (r, left_col, int(grid[r, left_col]), 0, 1)
            for r in range(r1, r0 - 1, -1)
            if int(grid[r, left_col]) in SIDE_COLORS
        ]
        markers.extend(left)

    return markers


def _band_fill(
    grid: np.ndarray, center: tuple[int, int], radius: int, palette: list[int]
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            dist = abs(r - center[0]) + abs(c - center[1])
            if dist >= radius or int(out[r, c]) == BLOCKER:
                continue
            layer = radius - dist
            color = int(palette[min(layer - 1, len(palette) - 1)])
            out[r, c] = color
    return out


def _ray_cells(
    grid: np.ndarray,
    center: tuple[int, int],
    radius: int,
    marker: tuple[int, int, int, int, int],
) -> tuple[list[tuple[int, int]], bool]:
    r, c, _color, dr, dc = marker
    rr, cc = r + dr, c + dc
    cells: list[tuple[int, int]] = []
    entered = False
    blocked = False

    while _in_bounds(grid.shape, rr, cc):
        if _inside(center, radius, rr, cc):
            entered = True
            if int(grid[rr, cc]) == BLOCKER:
                blocked = True
                break
            cells.append((int(rr), int(cc)))
        elif entered:
            break
        rr += dr
        cc += dc

    return cells, blocked


def _apply_rays(
    out: np.ndarray,
    grid: np.ndarray,
    center: tuple[int, int],
    radius: int,
    markers: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    for marker in markers:
        _r, _c, color, _dr, _dc = marker
        cells, _blocked = _ray_cells(grid, center, radius, marker)
        for rr, cc in cells:
            out[rr, cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a hollow diamond with concentric diagonal-palette bands, then overwrite them with side-marker rays stopped by blockers.

    Input structure:
    - The background is black, with one complete one-pixel-thick grey diamond frame.
    - A down-right diagonal of 2-4 singleton color markers ends one step above-left of the frame's top vertex.
    - The diamond interior contains black cells plus 1-3 maroon blocker pixels, and the frame bbox sides may each have up to two outside markers one cell away.

    Rule:
    - Read the diagonal colors from upper-left to lower-right and use them as outer-to-inner colors for concentric diamond layers strictly inside the frame.
    - If the interior has more layers than diagonal colors, reuse the last diagonal color for every remaining inner layer; blockers stay unchanged.
    - Process outside side markers in clockwise order: top left-to-right, right top-to-bottom, bottom right-to-left, left bottom-to-top. Each marker fires a perpendicular row/column ray that overwrites every strict interior cell until the ray hits a blocker or leaves the opposite side of the interior.

    Key insight:
    - The final interior is a superposition of two independent signals: diagonal markers define the default concentric diamond bands, while ordered side markers selectively overwrite those bands along blocker-limited rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    center, radius, bbox = _diamond_info(g)
    r0, _c0, _r1, top_c = bbox[0], bbox[1], bbox[2], center[1]
    palette, diag_coords = _read_diagonal_palette(g, (r0, top_c))
    out = _band_fill(g, center, radius, palette)
    markers = _ordered_side_markers(g, bbox, diag_coords)
    out = _apply_rays(out, g, center, radius, markers)
    return out


def _sample_margins(radius: int, diag_len: int) -> tuple[int, int, int, int]:
    top = diag_len + random.randint(1, 4)
    left = diag_len + random.randint(2, 6)
    bottom = random.randint(2, 5)
    right = random.randint(2, 5)
    return top, left, bottom, right


def _draw_diamond(
    grid: np.ndarray, center: tuple[int, int], radius: int
) -> tuple[int, int, int, int]:
    cr, cc = center
    for r in range(cr - radius, cr + radius + 1):
        for c in range(cc - radius, cc + radius + 1):
            if abs(r - cr) + abs(c - cc) == radius:
                grid[r, c] = FRAME
    return cr - radius, cc - radius, cr + radius, cc + radius


def _all_interior_cells(center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    cr, cc = center
    cells = []
    for r in range(cr - radius + 1, cr + radius):
        for c in range(cc - radius + 1, cc + radius):
            if _inside(center, radius, r, c):
                cells.append((int(r), int(c)))
    return cells


def _choose_marker_slot(
    available: list[int], used: set[int], limit: int
) -> list[int]:
    choices = [x for x in available if x not in used]
    random.shuffle(choices)
    take = min(limit, len(choices))
    return choices[:take]


def _place_diagonal(
    grid: np.ndarray, top_vertex: tuple[int, int], colors: list[int]
) -> set[tuple[int, int]]:
    tv_r, tv_c = top_vertex
    k = len(colors)
    coords = set()
    for i, color in enumerate(colors):
        r = tv_r - k + i
        c = tv_c - k + i
        grid[r, c] = int(color)
        coords.add((int(r), int(c)))
    return coords


def _available_side_positions(
    bbox: tuple[int, int, int, int], center: tuple[int, int]
) -> dict[str, list[int]]:
    r0, c0, r1, c1 = bbox
    cr, cc = center
    positions = {
        "top": list(range(c0 + 1, c1)),
        "bottom": list(range(c0 + 1, c1)),
        "left": list(range(r0 + 1, r1)),
        "right": list(range(r0 + 1, r1)),
    }
    if cc - 1 in positions["top"]:
        positions["top"].remove(cc - 1)
    return positions


def _marker_coord(side: str, offset: int, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0 - 1, offset
    if side == "bottom":
        return r1 + 1, offset
    if side == "left":
        return offset, c0 - 1
    if side == "right":
        return offset, c1 + 1
    raise ValueError(side)


def _marker_direction(side: str) -> tuple[int, int]:
    return {
        "top": (1, 0),
        "right": (0, -1),
        "bottom": (-1, 0),
        "left": (0, 1),
    }[side]


def _path_cells_for_side(
    side: str,
    offset: int,
    center: tuple[int, int],
    radius: int,
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r, c = _marker_coord(side, offset, bbox)
    dr, dc = _marker_direction(side)
    cells: list[tuple[int, int]] = []
    rr, cc = r + dr, c + dc
    entered = False
    while True:
        if not _in_bounds((bbox[2] + 2, bbox[3] + 2), rr, cc):
            break
        if _inside(center, radius, rr, cc):
            entered = True
            cells.append((int(rr), int(cc)))
        elif entered:
            break
        rr += dr
        cc += dc
    return cells


def _ordered_metadata(markers: list[dict]) -> list[dict]:
    order_key = {
        "top": lambda m: m["offset"],
        "right": lambda m: m["offset"],
        "bottom": lambda m: -m["offset"],
        "left": lambda m: -m["offset"],
    }
    ordered: list[dict] = []
    for side in ("top", "right", "bottom", "left"):
        same_side = [m for m in markers if m["side"] == side]
        same_side.sort(key=order_key[side])
        ordered.extend(same_side)
    return ordered


def _pick_crossing_pair(
    bbox: tuple[int, int, int, int], center: tuple[int, int], radius: int
) -> tuple[tuple[str, int], tuple[str, int], tuple[int, int]] | None:
    pos = _available_side_positions(bbox, center)
    vertical_sides = ["top", "bottom"]
    horizontal_sides = ["left", "right"]
    random.shuffle(vertical_sides)
    random.shuffle(horizontal_sides)
    for v_side in vertical_sides:
        for h_side in horizontal_sides:
            v_offsets = pos[v_side][:]
            h_offsets = pos[h_side][:]
            random.shuffle(v_offsets)
            random.shuffle(h_offsets)
            for c in v_offsets:
                v_path = set(_path_cells_for_side(v_side, c, center, radius, bbox))
                for r in h_offsets:
                    h_path = set(_path_cells_for_side(h_side, r, center, radius, bbox))
                    inter = sorted(v_path & h_path)
                    if inter:
                        rr, cc = random.choice(inter)
                        return (v_side, c), (h_side, r), (rr, cc)
    return None


def _place_markers(
    grid: np.ndarray, bbox: tuple[int, int, int, int], center: tuple[int, int], radius: int
) -> list[dict]:
    available = _available_side_positions(bbox, center)
    markers: list[dict] = []
    used_offsets = {side: set() for side in available}

    cross = _pick_crossing_pair(bbox, center, radius)
    if cross is not None and random.random() < 0.75:
        (v_side, v_offset), (h_side, h_offset), _cross_cell = cross
        colors = random.sample(SIDE_COLORS, 2)
        for side, offset, color in ((v_side, v_offset, colors[0]), (h_side, h_offset, colors[1])):
            markers.append({"side": side, "offset": offset, "color": int(color)})
            used_offsets[side].add(offset)

    target_total = random.randint(2, 6)
    if not markers and random.random() < 0.8:
        target_total = max(target_total, 3)

    sides = ["top", "right", "bottom", "left"]
    random.shuffle(sides)
    while len(markers) < target_total:
        progress = False
        for side in sides:
            if len(markers) >= target_total:
                break
            remaining = [x for x in available[side] if x not in used_offsets[side]]
            if not remaining:
                continue
            if sum(m["side"] == side for m in markers) >= 2:
                continue
            if random.random() < 0.55:
                offset = random.choice(remaining)
                color = random.choice(SIDE_COLORS)
                markers.append({"side": side, "offset": int(offset), "color": int(color)})
                used_offsets[side].add(offset)
                progress = True
        if not progress:
            break

    if not any(m["side"] in ("top", "bottom") for m in markers):
        side = random.choice(["top", "bottom"])
        offset = random.choice([x for x in available[side] if x not in used_offsets[side]])
        markers.append({"side": side, "offset": int(offset), "color": int(random.choice(SIDE_COLORS))})
        used_offsets[side].add(offset)
    if not any(m["side"] in ("left", "right") for m in markers):
        side = random.choice(["left", "right"])
        offset = random.choice([x for x in available[side] if x not in used_offsets[side]])
        markers.append({"side": side, "offset": int(offset), "color": int(random.choice(SIDE_COLORS))})
        used_offsets[side].add(offset)

    for marker in markers:
        r, c = _marker_coord(marker["side"], marker["offset"], bbox)
        grid[r, c] = int(marker["color"])

    return _ordered_metadata(markers)


def _marker_path_from_metadata(
    grid: np.ndarray, center: tuple[int, int], radius: int, bbox: tuple[int, int, int, int], marker: dict
) -> tuple[list[tuple[int, int]], bool]:
    r, c = _marker_coord(marker["side"], marker["offset"], bbox)
    dr, dc = _marker_direction(marker["side"])
    return _ray_cells(grid, center, radius, (r, c, marker["color"], dr, dc))


def _place_blockers(
    grid: np.ndarray,
    center: tuple[int, int],
    radius: int,
    bbox: tuple[int, int, int, int],
    markers: list[dict],
) -> list[tuple[int, int]]:
    interior = _all_interior_cells(center, radius)
    paths = []
    for marker in markers:
        cells, _blocked = _marker_path_from_metadata(grid, center, radius, bbox, marker)
        if len(cells) >= 2:
            paths.append((marker, cells))

    blockers: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    need_blocked_ray = bool(paths) and random.random() < 0.8
    if need_blocked_ray:
        marker, cells = random.choice(paths)
        candidate = random.choice(cells[:-1])
        blockers.append(candidate)
        used_rows.add(candidate[0])
        used_cols.add(candidate[1])

    target = random.randint(1, 3)
    random.shuffle(interior)
    for cell in interior:
        if len(blockers) >= target:
            break
        r, c = cell
        if r in used_rows or c in used_cols:
            continue
        blockers.append((int(r), int(c)))
        used_rows.add(int(r))
        used_cols.add(int(c))

    assert 1 <= len(blockers) <= 3
    for r, c in blockers:
        grid[r, c] = BLOCKER
    return blockers


def _coverage_counts(
    grid: np.ndarray,
    center: tuple[int, int],
    radius: int,
    bbox: tuple[int, int, int, int],
    markers: list[dict],
) -> dict[tuple[int, int], list[int]]:
    coverage: dict[tuple[int, int], list[int]] = {}
    for i, marker in enumerate(markers):
        cells, _blocked = _marker_path_from_metadata(grid, center, radius, bbox, marker)
        for cell in cells:
            coverage.setdefault(cell, []).append(i)
    return coverage


def _validate_case(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if _get_bg(g) != BG:
        return False

    center, radius, bbox = _diamond_info(g)
    top_vertex = (bbox[0], center[1])
    palette, diag_coords = _read_diagonal_palette(g, top_vertex)
    markers = _ordered_side_markers(g, bbox, diag_coords)
    if len(markers) < 2:
        return False

    blocker_coords = np.argwhere(g == BLOCKER)
    if not (1 <= blocker_coords.shape[0] <= 3):
        return False
    blocker_rows = {int(r) for r, _c in blocker_coords}
    blocker_cols = {int(c) for _r, c in blocker_coords}
    if len(blocker_rows) != blocker_coords.shape[0] or len(blocker_cols) != blocker_coords.shape[0]:
        return False
    if any(not _inside(center, radius, int(r), int(c)) for r, c in blocker_coords):
        return False

    base = _band_fill(g, center, radius, palette)
    out = main(g)

    interior_mask = np.zeros_like(g, dtype=bool)
    for r, c in _all_interior_cells(center, radius):
        interior_mask[r, c] = True
    interior_mask[g == BLOCKER] = False

    ray_changed = int(((out != base) & interior_mask).sum())
    unchanged = int(((out == base) & interior_mask).sum())
    if ray_changed == 0 or unchanged == 0:
        return False

    if len(palette) >= radius and np.all(base[interior_mask] == palette[-1]):
        return False

    coverage: dict[tuple[int, int], list[int]] = {}
    blocked_any = False
    for i, marker in enumerate(markers):
        cells, blocked = _ray_cells(g, center, radius, marker)
        blocked_any |= blocked
        for cell in cells:
            coverage.setdefault(cell, []).append(i)

    if not blocked_any:
        return False

    overlap_cells = [cell for cell, ids in coverage.items() if len(ids) >= 2]
    if not overlap_cells:
        return False

    latest_ok = False
    for cell in overlap_cells:
        ids = coverage[cell]
        final_color = int(out[cell])
        last_marker = markers[ids[-1]]
        if final_color == int(last_marker[2]):
            latest_ok = True
            break
    if not latest_ok:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    radius = random.randint(4, 6)
    diag_len = random.randint(2, min(4, radius))
    top, left, bottom, right = _sample_margins(radius, diag_len)
    h = top + (2 * radius + 1) + bottom
    w = left + (2 * radius + 1) + right
    if max(h, w) > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    center = (top + radius, left + radius)
    bbox = _draw_diamond(grid, center, radius)
    top_vertex = (bbox[0], center[1])

    diag_colors = random.sample(DIAG_COLORS, diag_len)
    diag_coords = _place_diagonal(grid, top_vertex, diag_colors)
    if (bbox[0] - 1, center[1] - 1) not in diag_coords:
        return None

    markers = _place_markers(grid, bbox, center, radius)
    _place_blockers(grid, center, radius, bbox, markers)

    if not _validate_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diamond-ray puzzle")
