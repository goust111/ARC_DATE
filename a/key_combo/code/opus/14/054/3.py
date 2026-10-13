from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

import numpy as np


BLACK = 0
GLYPH = 5
OBSTACLE = 7
SQUARE_COLORS = (1, 2, 3, 4, 6, 8, 9)
SIDES = ("top", "right", "bottom", "left")

Coord = Tuple[int, int]
BBox = Tuple[int, int, int, int]


def _in_bounds(shape: Tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_from_coords(coords: Iterable[Coord]) -> BBox:
    pts = list(coords)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _square_cells(square_bbox: BBox) -> List[Coord]:
    r0, c0, r1, c1 = square_bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _find_square(grid: np.ndarray) -> Tuple[BBox, int]:
    candidates: List[Tuple[BBox, int]] = []
    for color in sorted(set(int(v) for v in np.unique(grid)) - {BLACK, GLYPH, OBSTACLE}):
        coords = np.argwhere(grid == color)
        if coords.shape[0] != 9:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (int(r1) - int(r0) + 1, int(c1) - int(c0) + 1) != (3, 3):
            continue
        if np.all(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == color):
            candidates.append(((int(r0), int(c0), int(r1), int(c1)), int(color)))
    assert len(candidates) == 1, "expected exactly one solid 3x3 square"
    return candidates[0]


def _classify_glyph_pixels(grid: np.ndarray, square_bbox: BBox) -> Dict[str, List[Coord]]:
    r0, c0, r1, c1 = square_bbox
    sr = r0 + 1
    sc = c0 + 1
    groups: Dict[str, List[Coord]] = {side: [] for side in SIDES}
    for r, c in np.argwhere(grid == GLYPH):
        rr = int(r)
        cc = int(c)
        dr = rr - sr
        dc = cc - sc
        if abs(dr) >= abs(dc):
            side = "top" if dr < 0 else "bottom"
        else:
            side = "left" if dc < 0 else "right"
        groups[side].append((rr, cc))
    assert all(groups[side] for side in SIDES), "expected one grey glyph on each side"
    return groups


def _anchor(square_bbox: BBox, side: str) -> Coord:
    r0, c0, r1, c1 = square_bbox
    if side == "top":
        return r0, c0 + 1
    if side == "right":
        return r0 + 1, c1
    if side == "bottom":
        return r1, c0 + 1
    if side == "left":
        return r0 + 1, c0
    raise ValueError(side)


def _ray_directions(side: str, ray_type: int) -> List[Coord]:
    orth = {
        "top": (-1, 0),
        "right": (0, 1),
        "bottom": (1, 0),
        "left": (0, -1),
    }
    diagonals = {
        "top": [(-1, -1), (-1, 1)],
        "right": [(-1, 1), (1, 1)],
        "bottom": [(1, -1), (1, 1)],
        "left": [(-1, -1), (1, -1)],
    }
    if ray_type == 0:
        return []
    if ray_type == 1:
        return [orth[side]]
    if ray_type == 2:
        return diagonals[side]
    if ray_type == 3:
        return [orth[side], *diagonals[side]]
    raise ValueError(ray_type)


def _ray_path(shape: Tuple[int, int], anchor: Coord, direction: Coord) -> List[Coord]:
    dr, dc = direction
    r = anchor[0] + dr
    c = anchor[1] + dc
    cells: List[Coord] = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _paint_rays(
    out: np.ndarray,
    square_bbox: BBox,
    square_color: int,
    side_to_type: Dict[str, int],
    blockers: set[Coord],
) -> np.ndarray:
    for side in SIDES:
        anchor = _anchor(square_bbox, side)
        for direction in _ray_directions(side, side_to_type[side]):
            for cell in _ray_path(out.shape, anchor, direction):
                if cell in blockers:
                    break
                out[cell] = square_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode side glyphs into outward ray patterns around a central 3x3 square.

    Input structure:
    - The background is black, with one solid 3x3 colored square near the center.
    - Four separate 3x3 grey/black glyph blocks sit above, right, below, and left of the square.
    - Orange single-pixel obstacles are scattered away from the square and glyph blocks.

    Rule:
    - Count the grey pixels in each side glyph and compute that count modulo 4.
    - Use the glyph side to choose the square-edge anchor, then decode modulo 4 as: 0 = no ray, 1 = one orthogonal ray, 2 = two outward diagonals, 3 = orthogonal plus both diagonals.
    - Draw each ray in the square's color from the corresponding edge-center anchor outward, stopping just before an orange obstacle or the grid boundary; erase the grey glyph pixels to black.

    Key insight:
    - The glyphs are symbolic counters: their grey-pixel counts decode the ray pattern, while their spatial side relative to the square selects which edge emits that pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    square_bbox, square_color = _find_square(g)
    groups = _classify_glyph_pixels(g, square_bbox)
    side_to_type = {side: len(groups[side]) % 4 for side in SIDES}

    out = g.copy()
    out[g == GLYPH] = BLACK
    blockers = {(int(r), int(c)) for r, c in np.argwhere(g == OBSTACLE)}
    return _paint_rays(out, square_bbox, square_color, side_to_type, blockers)


def _window_cells(bbox: BBox) -> List[Coord]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _halo8(cells: Iterable[Coord], shape: Tuple[int, int]) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _choose_square_bbox(h: int, w: int) -> BBox:
    r0_lo = max(7, h // 2 - 2)
    r0_hi = min(h - 10, h // 2)
    c0_lo = max(7, w // 2 - 2)
    c0_hi = min(w - 10, w // 2)
    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)
    return r0, c0, r0 + 2, c0 + 2


def _place_glyph_windows(square_bbox: BBox, shape: Tuple[int, int]) -> Dict[str, BBox]:
    h, w = shape
    r0, c0, r1, c1 = square_bbox
    sr = r0 + 1
    sc = c0 + 1

    top_row = random.randint(1, min(3, r0 - 6))
    bottom_row = random.randint(r1 + 4, min(h - 4, r1 + 6))
    left_col = random.randint(1, min(3, c0 - 6))
    right_col = random.randint(c1 + 4, min(w - 4, c1 + 6))

    top_col = min(max(sc - 2 + random.randint(0, 2), 1), w - 4)
    bottom_col = min(max(sc - 2 + random.randint(0, 2), 1), w - 4)
    left_row = min(max(sr - 2 + random.randint(0, 2), 1), h - 4)
    right_row = min(max(sr - 2 + random.randint(0, 2), 1), h - 4)

    windows = {
        "top": (top_row, top_col, top_row + 2, top_col + 2),
        "right": (right_row, right_col, right_row + 2, right_col + 2),
        "bottom": (bottom_row, bottom_col, bottom_row + 2, bottom_col + 2),
        "left": (left_row, left_col, left_row + 2, left_col + 2),
    }

    sr_sq = r0 + 1
    sc_sq = c0 + 1
    for side, bbox in windows.items():
        for r, c in _window_cells(bbox):
            dr = r - sr_sq
            dc = c - sc_sq
            if side in ("top", "bottom"):
                assert abs(dr) > abs(dc), "vertical glyph must stay in its side cone"
            else:
                assert abs(dc) > abs(dr), "horizontal glyph must stay in its side cone"
    return windows


def _count_for_type(ray_type: int) -> int:
    if ray_type == 0:
        return 4
    if ray_type == 1:
        return random.choice((1, 5))
    if ray_type == 2:
        return random.choice((2, 6))
    if ray_type == 3:
        return random.choice((3, 7))
    raise ValueError(ray_type)


def _glyph_pattern(count: int) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    cells = random.sample([(r, c) for r in range(3) for c in range(3)], count)
    for r, c in cells:
        patch[r, c] = GLYPH
    return patch


def _obstacle_legal(
    cell: Coord,
    shape: Tuple[int, int],
    forbidden: set[Coord],
    chosen: set[Coord],
) -> bool:
    if cell in forbidden or cell in chosen:
        return False
    for rr in range(cell[0] - 1, cell[0] + 2):
        for cc in range(cell[1] - 1, cell[1] + 2):
            if _in_bounds(shape, rr, cc) and (rr, cc) in chosen:
                return False
    return True


def _choose_blockers(
    square_bbox: BBox,
    glyph_windows: Dict[str, BBox],
    side_to_type: Dict[str, int],
    shape: Tuple[int, int],
) -> Tuple[set[Coord], Dict[Tuple[str, Coord], str]]:
    forbidden = _halo8(_square_cells(square_bbox), shape)
    for bbox in glyph_windows.values():
        forbidden |= _halo8(_window_cells(bbox), shape)

    ray_records: List[Tuple[str, Coord, List[Coord], List[Coord]]] = []
    for side in SIDES:
        for direction in _ray_directions(side, side_to_type[side]):
            path = _ray_path(shape, _anchor(square_bbox, side), direction)
            blocker_cells = [cell for idx, cell in enumerate(path) if idx >= 1 and cell not in forbidden]
            ray_records.append((side, direction, path, blocker_cells))

    assert len(ray_records) >= 4
    rays_with_blockers = [rec for rec in ray_records if rec[3]]
    assert rays_with_blockers, "need at least one ray with a legal blocker position"

    block_count = random.randint(2, max(2, len(ray_records) - 1))
    random.shuffle(ray_records)

    chosen: set[Coord] = set()
    stop_kind: Dict[Tuple[str, Coord], str] = {}
    blocked_any = False
    edge_any = False

    for idx, (side, direction, path, blocker_cells) in enumerate(ray_records):
        remaining = len(ray_records) - idx - 1
        must_leave_edge = not edge_any and remaining == 0
        want_block = idx < block_count and blocker_cells and not must_leave_edge
        if not blocked_any and blocker_cells:
            want_block = True

        if want_block:
            legal = [cell for cell in blocker_cells if _obstacle_legal(cell, shape, forbidden, chosen)]
            if legal:
                blocker = random.choice(legal)
                chosen.add(blocker)
                stop_kind[(side, direction)] = "blocker"
                blocked_any = True
                continue
        stop_kind[(side, direction)] = "edge"
        edge_any = True

    assert blocked_any and edge_any, "need both blocker-stopped and edge-stopped rays"
    return chosen, stop_kind


def _add_extra_obstacles(
    grid: np.ndarray,
    square_bbox: BBox,
    glyph_windows: Dict[str, BBox],
    blockers: set[Coord],
) -> None:
    shape = grid.shape
    forbidden = _halo8(_square_cells(square_bbox), shape)
    for bbox in glyph_windows.values():
        forbidden |= _halo8(_window_cells(bbox), shape)

    occupied = set(blockers)
    for _ in range(random.randint(3, 8)):
        for _try in range(50):
            cell = (random.randint(0, shape[0] - 1), random.randint(0, shape[1] - 1))
            if _obstacle_legal(cell, shape, forbidden, occupied):
                occupied.add(cell)
                break

    for r, c in occupied:
        grid[r, c] = OBSTACLE


def _expected_from_meta(
    grid: np.ndarray,
    square_bbox: BBox,
    square_color: int,
    side_to_type: Dict[str, int],
) -> np.ndarray:
    out = grid.copy()
    out[out == GLYPH] = BLACK
    blockers = {(int(r), int(c)) for r, c in np.argwhere(grid == OBSTACLE)}
    return _paint_rays(out, square_bbox, square_color, side_to_type, blockers)


def _semantic_checks(
    grid: np.ndarray,
    square_bbox: BBox,
    square_color: int,
    glyph_windows: Dict[str, BBox],
    side_to_type: Dict[str, int],
) -> None:
    expected = _expected_from_meta(grid, square_bbox, square_color, side_to_type)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver must match semantic construction"
    assert np.count_nonzero(expected == square_color) > 9, "expected non-trivial ray output"

    no_block = grid.copy()
    no_block[no_block == OBSTACLE] = BLACK
    assert not np.array_equal(expected, _expected_from_meta(no_block, square_bbox, square_color, side_to_type))

    for side in SIDES:
        base = side_to_type[side]
        changed = False
        for alt in range(4):
            if alt == base:
                continue
            alt_map = dict(side_to_type)
            alt_map[side] = alt
            if not np.array_equal(expected, _expected_from_meta(grid, square_bbox, square_color, alt_map)):
                changed = True
                break
        assert changed, f"{side} glyph count must affect the output"

    rotated = {side: side_to_type[SIDES[(idx - 1) % 4]] for idx, side in enumerate(SIDES)}
    assert not np.array_equal(expected, _expected_from_meta(grid, square_bbox, square_color, rotated))

    groups = _classify_glyph_pixels(grid, square_bbox)
    for side in SIDES:
        assert len(groups[side]) % 4 == side_to_type[side]
        assert _bbox_from_coords(groups[side])[0] >= glyph_windows[side][0]
        assert _bbox_from_coords(groups[side])[1] >= glyph_windows[side][1]


def _generate_once() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    square_bbox = _choose_square_bbox(h, w)
    square_color = random.choice(SQUARE_COLORS)
    glyph_windows = _place_glyph_windows(square_bbox, (h, w))

    side_types = list(np.random.permutation([0, 1, 2, 3]))
    side_to_type = {side: side_types[idx] for idx, side in enumerate(SIDES)}

    grid = np.zeros((h, w), dtype=int)
    r0, c0, r1, c1 = square_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = square_color

    for side in SIDES:
        count = _count_for_type(side_to_type[side])
        patch = _glyph_pattern(count)
        wr0, wc0, _, _ = glyph_windows[side]
        grid[wr0 : wr0 + 3, wc0 : wc0 + 3] = patch

    blockers, _stop_kind = _choose_blockers(square_bbox, glyph_windows, side_to_type, grid.shape)
    for r, c in blockers:
        grid[r, c] = OBSTACLE
    _add_extra_obstacles(grid, square_bbox, glyph_windows, blockers)

    assert max(grid.shape) <= 30
    assert len(set(side_to_type.values())) == 4
    _semantic_checks(grid, square_bbox, square_color, glyph_windows, side_to_type)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            return _generate_once()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
