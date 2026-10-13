from __future__ import annotations

from collections import deque
from itertools import permutations
import random

import numpy as np


BG = 0
SOURCE = 1
RED = 2
GREEN = 3
DIVIDER = 4
CENTER = 5
ORANGE = 7
CORNER_COLORS = (RED, GREEN, ORANGE)

GRID_SIZES = (11, 13, 15, 17, 19, 21)
MIN_SHAPE_CELLS = 5
MAX_SHAPE_CELLS = 10
GEN_ATTEMPTS = 600

VALID_SUBSETS = (
    ("tl", "bl"),
    ("tr", "br"),
    ("tl", "br"),
    ("tr", "bl"),
    ("tl", "bl", "tr"),
    ("tl", "bl", "br"),
    ("tl", "tr", "br"),
    ("bl", "tr", "br"),
)
TIE_SUBSETS = (
    ("tr", "br"),
    ("tl", "tr", "br"),
    ("bl", "tr", "br"),
)


def _corner_positions(h: int, w: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }


def _neighbors(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    q: deque[tuple[int, int]] = deque([next(iter(cells))])
    seen = {q[0]}
    while q:
        cell = q.popleft()
        for nxt in _neighbors(cell):
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _tight_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _is_irregular_shape(cells: set[tuple[int, int]]) -> bool:
    mask = _tight_mask(cells)
    if bool(mask.all()):
        return False
    if np.array_equal(mask, np.rot90(mask, 2)):
        return False
    return True


def _nearest_corner(
    cell: tuple[int, int],
    corner_markers: dict[tuple[int, int], int],
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    r, c = cell
    distances = []
    for pos in corner_markers:
        pr, pc = pos
        distances.append((pos, abs(r - pr) + abs(c - pc)))
    min_dist = min(dist for _pos, dist in distances)
    tied = tuple(sorted(pos for pos, dist in distances if dist == min_dist))
    owner = min((pos for pos, dist in distances if dist == min_dist), key=lambda pos: (-pos[0], -pos[1]))
    return owner, tied


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add the point-reflected copy of the blue shape, but color each reflected pixel by its nearest corner marker.

    Input structure:
    - The grid has a black background, odd height and width, a gray center marker, and a yellow divider filling the center column except at the center.
    - A 4-connected irregular blue shape lies entirely on the left half.
    - Two or three differently colored corner markers sit in distinct corners, chosen from red, green, and orange.

    Rule:
    - Reflect every blue source pixel by 180 degrees around the gray center marker.
    - For each reflected position, compare Manhattan distances to the present corner markers.
    - Color that reflected pixel with the nearest marker's color, breaking ties by larger row index and then larger column index.
    - Erase all corner markers, while preserving the blue source shape, the gray center, and the yellow divider.

    Key insight:
    - The corner markers serve both as activation clues and as spatial color sources: geometry comes from point reflection, while color comes from nearest-corner ownership.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1

    cr, cc = h // 2, w // 2
    center_coords = np.argwhere(g == CENTER)
    assert center_coords.shape == (1, 2)
    assert tuple(map(int, center_coords[0])) == (cr, cc)

    divider = g[:, cc]
    assert int(divider[cr]) == CENTER
    assert np.all(divider[np.arange(h) != cr] == DIVIDER)

    corner_markers: dict[tuple[int, int], int] = {}
    for pos in _corner_positions(h, w).values():
        color = int(g[pos])
        if color in CORNER_COLORS:
            corner_markers[pos] = color
        else:
            assert color == BG
    assert 2 <= len(corner_markers) <= 3
    assert len(set(corner_markers.values())) == len(corner_markers)

    source_cells = {tuple(map(int, cell)) for cell in np.argwhere(g == SOURCE)}
    assert source_cells
    assert all(c < cc for _r, c in source_cells)
    assert _is_connected(source_cells)
    assert _is_irregular_shape(source_cells)
    assert not np.any(g[:, cc + 1 :] == SOURCE)

    out = g.copy()
    for pos in corner_markers:
        out[pos] = BG

    for r, c in sorted(source_cells):
        rr = 2 * cr - r
        rc = 2 * cc - c
        assert 0 <= rr < h and 0 <= rc < w
        assert rc > cc
        assert rc != cc
        assert (rr, rc) not in corner_markers
        owner, _tied = _nearest_corner((rr, rc), corner_markers)
        out[rr, rc] = corner_markers[owner]

    return out


def _allowed_target_cells(h: int, w: int) -> list[tuple[int, int]]:
    cc = w // 2
    return [(r, c) for r in range(1, h - 1) for c in range(cc + 1, w - 1)]


def _manhattan_path(
    a: tuple[int, int],
    b: tuple[int, int],
    vertical_first: bool,
) -> list[tuple[int, int]]:
    r, c = a
    rr, cc = b
    cells = [(r, c)]
    if vertical_first:
        while r != rr:
            r += 1 if rr > r else -1
            cells.append((r, c))
        while c != cc:
            c += 1 if cc > c else -1
            cells.append((r, c))
    else:
        while c != cc:
            c += 1 if cc > c else -1
            cells.append((r, c))
        while r != rr:
            r += 1 if rr > r else -1
            cells.append((r, c))
    return cells


def _best_anchor_shape(anchors: list[tuple[int, int]]) -> set[tuple[int, int]] | None:
    best: set[tuple[int, int]] | None = None
    count = len(anchors)
    for order in permutations(anchors):
        for bits in range(1 << max(0, count - 1)):
            shape = {order[0]}
            for i in range(count - 1):
                seg = _manhattan_path(order[i], order[i + 1], bool(bits & (1 << i)))
                shape.update(seg)
            if best is None or len(shape) < len(best):
                best = shape
    return best


def _frontier(shape: set[tuple[int, int]], allowed: set[tuple[int, int]]) -> list[tuple[int, int]]:
    frontier = set()
    for cell in shape:
        for nxt in _neighbors(cell):
            if nxt in allowed and nxt not in shape:
                frontier.add(nxt)
    return list(frontier)


def _grow_shape(
    base: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
    target_size: int,
) -> set[tuple[int, int]] | None:
    shape = set(base)
    while len(shape) < target_size:
        fringe = _frontier(shape, allowed)
        if not fringe:
            return None
        fringe.sort(key=lambda cell: (random.randint(0, 3), abs(cell[0]), abs(cell[1])))
        shape.add(random.choice(fringe[: min(8, len(fringe))]))
    return shape


def _territories(
    h: int,
    w: int,
    corner_markers: dict[tuple[int, int], int],
) -> tuple[dict[tuple[int, int], list[tuple[int, int]]], list[tuple[tuple[int, int], tuple[int, int], tuple[tuple[int, int], ...]]]]:
    territories = {corner: [] for corner in corner_markers}
    ties = []
    for cell in _allowed_target_cells(h, w):
        owner, tied = _nearest_corner(cell, corner_markers)
        territories[owner].append(cell)
        if len(tied) > 1:
            ties.append((cell, owner, tied))
    return territories, ties


def _central_pool(
    cells: list[tuple[int, int]],
    focus_r: int,
    focus_c: int,
    limit: int = 14,
) -> list[tuple[int, int]]:
    ranked = sorted(cells, key=lambda cell: (abs(cell[0] - focus_r) + abs(cell[1] - focus_c), abs(cell[0] - focus_r), abs(cell[1] - focus_c)))
    return ranked[: min(limit, len(ranked))]


def _make_corner_markers(h: int, w: int, tie_mode: bool) -> dict[tuple[int, int], int]:
    names = random.choice(TIE_SUBSETS if tie_mode else tuple(s for s in VALID_SUBSETS if s not in TIE_SUBSETS))
    colors = random.sample(CORNER_COLORS, len(names))
    positions = _corner_positions(h, w)
    return {positions[name]: color for name, color in zip(names, colors)}


def _build_target_shape(
    h: int,
    w: int,
    corner_markers: dict[tuple[int, int], int],
    tie_mode: bool,
) -> set[tuple[int, int]] | None:
    cr, cc = h // 2, w // 2
    focus_c = (cc + 1 + (w - 2)) // 2
    allowed = set(_allowed_target_cells(h, w))
    territories, tie_records = _territories(h, w, corner_markers)
    available_owners = [corner for corner, cells in territories.items() if cells]
    if len(available_owners) < 2:
        return None

    if tie_mode:
        tie_cells = [cell for cell, owner, tied in tie_records if owner in available_owners]
        if not tie_cells:
            return None
        primary_pool = _central_pool(tie_cells, cr, focus_c, limit=10)
    else:
        primary_owner = random.choice(available_owners)
        primary_pool = _central_pool(territories[primary_owner], cr, focus_c)

    for _ in range(300):
        if not primary_pool:
            return None
        first = random.choice(primary_pool)
        first_owner, first_tied = _nearest_corner(first, corner_markers)
        other_owners = [corner for corner in available_owners if corner != first_owner]
        if not other_owners:
            continue
        second_owner = random.choice(other_owners)
        second_pool = _central_pool(territories[second_owner], cr, focus_c)
        if not second_pool:
            continue
        second = random.choice(second_pool)
        if first[0] == second[0] or first[1] == second[1]:
            continue
        base = _best_anchor_shape([first, second])
        if base is None or len(base) > MAX_SHAPE_CELLS:
            continue
        min_size = max(MIN_SHAPE_CELLS, len(base))
        target_size = random.randint(min_size, MAX_SHAPE_CELLS)
        shape = _grow_shape(base, allowed, target_size)
        if shape is None:
            continue
        if not _is_connected(shape):
            continue
        if not _is_irregular_shape(shape):
            continue
        owners_in_shape = {_nearest_corner(cell, corner_markers)[0] for cell in shape}
        if len(owners_in_shape) < 2:
            continue
        if tie_mode:
            tied_cells = [cell for cell in shape if len(_nearest_corner(cell, corner_markers)[1]) > 1]
            if not tied_cells:
                continue
            if not any(_nearest_corner(cell, corner_markers)[0] == first_owner for cell in tied_cells):
                continue
        return shape
    return None


def _mirror_to_source(
    target_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    return {(h - 1 - r, w - 1 - c) for r, c in target_cells}


def _build_pair(
    h: int,
    w: int,
    target_cells: set[tuple[int, int]],
    corner_markers: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    cr, cc = h // 2, w // 2
    source_cells = _mirror_to_source(target_cells, h, w)

    inp = np.full((h, w), BG, dtype=int)
    inp[:, cc] = DIVIDER
    inp[cr, cc] = CENTER
    for r, c in source_cells:
        inp[r, c] = SOURCE
    for pos, color in corner_markers.items():
        inp[pos] = color

    out = inp.copy()
    for pos in corner_markers:
        out[pos] = BG
    for cell in target_cells:
        owner, _tied = _nearest_corner(cell, corner_markers)
        out[cell] = corner_markers[owner]
    return inp, out


def _validate_pair(
    inp: np.ndarray,
    out: np.ndarray,
    require_tie: bool,
) -> None:
    h, w = inp.shape
    cr, cc = h // 2, w // 2
    assert inp[cr, cc] == CENTER
    assert np.all(inp[:, cc][np.arange(h) != cr] == DIVIDER)
    assert np.all(out[:, cc][np.arange(h) != cr] == DIVIDER)
    assert out[cr, cc] == CENTER

    corners = {}
    for pos in _corner_positions(h, w).values():
        color = int(inp[pos])
        if color in CORNER_COLORS:
            corners[pos] = color
    assert 2 <= len(corners) <= 3
    assert len(set(corners.values())) == len(corners)

    source_cells = {tuple(map(int, cell)) for cell in np.argwhere(inp == SOURCE)}
    assert source_cells
    assert _is_connected(source_cells)
    assert _is_irregular_shape(source_cells)
    assert all(c < cc for _r, c in source_cells)
    target_cells = {(h - 1 - r, w - 1 - c) for r, c in source_cells}
    assert all(c > cc for _r, c in target_cells)
    assert target_cells.isdisjoint(corners)

    owners_used = set()
    tie_seen = False
    for cell in target_cells:
        owner, tied = _nearest_corner(cell, corners)
        owners_used.add(owner)
        tie_seen = tie_seen or len(tied) > 1
        assert out[cell] == corners[owner]
    assert len(owners_used) >= 2
    if require_tie:
        assert tie_seen

    for pos in corners:
        assert out[pos] == BG
    for cell in source_cells:
        assert out[cell] == SOURCE

    right_mask = np.zeros((h, w), dtype=bool)
    right_mask[:, cc + 1 :] = True
    for cell in target_cells:
        right_mask[cell] = False
    right_mask[:, cc] = False
    assert np.all(out[right_mask] == BG)


def _generate_pair() -> tuple[np.ndarray, np.ndarray, bool] | None:
    h = random.choice(GRID_SIZES)
    w = random.choice(GRID_SIZES)
    tie_mode = random.random() < 0.5
    corner_markers = _make_corner_markers(h, w, tie_mode)
    target_cells = _build_target_shape(h, w, corner_markers, tie_mode)
    if target_cells is None:
        return None
    inp, out = _build_pair(h, w, target_cells, corner_markers)
    _validate_pair(inp, out, require_tie=tie_mode)
    if not np.array_equal(main(inp), out):
        return None
    return inp, out, tie_mode


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pair = _generate_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
