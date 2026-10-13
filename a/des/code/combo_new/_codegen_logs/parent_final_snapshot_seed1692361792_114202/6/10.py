from __future__ import annotations

import random

import numpy as np


AZURE = 8
GREY = 5
RED = 2
PATTERN_COLORS = (1, 3, 4, 6, 7, 9)
MARKER_SIZE = 2
MIN_GRID = 18
MAX_GRID = 30
MIN_RADIUS = 4
MAX_RADIUS = 8
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 60
MASK_ATTEMPTS = 80
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MASK_SIGNS = {
    "tl": (1, 1),
    "tr": (1, -1),
    "bl": (-1, 1),
    "br": (-1, -1),
}


def _rotate_rel(dr: int, dc: int, turns: int) -> tuple[int, int]:
    rr, cc = int(dr), int(dc)
    for _ in range(int(turns) % 4):
        rr, cc = -cc, rr
    return rr, cc


def _rotate_point(r: int, c: int, center: tuple[int, int], turns: int) -> tuple[int, int]:
    cr, cc = center
    dr, dc = _rotate_rel(r - cr, c - cc, turns)
    return cr + dr, cc + dc


def _orbit(r: int, c: int, center: tuple[int, int]) -> set[tuple[int, int]]:
    return {_rotate_point(r, c, center, k) for k in range(4)}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _marker_corner(red_mask: np.ndarray) -> str:
    coords = np.argwhere(red_mask)
    assert coords.size > 0, "expected a red corner marker"
    h, w = red_mask.shape
    rs = coords[:, 0]
    cs = coords[:, 1]
    touches = []
    if int(rs.min()) == 0 and int(cs.min()) == 0:
        touches.append("tl")
    if int(rs.min()) == 0 and int(cs.max()) == w - 1:
        touches.append("tr")
    if int(rs.max()) == h - 1 and int(cs.min()) == 0:
        touches.append("bl")
    if int(rs.max()) == h - 1 and int(cs.max()) == w - 1:
        touches.append("br")
    assert len(touches) == 1, "red marker must occupy exactly one corner"
    return touches[0]


def _solid_rectangle_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a grey mask rectangle"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(np.all(sub)), "grey component must be a solid rectangle"
    assert int(mask.sum()) == sub.size, "grey mask must be a single rectangle"
    return r0, c0, r1, c1


def _center_matches_marker(corner: str, bbox: tuple[int, int, int, int], center: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    cr, cc = center
    if corner == "tl":
        return cr < r0 and cc < c0
    if corner == "tr":
        return cr < r0 and cc > c1
    if corner == "bl":
        return cr > r1 and cc < c0
    return cr > r1 and cc > c1


def _candidate_closure(
    grid: np.ndarray,
    visible: set[tuple[int, int]],
    pattern_color: int,
    grey_mask: np.ndarray,
    red_mask: np.ndarray,
    center: tuple[int, int],
) -> set[tuple[int, int]] | None:
    closure: set[tuple[int, int]] = set()
    for r, c in visible:
        orbit = _orbit(r, c, center)
        if any(not _in_bounds(grid.shape, rr, cc) for rr, cc in orbit):
            return None
        closure |= orbit

    if not any(grey_mask[r, c] for r, c in closure):
        return None

    for r, c in closure:
        if red_mask[r, c]:
            return None
        if not grey_mask[r, c] and int(grid[r, c]) != pattern_color:
            return None

    return closure


def _find_center_and_closure(grid: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    colors = set(map(int, np.unique(g)))
    extra = colors - {AZURE, GREY, RED}
    assert len(extra) == 1, "expected exactly one hidden-pattern color"
    pattern_color = int(next(iter(extra)))

    grey_mask = g == GREY
    red_mask = g == RED
    visible = {tuple(map(int, rc)) for rc in np.argwhere(g == pattern_color)}
    assert visible, "expected visible pattern pixels"

    bbox = _solid_rectangle_bbox(grey_mask)
    corner = _marker_corner(red_mask)

    candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []
    for cr in range(g.shape[0]):
        for cc in range(g.shape[1]):
            center = (cr, cc)
            if not _center_matches_marker(corner, bbox, center):
                continue
            closure = _candidate_closure(g, visible, pattern_color, grey_mask, red_mask, center)
            if closure is not None:
                candidates.append((center, closure))

    assert len(candidates) == 1, "symmetry center must be uniquely determined"
    center, closure = candidates[0]
    return center, closure, pattern_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the grey mask and reconstruct the hidden single-color figure using 90-degree rotational symmetry.

    Input structure:
    - The background is azure.
    - One large non-azure, non-grey figure is drawn in a single color and has fourfold rotational symmetry.
    - A solid grey rectangle hides part of that figure, while the rest of the figure remains visible.
    - A small red marker sits in one grid corner; its corner indicates which diagonal side of the grey mask contains the symmetry center.

    Rule:
    - Find the grey rectangle and the red corner marker.
    - Search for the unique rotation center that lies on the marker-indicated diagonal side of the mask and makes the visible figure consistent under 90-degree rotations, allowing only grey cells to be missing.
    - Close the visible figure under 90-degree, 180-degree, and 270-degree rotation around that center to infer the hidden cells.
    - Remove the grey mask: inferred figure cells become the figure color, all other mask cells become azure, and the red marker stays unchanged.

    Key insight:
    - The red marker does not point to the hidden patch itself; it constrains where the rotational center must be, and the actual repair comes from taking the visible figure's rotation orbit closure.
    """
    g = np.asarray(grid, dtype=int)
    center, closure, pattern_color = _find_center_and_closure(g)
    _ = center

    out = g.copy()
    out[out == GREY] = AZURE
    for r, c in closure:
        if out[r, c] != RED:
            out[r, c] = pattern_color
    return out


def _neighbors4(cell: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if -radius <= rr <= 0 and 0 <= cc <= radius:
            out.append((rr, cc))
    return out


def _grow_sector(radius: int, target_size: int) -> set[tuple[int, int]] | None:
    sector = {(0, 0)}
    frontier = set(_neighbors4((0, 0), radius))

    while len(sector) < target_size and frontier:
        choices = list(frontier)
        weights = [4 if (r < 0 and c > 0) else 1 for r, c in choices]
        cell = random.choices(choices, weights=weights, k=1)[0]
        frontier.remove(cell)
        sector.add(cell)
        for nxt in _neighbors4(cell, radius):
            if nxt not in sector:
                frontier.add(nxt)
        if len(sector) >= 3 and random.random() < 0.25:
            bonus = [nxt for nxt in _neighbors4(cell, radius) if nxt not in sector]
            if bonus:
                extra = random.choice(bonus)
                frontier.discard(extra)
                sector.add(extra)
                for nxt in _neighbors4(extra, radius):
                    if nxt not in sector:
                        frontier.add(nxt)

    if len(sector) != target_size:
        return None
    if not any(r < 0 and c > 0 for r, c in sector):
        return None
    if max(-r for r, _ in sector) < 2 or max(c for _, c in sector) < 2:
        return None
    return sector


def _sector_to_pattern(center: tuple[int, int], sector: set[tuple[int, int]]) -> set[tuple[int, int]]:
    cr, cc = center
    out: set[tuple[int, int]] = set()
    for dr, dc in sector:
        for turns in range(4):
            rr, rc = _rotate_rel(dr, dc, turns)
            out.add((cr + rr, cc + rc))
    return out


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _paint_marker(grid: np.ndarray, corner: str) -> None:
    if corner == "tl":
        grid[:MARKER_SIZE, :MARKER_SIZE] = RED
    elif corner == "tr":
        grid[:MARKER_SIZE, -MARKER_SIZE:] = RED
    elif corner == "bl":
        grid[-MARKER_SIZE:, :MARKER_SIZE] = RED
    else:
        grid[-MARKER_SIZE:, -MARKER_SIZE:] = RED


def _sample_pattern(shape: tuple[int, int], center: tuple[int, int]) -> set[tuple[int, int]] | None:
    h, w = shape
    cr, cc = center
    max_radius = min(cr - 3, cc - 3, h - 4 - cr, w - 4 - cc)
    if max_radius < MIN_RADIUS:
        return None

    radius = random.randint(MIN_RADIUS, min(MAX_RADIUS, max_radius))
    max_sector = (radius + 1) * (radius + 1)
    lo = max(radius + 3, 7)
    hi = min(max_sector - 1, radius * 2 + 8)
    if lo > hi:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        sector = _grow_sector(radius, random.randint(lo, hi))
        if sector is None:
            continue
        pattern = _sector_to_pattern(center, sector)
        if not all(_in_bounds(shape, r, c) for r, c in pattern):
            continue
        if not _is_connected(pattern):
            continue
        if len(pattern) < 24:
            continue
        return pattern
    return None


def _mask_ranges(limit: int, pivot: int, sign: int) -> list[tuple[int, int]]:
    ranges = []
    for size in range(2, min(8, limit) + 1):
        if sign > 0:
            start_lo = pivot + 1
            start_hi = limit - size - 1
        else:
            start_lo = 1
            start_hi = pivot - size
        if start_lo > start_hi:
            continue
        ranges.append((size, start_lo, start_hi))
    return ranges


def _choose_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    pattern: set[tuple[int, int]],
    corner: str,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    cr, cc = center
    sign_r, sign_c = MASK_SIGNS[corner]
    row_specs = _mask_ranges(h, cr, sign_r)
    col_specs = _mask_ranges(w, cc, sign_c)
    if not row_specs or not col_specs:
        return None

    for _ in range(MASK_ATTEMPTS):
        height, row_lo, row_hi = random.choice(row_specs)
        width, col_lo, col_hi = random.choice(col_specs)
        r0 = random.randint(row_lo, row_hi)
        c0 = random.randint(col_lo, col_hi)
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        inside = {(r, c) for r, c in pattern if r0 <= r <= r1 and c0 <= c <= c1}
        hidden = len(inside)
        area = height * width
        if hidden < 3:
            continue
        if hidden >= area:
            continue
        if hidden > len(pattern) // 3:
            continue
        return r0, c0, r1, c1
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    center = (random.randint(5, h - 6), random.randint(5, w - 6))
    corner = random.choice(("tl", "tr", "bl", "br"))
    pattern = _sample_pattern((h, w), center)
    if pattern is None:
        return None

    mask = _choose_mask((h, w), center, pattern, corner)
    if mask is None:
        return None
    r0, c0, r1, c1 = mask

    pattern_color = random.choice(PATTERN_COLORS)
    out = np.full((h, w), AZURE, dtype=int)
    for r, c in pattern:
        out[r, c] = pattern_color
    _paint_marker(out, corner)

    if np.count_nonzero(out == RED) != MARKER_SIZE * MARKER_SIZE:
        return None
    if any(out[r, c] == RED for r, c in pattern):
        return None

    inp = out.copy()
    inp[r0 : r1 + 1, c0 : c1 + 1] = GREY

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid masked rotational-symmetry puzzle")
