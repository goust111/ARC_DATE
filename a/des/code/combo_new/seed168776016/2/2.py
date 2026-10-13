import random
from collections import Counter

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
RECT_DIMS = ((2, 3), (3, 2))
GEN_RETRIES = 400
SIZE_MIN, SIZE_MAX = 11, 30
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 1, 4


def _get_bg(grid: np.ndarray) -> int:
    values = [int(v) for v in grid.ravel()]
    if BG in values:
        return BG
    return Counter(values).most_common(1)[0][0]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray | set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    pts = list(coords) if not isinstance(coords, np.ndarray) else [tuple(map(int, p)) for p in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return frozenset((int(r - min_r), int(c - min_c)) for r, c in pts)


def _shape_dims(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _rotate90(shape: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    h, _ = _shape_dims(shape)
    return _normalize({(c, h - 1 - r) for r, c in shape})


def _rotations(shape: frozenset[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    out: list[frozenset[tuple[int, int]]] = []
    cur = shape
    for _ in range(4):
        if cur not in out:
            out.append(cur)
        cur = _rotate90(cur)
    return out


def _is_l_triomino(coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != 3:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return len(_normalize(coords)) == 3


def _can_tile_rect(
    shape_a: frozenset[tuple[int, int]],
    shape_b: frozenset[tuple[int, int]],
    dims: tuple[int, int],
) -> bool:
    target = {(r, c) for r in range(dims[0]) for c in range(dims[1])}
    for rot_a in _rotations(shape_a):
        ha, wa = _shape_dims(rot_a)
        for rot_b in _rotations(shape_b):
            hb, wb = _shape_dims(rot_b)
            for ar in range(dims[0] - ha + 1):
                for ac in range(dims[1] - wa + 1):
                    placed_a = {(r + ar, c + ac) for r, c in rot_a}
                    for br in range(dims[0] - hb + 1):
                        for bc in range(dims[1] - wb + 1):
                            placed_b = {(r + br, c + bc) for r, c in rot_b}
                            if placed_a & placed_b:
                                continue
                            if placed_a | placed_b == target:
                                return True
    return False


def _parse_specs(grid: np.ndarray, bg: int) -> list[dict]:
    specs = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = _components(grid == color)
        anchors = []
        fragments = []
        for coords in comps:
            if int(coords.shape[0]) == 1:
                anchors.append(tuple(map(int, coords[0])))
            elif _is_l_triomino(coords):
                fragments.append(_normalize(coords))
            else:
                raise AssertionError(f"unexpected component for color {color}")
        assert len(anchors) == 1, f"expected one anchor for color {color}"
        assert len(fragments) == 2, f"expected two L fragments for color {color}"
        specs.append({"color": color, "anchor": anchors[0], "fragments": fragments})
    assert specs, "no colored objects found"
    return specs


def _choose_dims(spec: dict, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    ar, ac = spec["anchor"]
    candidates = []
    for dims in RECT_DIMS:
        if not _can_tile_rect(spec["fragments"][0], spec["fragments"][1], dims):
            continue
        if ar + dims[0] <= h and ac + dims[1] <= w:
            candidates.append(dims)

    if len(candidates) == 1:
        return candidates[0]
    if (2, 3) in candidates and ar + 2 == h:
        return 2, 3
    if (3, 2) in candidates and ac + 2 == w:
        return 3, 2
    assert candidates, f"anchor at {(ar, ac)} cannot host any rectangle"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Pair each color's two L triominoes, assemble them into a solid rectangle, and redraw that rectangle at the matching anchor.

    Input structure:
    - The background is black.
    - Each non-background color appears as exactly two disconnected 3-cell L triominoes and one isolated single-pixel anchor.
    - Every L fragment occupies three cells of a 2x2 box, so it is a missing-corner mini-square.

    Rule:
    - Group the non-background components by color.
    - For each color, treat the singleton as the destination anchor and the two 3-cell components as a pair of fragments.
    - Independently rotate the two fragments until they can tile a 2x3 or 3x2 rectangle; use the rectangle orientation that fits when the anchor is the rectangle's top-left corner.
    - Erase all original colored cells and keep only the solid anchored rectangles.

    Key insight:
    - Color gives the pairing, the two L triominoes encode a six-cell rectangle, and the lone pixel is a placement anchor rather than part of the source shape.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = np.full(g.shape, bg, dtype=int)

    for spec in _parse_specs(g, bg):
        rh, rw = _choose_dims(spec, g.shape)
        ar, ac = spec["anchor"]
        out[ar : ar + rh, ac : ac + rw] = int(spec["color"])
    return out


def _make_l_patch(color: int, missing_corner: int) -> np.ndarray:
    patch = np.full((2, 2), int(color), dtype=int)
    corners = ((0, 0), (0, 1), (1, 0), (1, 1))
    mr, mc = corners[missing_corner]
    patch[mr, mc] = BG
    return patch


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    sub = grid[top : top + h, left : left + w]
    sub[mask] = patch[mask]


def _area_is_free(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    rows, cols = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(rows, top + h + pad)
    c1 = min(cols, left + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _reserve_area(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    rows, cols = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(rows, top + h + pad)
    c1 = min(cols, left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _band_max_start(length: int, extent: int, needs_gap_for_other_band: bool) -> int:
    return length - extent - 3 if needs_gap_for_other_band else length - extent


def _starts_possible(length: int, count: int, extent: int, needs_gap_for_other_band: bool) -> bool:
    if count == 0:
        return True
    max_start = _band_max_start(length, extent, needs_gap_for_other_band)
    if max_start < 0:
        return False
    total_len = max_start + extent
    min_span = extent * count + (count - 1)
    return total_len >= min_span


def _sample_band_starts(
    length: int,
    count: int,
    extent: int,
    needs_gap_for_other_band: bool,
) -> list[int] | None:
    if count == 0:
        return []
    if not _starts_possible(length, count, extent, needs_gap_for_other_band):
        return None

    max_start = _band_max_start(length, extent, needs_gap_for_other_band)
    total_len = max_start + extent
    min_span = extent * count + (count - 1)
    slack = total_len - min_span
    extras = [0] * (count + 1)
    for _ in range(slack):
        extras[random.randrange(count + 1)] += 1

    starts = []
    cursor = extras[0]
    for i in range(count):
        starts.append(cursor)
        cursor += extent + 1 + extras[i + 1]
    return starts


def _feasible_lengths(count: int, extent: int, needs_gap_for_other_band: bool) -> list[int]:
    return [
        length
        for length in range(SIZE_MIN, SIZE_MAX + 1)
        if _starts_possible(length, count, extent, needs_gap_for_other_band)
    ]


def _place_fragments(grid: np.ndarray, occupied: np.ndarray, color: int) -> bool:
    positions = [(r, c) for r in range(grid.shape[0] - 1) for c in range(grid.shape[1] - 1)]
    random.shuffle(positions)
    placed = 0
    while positions and placed < 2:
        top, left = positions.pop()
        if not _area_is_free(occupied, top, left, 2, 2):
            continue
        patch = _make_l_patch(color, random.randrange(4))
        _paste(grid, patch, top, left)
        _reserve_area(occupied, top, left, 2, 2)
        placed += 1
    return placed == 2


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(COLOR_POOL, n_colors)
    dims_by_color = {color: random.choice(RECT_DIMS) for color in colors}

    horizontal_colors = [color for color in colors if dims_by_color[color] == (2, 3)]
    vertical_colors = [color for color in colors if dims_by_color[color] == (3, 2)]

    heights = _feasible_lengths(len(vertical_colors), 3, bool(horizontal_colors))
    widths = _feasible_lengths(len(horizontal_colors), 3, bool(vertical_colors))
    if not heights or not widths:
        return None
    h = random.choice(heights)
    w = random.choice(widths)

    h_starts = _sample_band_starts(w, len(horizontal_colors), 3, bool(vertical_colors))
    v_starts = _sample_band_starts(h, len(vertical_colors), 3, bool(horizontal_colors))
    if h_starts is None or v_starts is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    expected = np.full((h, w), BG, dtype=int)

    random.shuffle(horizontal_colors)
    random.shuffle(vertical_colors)

    for color, start in zip(horizontal_colors, h_starts):
        ar, ac = h - 2, start
        grid[ar, ac] = color
        expected[ar : ar + 2, ac : ac + 3] = color
        _reserve_area(occupied, ar, ac, 2, 3)

    for color, start in zip(vertical_colors, v_starts):
        ar, ac = start, w - 2
        grid[ar, ac] = color
        expected[ar : ar + 3, ac : ac + 2] = color
        _reserve_area(occupied, ar, ac, 3, 2)

    color_order = colors[:]
    random.shuffle(color_order)
    for color in color_order:
        if not _place_fragments(grid, occupied, color):
            return None

    try:
        actual = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
