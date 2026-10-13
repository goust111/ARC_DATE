from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).ravel()))
    assert counts, "expected at least one value"
    best_count = max(counts.values())
    best = [color for color, count in counts.items() if count == best_count]
    assert len(best) == 1, "most common color must be unique"
    return int(best[0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> BBox:
    coords = np.argwhere(mask)
    assert coords.size > 0, "cannot take bbox of an empty mask"
    return _bbox_coords(coords)


def _crop_bool(mask: np.ndarray) -> tuple[np.ndarray, Coord, BBox]:
    b = _bbox(mask)
    r0, c0, r1, c1 = b
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0), b


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int) -> list[Coord]:
    h, w = shape
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out = []
    for dr, dc in steps:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for start_r, start_c in np.argwhere(mask):
        sr, sc = int(start_r), int(start_c)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[Coord] = deque([(sr, sc)])
        coords: list[Coord] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in _neighbors(r, c, (h, w), connectivity):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rs, cs = coords[:, 0], coords[:, 1]
    return bool((rs == 0).any() or (rs == h - 1).any() or (cs == 0).any() or (cs == w - 1).any())


def _coords_form_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    return int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _assemble_tile(
    a_mask: np.ndarray,
    b_mask: np.ndarray,
    a_color: int,
    b_color: int,
) -> tuple[np.ndarray, Coord, Coord]:
    """Return the unique zero-gap rectangular assembly and its origin relative to A."""
    a_pts = [tuple(map(int, p)) for p in np.argwhere(a_mask)]
    b_pts = [tuple(map(int, p)) for p in np.argwhere(b_mask)]
    assert a_pts and b_pts, "both pieces must be non-empty"
    a_set = set(a_pts)
    ah, aw = a_mask.shape
    bh, bw = b_mask.shape

    best_holes: int | None = None
    best: list[tuple[int, int, BBox]] = []
    for dr in range(-bh, ah + 1):
        for dc in range(-bw, aw + 1):
            shifted_b = [(r + dr, c + dc) for r, c in b_pts]
            if any(pt in a_set for pt in shifted_b):
                continue
            coords = a_pts + shifted_b
            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            holes = area - len(coords)
            if best_holes is None or holes < best_holes:
                best_holes = holes
                best = [(dr, dc, (r0, c0, r1, c1))]
            elif holes == best_holes:
                best.append((dr, dc, (r0, c0, r1, c1)))

    assert best_holes == 0, "pieces must have a gap-free rectangular fit"
    assert len(best) == 1, "gap-free densest fit must be unique"
    dr, dc, (r0, c0, r1, c1) = best[0]
    tile = np.full((r1 - r0 + 1, c1 - c0 + 1), -1, dtype=int)
    for r, c in a_pts:
        tile[r - r0, c - c0] = int(a_color)
    for r, c in b_pts:
        tile[r + dr - r0, c + dc - c0] = int(b_color)
    assert not np.any(tile < 0)
    return tile, (int(r0), int(c0)), (int(dr), int(dc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the outside loose piece with the cavity seed, then tile the cavity with that unit.

    Input structure:
    - A unique dominant wall color forms a solid rectangular annulus around one rectangular cavity.
    - The cavity contains one monochrome seed fragment on a uniform empty cavity color.
    - The exterior background contains one loose monochrome fragment of a different color.
    - The two fragments can be translated, without rotation, into one gap-free rectangular tile.

    Rule:
    - Identify the cavity fragment as piece A and the exterior loose fragment as piece B.
    - Find the unique non-overlapping translation of B relative to A whose cropped union has the fewest empty cells, requiring a solid rectangle.
    - Repeat that assembled two-color tile across the whole cavity using piece A's current position as the phase anchor, clipping at the cavity boundary.
    - Erase the original exterior loose fragment and leave the wall and all other exterior background cells unchanged.

    Key insight:
    - The outside fragment is the missing half of the periodic tile; the seed fragment fixes both the complementary fit and the tiling phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    wall = _unique_mode(g)
    non_wall = g != wall
    comps = _components(non_wall, connectivity=4)
    assert comps, "expected non-wall regions"

    outside_mask = np.zeros(g.shape, dtype=bool)
    cavity_records: list[tuple[np.ndarray, BBox]] = []
    for coords in comps:
        if _touches_border(coords, (h, w)):
            outside_mask[coords[:, 0], coords[:, 1]] = True
        else:
            assert _coords_form_rectangle(coords), "enclosed non-wall region must be rectangular"
            cavity_records.append((coords, _bbox_coords(coords)))

    assert len(cavity_records) == 1, "expected exactly one enclosed rectangular cavity"
    cavity_coords, cavity_bbox = cavity_records[0]
    cavity_mask = np.zeros(g.shape, dtype=bool)
    cavity_mask[cavity_coords[:, 0], cavity_coords[:, 1]] = True
    cr0, cc0, cr1, cc1 = cavity_bbox
    assert np.all(cavity_mask[cr0 : cr1 + 1, cc0 : cc1 + 1])

    cavity = g[cr0 : cr1 + 1, cc0 : cc1 + 1]
    empty = _unique_mode(cavity)
    seed_colors = set(map(int, cavity.ravel())) - {int(empty)}
    assert len(seed_colors) == 1, "cavity must contain exactly one fragment color"
    a_color = int(next(iter(seed_colors)))
    a_global = (g == a_color) & cavity_mask
    assert len(_components(a_global, connectivity=8)) == 1

    assert outside_mask.any(), "expected exterior background"
    outside_bg = _unique_mode(g[outside_mask])
    outside_colors = set(map(int, g[outside_mask])) - {int(outside_bg)}
    assert len(outside_colors) == 1, "exterior must contain exactly one loose fragment color"
    b_color = int(next(iter(outside_colors)))
    assert b_color != a_color
    b_global = (g == b_color) & outside_mask
    assert b_global.any()
    assert not ((g == b_color) & cavity_mask).any()
    assert len(_components(b_global, connectivity=8)) == 1

    a_crop, a_top, _ = _crop_bool(a_global)
    b_crop, _, _ = _crop_bool(b_global)
    tile, rel_origin, _ = _assemble_tile(a_crop, b_crop, a_color, b_color)
    tile_h, tile_w = tile.shape
    phase_r = int(a_top[0] + rel_origin[0])
    phase_c = int(a_top[1] + rel_origin[1])

    out = g.copy()
    out[b_global] = int(outside_bg)

    rr = np.arange(cr0, cr1 + 1)[:, None]
    cc = np.arange(cc0, cc1 + 1)[None, :]
    out[cr0 : cr1 + 1, cc0 : cc1 + 1] = tile[
        (rr - phase_r) % tile_h,
        (cc - phase_c) % tile_w,
    ]
    return out


# ------------------------------ generator ------------------------------

_MAX_ATTEMPTS = 600


def _is_connected(mask: np.ndarray, connectivity: int = 4) -> bool:
    return bool(mask.any()) and len(_components(mask, connectivity=connectivity)) == 1


def _solid_crop(mask: np.ndarray) -> bool:
    crop, _, _ = _crop_bool(mask)
    return bool(np.all(crop))


def _grow_subset(height: int, width: int, target: int) -> set[Coord] | None:
    start = (random.randrange(height), random.randrange(width))
    cells: set[Coord] = {start}
    frontier: set[Coord] = set(_neighbors(start[0], start[1], (height, width), 4))
    for _ in range(height * width * 20):
        if len(cells) >= target:
            return cells
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors(cell[0], cell[1], (height, width), 4):
            if nb not in cells:
                frontier.add(nb)
    return None


def _transform_pair(a_mask: np.ndarray, b_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randrange(4)
    a = np.rot90(a_mask, k)
    b = np.rot90(b_mask, k)
    if random.random() < 0.5:
        a = np.flipud(a)
        b = np.flipud(b)
    if random.random() < 0.5:
        a = np.fliplr(a)
        b = np.fliplr(b)
    return a.copy(), b.copy()


def _random_tile_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(500):
        tile_h = random.randint(2, 5)
        tile_w = random.randint(3, 6)
        area = tile_h * tile_w
        lo = max(2, area // 4)
        hi = min(area - 2, (3 * area) // 4)
        if lo > hi:
            continue
        target = random.randint(lo, hi)
        cells = _grow_subset(tile_h, tile_w, target)
        if cells is None:
            continue

        a_mask = np.zeros((tile_h, tile_w), dtype=bool)
        for r, c in cells:
            a_mask[r, c] = True
        b_mask = ~a_mask
        a_mask, b_mask = _transform_pair(a_mask, b_mask)

        if not (_is_connected(a_mask, 4) and _is_connected(b_mask, 4)):
            continue
        if int(a_mask.sum()) < 2 or int(b_mask.sum()) < 2:
            continue
        if _solid_crop(a_mask) and _solid_crop(b_mask):
            continue

        a_crop, _, a_bbox = _crop_bool(a_mask)
        b_crop, _, _ = _crop_bool(b_mask)
        try:
            tile, rel_origin, _ = _assemble_tile(a_crop, b_crop, 1, 2)
        except AssertionError:
            continue

        expected = np.full(a_mask.shape, 2, dtype=int)
        expected[a_mask] = 1
        expected_origin = (-a_bbox[0], -a_bbox[1])
        if tile.shape != expected.shape:
            continue
        if rel_origin != expected_origin:
            continue
        if not np.array_equal(tile, expected):
            continue
        return a_mask, b_mask, expected
    return None


def _choose_length(tile_len: int, piece_len: int, upper: int) -> int | None:
    low = max(piece_len + 3, tile_len * 2)
    choices = list(range(low, upper + 1))
    if not choices:
        return None
    non_multiple = [v for v in choices if v % tile_len != 0]
    if non_multiple and random.random() < 0.7:
        return random.choice(non_multiple)
    return random.choice(choices)


def _dominant_is(grid: np.ndarray, color: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[int(color)] > max(count for col, count in counts.items() if int(col) != int(color))


def _render_expected(
    inp: np.ndarray,
    tile: np.ndarray,
    cavity_bbox: BBox,
    phase: Coord,
    loose_mask: np.ndarray,
    outside_bg: int,
) -> np.ndarray:
    out = inp.copy()
    out[loose_mask] = int(outside_bg)
    cr0, cc0, cr1, cc1 = cavity_bbox
    tile_h, tile_w = tile.shape
    rr = np.arange(cr0, cr1 + 1)[:, None]
    cc = np.arange(cc0, cc1 + 1)[None, :]
    out[cr0 : cr1 + 1, cc0 : cc1 + 1] = tile[
        (rr - phase[0]) % tile_h,
        (cc - phase[1]) % tile_w,
    ]
    return out


def _place_loose_piece(
    grid: np.ndarray,
    side: str,
    outer_bbox: BBox,
    piece: np.ndarray,
    color: int,
) -> np.ndarray | None:
    h, w = grid.shape
    or0, oc0, or1, oc1 = outer_bbox
    ph, pw = piece.shape

    if side == "top":
        r_min, r_max = 0, or0 - ph - 1
        c_min, c_max = 0, w - pw
    elif side == "bottom":
        r_min, r_max = or1 + 2, h - ph
        c_min, c_max = 0, w - pw
    elif side == "left":
        r_min, r_max = 0, h - ph
        c_min, c_max = 0, oc0 - pw - 1
    else:
        r_min, r_max = 0, h - ph
        c_min, c_max = oc1 + 2, w - pw

    if r_min > r_max or c_min > c_max:
        return None
    top = random.randint(r_min, r_max)
    left = random.randint(c_min, c_max)
    loose_mask = np.zeros(grid.shape, dtype=bool)
    pts = np.argwhere(piece)
    for rr, cc in pts:
        r, c = top + int(rr), left + int(cc)
        if grid[r, c] != grid[0, 0]:
            return None
        grid[r, c] = int(color)
        loose_mask[r, c] = True
    return loose_mask


def _gen_attempt() -> np.ndarray | None:
    masks = _random_tile_masks()
    if masks is None:
        return None
    a_mask, b_mask, _ = masks
    tile_h, tile_w = a_mask.shape
    a_crop, _, a_bbox = _crop_bool(a_mask)
    b_crop, _, _ = _crop_bool(b_mask)
    ah = a_bbox[2] - a_bbox[0] + 1
    aw = a_bbox[3] - a_bbox[1] + 1

    cavity_h = _choose_length(tile_h, ah, 15)
    cavity_w = _choose_length(tile_w, aw, 17)
    if cavity_h is None or cavity_w is None:
        return None

    min_phase_r = 1 - a_bbox[0]
    max_phase_r = cavity_h - 2 - a_bbox[2]
    min_phase_c = 1 - a_bbox[1]
    max_phase_c = cavity_w - 2 - a_bbox[3]
    if min_phase_r > max_phase_r or min_phase_c > max_phase_c:
        return None
    rel_phase_r = random.randint(min_phase_r, max_phase_r)
    rel_phase_c = random.randint(min_phase_c, max_phase_c)

    colors = list(range(10))
    random.shuffle(colors)
    wall = colors[0]
    outside_bg = colors[1]
    if random.random() < 0.4:
        empty = outside_bg
        next_idx = 2
    else:
        empty = colors[2]
        next_idx = 3
    a_color = colors[next_idx]
    b_color = colors[next_idx + 1]
    tile = np.full(a_mask.shape, int(b_color), dtype=int)
    tile[a_mask] = int(a_color)

    top_wall = random.randint(2, 4)
    bottom_wall = random.randint(2, 4)
    left_wall = random.randint(2, 4)
    right_wall = random.randint(2, 4)
    outer_h = cavity_h + top_wall + bottom_wall
    outer_w = cavity_w + left_wall + right_wall

    side = random.choice(["top", "bottom", "left", "right"])
    margins = {
        "top": random.randint(1, 2),
        "bottom": random.randint(1, 2),
        "left": random.randint(1, 2),
        "right": random.randint(1, 2),
    }
    if side == "top":
        margins["top"] = max(margins["top"], b_crop.shape[0] + 2)
    elif side == "bottom":
        margins["bottom"] = max(margins["bottom"], b_crop.shape[0] + 2)
    elif side == "left":
        margins["left"] = max(margins["left"], b_crop.shape[1] + 2)
    else:
        margins["right"] = max(margins["right"], b_crop.shape[1] + 2)

    height = outer_h + margins["top"] + margins["bottom"]
    width = outer_w + margins["left"] + margins["right"]
    if height > 30 or width > 30:
        return None

    inp = np.full((height, width), int(outside_bg), dtype=int)
    or0, oc0 = margins["top"], margins["left"]
    or1, oc1 = or0 + outer_h - 1, oc0 + outer_w - 1
    inp[or0 : or1 + 1, oc0 : oc1 + 1] = int(wall)

    cr0 = or0 + top_wall
    cc0 = oc0 + left_wall
    cr1 = cr0 + cavity_h - 1
    cc1 = cc0 + cavity_w - 1
    inp[cr0 : cr1 + 1, cc0 : cc1 + 1] = int(empty)

    phase = (cr0 + rel_phase_r, cc0 + rel_phase_c)
    for rr, cc in np.argwhere(a_mask):
        inp[phase[0] + int(rr), phase[1] + int(cc)] = int(a_color)

    loose_mask = _place_loose_piece(inp, side, (or0, oc0, or1, oc1), b_crop, b_color)
    if loose_mask is None:
        return None
    if not _dominant_is(inp, wall):
        return None

    expected = _render_expected(inp, tile, (cr0, cc0, cr1, cc1), phase, loose_mask, outside_bg)
    if np.array_equal(inp, expected):
        return None
    cavity_out = expected[cr0 : cr1 + 1, cc0 : cc1 + 1]
    if not (np.any(cavity_out == a_color) and np.any(cavity_out == b_color)):
        return None
    if np.any(expected[loose_mask] != outside_bg):
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid two-piece tiling input")
