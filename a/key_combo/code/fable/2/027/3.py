from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, find_components, in_bounds, neighbors4


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_POLY_AREA = 8
GEN_RETRIES = 400


def _normalize_shape(coords) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))


def _shape_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _shape_area(coords: tuple[tuple[int, int], ...]) -> int:
    return len(coords)


def _shape_orientations(coords) -> tuple[tuple[tuple[int, int], ...], ...]:
    pts = tuple((int(r), int(c)) for r, c in coords)
    out = set()
    for k in range(8):
        transformed = []
        for r, c in pts:
            if k == 0:
                pt = (r, c)
            elif k == 1:
                pt = (r, -c)
            elif k == 2:
                pt = (-r, c)
            elif k == 3:
                pt = (-r, -c)
            elif k == 4:
                pt = (c, r)
            elif k == 5:
                pt = (c, -r)
            elif k == 6:
                pt = (-c, r)
            else:
                pt = (-c, -r)
            transformed.append(pt)
        out.add(_normalize_shape(transformed))
    return tuple(sorted(out))


def _canonical_shape(coords) -> tuple[tuple[int, int], ...]:
    return _shape_orientations(coords)[0]


@lru_cache(maxsize=None)
def _polyominoes(area: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    area = int(area)
    if area == 1:
        return (((0, 0),),)

    seen = set()
    out = []
    for poly in _polyominoes(area - 1):
        cells = set(poly)
        frontier = set()
        for r, c in poly:
            for dr, dc in DIR4:
                nb = (r + dr, c + dc)
                if nb not in cells:
                    frontier.add(nb)
        for nb in frontier:
            cand = _canonical_shape(cells | {nb})
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    return tuple(sorted(out))


@lru_cache(maxsize=None)
def _breach_catalog(width: int, max_height: int = 5) -> dict[int, tuple[tuple[tuple[int, int], ...], ...]]:
    width = int(width)
    max_height = int(max_height)
    out: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {}
    for area in range(width, MAX_POLY_AREA + 1):
        shapes = []
        seen = set()
        for poly in _polyominoes(area):
            for orient in _shape_orientations(poly):
                h, w = _shape_dims(orient)
                if w == width and h <= max_height and orient not in seen:
                    seen.add(orient)
                    shapes.append(orient)
        if shapes:
            out[area] = tuple(sorted(shapes))
    return out


@lru_cache(maxsize=None)
def _piece_catalog(area: int, max_h: int = 5, max_w: int = 5) -> tuple[tuple[tuple[int, int], ...], ...]:
    area = int(area)
    max_h = int(max_h)
    max_w = int(max_w)
    shapes = []
    seen = set()
    for poly in _polyominoes(area):
        for orient in _shape_orientations(poly):
            h, w = _shape_dims(orient)
            if h <= max_h and w <= max_w and orient not in seen:
                seen.add(orient)
                shapes.append(orient)
    return tuple(sorted(shapes))


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], anchor: tuple[int, int], color: int) -> None:
    r0, c0 = map(int, anchor)
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = int(color)


def _absolute_coords(shape: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0 = map(int, anchor)
    return tuple(sorted((r0 + dr, c0 + dc) for dr, dc in shape))


def _mark_occupied(occupied: np.ndarray, shape: tuple[tuple[int, int], ...], anchor: tuple[int, int], pad: int = 1) -> None:
    h, w = occupied.shape
    r0, c0 = map(int, anchor)
    for dr, dc in shape:
        rr = r0 + dr
        cc = c0 + dc
        for pr in range(rr - pad, rr + pad + 1):
            for pc in range(cc - pad, cc + pad + 1):
                if 0 <= pr < h and 0 <= pc < w:
                    occupied[pr, pc] = True


def _can_place(
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    anchor: tuple[int, int],
    height: int,
    width: int,
    pad: int = 1,
) -> bool:
    r0, c0 = map(int, anchor)
    for dr, dc in shape:
        rr = r0 + dr
        cc = c0 + dc
        if not (0 <= rr < height and 0 <= cc < width):
            return False
        for pr in range(rr - pad, rr + pad + 1):
            for pc in range(cc - pad, cc + pad + 1):
                if 0 <= pr < height and 0 <= pc < width and occupied[pr, pc]:
                    return False
    return True


def _sample_anchor(
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    height: int,
    width: int,
    rng: random.Random,
    pad: int = 1,
) -> tuple[int, int] | None:
    sh, sw = _shape_dims(shape)
    anchors = [(r, c) for r in range(height - sh + 1) for c in range(width - sw + 1)]
    rng.shuffle(anchors)
    for anchor in anchors:
        if _can_place(occupied, shape, anchor, height, width, pad=pad):
            return anchor
    return None


def _flood_from_source(out: np.ndarray, source: tuple[int, int], fill_color: int) -> None:
    sr, sc = map(int, source)
    q = deque([(sr, sc)])
    seen = {(sr, sc)}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(out.shape, nr, nc) or (nr, nc) in seen:
                continue
            if int(out[nr, nc]) != Color.BLACK:
                continue
            out[nr, nc] = int(fill_color)
            seen.add((nr, nc))
            q.append((nr, nc))


def _wall_color(g: np.ndarray) -> int:
    counts = Counter(map(int, g.ravel()))
    fg = [(n, color) for color, n in counts.items() if color != Color.BLACK]
    assert fg, "expected at least one foreground color"
    max_count = max(n for n, _color in fg)
    colors = [color for n, color in fg if n == max_count]
    assert len(colors) == 1, "wall color must be uniquely most frequent"
    return int(colors[0])


def _parse_scene(g: np.ndarray) -> dict:
    wall_color = _wall_color(g)
    wall_mask = g == wall_color
    wall_cols = sorted({int(c) for _r, c in np.argwhere(wall_mask)})
    assert wall_cols and wall_cols == list(range(wall_cols[0], wall_cols[-1] + 1))
    c0 = wall_cols[0]
    c1 = wall_cols[-1]
    wall_width = c1 - c0 + 1

    breach_shapes = []
    for comp in components_mask(g[:, c0 : c1 + 1] == Color.BLACK, connectivity=4):
        cols = comp[:, 1]
        if int(cols.min()) == 0 and int(cols.max()) == wall_width - 1:
            breach_shapes.append(tuple(sorted((int(r), int(c + c0)) for r, c in comp)))

    objects = find_components(g, bg=Color.BLACK, connectivity=4, monochromatic=True)
    left_objects = [
        obj
        for obj in objects
        if int(obj["color"]) != wall_color and int(obj["bbox"][3]) < c0
    ]
    source_objs = [obj for obj in left_objects if int(obj["area"]) == 1]
    piece_objs = [obj for obj in left_objects if int(obj["area"]) > 1]
    assert len(source_objs) == 1, "expected exactly one source pixel"

    source_obj = source_objs[0]
    source_pos = tuple(map(int, source_obj["coords"][0]))
    return {
        "wall_color": wall_color,
        "wall_c0": c0,
        "wall_c1": c1,
        "breaches": breach_shapes,
        "pieces": piece_objs,
        "source_pos": source_pos,
        "source_color": int(source_obj["color"]),
    }


def _piece_signature(obj: dict) -> tuple[tuple[int, int], ...]:
    return _canonical_shape(tuple(map(tuple, obj["coords"])))


def _coords_signature(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _canonical_shape(coords)


def _coords_area(coords: tuple[tuple[int, int], ...]) -> int:
    return len(coords)


def _alt_flood_only(inp: np.ndarray, source_pos: tuple[int, int], source_color: int) -> np.ndarray:
    out = inp.copy()
    _flood_from_source(out, source_pos, source_color)
    return out


def _alt_close_all_breaches(inp: np.ndarray, parsed: dict) -> np.ndarray:
    out = inp.copy()
    for breach in parsed["breaches"]:
        for r, c in breach:
            out[r, c] = parsed["wall_color"]
    _flood_from_source(out, parsed["source_pos"], parsed["source_color"])
    return out


def _alt_erase_all_pieces(inp: np.ndarray, parsed: dict) -> np.ndarray:
    out = inp.copy()
    breach_by_area = {_coords_area(breach): breach for breach in parsed["breaches"]}
    for piece in parsed["pieces"]:
        for r, c in piece["coords"]:
            out[int(r), int(c)] = Color.BLACK
        if int(piece["area"]) in breach_by_area:
            for r, c in breach_by_area[int(piece["area"])]:
                out[r, c] = parsed["wall_color"]
    _flood_from_source(out, parsed["source_pos"], parsed["source_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair every wall breach whose area matches a loose piece, then flood water through the one unrepaired breach.

    Input structure:
    - Black is the background, and a multi-pixel-thick vertical wall band of one color splits the grid into left and right zones.
    - Inside the wall band, several black 4-connected breach shapes cut all the way through the band; their areas are all distinct.
    - In the left zone, one non-wall single pixel is the water source, and the other non-wall monochrome 4-connected objects are loose pieces.

    Rule:
    - Identify the wall color as the unique most frequent non-black color, then collect every black wall-band component that touches both wall-band sides as a breach.
    - For each loose piece, if its area equals the area of a unique breach, erase the piece from its original position and fill that entire breach with the wall color; pieces with no area match stay in place.
    - Using the repaired wall and the remaining pieces as obstacles, start a 4-connected flood from the source pixel and recolor every reachable black cell to the source color.

    Key insight:
    - Shape is a distractor: pieces repair breaches by area only, and that repair step determines which single breach still leaks water into the right side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    parsed = _parse_scene(g)
    wall_color = parsed["wall_color"]
    breach_by_area = {_coords_area(breach): breach for breach in parsed["breaches"]}

    out = g.copy()
    for piece in parsed["pieces"]:
        area = int(piece["area"])
        if area not in breach_by_area:
            continue
        for r, c in piece["coords"]:
            out[int(r), int(c)] = Color.BLACK
        for r, c in breach_by_area[area]:
            out[r, c] = wall_color

    _flood_from_source(out, parsed["source_pos"], parsed["source_color"])
    return out


def _expected_from_layout(layout: dict) -> np.ndarray:
    out = layout["input"].copy()
    for piece in layout["pieces"]:
        if piece["matched_breach_idx"] is None:
            continue
        for r, c in piece["coords"]:
            out[r, c] = Color.BLACK
        breach = layout["breaches"][piece["matched_breach_idx"]]
        for r, c in breach["coords"]:
            out[r, c] = layout["wall_color"]
    _flood_from_source(out, layout["source_pos"], layout["source_color"])
    return out


def _semantic_check(layout: dict, expected: np.ndarray) -> bool:
    inp = layout["input"]
    try:
        parsed = _parse_scene(inp)
    except Exception:
        return False

    if not np.array_equal(main(inp), expected):
        return False

    breaches = parsed["breaches"]
    pieces = parsed["pieces"]
    breach_areas = [_coords_area(breach) for breach in breaches]
    if len(breaches) < 2 or len(set(breach_areas)) != len(breach_areas):
        return False

    matched_pieces = [piece for piece in pieces if int(piece["area"]) in set(breach_areas)]
    unmatched_pieces = [piece for piece in pieces if int(piece["area"]) not in set(breach_areas)]
    if len(matched_pieces) != len(breaches) - 1:
        return False
    if len(unmatched_pieces) < 1:
        return False

    matched_areas = {int(piece["area"]) for piece in matched_pieces}
    if len(set(breach_areas) - matched_areas) != 1:
        return False

    breach_sigs = {_coords_signature(breach) for breach in breaches}
    if any(_piece_signature(piece) in breach_sigs for piece in pieces):
        return False

    source_color = parsed["source_color"]
    c1 = parsed["wall_c1"]
    c0 = parsed["wall_c0"]
    if not np.any(expected[:, c1 + 1 :] == source_color):
        return False
    if not np.any((inp[:, :c0] == Color.BLACK) & (expected[:, :c0] == source_color)):
        return False

    if np.array_equal(_alt_flood_only(inp, parsed["source_pos"], source_color), expected):
        return False
    if np.array_equal(_alt_close_all_breaches(inp, parsed), expected):
        return False
    if np.array_equal(_alt_erase_all_pieces(inp, parsed), expected):
        return False
    return True


def _make_piece_specs(
    breach_shapes: list[tuple[tuple[int, int], ...]],
    breach_areas: list[int],
    open_breach_idx: int,
    rng: random.Random,
) -> list[dict] | None:
    breach_sigs = {_canonical_shape(shape) for shape in breach_shapes}
    pieces = []
    for idx, area in enumerate(breach_areas):
        if idx == open_breach_idx:
            continue
        candidates = [
            shape
            for shape in _piece_catalog(area)
            if _canonical_shape(shape) not in breach_sigs
        ]
        if not candidates:
            return None
        pieces.append(
            {
                "shape": rng.choice(candidates),
                "area": area,
                "matched_breach_idx": idx,
            }
        )

    extra_pool = [area for area in range(2, MAX_POLY_AREA + 1) if area not in set(breach_areas) and _piece_catalog(area)]
    if not extra_pool:
        return None
    n_extra = 1 if len(extra_pool) == 1 else (2 if rng.random() < 0.45 else 1)
    n_extra = min(n_extra, len(extra_pool))
    for area in rng.sample(extra_pool, n_extra):
        pieces.append(
            {
                "shape": rng.choice(_piece_catalog(area)),
                "area": area,
                "matched_breach_idx": None,
            }
        )
    return pieces


def _build_attempt(rng: random.Random) -> dict | None:
    wall_width = rng.randint(3, 5)
    breach_catalog = _breach_catalog(wall_width)
    area_pool = sorted(breach_catalog)
    max_breaches = min(4, len(area_pool))
    if max_breaches < 2:
        return None

    n_breaches = rng.choice([k for k in (2, 3, 4) if k <= max_breaches])
    breach_areas = rng.sample(area_pool, n_breaches)
    breach_shapes = [rng.choice(breach_catalog[area]) for area in breach_areas]

    piece_specs = _make_piece_specs(breach_shapes, breach_areas, rng.randrange(n_breaches), rng)
    if piece_specs is None:
        return None

    breach_heights = [_shape_dims(shape)[0] for shape in breach_shapes]
    min_h = sum(breach_heights) + (n_breaches + 1)
    h_lo = max(16, min_h)
    h_hi = min(28, min_h + 7)
    if h_lo > h_hi:
        return None
    height = rng.randint(h_lo, h_hi)

    gap_slots = [1] * (n_breaches + 1)
    extra_rows = height - (sum(breach_heights) + sum(gap_slots))
    while extra_rows > 0:
        gap_slots[rng.randrange(len(gap_slots))] += 1
        extra_rows -= 1

    row_starts = []
    cursor = gap_slots[0]
    for i, shape in enumerate(breach_shapes):
        row_starts.append(cursor)
        cursor += _shape_dims(shape)[0]
        if i < n_breaches - 1:
            cursor += gap_slots[i + 1]

    left_width = rng.randint(10, 14)
    right_hi = min(10, 30 - left_width - wall_width)
    if right_hi < 6:
        return None
    right_width = rng.randint(6, right_hi)
    width = left_width + wall_width + right_width
    wall_c0 = left_width

    pieces = [dict(spec) for spec in piece_specs]
    available_colors = rng.sample(list(Color.NOT_BLACK), 2 + len(pieces))
    wall_color = int(available_colors[0])
    source_color = int(available_colors[1])
    for piece, color in zip(pieces, available_colors[2:]):
        piece["color"] = int(color)

    place_width = left_width - 1
    occupied = np.zeros((height, place_width), dtype=bool)
    pieces.sort(key=lambda piece: (_shape_area(piece["shape"]), _shape_dims(piece["shape"])[0] * _shape_dims(piece["shape"])[1]), reverse=True)
    for piece in pieces:
        anchor = _sample_anchor(occupied, piece["shape"], height, place_width, rng, pad=1)
        if anchor is None:
            return None
        piece["anchor"] = anchor
        piece["coords"] = _absolute_coords(piece["shape"], anchor)
        _mark_occupied(occupied, piece["shape"], anchor, pad=1)

    source_anchor = _sample_anchor(occupied, ((0, 0),), height, place_width, rng, pad=1)
    if source_anchor is None:
        return None
    source_pos = tuple(map(int, source_anchor))

    inp = np.full((height, width), Color.BLACK, dtype=int)
    inp[:, wall_c0 : wall_c0 + wall_width] = wall_color

    breaches = []
    for area, shape, row0 in zip(breach_areas, breach_shapes, row_starts):
        coords = _absolute_coords(shape, (row0, wall_c0))
        for r, c in coords:
            inp[r, c] = Color.BLACK
        breaches.append({"area": area, "shape": shape, "coords": coords})

    for piece in pieces:
        _paint_shape(inp, piece["shape"], piece["anchor"], piece["color"])

    inp[source_pos] = source_color

    return {
        "input": inp,
        "wall_color": wall_color,
        "source_color": source_color,
        "source_pos": source_pos,
        "breaches": breaches,
        "pieces": pieces,
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        layout = _build_attempt(rng)
        if layout is None:
            continue
        expected = _expected_from_layout(layout)
        if _semantic_check(layout, expected):
            return layout["input"]
    raise RuntimeError("failed to generate a valid wall-repair leak puzzle")
