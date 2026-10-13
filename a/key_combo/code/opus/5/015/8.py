from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


BG = 5
BLACK = 0
MARKER_L_COLOR = 6
PIECE_COLORS = (1, 2, 3, 4, 7, 8, 9)

GEN_ATTEMPTS = 400
PLACE_TRIES = 300
UNIQUE_LIMIT = 2

SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
)


def _shape_key(coords: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    pts = np.asarray(coords, dtype=int)
    origin = pts.min(axis=0)
    norm = pts - origin
    h = int(norm[:, 0].max()) + 1
    w = int(norm[:, 1].max()) + 1
    cells = tuple(sorted((int(r), int(c)) for r, c in norm))
    return h, w, cells


def _touches_bbox_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        (coords[:, 0] == 0).any()
        or (coords[:, 0] == h - 1).any()
        or (coords[:, 1] == 0).any()
        or (coords[:, 1] == w - 1).any()
    )


def _corner_kind_from_missing(missing: tuple[int, int], marker_bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = marker_bbox
    if missing == (r0, c0):
        return "BR"
    if missing == (r0, c1):
        return "BL"
    if missing == (r1, c0):
        return "TR"
    return "TL"


def _find_l_marker(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int], str]:
    comps = components_mask(grid == MARKER_L_COLOR, connectivity=4)
    for coords in comps:
        if int(coords.shape[0]) != 3:
            continue
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        cells = {tuple(map(int, p)) for p in coords}
        all_cells = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        missing = tuple((all_cells - cells).pop())
        return coords, missing, box, _corner_kind_from_missing(missing, box)
    raise AssertionError("missing L marker")


def _decompose_input(grid: np.ndarray) -> tuple[list[np.ndarray], list[dict], tuple[int, int], str]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG

    l_coords, missing, _box, corner_kind = _find_l_marker(g)
    l_set = {tuple(map(int, p)) for p in l_coords}
    occ_mask = g != bg
    components = components_mask(occ_mask, connectivity=4)

    pieces: list[np.ndarray] = []
    markers: list[dict] = []
    for coords in components:
        coord_set = {tuple(map(int, p)) for p in coords}
        if coord_set == l_set:
            continue
        box = bbox_coords(coords)
        patch = crop_bbox(g, box)
        h, w = patch.shape
        area = int(coords.shape[0])
        if area == h * w and min(h, w) >= 4:
            pieces.append(patch)
            continue
        color_vals = {int(g[r, c]) for r, c in coord_set}
        assert len(color_vals) == 1
        markers.append(
            {
                "coords": coords,
                "color": next(iter(color_vals)),
                "bbox": box,
            }
        )

    assert 3 <= len(pieces) <= 4
    assert 2 <= len(markers) <= 3
    return pieces, markers, missing, corner_kind


def _rect_color(pieces: list[np.ndarray]) -> int:
    pixels: list[int] = []
    for piece in pieces:
        pixels.extend(map(int, piece.ravel()))
    return int(Counter(pixels).most_common(1)[0][0])


def _compat(v1: np.ndarray, v2: np.ndarray, rect_color: int) -> bool:
    mask = (v1 != rect_color) & (v2 != rect_color)
    return bool((v1[mask] == v2[mask]).all())


def _candidate_dims(total_area: int, grid_shape: tuple[int, int], missing: tuple[int, int], corner_kind: str) -> list[tuple[int, int]]:
    gh, gw = grid_shape
    mr, mc = missing

    def fits(h: int, w: int) -> bool:
        if corner_kind == "TL":
            r0, c0 = mr, mc
        elif corner_kind == "TR":
            r0, c0 = mr, mc - w + 1
        elif corner_kind == "BL":
            r0, c0 = mr - h + 1, mc
        else:
            r0, c0 = mr - h + 1, mc - w + 1
        return 0 <= r0 and 0 <= c0 and r0 + h <= gh and c0 + w <= gw

    dims = []
    for h in range(1, int(total_area ** 0.5) + 1):
        if total_area % h != 0:
            continue
        for hh, ww in ((h, total_area // h), (total_area // h, h)):
            if fits(int(hh), int(ww)):
                dims.append((int(hh), int(ww)))
    dims = sorted(set(dims), key=lambda x: (x[0] * x[1], x[0], x[1]))
    return dims


def _assemble_outputs(
    pieces: list[np.ndarray],
    rect_color: int,
    grid_shape: tuple[int, int],
    missing: tuple[int, int],
    corner_kind: str,
    *,
    limit: int = UNIQUE_LIMIT,
) -> list[np.ndarray]:
    total_area = int(sum(piece.shape[0] * piece.shape[1] for piece in pieces))
    dims = _candidate_dims(total_area, grid_shape, missing, corner_kind)
    outputs: dict[bytes, np.ndarray] = {}

    for h, w in dims:
        canvas = np.full((h, w), -1, dtype=int)
        used = [False] * len(pieces)
        order = sorted(range(len(pieces)), key=lambda i: (-pieces[i].shape[0] * pieces[i].shape[1], -pieces[i].shape[0], -pieces[i].shape[1], i))

        def boundary_ok(piece: np.ndarray, r0: int, c0: int) -> bool:
            ph, pw = piece.shape
            if r0 == 0 and not np.all(piece[0, :] == rect_color):
                return False
            if r0 + ph == h and not np.all(piece[-1, :] == rect_color):
                return False
            if c0 == 0 and not np.all(piece[:, 0] == rect_color):
                return False
            if c0 + pw == w and not np.all(piece[:, -1] == rect_color):
                return False
            return True

        def can_place(idx: int, r0: int, c0: int) -> bool:
            piece = pieces[idx]
            ph, pw = piece.shape
            if r0 + ph > h or c0 + pw > w:
                return False
            if (canvas[r0 : r0 + ph, c0 : c0 + pw] != -1).any():
                return False
            if not boundary_ok(piece, r0, c0):
                return False
            if c0 > 0:
                left = canvas[r0 : r0 + ph, c0 - 1]
                mask = left != -1
                if mask.any() and not _compat(left[mask], piece[:, 0][mask], rect_color):
                    return False
            if c0 + pw < w:
                right = canvas[r0 : r0 + ph, c0 + pw]
                mask = right != -1
                if mask.any() and not _compat(right[mask], piece[:, -1][mask], rect_color):
                    return False
            if r0 > 0:
                up = canvas[r0 - 1, c0 : c0 + pw]
                mask = up != -1
                if mask.any() and not _compat(up[mask], piece[0, :][mask], rect_color):
                    return False
            if r0 + ph < h:
                down = canvas[r0 + ph, c0 : c0 + pw]
                mask = down != -1
                if mask.any() and not _compat(down[mask], piece[-1, :][mask], rect_color):
                    return False
            return True

        def next_empty() -> tuple[int, int] | None:
            coords = np.argwhere(canvas == -1)
            if coords.size == 0:
                return None
            return tuple(map(int, coords[0]))

        def bt() -> None:
            if len(outputs) >= limit:
                return
            nxt = next_empty()
            if nxt is None:
                key = canvas.tobytes()
                outputs.setdefault(key, canvas.copy())
                return
            r0, c0 = nxt
            for idx in order:
                if used[idx]:
                    continue
                if not can_place(idx, r0, c0):
                    continue
                piece = pieces[idx]
                ph, pw = piece.shape
                used[idx] = True
                canvas[r0 : r0 + ph, c0 : c0 + pw] = piece
                bt()
                canvas[r0 : r0 + ph, c0 : c0 + pw] = -1
                used[idx] = False
                if len(outputs) >= limit:
                    return

        bt()
        if len(outputs) >= limit:
            break

    return list(outputs.values())


def _hole_components(rect: np.ndarray) -> list[np.ndarray]:
    holes = []
    for coords in components_mask(rect == BLACK, connectivity=4):
        if _touches_bbox_edge(coords, rect.shape):
            continue
        holes.append(coords)
    return holes


def _marker_shape_map(markers: list[dict]) -> dict[tuple[int, int, tuple[tuple[int, int], ...]], int]:
    by_shape: dict[tuple[int, int, tuple[tuple[int, int], ...]], set[int]] = {}
    for marker in markers:
        key = _shape_key(marker["coords"])
        by_shape.setdefault(key, set()).add(int(marker["color"]))
    shape_map = {}
    for key, colors in by_shape.items():
        if len(colors) == 1:
            shape_map[key] = next(iter(colors))
    return shape_map


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    pieces, markers, missing, corner_kind = _decompose_input(g)
    rect_color = _rect_color(pieces)
    assemblies = _assemble_outputs(pieces, rect_color, g.shape, missing, corner_kind, limit=UNIQUE_LIMIT)
    assert len(assemblies) == 1
    assembled = assemblies[0]
    holes = _hole_components(assembled)
    shape_map = _marker_shape_map(markers)
    out = assembled.copy()
    matched = 0
    for hole in holes:
        key = _shape_key(hole)
        if key not in shape_map:
            continue
        out[hole[:, 0], hole[:, 1]] = int(shape_map[key])
        matched += 1
    return {
        "pieces": pieces,
        "markers": markers,
        "missing": missing,
        "corner_kind": corner_kind,
        "rect_color": rect_color,
        "assembled": assembled,
        "holes": holes,
        "shape_map": shape_map,
        "output": out,
        "assembly_count": len(assemblies),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered rectangles, then fill each final black hole with the color of the matching external marker shape.

    Input structure:
    - The background is gray, with 3-4 scattered axis-aligned rectangular pieces whose cells are all non-gray.
    - A 3-cell magenta L marker indicates which corner the assembled rectangle is anchored to.
    - Separate small monochromatic marker shapes lie away from the pieces.

    Rule:
    - Identify the rectangular pieces and let the rectangle color be the most common color across all piece cells.
    - Translate the pieces without rotation so they tile one rectangle at the L-marked corner; along shared edges, rectangle-color cells are wildcards and every other paired edge color must match exactly.
    - In the assembled rectangle, find every internal 4-connected black region, normalize its shape, and recolor it with the color of the external marker that has the same normalized shape.
    - Leave unmatched holes black and return the assembled rectangle cropped to its bounding box.

    Key insight:
    - The relevant hole shapes are created only after the piece assembly, so the hole-to-marker matching must be done on the post-assembly rectangle rather than on any single fragment.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]


def _piece_layout(rng: random.Random) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    if rng.random() < 0.5:
        h1, h2 = rng.randint(4, 6), rng.randint(4, 6)
        w1, w2 = rng.randint(4, 6), rng.randint(4, 6)
        return h1 + h2, w1 + w2, [
            (0, 0, h1, w1),
            (0, w1, h1, w2),
            (h1, 0, h2, w1),
            (h1, w1, h2, w2),
        ]

    orient = rng.choice(("left", "right", "top", "bottom"))
    if orient in ("left", "right"):
        h1, h2 = rng.randint(4, 6), rng.randint(4, 6)
        w_big, w_side = rng.randint(4, 6), rng.randint(4, 6)
        h = h1 + h2
        if orient == "left":
            boxes = [
                (0, 0, h, w_big),
                (0, w_big, h1, w_side),
                (h1, w_big, h2, w_side),
            ]
        else:
            boxes = [
                (0, 0, h1, w_side),
                (h1, 0, h2, w_side),
                (0, w_side, h, w_big),
            ]
        return h, w_big + w_side, boxes

    w1, w2 = rng.randint(4, 6), rng.randint(4, 6)
    h_big, h_side = rng.randint(4, 6), rng.randint(4, 6)
    w = w1 + w2
    if orient == "top":
        boxes = [
            (0, 0, h_big, w),
            (h_big, 0, h_side, w1),
            (h_big, w1, h_side, w2),
        ]
    else:
        boxes = [
            (0, 0, h_side, w1),
            (0, w1, h_side, w2),
            (h_side, 0, h_big, w),
        ]
    return h_big + h_side, w, boxes


def _piece_id_grid(h: int, w: int, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.full((h, w), -1, dtype=int)
    for idx, (r0, c0, hh, ww) in enumerate(boxes):
        out[r0 : r0 + hh, c0 : c0 + ww] = idx
    return out


def _shared_seams(boxes: list[tuple[int, int, int, int]]) -> list[tuple[str, int, int, int]]:
    seams = []
    for i, (r0, c0, h0, w0) in enumerate(boxes):
        for j, (r1, c1, h1, w1) in enumerate(boxes):
            if j <= i:
                continue
            if c0 + w0 == c1 or c1 + w1 == c0:
                left, right = (i, j) if c0 < c1 else (j, i)
                a = boxes[left]
                b = boxes[right]
                rr0 = max(a[0], b[0])
                rr1 = min(a[0] + a[2], b[0] + b[2])
                if rr0 < rr1:
                    seams.append(("v", a[1] + a[3] - 1, rr0, rr1))
            if r0 + h0 == r1 or r1 + h1 == r0:
                top, bottom = (i, j) if r0 < r1 else (j, i)
                a = boxes[top]
                b = boxes[bottom]
                cc0 = max(a[1], b[1])
                cc1 = min(a[1] + a[3], b[1] + b[3])
                if cc0 < cc1:
                    seams.append(("h", a[0] + a[2] - 1, cc0, cc1))
    uniq = []
    seen = set()
    for seam in seams:
        if seam not in seen:
            uniq.append(seam)
            seen.add(seam)
    return uniq


def _paint_seam_signatures(rect: np.ndarray, seams: list[tuple[str, int, int, int]], feature_colors: tuple[int, int], rng: random.Random) -> None:
    for orient, fixed, start, end in seams:
        seg_len = end - start
        if seg_len <= 1:
            continue
        count = rng.randint(1, min(3, seg_len))
        offsets = rng.sample(list(range(start, end)), count)
        for pos in offsets:
            color = rng.choice(feature_colors)
            if orient == "v":
                rect[pos, fixed] = color
                rect[pos, fixed + 1] = color
            else:
                rect[fixed, pos] = color
                rect[fixed + 1, pos] = color


def _shape_cells(shape_key: tuple[int, int, tuple[tuple[int, int], ...]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in shape_key[2]]


def _hole_candidates(
    shape_key: tuple[int, int, tuple[tuple[int, int], ...]],
    piece_ids: np.ndarray,
    occupied: np.ndarray,
    *,
    require_crossing: bool | None,
) -> list[list[tuple[int, int]]]:
    h, w, cells = shape_key
    gh, gw = piece_ids.shape
    out = []
    for r0 in range(1, gh - h):
        for c0 in range(1, gw - w):
            placed = [(r0 + r, c0 + c) for r, c in cells]
            if any(occupied[r, c] for r, c in placed):
                continue
            ids = {int(piece_ids[r, c]) for r, c in placed}
            crossing = len(ids) > 1
            if require_crossing is not None and crossing != require_crossing:
                continue
            rs = [r for r, _c in placed]
            cs = [c for _r, c in placed]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)
            if occupied[max(0, rmin - 1) : min(gh, rmax + 2), max(0, cmin - 1) : min(gw, cmax + 2)].any():
                continue
            out.append(placed)
    return out


def _patch_from_shape(shape_key: tuple[int, int, tuple[tuple[int, int], ...]], color: int) -> np.ndarray:
    h, w, cells = shape_key
    patch = np.full((h, w), BG, dtype=int)
    for r, c in cells:
        patch[r, c] = int(color)
    return patch


def _piece_hole_keys(piece: np.ndarray) -> set[tuple[int, int, tuple[tuple[int, int], ...]]]:
    out = set()
    for coords in components_mask(piece == BLACK, connectivity=4):
        if _touches_bbox_edge(coords, piece.shape):
            continue
        out.add(_shape_key(coords))
    return out


def _marker_cells_for_corner(rect_box: tuple[int, int, int, int], corner_kind: str) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    r0, c0, r1, c1 = rect_box
    if corner_kind == "TL":
        cells = [(r0 - 1, c0 - 1), (r0 - 1, c0), (r0, c0 - 1)]
        missing = (r0, c0)
    elif corner_kind == "TR":
        cells = [(r0 - 1, c1), (r0 - 1, c1 + 1), (r0, c1 + 1)]
        missing = (r0, c1)
    elif corner_kind == "BL":
        cells = [(r1, c0 - 1), (r1 + 1, c0 - 1), (r1 + 1, c0)]
        missing = (r1, c0)
    else:
        cells = [(r1, c1 + 1), (r1 + 1, c1), (r1 + 1, c1 + 1)]
        missing = (r1, c1)
    return cells, missing


def _place_patch_random(canvas: np.ndarray, blocked: np.ndarray, patch: np.ndarray, rng: random.Random, *, padding: int = 1, transparent: int | None = None) -> bool:
    h, w = patch.shape
    gh, gw = canvas.shape
    for _ in range(PLACE_TRIES):
        r0 = rng.randint(0, gh - h)
        c0 = rng.randint(0, gw - w)
        rr0 = max(0, r0 - padding)
        cc0 = max(0, c0 - padding)
        rr1 = min(gh, r0 + h + padding)
        cc1 = min(gw, c0 + w + padding)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        paste(canvas, patch, r0, c0, transparent=transparent)
        blocked[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    rect_h, rect_w, boxes = _piece_layout(rng)
    piece_count = len(boxes)
    piece_ids = _piece_id_grid(rect_h, rect_w, boxes)

    piece_colors = list(PIECE_COLORS)
    rng.shuffle(piece_colors)
    rect_color = piece_colors[0]
    feature_colors = (piece_colors[1], piece_colors[2])
    marker_pool = [c for c in PIECE_COLORS if c not in {rect_color, *feature_colors}]

    rect = np.full((rect_h, rect_w), rect_color, dtype=int)
    seams = _shared_seams(boxes)
    _paint_seam_signatures(rect, seams, feature_colors, rng)

    occupied_holes = np.zeros((rect_h, rect_w), dtype=bool)
    distinct_shapes = rng.sample(list(SHAPES), 2)
    first_shape = _shape_key(distinct_shapes[0])
    second_shape = _shape_key(distinct_shapes[1])

    first_candidates = _hole_candidates(first_shape, piece_ids, occupied_holes, require_crossing=True)
    if not first_candidates:
        raise AssertionError("no crossing-hole placement")
    first_cells = rng.choice(first_candidates)
    for r, c in first_cells:
        occupied_holes[r, c] = True
        rect[r, c] = BLACK

    second_cross = rng.random() < 0.5
    second_candidates = _hole_candidates(second_shape, piece_ids, occupied_holes, require_crossing=second_cross)
    if not second_candidates:
        second_candidates = _hole_candidates(second_shape, piece_ids, occupied_holes, require_crossing=None)
    if not second_candidates:
        raise AssertionError("no second-hole placement")
    second_cells = rng.choice(second_candidates)
    for r, c in second_cells:
        occupied_holes[r, c] = True
        rect[r, c] = BLACK

    for idx, (r0, c0, h, w) in enumerate(boxes):
        inner = [(r, c) for r in range(r0 + 1, r0 + h - 1) for c in range(c0 + 1, c0 + w - 1) if rect[r, c] == rect_color]
        rng.shuffle(inner)
        for pos, (r, c) in enumerate(inner[: rng.randint(1, 2)]):
            rect[r, c] = feature_colors[(idx + pos) % len(feature_colors)]

    holes = _hole_components(rect)
    hole_keys = [_shape_key(hole) for hole in holes]
    assert len(holes) == 2
    assert len(set(hole_keys)) == 2

    marker_count = 2 + int(rng.random() < 0.5)
    marker_colors = rng.sample(marker_pool, marker_count)
    marker_specs = [(hole_keys[i], marker_colors[i]) for i in range(2)]
    if marker_count == 3:
        distractor_shape = next(_shape_key(shape) for shape in SHAPES if _shape_key(shape) not in set(hole_keys))
        marker_specs.append((distractor_shape, marker_colors[2]))

    filled = rect.copy()
    for hole, (_shape, color) in zip(holes, marker_specs[:2]):
        filled[hole[:, 0], hole[:, 1]] = int(color)

    pieces = [rect[r0 : r0 + h, c0 : c0 + w].copy() for r0, c0, h, w in boxes]
    assert _rect_color(pieces) == rect_color

    piece_hole_keys = set()
    for piece in pieces:
        piece_hole_keys |= _piece_hole_keys(piece)
    assert hole_keys[0] not in piece_hole_keys or hole_keys[1] not in piece_hole_keys

    grid_h = rng.randint(max(18, rect_h + 8), min(28, 30))
    grid_w = rng.randint(max(18, rect_w + 8), min(28, 30))
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    corner_kind = rng.choice(("TL", "TR", "BL", "BR"))
    row_low = 2
    row_high = grid_h - rect_h - 3
    col_low = 2
    col_high = grid_w - rect_w - 3
    if row_high < row_low or col_high < col_low:
        raise AssertionError("canvas too small")
    rr = rng.randint(row_low, row_high)
    cc = rng.randint(col_low, col_high)
    rect_box = (rr, cc, rr + rect_h - 1, cc + rect_w - 1)
    blocked[max(0, rr - 1) : min(grid_h, rr + rect_h + 1), max(0, cc - 1) : min(grid_w, cc + rect_w + 1)] = True

    marker_cells, missing = _marker_cells_for_corner(rect_box, corner_kind)
    if not all(0 <= r < grid_h and 0 <= c < grid_w for r, c in marker_cells):
        raise AssertionError("L marker out of bounds")
    for r, c in marker_cells:
        grid[r, c] = MARKER_L_COLOR
    for r, c in marker_cells:
        blocked[max(0, r - 1) : min(grid_h, r + 2), max(0, c - 1) : min(grid_w, c + 2)] = True

    piece_order = sorted(pieces, key=lambda p: (-p.shape[0] * p.shape[1], -p.shape[0], -p.shape[1]))
    for piece in piece_order:
        if not _place_patch_random(grid, blocked, piece, rng, padding=1):
            raise AssertionError("piece placement failed")

    for shape_key, color in marker_specs:
        patch = _patch_from_shape(shape_key, color)
        if not _place_patch_random(grid, blocked, patch, rng, padding=1, transparent=BG):
            raise AssertionError("marker placement failed")

    meta = {
        "rect": rect,
        "filled": filled,
        "pieces": pieces,
        "hole_keys": hole_keys,
        "marker_specs": marker_specs,
        "missing": missing,
        "corner_kind": corner_kind,
        "piece_count": piece_count,
    }
    return grid, filled, meta


def _validate_case(grid: np.ndarray, expected: np.ndarray, meta: dict) -> None:
    details = _solve_details(grid)
    assert np.array_equal(details["output"], expected)
    assert details["assembly_count"] == 1
    assert len(details["holes"]) == 2
    assert {_shape_key(hole) for hole in details["holes"]} == set(meta["hole_keys"])
    assert int(sum(key in details["shape_map"] for key in meta["hole_keys"])) == 2
    piece_holes = set()
    for piece in details["pieces"]:
        piece_holes |= _piece_hole_keys(piece)
    assert any(key not in piece_holes for key in meta["hole_keys"])
    if len(meta["marker_specs"]) == 3:
        distractor_key = meta["marker_specs"][2][0]
        assert distractor_key not in meta["hole_keys"]


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30))
        try:
            grid, expected, meta = _build_case(rng)
            _validate_case(grid, expected, meta)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
