import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

GEN_OUT_SHAPES = [(3, 4), (3, 5), (4, 4), (3, 6), (4, 5), (4, 6), (5, 5)]
GEN_GRID_ROWS = (18, 22)
GEN_GRID_COLS = (18, 22)
GEN_FRAME_SIDE = (8, 10)
GEN_PIECE_COUNT = (4, 6)
GEN_PIECE_AREA = (3, 6)
GEN_ATTEMPTS = 400
GEN_PARTITION_ATTEMPTS = 300
GEN_LAYOUT_ATTEMPTS = 160
GEN_CACHE_TARGET = 24
_GEN_CACHE: list[np.ndarray] = []
_GEN_CURSOR = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def component_coords(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    out = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(sorted(pts), dtype=int))
    return out


def find_monochrome_components(grid: np.ndarray, colors: list[int], connectivity: int = 4) -> list[dict]:
    out = []
    for color in colors:
        comps = component_coords(grid == int(color), connectivity=connectivity)
        for coords in comps:
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def compact_shape(area: int) -> tuple[int, int]:
    best = None
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        cand = (abs(w - h), -w, h, w)
        if best is None or cand < best[0]:
            best = (cand, (h, w))
    assert best is not None
    return best[1]


def frame_mask_and_bbox(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mask = grid == FRAME
    assert mask.any(), "expected a grey frame"
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1 and side >= 3, "expected a square frame"
    expected = np.zeros_like(mask)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    assert np.array_equal(mask, expected), "grey pixels must form one 1-cell-thick square frame"
    return mask, (r0, c0, r1, c1)


def manhattan_distance_map(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    H, W = mask.shape
    out = np.empty((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            out[r, c] = min(abs(r - int(rr)) + abs(c - int(cc)) for rr, cc in pts)
    return out


def _hflip(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    w = int(arr[:, 1].max()) + 1
    out = arr.copy()
    out[:, 1] = w - 1 - out[:, 1]
    return normalize_coords(out)


def _vflip(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    h = int(arr[:, 0].max()) + 1
    out = arr.copy()
    out[:, 0] = h - 1 - out[:, 0]
    return normalize_coords(out)


def _rot90cw(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    h = int(arr[:, 0].max()) + 1
    out = np.zeros_like(arr)
    out[:, 0] = arr[:, 1]
    out[:, 1] = h - 1 - arr[:, 0]
    return normalize_coords(out)


def _rot180(coords: np.ndarray) -> np.ndarray:
    return _rot90cw(_rot90cw(coords))


def _rot270cw(coords: np.ndarray) -> np.ndarray:
    return _rot90cw(_rot180(coords))


def ordered_orientations(coords: np.ndarray) -> list[np.ndarray]:
    base = normalize_coords(coords)
    rot90 = _rot90cw(base)
    rot270 = _rot270cw(base)
    return [
        base,
        _hflip(base),
        _vflip(base),
        rot90,
        _hflip(rot90),
        _rot180(base),
        rot270,
        _hflip(rot270),
    ]


def first_fit_pack(
    pieces: list[dict],
    order: list[int],
    out_shape: tuple[int, int],
) -> tuple[np.ndarray | None, list[np.ndarray] | None]:
    H, W = out_shape
    occupied = np.zeros((H, W), dtype=bool)
    out = np.full((H, W), BG, dtype=int)
    placements = []
    for idx in order:
        piece = pieces[idx]
        placed = None
        for orient in ordered_orientations(piece["coords"]):
            h = int(orient[:, 0].max()) + 1
            w = int(orient[:, 1].max()) + 1
            for r0 in range(H - h + 1):
                for c0 in range(W - w + 1):
                    cells = orient + np.array([r0, c0], dtype=int)
                    if occupied[cells[:, 0], cells[:, 1]].any():
                        continue
                    placed = cells
                    break
                if placed is not None:
                    break
            if placed is not None:
                break
        if placed is None:
            return None, None
        occupied[placed[:, 0], placed[:, 1]] = True
        out[placed[:, 0], placed[:, 1]] = int(piece["color"])
        placements.append(placed)
    if not occupied.all():
        return None, None
    return out, placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack colored fragments into a compact rectangle using distance to a grey frame as the placement order.

    Input structure:
    - The grid has black background, one hollow 1-cell-thick grey square frame, and 4-6 scattered monochrome 4-connected fragments in non-grey colors.
    - Fragments may lie inside or outside the frame, but none touches the grey border.
    - The grey frame is the only grey object and serves only as a spatial reference.

    Rule:
    - For each fragment, compute the minimum Manhattan distance from any fragment cell to any grey frame cell.
    - Sort fragments by that frame distance from small to large.
    - Let the output rectangle area equal the total fragment area; among all factor pairs choose the rectangle with minimal |width-height|, breaking ties by width >= height.
    - Process fragments in sorted order. For each fragment, try these 8 transforms in order: original, horizontal flip, vertical flip, 90 clockwise, 90 clockwise then horizontal flip, 180, 270 clockwise, 270 clockwise then horizontal flip. For each transform, scan positions from the output rectangle's top-left row by row and place the fragment at the first non-overlapping fit.
    - Preserve fragment colors. Discard the grey frame and all background.

    Key insight:
    - The frame never appears in the output; its only job is to impose a hidden distance-based ordering that replaces more obvious cues like area, color, or input position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_mask, _frame_bbox = frame_mask_and_bbox(g)
    dist_map = manhattan_distance_map(frame_mask)

    colors = sorted(set(map(int, g.ravel())) - {BG, FRAME})
    components = find_monochrome_components(g, colors, connectivity=4)
    assert 4 <= len(components) <= 6, "expected 4-6 fragments"

    pieces = []
    distances = []
    for comp in components:
        coords = np.asarray(comp["coords"], dtype=int)
        r0, c0, _r1, _c1 = comp["bbox"]
        local = normalize_coords(coords - np.array([r0, c0], dtype=int))
        d = int(min(dist_map[int(r), int(c)] for r, c in coords))
        assert d >= 1, "fragments must not touch the frame"
        pieces.append(
            {
                "color": int(comp["color"]),
                "coords": local,
                "bbox": tuple(map(int, comp["bbox"])),
                "area": int(comp["area"]),
                "distance": d,
            }
        )
        distances.append(d)

    assert len(set(distances)) == len(distances), "fragment distances to the frame must be unique"

    total_area = sum(piece["area"] for piece in pieces)
    out_shape = compact_shape(total_area)
    order = sorted(range(len(pieces)), key=lambda i: (pieces[i]["distance"], pieces[i]["bbox"][0], pieces[i]["bbox"][1]))
    out, placements = first_fit_pack(pieces, order, out_shape)
    assert out is not None and placements is not None, "expected fragments to fill a compact rectangle"
    assert np.all(out != BG), "output rectangle must be full"
    return out


def _piece_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(map(tuple, normalize_coords(coords)))


def _all_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = shape
    return [(r, c) for r in range(H) for c in range(W)]


def _is_connected_cells(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _random_connected_subset(cells: set[tuple[int, int]], size: int) -> set[tuple[int, int]] | None:
    start = random.choice(tuple(cells))
    subset = {start}
    while len(subset) < size:
        frontier = set()
        for r, c in subset:
            for dr, dc in DIR4:
                nb = (r + dr, c + dc)
                if nb in cells and nb not in subset:
                    frontier.add(nb)
        if not frontier:
            return None
        subset.add(random.choice(tuple(frontier)))
    return subset


def _sample_sizes(area: int, count: int) -> tuple[int, ...] | None:
    options = []

    def rec(k: int, rem: int, cur: list[int]) -> None:
        if k == count:
            if rem == 0:
                options.append(tuple(cur))
            return
        left = count - k - 1
        for size in range(GEN_PIECE_AREA[0], GEN_PIECE_AREA[1] + 1):
            nxt = rem - size
            if nxt < left * GEN_PIECE_AREA[0] or nxt > left * GEN_PIECE_AREA[1]:
                continue
            cur.append(size)
            rec(k + 1, nxt, cur)
            cur.pop()

    rec(0, area, [])
    options = [opt for opt in options if len(set(opt)) >= 2 or area == 12]
    return random.choice(options) if options else None


def _partition_rectangle(shape: tuple[int, int], sizes: tuple[int, ...]) -> list[np.ndarray] | None:
    cells_all = set(_all_cells(shape))
    for _ in range(GEN_PARTITION_ATTEMPTS):
        remaining = set(cells_all)
        parts: list[set[tuple[int, int]]] = []
        ok = True
        for size in sizes[:-1]:
            picked = None
            for _inner in range(60):
                subset = _random_connected_subset(remaining, int(size))
                if subset is None:
                    continue
                rest = remaining - subset
                if not rest or _is_connected_cells(rest):
                    picked = subset
                    break
            if picked is None:
                ok = False
                break
            parts.append(picked)
            remaining -= picked
        if not ok:
            continue
        if len(remaining) != sizes[-1] or not _is_connected_cells(remaining):
            continue
        parts.append(remaining)
        coords_list = [normalize_coords(np.asarray(sorted(part), dtype=int)) for part in parts]
        non_rect = 0
        for coords in coords_list:
            h = int(coords[:, 0].max()) + 1
            w = int(coords[:, 1].max()) + 1
            if h * w != len(coords):
                non_rect += 1
        if non_rect >= 2:
            return coords_list
    return None


def _coords_to_patch(coords: np.ndarray, color: int) -> np.ndarray:
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    patch = np.full((h, w), BG, dtype=int)
    patch[coords[:, 0], coords[:, 1]] = int(color)
    return patch


def _frame_distance_for_cells(dist_map: np.ndarray, cells: np.ndarray) -> int:
    return int(dist_map[cells[:, 0], cells[:, 1]].min())


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        for dr, dc in DIR8:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = True
    return out


def _enumerate_candidates(
    coords: np.ndarray,
    grid_shape: tuple[int, int],
    frame_mask: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    dist_map: np.ndarray,
    region: str,
) -> list[dict]:
    H, W = grid_shape
    r0f, c0f, r1f, c1f = frame_bbox
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    candidates = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            cells = coords + np.array([r0, c0], dtype=int)
            rr = cells[:, 0]
            cc = cells[:, 1]
            if frame_mask[rr, cc].any():
                continue
            inside = bool(np.all((rr > r0f) & (rr < r1f) & (cc > c0f) & (cc < c1f)))
            if region == "inside" and not inside:
                continue
            if region == "outside" and inside:
                continue
            dist = _frame_distance_for_cells(dist_map, cells)
            pad = np.zeros((H, W), dtype=bool)
            pad[rr, cc] = True
            pad = _dilate8(pad)
            candidates.append(
                {
                    "cells": cells,
                    "bbox": (int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())),
                    "distance": dist,
                    "pad": pad,
                }
            )
    random.shuffle(candidates)
    candidates.sort(key=lambda cand: (cand["distance"], cand["bbox"][0], cand["bbox"][1]))
    trimmed = []
    per_distance = Counter()
    for cand in candidates:
        d = int(cand["distance"])
        if per_distance[d] >= 10:
            continue
        trimmed.append(cand)
        per_distance[d] += 1
    return trimmed


def _choose_layout(
    pieces: list[dict],
    grid_shape: tuple[int, int],
    frame_mask: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    dist_map: np.ndarray,
) -> list[dict] | None:
    n = len(pieces)
    order_by_area = sorted(range(n), key=lambda i: (pieces[i]["area"], i))
    inside_count = 2 if n >= 5 and random.random() < 0.4 else 1
    inside_ids = set(order_by_area[:inside_count])
    regions = ["inside" if i in inside_ids else "outside" for i in range(n)]

    candidates_by_piece = []
    for i, piece in enumerate(pieces):
        cands = _enumerate_candidates(piece["coords"], grid_shape, frame_mask, frame_bbox, dist_map, regions[i])
        if not cands:
            return None
        candidates_by_piece.append(cands)

    piece_order = sorted(range(n), key=lambda i: (len(candidates_by_piece[i]), -pieces[i]["area"]))
    forbidden = np.zeros(grid_shape, dtype=bool)
    used_distances: set[int] = set()
    chosen = [None] * n

    def dfs(k: int) -> bool:
        if k == n:
            return True
        pid = piece_order[k]
        cands = candidates_by_piece[pid][:]
        random.shuffle(cands)
        cands.sort(key=lambda cand: (cand["distance"], random.random()))
        for cand in cands:
            if int(cand["distance"]) in used_distances:
                continue
            rr = cand["cells"][:, 0]
            cc = cand["cells"][:, 1]
            if forbidden[rr, cc].any():
                continue
            prev_forbidden = forbidden.copy()
            forbidden[:] = forbidden | cand["pad"]
            used_distances.add(int(cand["distance"]))
            chosen[pid] = cand
            if dfs(k + 1):
                return True
            chosen[pid] = None
            used_distances.remove(int(cand["distance"]))
            forbidden[:] = prev_forbidden
        return False

    if not dfs(0):
        return None
    return chosen


def _order_from_metadata(meta: list[dict], mode: str) -> list[int]:
    if mode == "distance":
        return sorted(range(len(meta)), key=lambda i: (meta[i]["distance"], meta[i]["bbox"][0], meta[i]["bbox"][1]))
    if mode == "area_desc":
        return sorted(range(len(meta)), key=lambda i: (-meta[i]["area"], meta[i]["bbox"][1], meta[i]["bbox"][0]))
    if mode == "color":
        return sorted(range(len(meta)), key=lambda i: (meta[i]["color"], meta[i]["bbox"][0], meta[i]["bbox"][1]))
    if mode == "scan":
        return sorted(range(len(meta)), key=lambda i: (meta[i]["bbox"][0], meta[i]["bbox"][1], meta[i]["color"]))
    if mode == "left_right":
        return sorted(range(len(meta)), key=lambda i: (meta[i]["bbox"][1], meta[i]["bbox"][0], meta[i]["color"]))
    raise ValueError(mode)


def _pack_output_for_order(pieces: list[dict], order: list[int]) -> np.ndarray | None:
    total_area = sum(piece["area"] for piece in pieces)
    out_shape = compact_shape(total_area)
    out, _ = first_fit_pack(pieces, order, out_shape)
    return out


def _make_frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _sample_frame_bbox(grid_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    H, W = grid_shape
    side = random.randint(*GEN_FRAME_SIDE)
    if side + 6 > H or side + 6 > W:
        return None
    r0_min, r0_max = 2, H - side - 3
    c0_min, c0_max = 2, W - side - 3
    if r0_min > r0_max or c0_min > c0_max:
        return None
    r0 = random.randint(r0_min, r0_max)
    c0 = random.randint(c0_min, c0_max)
    return (r0, c0, r0 + side - 1, c0 + side - 1)


def _build_generator_pieces() -> list[dict] | None:
    out_shape = random.choice(GEN_OUT_SHAPES)
    area = out_shape[0] * out_shape[1]
    possible_counts = [
        k
        for k in range(GEN_PIECE_COUNT[0], GEN_PIECE_COUNT[1] + 1)
        if GEN_PIECE_AREA[0] * k <= area <= GEN_PIECE_AREA[1] * k
    ]
    if not possible_counts:
        return None
    count = random.choice(possible_counts)
    sizes = _sample_sizes(area, count)
    if sizes is None:
        return None
    coords_list = _partition_rectangle(out_shape, sizes)
    if coords_list is None:
        return None

    pieces = []
    colors = random.sample(PALETTE, count)
    for color, coords in zip(colors, coords_list):
        orient = random.choice(ordered_orientations(coords))
        pieces.append(
            {
                "color": int(color),
                "coords": orient,
                "area": int(len(orient)),
                "shape_key": _piece_signature(orient),
            }
        )
    return pieces


def _anti_shortcut_ok(pieces: list[dict], truth: np.ndarray, meta: list[dict]) -> bool:
    true_order = _order_from_metadata(meta, "distance")
    area_order = _order_from_metadata(meta, "area_desc")
    color_order = _order_from_metadata(meta, "color")
    scan_order = _order_from_metadata(meta, "scan")
    left_order = _order_from_metadata(meta, "left_right")

    if true_order == area_order or true_order == color_order:
        return False

    area_out = _pack_output_for_order(pieces, area_order)
    color_out = _pack_output_for_order(pieces, color_order)
    scan_out = _pack_output_for_order(pieces, scan_order)
    left_out = _pack_output_for_order(pieces, left_order)

    if area_out is None or np.array_equal(area_out, truth):
        return False

    changed = 0
    for candidate in (color_out, scan_out, left_out):
        if candidate is None or not np.array_equal(candidate, truth):
            changed += 1
    return changed >= 2


def _attempt() -> np.ndarray | None:
    pieces = _build_generator_pieces()
    if pieces is None:
        return None

    H = random.randint(*GEN_GRID_ROWS)
    W = random.randint(*GEN_GRID_COLS)
    frame_bbox = _sample_frame_bbox((H, W))
    if frame_bbox is None:
        return None
    frame_mask = _make_frame_mask((H, W), frame_bbox)
    dist_map = manhattan_distance_map(frame_mask)

    layout = _choose_layout(pieces, (H, W), frame_mask, frame_bbox, dist_map)
    if layout is None:
        return None

    inp = np.zeros((H, W), dtype=int)
    inp[frame_mask] = FRAME

    meta = []
    for piece, chosen in zip(pieces, layout):
        cells = chosen["cells"]
        inp[cells[:, 0], cells[:, 1]] = int(piece["color"])
        meta.append(
            {
                "color": int(piece["color"]),
                "coords": piece["coords"],
                "area": int(piece["area"]),
                "bbox": chosen["bbox"],
                "distance": int(chosen["distance"]),
            }
        )

    distances = [m["distance"] for m in meta]
    if len(set(distances)) != len(distances):
        return None
    if not (any(np.all((cand["cells"][:, 0] > frame_bbox[0]) & (cand["cells"][:, 0] < frame_bbox[2]) &
                       (cand["cells"][:, 1] > frame_bbox[1]) & (cand["cells"][:, 1] < frame_bbox[3])) for cand in layout)
            and any(not np.all((cand["cells"][:, 0] > frame_bbox[0]) & (cand["cells"][:, 0] < frame_bbox[2]) &
                               (cand["cells"][:, 1] > frame_bbox[1]) & (cand["cells"][:, 1] < frame_bbox[3])) for cand in layout)):
        return None

    true_order = _order_from_metadata(meta, "distance")
    truth = _pack_output_for_order(pieces, true_order)
    if truth is None:
        return None
    if not _anti_shortcut_ok(pieces, truth, meta):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, truth):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CURSOR
    if len(_GEN_CACHE) >= GEN_CACHE_TARGET:
        grid = _GEN_CACHE[_GEN_CURSOR % len(_GEN_CACHE)].copy()
        _GEN_CURSOR += 1
        return grid
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            _GEN_CACHE.append(grid.copy())
            _GEN_CURSOR += 1
            return grid
    raise RuntimeError("failed to generate a valid frame-distance packing sample")
