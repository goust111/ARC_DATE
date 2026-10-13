from __future__ import annotations

import random

import numpy as np


BG = 0
FENCE = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_ROT_CW = {"top": 0, "right": 1, "bottom": 2, "left": 3}
CANVAS_LIMIT = 30
GEN_RETRIES = 400

SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _crop_component(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -turns)


def _canonical_shape(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append((rot.shape, rot.astype(np.uint8).tobytes()))
    return min(variants)


def _shapes_match_by_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    for k in range(4):
        if np.array_equal(_rotate_cw(a, k), b):
            return True
    return False


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _edge_gap(edge: np.ndarray) -> tuple[int, int] | None:
    missing = np.flatnonzero(~edge)
    if missing.size == 0:
        return None
    if missing[0] == 0 or missing[-1] == len(edge) - 1:
        return None
    if missing[-1] - missing[0] + 1 != missing.size:
        return None
    return int(missing[0]), int(missing[-1])


def _analyze_fence_component(mask: np.ndarray, color: int) -> dict | None:
    if mask.sum() < 10:
        return None
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub[1:-1, 1:-1]):
        return None

    gap_info = {
        "top": _edge_gap(sub[0, :]),
        "right": _edge_gap(sub[:, -1]),
        "bottom": _edge_gap(sub[-1, :]),
        "left": _edge_gap(sub[:, 0]),
    }
    sides = [side for side, gap in gap_info.items() if gap is not None]
    if len(sides) != 1:
        return None

    side = sides[0]
    for other in SIDES:
        if other == side:
            continue
        if other == "top" and not np.all(sub[0, :]):
            return None
        if other == "right" and not np.all(sub[:, -1]):
            return None
        if other == "bottom" and not np.all(sub[-1, :]):
            return None
        if other == "left" and not np.all(sub[:, 0]):
            return None

    expected = np.zeros_like(sub, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    gap = gap_info[side]
    assert gap is not None
    lo, hi = gap
    if side == "top":
        expected[0, lo : hi + 1] = False
    elif side == "right":
        expected[lo : hi + 1, -1] = False
    elif side == "bottom":
        expected[-1, lo : hi + 1] = False
    else:
        expected[lo : hi + 1, 0] = False
    if not np.array_equal(sub, expected):
        return None

    return {"color": color, "bbox": (r0, c0, r1, c1), "gap_side": side}


def _find_fences(grid: np.ndarray, bg: int) -> list[dict]:
    infos: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components(grid == color):
            mask = np.zeros_like(grid, dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            info = _analyze_fence_component(mask, color)
            if info is not None:
                infos.append(info)
    infos.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    assert infos, "expected at least one fence"
    return infos


def _extract_pieces(grid: np.ndarray, bg: int, fences: list[dict]) -> list[dict]:
    outside = np.ones_like(grid, dtype=bool)
    for info in fences:
        r0, c0, r1, c1 = info["bbox"]
        outside[r0 : r1 + 1, c0 : c1 + 1] = False
    mask = (grid != bg) & outside
    pieces = []
    for coords in _components(mask):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1
        color = colors.pop()
        shape = _mask_from_coords(coords)
        pieces.append({"color": color, "shape": shape})
    assert pieces, "expected outside candidate pieces"
    return pieces


def _fill_fence_interior(
    interior: np.ndarray,
    pieces: list[dict],
    used: set[int],
    bg: int,
) -> np.ndarray:
    filled = interior.copy()
    hole_mask = filled == bg
    holes = _components(hole_mask)
    assert holes, "expected at least one hole per fence"
    for coords in holes:
        local = np.zeros_like(hole_mask, dtype=bool)
        local[coords[:, 0], coords[:, 1]] = True
        shape, (r0, c0, r1, c1) = _crop_component(local)
        matches = []
        for idx, piece in enumerate(pieces):
            if idx in used:
                continue
            if _shapes_match_by_rotation(piece["shape"], shape):
                matches.append(idx)
        assert len(matches) == 1, "hole-to-piece match must be unique"
        idx = matches[0]
        used.add(idx)
        patch = filled[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[shape] = int(pieces[idx]["color"])
        filled[r0 : r1 + 1, c0 : c1 + 1] = patch
    return filled


def _assemble_horizontal(parts: list[np.ndarray], bg: int) -> np.ndarray:
    assert parts
    height = max(part.shape[0] for part in parts)
    width = sum(part.shape[1] for part in parts) + max(0, len(parts) - 1)
    out = np.full((height, width), bg, dtype=int)
    col = 0
    for part in parts:
        h, w = part.shape
        out[:h, col : col + w] = part
        col += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill fence-internal holes from matching outside pieces, then rotate each completed interior by the fence-gap direction.

    Input structure:
    - Background is black.
    - Several one-pixel-thick grey rectangular fences appear on the grid; each fence has exactly one contiguous gap on one side.
    - Inside each fence are colored 4-connected blocks together with one or more black hole regions.
    - Outside the fences are small monochrome 4-connected candidate pieces whose shapes match those hole regions up to rotation.

    Rule:
    - Detect every rectangular fence and crop its interior.
    - For each interior black hole, find the unique outside piece with the same shape under a 0/90/180/270 degree rotation and fill the hole with that piece's color.
    - Read the fence-gap side as a clockwise rotation code for the fully filled interior: top=0, right=90, bottom=180, left=270.
    - Output only the rotated interiors, ordered from left to right then top to bottom, packed horizontally with one black separator column.

    Key insight:
    - The outside pieces are shape-coded color keys for the internal holes, while the fence opening independently encodes the final rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = BG

    fences = _find_fences(g, bg)
    pieces = _extract_pieces(g, bg, fences)

    used_pieces: set[int] = set()
    outputs: list[np.ndarray] = []
    for info in fences:
        r0, c0, r1, c1 = info["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        filled = _fill_fence_interior(interior, pieces, used_pieces, bg)
        rotated = _rotate_cw(filled, SIDE_TO_ROT_CW[info["gap_side"]])
        outputs.append(rotated)

    return _assemble_horizontal(outputs, bg)


def _paint_regions(h: int, w: int, hole_mask: np.ndarray, forbidden: set[int]) -> np.ndarray:
    free = [(r, c) for r in range(h) for c in range(w) if not hole_mask[r, c]]
    assert free
    region_pool = [c for c in COLORS if c not in forbidden]
    if len(region_pool) < 2:
        region_pool = list(COLORS)
    region_count = random.randint(2, min(4, len(region_pool), len(free)))
    region_colors = random.sample(region_pool, region_count)
    seeds = random.sample(free, region_count)

    out = np.full((h, w), BG, dtype=int)
    for r, c in free:
        best_idx = 0
        best_key = None
        for idx, (sr, sc) in enumerate(seeds):
            key = (abs(r - sr) + abs(c - sc), idx)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
        out[r, c] = region_colors[best_idx]
    return out


def _place_holes(h: int, w: int, base_shapes: list[np.ndarray]) -> list[dict] | None:
    blocked = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []
    for base in base_shapes:
        hole = _rotate_cw(base, random.randint(0, 3))
        hh, hw = hole.shape
        if hh > h - 2 or hw > w - 2:
            return None
        placed = False
        for _ in range(100):
            r = random.randint(1, h - hh - 1)
            c = random.randint(1, w - hw - 1)
            pr0 = max(0, r - 1)
            pc0 = max(0, c - 1)
            pr1 = min(h, r + hh + 1)
            pc1 = min(w, c + hw + 1)
            if blocked[pr0:pr1, pc0:pc1].any():
                continue
            blocked[pr0:pr1, pc0:pc1] = True
            specs.append({"mask": hole, "r": r, "c": c, "base": base})
            placed = True
            break
        if not placed:
            return None
    return specs


def _carve_gap(frame: np.ndarray, side: str) -> None:
    h, w = frame.shape
    if side in {"top", "bottom"}:
        gap_len = 1 if w <= 5 else random.randint(1, min(2, w - 3))
        start = random.randint(1, w - gap_len - 1)
        row = 0 if side == "top" else h - 1
        frame[row, start : start + gap_len] = BG
    else:
        gap_len = 1 if h <= 5 else random.randint(1, min(2, h - 3))
        start = random.randint(1, h - gap_len - 1)
        col = 0 if side == "left" else w - 1
        frame[start : start + gap_len, col] = BG


def _make_fence_case(base_shapes: list[np.ndarray], piece_colors: list[int], size_cap: int) -> dict | None:
    min_h = max(5, max(shape.shape[0] for shape in base_shapes) + 2)
    min_w = max(5, max(shape.shape[1] for shape in base_shapes) + 2)
    if len(base_shapes) > 1:
        min_h = max(min_h, 6)
        min_w = max(min_w, 6)
    if min_h > size_cap or min_w > size_cap:
        return None
    h = random.randint(min_h, size_cap)
    w = random.randint(min_w, size_cap)

    hole_specs = _place_holes(h, w, base_shapes)
    if hole_specs is None:
        return None

    piece_color_set = set(piece_colors)
    interior = _paint_regions(h, w, np.zeros((h, w), dtype=bool), piece_color_set)
    hole_mask = np.zeros((h, w), dtype=bool)
    for spec in hole_specs:
        hole = spec["mask"]
        r = spec["r"]
        c = spec["c"]
        hh, hw = hole.shape
        hole_mask[r : r + hh, c : c + hw] |= hole
    interior[hole_mask] = BG

    filled = interior.copy()
    pieces = []
    for spec, piece_color in zip(hole_specs, piece_colors, strict=False):
        r = spec["r"]
        c = spec["c"]
        hole = spec["mask"]
        hh, hw = hole.shape
        patch = filled[r : r + hh, c : c + hw].copy()
        patch[hole] = int(piece_color)
        filled[r : r + hh, c : c + hw] = patch
        piece_mask = _rotate_cw(spec["base"], random.randint(0, 3))
        pieces.append({"mask": piece_mask, "color": int(piece_color)})

    frame = np.full((h + 2, w + 2), BG, dtype=int)
    frame[0, :] = FENCE
    frame[-1, :] = FENCE
    frame[:, 0] = FENCE
    frame[:, -1] = FENCE
    frame[1:-1, 1:-1] = interior
    side = random.choice(SIDES)
    _carve_gap(frame, side)

    rotated = _rotate_cw(filled, SIDE_TO_ROT_CW[side])
    return {"frame": frame, "pieces": pieces, "expected": rotated}


def _place_bbox(blocked: np.ndarray, obj_h: int, obj_w: int, used_cols: set[int] | None = None) -> tuple[int, int] | None:
    h, w = blocked.shape
    if obj_h + 2 > h or obj_w + 2 > w:
        return None
    for _ in range(250):
        r = random.randint(1, h - obj_h - 1)
        c = random.randint(1, w - obj_w - 1)
        if used_cols is not None and c in used_cols:
            continue
        if blocked[r - 1 : r + obj_h + 1, c - 1 : c + obj_w + 1].any():
            continue
        blocked[r - 1 : r + obj_h + 1, c - 1 : c + obj_w + 1] = True
        if used_cols is not None:
            used_cols.add(c)
        return r, c
    return None


def _paste_mask(grid: np.ndarray, arr: np.ndarray, r: int, c: int, mask: np.ndarray | None = None) -> None:
    h, w = arr.shape
    if mask is None:
        grid[r : r + h, c : c + w] = arr
    else:
        patch = grid[r : r + h, c : c + w].copy()
        patch[mask] = arr[mask]
        grid[r : r + h, c : c + w] = patch


def _trim_grid(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0, r1, c1 = _bbox(coords)
    r0 = max(0, r0 - 1)
    c0 = max(0, c0 - 1)
    r1 = min(grid.shape[0] - 1, r1 + 1)
    c1 = min(grid.shape[1] - 1, c1 + 1)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_fences = random.choice([1, 2, 3])
    size_cap = {1: 8, 2: 7, 3: 6}[n_fences]
    hole_counts = [random.choice([1, 2]) for _ in range(n_fences)]
    total_holes = sum(hole_counts)
    unique_shapes = []
    seen_keys = set()
    for shape in SHAPES:
        key = _canonical_shape(shape)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_shapes.append(shape)
    if total_holes > len(unique_shapes):
        return None

    base_shapes = random.sample(unique_shapes, total_holes)
    piece_colors = random.sample(list(COLORS), total_holes)

    fences: list[dict] = []
    cursor = 0
    for hole_count in hole_counts:
        shape_slice = [shape.copy() for shape in base_shapes[cursor : cursor + hole_count]]
        color_slice = piece_colors[cursor : cursor + hole_count]
        cursor += hole_count
        fence = _make_fence_case(shape_slice, color_slice, size_cap)
        if fence is None:
            return None
        fences.append(fence)

    canvas = np.full((CANVAS_LIMIT, CANVAS_LIMIT), BG, dtype=int)
    blocked = np.zeros((CANVAS_LIMIT, CANVAS_LIMIT), dtype=bool)
    used_cols: set[int] = set()
    placements: list[dict] = []
    loose_pieces: list[dict] = []
    for fence in fences:
        fh, fw = fence["frame"].shape
        pos = _place_bbox(blocked, fh, fw, used_cols=used_cols)
        if pos is None:
            return None
        r, c = pos
        _paste_mask(canvas, fence["frame"], r, c)
        placements.append({"bbox": (r, c, r + fh - 1, c + fw - 1), "expected": fence["expected"]})
        loose_pieces.extend(fence["pieces"])

    for piece in loose_pieces:
        mask = piece["mask"]
        ph, pw = mask.shape
        pos = _place_bbox(blocked, ph, pw)
        if pos is None:
            return None
        r, c = pos
        arr = np.where(mask, piece["color"], BG)
        _paste_mask(canvas, arr, r, c, mask=mask)

    inp = _trim_grid(canvas)
    ordered = [item["expected"] for item in sorted(placements, key=lambda item: (item["bbox"][1], item["bbox"][0]))]
    out = _assemble_horizontal(ordered, BG)
    if max(inp.shape) > CANVAS_LIMIT or max(out.shape) > CANVAS_LIMIT:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _build_case()
        if case is None:
            continue
        inp, expected = case
        try:
            got = main(inp)
        except AssertionError:
            continue
        if np.array_equal(got, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
