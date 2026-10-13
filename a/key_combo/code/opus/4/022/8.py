from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
FRAME = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    comps: list[dict] = []
    if monochromatic:
        colors = sorted(set(map(int, grid.reshape(-1))) - {int(bg)})
        for color in colors:
            for coords in _components_mask(grid == int(color), connectivity=connectivity):
                comps.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": _bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return comps

    for coords in _components_mask(grid != int(bg), connectivity=connectivity):
        comps.append(
            {
                "color": None,
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    return out


def _random_free_location(canvas: np.ndarray, patch: np.ndarray, *, transparent: int, border: int = 0, padding: int = 0) -> Coord:
    h, w = canvas.shape
    ph, pw = patch.shape
    occ = canvas != int(transparent)
    if padding > 0:
        for _ in range(padding):
            occ = _dilate8(occ)
    mask = patch != int(transparent)
    candidates: list[Coord] = []
    for r0 in range(border, h - ph - border + 1):
        for c0 in range(border, w - pw - border + 1):
            if not bool((occ[r0 : r0 + ph, c0 : c0 + pw] & mask).any()):
                candidates.append((r0, c0))
    if not candidates:
        raise ValueError("no free location")
    return random.choice(candidates)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for v in (rot, np.fliplr(rot)):
            key = (tuple(map(int, v.shape)), v.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(v.copy())
    return variants


def _canonical_mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, v.shape)), v.tobytes()) for v in _mask_variants(mask))


def _find_frame(grid: np.ndarray) -> tuple[int, BBox]:
    bg = BG
    candidates: list[tuple[int, BBox, int]] = []
    comps = _find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    preferred = [comp for comp in comps if int(comp["color"]) == FRAME]
    for comp in preferred or comps:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros(sub.shape, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == color):
            continue
        if not np.any(sub[1:-1, 1:-1] == int(bg)):
            continue
        candidates.append((color, (r0, c0, r1, c1), int(comp["area"])))
    assert candidates, "frame not found"
    color, bbox, _area = max(candidates, key=lambda item: item[2])
    return int(color), bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each scattered piece into the matching frame hole, then recolor holes by area rank using the palette row.

    Input structure:
    - Black background surrounds one large gray frame whose rectangular bbox contains several black 4-connected holes.
    - Outside the frame are one monochrome piece per hole, each matching a hole under rotation/flip, and one horizontal palette row below the frame.
    - Hole areas are all distinct, piece colors are all distinct, and palette colors are distinct.

    Rule:
    - Find the gray frame and extract every black 4-connected hole inside its bbox.
    - Find the outside monochrome pieces and match each one to exactly one hole by exact shape under the 8 dihedral transforms.
    - Sort the holes by area from smallest to largest and read the palette row from left to right.
    - Fill each matched hole and overwrite the piece color with the palette color assigned to that hole's area rank.
    - Return only the frame bbox with the gray frame preserved and the holes filled.

    Key insight:
    - Piece colors identify which outside object is a piece, but final colors come only from the palette ordered by hole-area rank after shape matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = BG
    frame_color, frame_bbox = _find_frame(g)
    fr0, fc0, fr1, fc1 = frame_bbox
    frame_crop = _crop_bbox(g, frame_bbox)

    hole_coords = _components_mask(frame_crop == int(bg), connectivity=4)
    assert 3 <= len(hole_coords) <= 5, "expected 3..5 holes"
    holes: list[dict] = []
    for coords in hole_coords:
        rr0, cc0, rr1, cc1 = _bbox_coords(coords)
        assert rr0 > 0 and cc0 > 0 and rr1 < frame_crop.shape[0] - 1 and cc1 < frame_crop.shape[1] - 1
        mask = _tight_mask(coords)
        holes.append(
            {
                "coords": coords,
                "mask": mask,
                "bbox": (rr0, cc0, rr1, cc1),
                "area": int(coords.shape[0]),
            }
        )
    hole_areas = [hole["area"] for hole in holes]
    assert len(set(hole_areas)) == len(hole_areas), "hole areas must be unique"

    outside = g.copy()
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = bg
    comps = _find_components(outside, bg=bg, connectivity=4, monochromatic=True)

    singletons = [comp for comp in comps if int(comp["area"]) == 1]
    row_groups: dict[int, list[dict]] = {}
    for comp in singletons:
        r, c = map(int, comp["coords"][0])
        if r <= fr1:
            continue
        row_groups.setdefault(r, []).append(comp)
    palette_rows = []
    for r, row_comps in row_groups.items():
        row_comps = sorted(row_comps, key=lambda comp: int(comp["coords"][0, 1]))
        cols = [int(comp["coords"][0, 1]) for comp in row_comps]
        if len(row_comps) != len(holes):
            continue
        if len(set(int(comp["color"]) for comp in row_comps)) != len(holes):
            continue
        if cols != sorted(cols):
            continue
        palette_rows.append((r, row_comps))
    assert len(palette_rows) == 1, f"expected one palette row, found {len(palette_rows)}"
    _palette_r, palette_comps = palette_rows[0]
    palette = [int(comp["color"]) for comp in palette_comps]

    palette_cells = {tuple(map(int, comp["coords"][0])) for comp in palette_comps}
    pieces: list[dict] = []
    for comp in comps:
        coords_list = [tuple(map(int, rc)) for rc in comp["coords"]]
        if len(coords_list) == 1 and coords_list[0] in palette_cells:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        mask = _tight_mask(comp["coords"])
        pieces.append(
            {
                "color": int(comp["color"]),
                "mask": mask,
                "bbox": (r0, c0, r1, c1),
                "area": int(comp["area"]),
            }
        )
    assert len(pieces) == len(holes), "piece count must equal hole count"
    assert len({piece["color"] for piece in pieces}) == len(pieces), "piece colors must be unique"

    hole_candidates: list[list[int]] = []
    for hole in holes:
        candidates = []
        for pi, piece in enumerate(pieces):
            if int(piece["area"]) != int(hole["area"]):
                continue
            if any(
                variant.shape == hole["mask"].shape and np.array_equal(variant, hole["mask"])
                for variant in _mask_variants(piece["mask"])
            ):
                candidates.append(pi)
        assert candidates, "every hole must match at least one piece"
        hole_candidates.append(candidates)

    order = sorted(range(len(holes)), key=lambda idx: (len(hole_candidates[idx]), holes[idx]["area"]))
    used: set[int] = set()
    assignment: dict[int, int] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        hi = order[pos]
        for pi in hole_candidates[hi]:
            if pi in used:
                continue
            used.add(pi)
            assignment[hi] = pi
            if backtrack(pos + 1):
                return True
            used.remove(pi)
            del assignment[hi]
        return False

    assert backtrack(0), "piece-hole matching must be bijective"
    assert len(assignment) == len(holes)

    out = frame_crop.copy()
    ranked = sorted(range(len(holes)), key=lambda idx: int(holes[idx]["area"]))
    for rank, hi in enumerate(ranked):
        _matched_piece = pieces[assignment[hi]]
        color = int(palette[rank])
        coords = holes[hi]["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _shape_from_cells(cells: set[Coord]) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    return _tight_mask(coords)


def _grow_connected_shape(area: int) -> np.ndarray | None:
    for _ in range(120):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < area and frontier:
            rr, cc = random.choice(tuple(frontier))
            frontier.remove((rr, cc))
            if (rr, cc) in cells:
                continue
            cells.add((rr, cc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (rr + dr, cc + dc)
                if nxt not in cells:
                    frontier.add(nxt)
        if len(cells) != area:
            continue
        mask = _shape_from_cells(cells)
        if mask.shape[0] > 5 or mask.shape[1] > 5:
            continue
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if np.all(mask):
            continue
        return mask
    return None


def _generate_shapes(n_holes: int) -> list[np.ndarray] | None:
    area_choices = random.sample(list(range(4, 13)), n_holes)
    shapes: list[np.ndarray] = []
    canon_seen: set[tuple[tuple[int, int], bytes]] = set()
    asymmetric_count = 0
    for area in area_choices:
        picked = None
        for _ in range(120):
            mask = _grow_connected_shape(area)
            if mask is None:
                continue
            canon = _canonical_mask_key(mask)
            if canon in canon_seen:
                continue
            picked = mask
            canon_seen.add(canon)
            if len(_mask_variants(mask)) > 1:
                asymmetric_count += 1
            break
        if picked is None:
            return None
        shapes.append(picked)
    if asymmetric_count == 0:
        return None
    return shapes


def _place_holes(shapes: list[np.ndarray]) -> tuple[np.ndarray, list[dict]] | None:
    max_h = max(int(shape.shape[0]) for shape in shapes)
    max_w = max(int(shape.shape[1]) for shape in shapes)
    total_area = sum(int(shape.sum()) for shape in shapes)
    for _ in range(120):
        inner_h = random.randint(max(max_h + 2, 9), min(16, max_h + 7 + len(shapes)))
        inner_w = random.randint(max(max_w + 2, 9), min(16, max_w + 7 + len(shapes)))
        if inner_h * inner_w < total_area + 10 * len(shapes):
            continue
        occ = np.zeros((inner_h, inner_w), dtype=int)
        placed: list[dict] = []
        okay = True
        order = list(range(len(shapes)))
        random.shuffle(order)
        for idx in order:
            shape = shapes[idx]
            patch = shape.astype(int)
            candidates: list[Coord] = []
            ph, pw = shape.shape
            blocked = _dilate8(occ != 0)
            for r0 in range(1, inner_h - ph):
                for c0 in range(1, inner_w - pw):
                    if bool((blocked[r0 : r0 + ph, c0 : c0 + pw] & shape).any()):
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                okay = False
                break
            r0, c0 = random.choice(candidates)
            occ[r0 : r0 + ph, c0 : c0 + pw][shape] = idx + 1
            placed.append({"index": idx, "shape": shape, "top": r0, "left": c0, "area": int(shape.sum())})
        if not okay:
            continue
        placed.sort(key=lambda item: int(item["index"]))
        order_by_position = sorted(range(len(placed)), key=lambda i: (placed[i]["top"], placed[i]["left"]))
        order_by_area = sorted(range(len(placed)), key=lambda i: int(placed[i]["area"]))
        if order_by_position == order_by_area:
            continue
        return occ, placed
    return None


def _build_frame(inner_occ: np.ndarray, placed_holes: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    inner_h, inner_w = inner_occ.shape
    frame = np.full((inner_h + 2, inner_w + 2), FRAME, dtype=int)
    frame[1:-1, 1:-1][inner_occ != 0] = BG
    expected = frame.copy()
    ranked = sorted(placed_holes, key=lambda item: int(item["area"]))
    return frame, expected, ranked


def _choose_piece_and_palette_colors(n_holes: int) -> tuple[list[int], list[int]]:
    piece_colors = random.sample(COLOR_POOL, n_holes)
    overlap = random.randint(1, max(1, n_holes - 1))
    shared = random.sample(piece_colors, overlap)
    remaining = [c for c in COLOR_POOL if c not in shared]
    extra = random.sample(remaining, n_holes - overlap)
    palette = shared + extra
    random.shuffle(palette)
    return piece_colors, palette


def _render_piece(shape: np.ndarray, color: int, *, require_change: bool = False) -> tuple[np.ndarray, bool]:
    variants = _mask_variants(shape)
    random.shuffle(variants)
    for variant in variants:
        changed = variant.shape != shape.shape or not np.array_equal(variant, shape)
        if require_change and not changed:
            continue
        patch = np.zeros(variant.shape, dtype=int)
        patch[variant] = int(color)
        return patch, changed
    variant = random.choice(variants)
    patch = np.zeros(variant.shape, dtype=int)
    patch[variant] = int(color)
    changed = variant.shape != shape.shape or not np.array_equal(variant, shape)
    return patch, changed


def _build_input() -> tuple[np.ndarray, np.ndarray] | None:
    n_holes = random.randint(3, 5)
    shapes = _generate_shapes(n_holes)
    if shapes is None:
        return None
    placement = _place_holes(shapes)
    if placement is None:
        return None
    inner_occ, placed_holes = placement
    frame, expected, ranked = _build_frame(inner_occ, placed_holes)

    piece_colors, palette = _choose_piece_and_palette_colors(n_holes)
    rank_to_piece_color = [piece_colors[item["index"]] for item in ranked]
    if all(int(pc) == int(pp) for pc, pp in zip(rank_to_piece_color, palette)):
        return None

    for rank, item in enumerate(ranked):
        color = int(palette[rank])
        top = int(item["top"]) + 1
        left = int(item["left"]) + 1
        mask = item["shape"]
        expected[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = color

    fh, fw = frame.shape
    canvas_h = random.randint(max(fh + 6, 20), min(29, fh + 11))
    canvas_w = random.randint(max(fw + 7, 21), min(29, fw + 13))
    if canvas_h <= fh + 3 or canvas_w <= fw + 3:
        return None
    grid = np.zeros((canvas_h, canvas_w), dtype=int)

    frame_top = random.randint(1, canvas_h - fh - 4)
    frame_left = random.randint(1, canvas_w - fw - 1)
    _paste(grid, frame, frame_top, frame_left)

    palette_row = random.randint(frame_top + fh + 1, canvas_h - 2)
    palette_cols = list(range(frame_left + 1, frame_left + 1 + 2 * n_holes, 2))
    if palette_cols[-1] >= canvas_w - 1:
        return None
    for c, color in zip(palette_cols, palette):
        grid[palette_row, c] = int(color)

    transformed_any = False
    piece_order = list(range(n_holes))
    random.shuffle(piece_order)
    change_candidates = [i for i, shape in enumerate(shapes) if len(_mask_variants(shape)) > 1]
    must_change = random.choice(change_candidates) if change_candidates else None
    for idx in piece_order:
        patch, changed = _render_piece(shapes[idx], piece_colors[idx], require_change=(idx == must_change))
        transformed_any |= changed
        try:
            r0, c0 = _random_free_location(grid, patch, transparent=BG, border=1, padding=1)
        except ValueError:
            return None
        _paste(grid, patch, r0, c0, transparent=BG)
    if must_change is not None and not transformed_any:
        return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    if tuple(map(int, solved.shape)) != tuple(map(int, frame.shape)):
        return None

    frame_crop = solved
    hole_colors = [int(frame_crop[item["top"] + 1, item["left"] + 1]) for item in ranked]
    if hole_colors != palette:
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_input()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
