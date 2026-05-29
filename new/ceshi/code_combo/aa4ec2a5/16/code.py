# Combo variant of aa4ec2a5 (index 16)
# combo_spec: D3/2+D13/1
# Name: 标记像素指示类别+特殊标记像素指示每个对象类别

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the combo variant
# a) Concrete behavior:
#    - Each blue object now carries a two-pixel marker ray attached outside its bbox.
#    - The inner marker (4-neighbor to the object) selects the processing branch:
#      green (3) => treat as hollow, orange (7) => treat as solid.
#    - The outer marker (one more step along the same ray) selects the color family:
#      red (2) => original-looking palette, maroon (9) => alternate palette.
#    - Example: consider a 3x3 ring. In the original task it must become teal with a magenta
#      center. Here, if its inner marker is orange it stays "solid-treated" so the center remains
#      background and receives an inner outline; if its outer marker is maroon that outline turns
#      maroon instead of red and the kept/recolored body uses the alternate palette.
# b) Complexity calibration:
#    - Only two status markers and two palette markers are used, so the symbolic systems are small.
#    - The generator enforces both status values, both palette values, and cross-combinations, so
#      the task cannot be solved by only reading one of the two marker pixels.
# c) Implicit assumptions audited from the old main():
#    - `find_components(..., monochromatic=True)` must now be restricted to blue cells only;
#      otherwise the four marker colors would be mistaken for foreground objects.
#    - `_interior_holes(...)` is no longer the category test. It is consulted only after the inner
#      marker chooses the hollow-treatment branch.
#    - The old fixed-color writes are invalid because the outer marker now chooses the palette.
#    - Outline color can no longer be hard-coded red; it depends on the paired outer marker.
# d) Placement legality:
#    - Blue objects are still generated first as disjoint 4-connected components.
#    - A marker pair is placed on a straight outward ray from a bbox edge cell, with two free cells
#      available, so the pair is visually attached to exactly one object and never inside a hole.
#    - Extra border/spacing is kept so the second marker cell does not collide with another object.
# e) Visual observability:
#    - Every sample contains both inner-marker meanings and both outer-marker meanings.
#    - At least one object contradicts topology, so "has a real hole" is visibly not the deciding
#      factor anymore.
#    - At least one pair of objects shares a status but differs in palette, and another pair shares
#      a palette but differs in status, making the semantic split human-visible.
# f) Visual cleanliness:
#    - The two markers form a short straight ray instead of loose pixels, so pairing is obvious.
#    - Markers occupy cells that would otherwise be background/outline cells, which keeps them legible.
#    - Marker rays are kept separated from each other, avoiding clutter around object borders.
# g) New solver assumptions enforced by the generator:
#    - Each blue object has exactly one valid marker ray whose first cell is in {3, 7} and second
#      cell is in {2, 9}.
#    - The ray points straight away from a bbox side, so solver-side detection by local ray search
#      is unambiguous.
#    - The fallback branch/palette only exists for malformed inputs; generated samples satisfy the
#      stronger one-ray-per-object contract.


FG_COLOR = 1
BG_COLOR = 4

STATUS_MARKER_HOLLOW = 3
STATUS_MARKER_SOLID = 7
STATUS_MARKER_COLORS = {STATUS_MARKER_HOLLOW, STATUS_MARKER_SOLID}

PALETTE_MARKER_RED = 2
PALETTE_MARKER_MAROON = 9
PALETTE_MARKER_COLORS = {PALETTE_MARKER_RED, PALETTE_MARKER_MAROON}

PALETTES = {
    PALETTE_MARKER_RED: {
        "solid": 1,
        "hollow": 8,
        "hole": 6,
        "outline": 2,
    },
    PALETTE_MARKER_MAROON: {
        "solid": 3,
        "hollow": 5,
        "hole": 7,
        "outline": 9,
    },
}

CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _interior_holes(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]

    holes: list[np.ndarray] = []
    for hole in components_mask(~sub, connectivity=4):
        touches_edge = np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == sub.shape[0] - 1)
            | (hole[:, 1] == sub.shape[1] - 1)
        )
        if not bool(touches_edge):
            shifted = hole.copy()
            shifted[:, 0] += int(r0)
            shifted[:, 1] += int(c0)
            holes.append(shifted)
    return holes


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outline = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            outline[rr[ok], cc[ok]] = True
    outline &= ~mask
    return outline


def _marker_pairs_for_component(
    grid: np.ndarray, coords: np.ndarray
) -> list[tuple[tuple[int, int], tuple[int, int], int, int]]:
    h, w = grid.shape
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    pairs: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []

    for r, c in coords:
        for dr, dc in CARDINAL_DIRS:
            sr = int(r + dr)
            sc = int(c + dc)
            pr = int(r + 2 * dr)
            pc = int(c + 2 * dc)
            if not (0 <= sr < h and 0 <= sc < w and 0 <= pr < h and 0 <= pc < w):
                continue
            status_color = int(grid[sr, sc])
            palette_color = int(grid[pr, pc])
            if status_color not in STATUS_MARKER_COLORS or palette_color not in PALETTE_MARKER_COLORS:
                continue
            key = ((sr, sc), (pr, pc))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(((sr, sc), (pr, pc), status_color, palette_color))

    return pairs


def _find_marker_pair(grid: np.ndarray, coords: np.ndarray) -> tuple[int, int]:
    pairs = _marker_pairs_for_component(grid, coords)

    if pairs:
        return pairs[0][2], pairs[0][3]
    return STATUS_MARKER_SOLID, PALETTE_MARKER_RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Outline each blue object, but decode two nearby marker pixels to decide how it transforms.

    Input structure:
    - A mostly uniform background contains several blue 4-connected components.
    - Every blue component has a two-pixel straight marker ray outside its bbox.
    - The marker pixel touching the object is green (3) or orange (7), and the next pixel farther
      away is red (2) or maroon (9).
    - Some blue components may physically contain enclosed background holes, but topology is no
      longer enough to determine the final treatment by itself.

    Rule:
    - Draw a one-cell 8-neighborhood outline around every blue component, using the color chosen by
      the outer marker pixel.
    - If the inner marker is green, process the object with the hollow branch selected by that
      object's palette: recolor the body, fill any enclosed holes with the palette's hole color,
      and treat those holes as filled when computing the outline.
    - If the inner marker is orange, process the object with the solid branch selected by that
      object's palette: recolor/keep only the object's blue cells and ignore real holes, so holes
      stay background and can receive inner outline cells.

    Key insight:
    - The two marker pixels have different jobs. The inner one chooses hollow-vs-solid behavior,
      while the outer one chooses which color family that behavior uses.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[FG_COLOR]):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        status_color, palette_color = _find_marker_pair(g, coords)
        palette = PALETTES.get(palette_color, PALETTES[PALETTE_MARKER_RED])
        treat_as_hollow = status_color == STATUS_MARKER_HOLLOW

        solid = mask.copy()
        if treat_as_hollow:
            out[mask] = palette["hollow"]
            for hole in _interior_holes(mask, coords):
                out[hole[:, 0], hole[:, 1]] = palette["hole"]
                solid[hole[:, 0], hole[:, 1]] = True
        else:
            out[mask] = palette["solid"]

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = palette["outline"]

    return out


# --------------- generator ---------------

# Canvas / grid bounds
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)

# Component counts
COMP_COUNT_RANGE = (3, 4)
TOPO_HOLLOW_COUNT_RANGE = (1, 2)
MARKED_HOLLOW_COUNT_RANGE = (1, 2)

# Polyomino construction (solid shapes)
POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

# Small solid shapes
SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

# Hollow component construction
HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 50
HOLLOW_MIN_AREA = 10

# Placement
CHEBYSHEV_MIN = 5
BORDER_MIN = 2
PLACEMENT_RETRIES = 500
MARKER_NEIGHBOR_MIN = 2

# Top-level generation
GEN_OUTER_RETRIES = 400
ROLE_RETRIES = 120


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _has_interior_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for bg_reg in components_mask(~mask, connectivity=4):
        touches_edge = np.any(
            (bg_reg[:, 0] == 0)
            | (bg_reg[:, 1] == 0)
            | (bg_reg[:, 0] == h - 1)
            | (bg_reg[:, 1] == w - 1)
        )
        if not bool(touches_edge):
            return True
    return False


def _rect_dims(area: int) -> tuple[int, int] | None:
    factors = [
        (i, area // i)
        for i in range(1, int(area**0.5) + 1)
        if area % i == 0 and i <= HOLE_DIM_MAX and area // i <= HOLE_DIM_MAX
    ]
    if not factors:
        return None
    h, w = random.choice(factors)
    if random.random() < 0.5:
        h, w = w, h
    return h, w


def _chebyshev_conflict(rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, pm: np.ndarray) -> bool:
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _make_polyomino() -> np.ndarray | None:
    num_rects = random.randint(*POLY_NUM_RECTS_RANGE)
    rects = [
        (random.randint(*POLY_RECT_DIM_RANGE), random.randint(*POLY_RECT_DIM_RANGE))
        for _ in range(num_rects)
    ]

    canvas_h = sum(r[0] for r in rects) + POLY_CANVAS_PAD
    canvas_w = sum(r[1] for r in rects) + POLY_CANVAS_PAD
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    r0, c0 = 2, 2
    mask[r0 : r0 + rects[0][0], c0 : c0 + rects[0][1]] = True

    for i in range(1, num_rects):
        rh, rw = rects[i]
        for _ in range(POLY_PLACEMENT_RETRIES):
            nr = random.randint(max(0, r0 - rh + 1), r0 + rects[0][0] + 2)
            nc = random.randint(max(0, c0 - rw + 1), c0 + rects[0][1] + 2)
            if nr + rh > canvas_h or nc + rw > canvas_w or nr < 0 or nc < 0:
                continue
            if mask[nr : nr + rh, nc : nc + rw].any():
                mask[nr : nr + rh, nc : nc + rw] = True
                break

    mask = _crop_mask(mask)
    if mask is None or not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _extend_mask(mask: np.ndarray, base_h: int, base_w: int) -> np.ndarray:
    if random.random() >= HOLLOW_EXTENSION_PROB:
        return mask

    side = random.choice(["top", "bottom", "left", "right"])
    ext_h = random.randint(*HOLLOW_EXT_H_RANGE)

    if side in ("top", "bottom"):
        ext_w = min(random.randint(2, max(2, base_w)), base_w)
        offset = random.randint(0, base_w - ext_w)
        new_mask = np.zeros((base_h + ext_h, base_w), dtype=bool)
        if side == "top":
            new_mask[ext_h:, :] = mask
            new_mask[0:ext_h, offset : offset + ext_w] = True
        else:
            new_mask[:base_h, :] = mask
            new_mask[base_h : base_h + ext_h, offset : offset + ext_w] = True
    else:
        ext_h_actual = min(ext_h, base_h)
        ext_w_actual = random.randint(*HOLLOW_EXT_SIDE_W_RANGE)
        offset = random.randint(0, base_h - ext_h_actual)
        new_mask = np.zeros((base_h, base_w + ext_w_actual), dtype=bool)
        if side == "left":
            new_mask[:, ext_w_actual:] = mask
            new_mask[offset : offset + ext_h_actual, 0:ext_w_actual] = True
        else:
            new_mask[:, :base_w] = mask
            new_mask[offset : offset + ext_h_actual, base_w : base_w + ext_w_actual] = True

    return new_mask


def _carve_hole(mask: np.ndarray, hh: int, hw: int) -> np.ndarray | None:
    mh, mw = mask.shape
    for _ in range(HOLLOW_HOLE_SEARCH_RETRIES):
        hr = random.randint(1, mh - hh - 1)
        hc = random.randint(1, mw - hw - 1)
        if not mask[hr : hr + hh, hc : hc + hw].all():
            continue
        if (
            not mask[hr - 1, hc : hc + hw].any()
            or not mask[hr + hh, hc : hc + hw].any()
            or not mask[hr : hr + hh, hc - 1].any()
            or not mask[hr : hr + hh, hc + hw].any()
        ):
            continue

        carved = mask.copy()
        carved[hr : hr + hh, hc : hc + hw] = False
        if not is_contiguous(carved.astype(int), background=0, connectivity=4):
            continue
        cropped = _crop_mask(carved)
        if cropped is not None and _has_interior_hole(cropped):
            return cropped
    return None


def _make_hollow_component(target_hole_area: int) -> np.ndarray | None:
    dims = _rect_dims(target_hole_area)
    if dims is None:
        return None
    hh, hw = dims

    pad_top = random.randint(*HOLLOW_PAD_RANGE)
    pad_bot = random.randint(*HOLLOW_PAD_RANGE)
    pad_left = random.randint(*HOLLOW_PAD_RANGE)
    pad_right = random.randint(*HOLLOW_PAD_RANGE)

    base_h = hh + pad_top + pad_bot
    base_w = hw + pad_left + pad_right
    mask = np.ones((base_h, base_w), dtype=bool)

    num_notches = random.randint(0, HOLLOW_MAX_NOTCHES)
    for _ in range(num_notches):
        corner = random.choice(["tl", "tr", "bl", "br"])
        nh = random.randint(1, max(1, pad_top - 1) if corner[0] == "t" else max(1, pad_bot - 1))
        nw = random.randint(1, max(1, pad_left - 1) if corner[1] == "l" else max(1, pad_right - 1))
        if corner == "tl":
            mask[0:nh, 0:nw] = False
        elif corner == "tr":
            mask[0:nh, base_w - nw : base_w] = False
        elif corner == "bl":
            mask[base_h - nh : base_h, 0:nw] = False
        else:
            mask[base_h - nh : base_h, base_w - nw : base_w] = False

    mask = _extend_mask(mask, base_h, base_w)
    return _carve_hole(mask, hh, hw)


def _make_small_shape() -> np.ndarray:
    size = random.randint(*SMALL_SHAPE_SIZE_RANGE)
    if size == 1:
        return np.ones((1, 1), dtype=bool)
    if size == 2:
        return np.ones((1, 2), dtype=bool) if random.random() < 0.5 else np.ones((2, 1), dtype=bool)
    if size == 3:
        return random.choice(
            [
                np.array([[1, 1], [1, 0]], dtype=bool),
                np.array([[1, 0], [1, 1]], dtype=bool),
                np.array([[1, 1, 1]], dtype=bool),
                np.array([[1], [1], [1]], dtype=bool),
            ]
        )
    return np.ones((2, 2), dtype=bool)


def _choose_role_sets(
    num_components: int,
) -> tuple[set[int], set[int], dict[int, int]] | None:
    indices = list(range(num_components))
    topo_hi = min(TOPO_HOLLOW_COUNT_RANGE[1], num_components - 1)
    marked_hi = min(MARKED_HOLLOW_COUNT_RANGE[1], num_components - 1)

    for _ in range(ROLE_RETRIES):
        topo_count = random.randint(TOPO_HOLLOW_COUNT_RANGE[0], topo_hi)
        marked_count = random.randint(MARKED_HOLLOW_COUNT_RANGE[0], marked_hi)
        topo_hollow = set(random.sample(indices, topo_count))
        marked_hollow = set(random.sample(indices, marked_count))
        palette_by_idx = {
            idx: random.choice([PALETTE_MARKER_RED, PALETTE_MARKER_MAROON]) for idx in indices
        }

        if topo_hollow == marked_hollow:
            continue
        if len(set(palette_by_idx.values())) < 2:
            continue

        same_status_diff_palette = any(
            ((i in marked_hollow) == (j in marked_hollow)) and (palette_by_idx[i] != palette_by_idx[j])
            for i in indices
            for j in indices
            if i < j
        )
        same_palette_diff_status = any(
            (palette_by_idx[i] == palette_by_idx[j]) and ((i in marked_hollow) != (j in marked_hollow))
            for i in indices
            for j in indices
            if i < j
        )
        if not same_status_diff_palette or not same_palette_diff_status:
            continue

        return topo_hollow, marked_hollow, palette_by_idx

    return None


def _global_coords(r0: int, c0: int, mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return np.column_stack((rows + r0, cols + c0))


def _ray_cells_free(
    shape: tuple[int, int],
    coords_set: set[tuple[int, int]],
    occupied: np.ndarray,
    status_pos: tuple[int, int],
    palette_pos: tuple[int, int],
) -> bool:
    h, w = shape
    for r, c in (status_pos, palette_pos):
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False

    sr, sc = status_pos
    pr, pc = palette_pos

    attached = False
    for nr, nc in ((sr - 1, sc), (sr + 1, sc), (sr, sc - 1), (sr, sc + 1)):
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if occupied[nr, nc] and (nr, nc) not in coords_set:
            return False
        if (nr, nc) in coords_set:
            attached = True
    if not attached:
        return False

    for nr, nc in ((pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)):
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if occupied[nr, nc]:
            return False

    return True


def _marker_ray_candidates(
    shape: tuple[int, int], coords: np.ndarray, occupied: np.ndarray
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    coords_set = {tuple(map(int, rc)) for rc in coords}
    candidates: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for r, c in coords_set:
        if r == r0:
            status_pos = (r - 1, c)
            palette_pos = (r - 2, c)
            if _ray_cells_free(shape, coords_set, occupied, status_pos, palette_pos):
                candidates.add((status_pos, palette_pos))
        if r == r1:
            status_pos = (r + 1, c)
            palette_pos = (r + 2, c)
            if _ray_cells_free(shape, coords_set, occupied, status_pos, palette_pos):
                candidates.add((status_pos, palette_pos))
        if c == c0:
            status_pos = (r, c - 1)
            palette_pos = (r, c - 2)
            if _ray_cells_free(shape, coords_set, occupied, status_pos, palette_pos):
                candidates.add((status_pos, palette_pos))
        if c == c1:
            status_pos = (r, c + 1)
            palette_pos = (r, c + 2)
            if _ray_cells_free(shape, coords_set, occupied, status_pos, palette_pos):
                candidates.add((status_pos, palette_pos))

    return list(candidates)


def _markers_too_close(
    marker_positions: list[tuple[int, int]], status_pos: tuple[int, int], palette_pos: tuple[int, int]
) -> bool:
    for mr, mc in marker_positions:
        if max(abs(mr - status_pos[0]), abs(mc - status_pos[1])) < MARKER_NEIGHBOR_MIN:
            return True
        if max(abs(mr - palette_pos[0]), abs(mc - palette_pos[1])) < MARKER_NEIGHBOR_MIN:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    role_sets = _choose_role_sets(num_components)
    if role_sets is None:
        return None
    topo_hollow, marked_hollow, palette_by_idx = role_sets

    components: list[np.ndarray] = []
    for i in range(num_components):
        if i in topo_hollow:
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
                if mask is not None and int(mask.sum()) >= HOLLOW_MIN_AREA:
                    components.append(mask)
                    found = True
                    break
            if not found:
                return None
        elif random.random() < SMALL_SHAPE_PROB:
            components.append(_make_small_shape())
        else:
            for _ in range(POLY_GEN_RETRIES):
                mask = _make_polyomino()
                if mask is not None:
                    components.append(mask)
                    break
            else:
                return None

    if len(components) != num_components:
        return None

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[dict[str, object]] = []
    order = sorted(range(num_components), key=lambda i: -int(components[i].sum()))

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape
        placed_ok = False

        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, H - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, W - mw - BORDER_MIN))
            if r + mh > H - BORDER_MIN or c + mw > W - BORDER_MIN:
                continue

            rows_a, cols_a = np.where(mask)
            rows_a, cols_a = rows_a + r, cols_a + c

            conflict = False
            for other in placed:
                pr = int(other["r"])
                pc = int(other["c"])
                pm = other["mask"]
                assert isinstance(pm, np.ndarray)

                row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + pm.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_a, cols_a, pr, pc, pm):
                    conflict = True
                    break

            if not conflict:
                placed.append({"idx": idx, "r": r, "c": c, "mask": mask})
                placed_ok = True
                break

        if not placed_ok:
            return None

    for item in placed:
        r = int(item["r"])
        c = int(item["c"])
        mask = item["mask"]
        assert isinstance(mask, np.ndarray)
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    occupied = grid != BG_COLOR
    marker_positions: list[tuple[int, int]] = []
    marker_specs: list[tuple[int, int, list[tuple[tuple[int, int], tuple[int, int]]]]] = []

    for item in placed:
        idx = int(item["idx"])
        r = int(item["r"])
        c = int(item["c"])
        mask = item["mask"]
        assert isinstance(mask, np.ndarray)
        coords = _global_coords(r, c, mask)
        candidates = _marker_ray_candidates(grid.shape, coords, occupied)
        if not candidates:
            return None
        status_color = STATUS_MARKER_HOLLOW if idx in marked_hollow else STATUS_MARKER_SOLID
        palette_color = palette_by_idx[idx]
        marker_specs.append((status_color, palette_color, candidates))

    marker_specs.sort(key=lambda spec: len(spec[2]))
    for status_color, palette_color, candidates in marker_specs:
        random.shuffle(candidates)
        placed_ok = False
        for status_pos, palette_pos in candidates:
            if occupied[status_pos] or occupied[palette_pos]:
                continue
            if _markers_too_close(marker_positions, status_pos, palette_pos):
                continue
            grid[status_pos] = status_color
            grid[palette_pos] = palette_color
            occupied[status_pos] = True
            occupied[palette_pos] = True
            marker_positions.extend([status_pos, palette_pos])
            placed_ok = True
            break
        if not placed_ok:
            return None

    blue_components = find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True, colors=[FG_COLOR])
    if len(blue_components) != num_components:
        return None
    if any(len(_marker_pairs_for_component(grid, comp["coords"])) != 1 for comp in blue_components):
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
