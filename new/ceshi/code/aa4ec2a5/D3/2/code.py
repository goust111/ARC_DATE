# Variant of aa4ec2a5
# Dimension: D3 — 源对象的连通性与形态 (src)
# Change: 蓝色4连通对象（可能有多个），部分对象内部封闭了背景色洞... → 对象附带标记像素指示类别

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the D3 variant
# a) Concrete behavior:
#    - Every blue 4-connected object now has one single-pixel marker attached on the outside of
#      its bbox by 4-neighbor contact.
#    - Green (3) means "apply the hollow-object branch"; orange (7) means "apply the solid branch".
#    - The branch is chosen from the marker, not from topology. A green-marked solid object turns
#      teal even with no hole, while an orange-marked holed object keeps its hole as background and
#      gets a red inner outline around that hole.
#    - Example: a 3x3 ring with an orange marker used to become teal with a magenta center in the
#      original task; here it stays blue, the center stays background, and the inner border becomes red.
# b) Complexity calibration:
#    - Only two marker colors are used, so the symbolic mapping is learnable from 2-4 examples.
#    - Each grid contains both marker colors and at least one marker/topology mismatch, so the new
#      rule is visible and does not collapse back to "detect holes".
# c) Implicit assumptions from the old main():
#    - `find_components(..., monochromatic=True)` still works, but it must only iterate blue objects;
#      otherwise markers would be mistaken for objects.
#    - `_interior_holes(...)` can no longer decide the branch; it is only consulted after the marker
#      selects the hollow-treatment path.
#    - The old unconditional "holes imply teal/magenta" writeback would be wrong, because a real hole
#      may now be intentionally ignored by an orange marker.
# d) Placement legality:
#    - Blue objects keep the original non-overlap / spacing logic.
#    - Markers are placed only on exterior bbox-adjacent cells, never inside topological holes, so
#      solver-side marker lookup is unambiguous and visually attached to the intended object.
# e) Visual observability:
#    - Each sample contains both green and orange markers.
#    - Mismatches guarantee that at least one object contradicts what pure topology would predict.
# f) Visual cleanliness:
#    - Markers are isolated single pixels outside the objects, so they do not merge with blue shapes.
#    - Red outlines are still drawn only on background cells, so markers remain visible instead of
#      being overwritten.
# g) New solver assumptions audited:
#    - Each blue object has exactly one adjacent marker in {3, 7}; generator enforces this.
#    - A marker is 4-neighbor to its object and not placed in an internal cavity.
#    - The solver defaults to the solid branch if malformed input lacks a green marker; generated
#      inputs always satisfy the stronger one-marker-per-object condition.


MARKER_HOLLOW = 3
MARKER_SOLID = 7
MARKER_COLORS = {MARKER_HOLLOW, MARKER_SOLID}


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


def _marker_colors_for_component(grid: np.ndarray, coords: np.ndarray) -> set[int]:
    h, w = grid.shape
    colors: set[int] = set()
    seen: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            val = int(grid[rr, cc])
            if val in MARKER_COLORS:
                colors.add(val)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Outline each blue object, but choose its interior treatment from a nearby marker pixel.

    Input structure:
    - A mostly uniform background contains several blue 4-connected components.
    - Every blue component has one adjacent single-pixel marker: green (3) or orange (7).
    - Some blue components may physically contain enclosed background holes, but that topology is
      no longer the category signal.

    Rule:
    - Draw a one-cell 8-neighborhood red outline around every blue component.
    - A green-adjacent component is processed with the hollow branch: recolor the blue cells to
      teal (8), recolor any enclosed background holes to magenta (6), and treat those holes as
      filled when computing the outline.
    - An orange-adjacent component is processed with the solid branch: keep it blue (1) and ignore
      any real holes, so those holes stay background and may receive red inner outline cells.

    Key insight:
    - The local marker color encodes which branch to use; the solver must decode that symbol rather
      than infer the class from the object's topology.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[1]):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        marker_colors = _marker_colors_for_component(g, coords)
        treat_as_hollow = MARKER_HOLLOW in marker_colors

        solid = mask.copy()
        if treat_as_hollow:
            out[mask] = 8
            for hole in _interior_holes(mask, coords):
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator ---------------

# Canvas / grid bounds
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1

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
CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300
MARKER_NEIGHBOR_MIN = 2

# Top-level generation
GEN_OUTER_RETRIES = 300
ROLE_RETRIES = 80


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


def _choose_role_sets(num_components: int) -> tuple[set[int], set[int]] | None:
    indices = list(range(num_components))
    topo_hi = min(TOPO_HOLLOW_COUNT_RANGE[1], num_components - 1)
    marked_hi = min(MARKED_HOLLOW_COUNT_RANGE[1], num_components - 1)
    for _ in range(ROLE_RETRIES):
        topo_count = random.randint(TOPO_HOLLOW_COUNT_RANGE[0], topo_hi)
        marked_count = random.randint(MARKED_HOLLOW_COUNT_RANGE[0], marked_hi)
        topo_hollow = set(random.sample(indices, topo_count))
        marked_hollow = set(random.sample(indices, marked_count))
        if topo_hollow != marked_hollow:
            return topo_hollow, marked_hollow
    return None


def _global_coords(r0: int, c0: int, mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return np.column_stack((rows + r0, cols + c0))


def _marker_candidates(shape: tuple[int, int], coords: np.ndarray, occupied: np.ndarray) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox_coords(coords)
    coord_set = {tuple(map(int, rc)) for rc in coords}
    candidates: set[tuple[int, int]] = set()

    for r, c in coord_set:
        if r == r0 and r - 1 >= 0 and not occupied[r - 1, c]:
            candidates.add((r - 1, c))
        if r == r1 and r + 1 < h and not occupied[r + 1, c]:
            candidates.add((r + 1, c))
        if c == c0 and c - 1 >= 0 and not occupied[r, c - 1]:
            candidates.add((r, c - 1))
        if c == c1 and c + 1 < w and not occupied[r, c + 1]:
            candidates.add((r, c + 1))

    return list(candidates)


def _markers_too_close(marker_positions: list[tuple[int, int]], r: int, c: int) -> bool:
    for mr, mc in marker_positions:
        if max(abs(mr - r), abs(mc - c)) < MARKER_NEIGHBOR_MIN:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    role_sets = _choose_role_sets(num_components)
    if role_sets is None:
        return None
    topo_hollow, marked_hollow = role_sets

    components: list[np.ndarray] = []
    for i in range(num_components):
        if i in topo_hollow:
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
                if mask is not None and mask.sum() >= HOLLOW_MIN_AREA:
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
    marker_specs: list[tuple[int, int, list[tuple[int, int]]]] = []

    for item in placed:
        idx = int(item["idx"])
        r = int(item["r"])
        c = int(item["c"])
        mask = item["mask"]
        assert isinstance(mask, np.ndarray)
        coords = _global_coords(r, c, mask)
        candidates = _marker_candidates(grid.shape, coords, occupied)
        if not candidates:
            return None
        color = MARKER_HOLLOW if idx in marked_hollow else MARKER_SOLID
        marker_specs.append((color, idx, candidates))

    marker_specs.sort(key=lambda spec: len(spec[2]))
    for color, _idx, candidates in marker_specs:
        random.shuffle(candidates)
        placed_ok = False
        for mr, mc in candidates:
            if occupied[mr, mc]:
                continue
            if _markers_too_close(marker_positions, mr, mc):
                continue
            grid[mr, mc] = color
            occupied[mr, mc] = True
            marker_positions.append((mr, mc))
            placed_ok = True
            break
        if not placed_ok:
            return None

    blue_components = find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True, colors=[FG_COLOR])
    if len(blue_components) != num_components:
        return None
    if any(len(_marker_colors_for_component(grid, comp["coords"])) != 1 for comp in blue_components):
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
