# Combo variant of 53fb4810 (index 24)
# combo_spec: D3/2+D5/0+D15/2
# Name: 大瓦片镜像多色精灵

from __future__ import annotations

import itertools

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# a) Concrete behavior:
#    Start from the original rule "a marker attached to a plus launches a ray away from
#    that plus". This combo changes three linked parts at once. First, the plus is no
#    longer monochrome: every repeated anchor is a two-color sprite with a core color and
#    a boundary / arm-tip color. Second, the marker is no longer a tiny strip: it is a
#    larger filled rectangle such as 3x2, 2x3, 3x3, 4x2, or 4x3 whose cross-axis span
#    still matches the chosen nozzle exactly. Third, repetition is no longer plain
#    periodic copy; tile units alternate original / mirrored / original / mirrored along
#    the ray axis. Example: if a 3x2 seed touches the right tip color of a plus, the
#    output shows [seed][left-right mirror][seed][left-right mirror] until the border.
#    The input keeps exactly one fully expanded clue ray so that the alternating mirror
#    rhythm is visibly demonstrated.
# b) Complexity calibration:
#    Large seeds are restricted to a small family: nozzle widths 2 or 3, along-ray sizes
#    2 to 4, area at least 6, at least 2 colors, and non-trivial row/column variation.
#    This makes the combo visually distinct from the original narrow markers without
#    exploding the search space. The two-color plus template is shared across all anchors
#    in a grid, so the "composite sprite" idea repeats often enough for a human to see.
# c) Implicit assumption audit:
#    - The original solver assumed one color alone identifies every plus. That fails
#      under D15/2, so `main()` must search for a color pair whose union forms several
#      repeated large symmetric components.
#    - The original adjacency scan over all plus pixels becomes too permissive. In this
#      combo the marker is attached specifically to the tip-color boundary cells, so
#      candidate markers are collected only from those cells.
#    - The old propagation helper used a simple modulo repeat. D3/2 breaks that because
#      each tile unit depends on parity; the helper must step one full tile at a time and
#      choose original vs mirrored orientation by unit index.
#    - `crop_bbox(...)` still assumes the marker component's bbox is the repeating period.
#      To preserve that invariant with D5/0, the generator keeps every compact seed as a
#      fully filled rectangle. The pre-expanded clue ray is also safe because its bbox
#      already reaches the border, so re-extension adds nothing.
#    - The old assert-based nozzle checks can crash during generation retries. Here they
#      are explicit guards because the combo introduces more degrees of freedom.
# d) Placement legality:
#    Generation remains output-first. Each plus sprite uses the same footprint and the
#    same two role colors; each ray is projected to the boundary before any erasing step.
#    Placements are rejected if a plus overlaps an existing object, touches an existing
#    ray too closely, or if its full projected alternating ray would hit another plus or
#    ray. Marker colors always exclude both plus colors, so the solver's pair-detection
#    logic stays clean.
# e) Visual observability:
#    A human can infer the rule from a few examples because all three changes are visible:
#    anchors are clearly two-color objects, seeds are visibly larger textured rectangles,
#    and the single pre-expanded clue ray directly shows the mirror alternation. The
#    generator forbids seeds that equal their directional mirror, so alternation always
#    changes appearance rather than being a hidden parity detail.
# f) Visual cleanliness:
#    Rays do not overlap one another or plus cells, and pluses are kept away from existing
#    rays. Only two colors are reserved for the anchors; the seed palette excludes them.
#    This keeps anchors, compact seeds, and repeated bands easy to separate by eye.
# g) Change inferability:
#    No rule depends on numeric color ids. Colors only encode structural roles: two shared
#    anchor colors, then seed colors copied under geometric mirroring. Because the clue ray
#    exhibits the same transformation used everywhere else, the combo remains learnable.
# Main/generator contract after the change:
#    `main()` assumes the repeated object is a two-color plus union, the marker touches
#    the shared tip color, the seed bbox is a full rectangular period whose cross span
#    matches the nozzle, and the correct continuation is tile-by-tile mirror alternation.
#    `generate_input()` guarantees those conditions by construction.


def _is_symmetric_plus_mask(mask: np.ndarray) -> bool:
    """Heuristic footprint check for the repeated plus-like anchor components."""
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return False
    if not mask.any():
        return False
    return bool(np.array_equal(mask, mask[::-1, :]) and np.array_equal(mask, mask[:, ::-1]))


def detect_plus_color_pair(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> tuple[int, int] | None:
    """
    Choose the pair of non-bg colors whose union forms the repeated plus components.

    Candidate pair requirements:
    - at least two big components exist,
    - every big component has bbox >= (min_h x min_w),
    - each big component is LR+UD symmetric,
    - all big components share the same cropped footprint.
    Tie-break: more matching components, then larger total area.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    colors = sorted(set(map(int, grid.flatten())) - {bg})
    best_key: tuple[int, int] | None = None
    best_pair: tuple[int, int] | None = None

    for a, b in itertools.combinations(colors, 2):
        union = (grid == int(a)) | (grid == int(b))
        lab, n = label_mask(union, connectivity=int(connectivity))
        big_masks: list[np.ndarray] = []
        total_area = 0

        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < int(min_h) or w < int(min_w):
                continue
            comp = lab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(k)
            if not _is_symmetric_plus_mask(comp):
                continue
            big_masks.append(comp)
            total_area += int(coords.shape[0])

        if len(big_masks) < 2:
            continue
        if any(not np.array_equal(m, big_masks[0]) for m in big_masks[1:]):
            continue

        key = (int(len(big_masks)), int(total_area))
        if best_key is None or key > best_key:
            best_key = key
            best_pair = (int(a), int(b))

    return best_pair


def detect_tip_color(
    grid: np.ndarray,
    plus_lab: np.ndarray,
    n_plus: int,
    colors: tuple[int, int],
) -> int | None:
    """Identify which plus color occupies the outer nozzle edges for every anchor."""
    for col in colors:
        ok = True
        for pk in range(1, int(n_plus) + 1):
            plus_mask = plus_lab == int(pk)
            coords = np.argwhere(plus_mask)
            if coords.size == 0:
                ok = False
                break
            r0, c0, r1, c1 = bbox_coords(coords)
            sub_mask = plus_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
            sub_grid = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

            top_cols = np.where(sub_mask[0])[0]
            bottom_cols = np.where(sub_mask[-1])[0]
            left_rows = np.where(sub_mask[:, 0])[0]
            right_rows = np.where(sub_mask[:, -1])[0]
            if (
                top_cols.size == 0
                or bottom_cols.size == 0
                or left_rows.size == 0
                or right_rows.size == 0
            ):
                ok = False
                break
            if not np.all(sub_grid[0, top_cols] == int(col)):
                ok = False
                break
            if not np.all(sub_grid[-1, bottom_cols] == int(col)):
                ok = False
                break
            if not np.all(sub_grid[left_rows, 0] == int(col)):
                ok = False
                break
            if not np.all(sub_grid[right_rows, -1] == int(col)):
                ok = False
                break
        if ok:
            return int(col)
    return None


def _mirror_tile_for_direction(tile: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    """Mirror a tile along the ray axis."""
    dr, dc = map(int, direction)
    if dc != 0:
        return np.fliplr(tile)
    if dr != 0:
        return np.flipud(tile)
    raise ValueError("_mirror_tile_for_direction: zero direction")


def _tile_variant(tile: np.ndarray, direction: tuple[int, int], unit_index: int) -> np.ndarray:
    """Even units use the original tile; odd units use the directional mirror."""
    return _mirror_tile_for_direction(tile, direction) if unit_index % 2 == 1 else tile


def extend_alternating_mirror_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Extend a ray by repeating full tile units with original/mirror alternation."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    dr = 0 if dr == 0 else int(dr / abs(dr))
    dc = 0 if dc == 0 else int(dc / abs(dc))
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    def put(rr: int, cc: int, v: int) -> None:
        if not (0 <= rr < H and 0 <= cc < W):
            return
        if int(v) == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    step_r = dr * h
    step_c = dc * w
    unit = 1
    while True:
        base_r = r0 + unit * step_r
        base_c = c0 + unit * step_c
        if dr == 1 and base_r >= H:
            break
        if dr == -1 and base_r + h <= 0:
            break
        if dc == 1 and base_c >= W:
            break
        if dc == -1 and base_c + w <= 0:
            break

        cur = _tile_variant(tile, (dr, dc), unit)
        for i in range(h):
            for j in range(w):
                put(base_r + i, base_c + j, int(cur[i, j]))
        unit += 1


def _seed_matches_nozzle(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> bool:
    """Check that the seed cross-axis span matches the chosen nozzle exactly."""
    r0, c0, r1, c1 = map(int, seed_bbox)
    dr, dc = map(int, direction)
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    pr0 = int(pr0)
    pc0 = int(pc0)
    pr1 = int(pr1)
    pc1 = int(pc1)

    if dc == 0:
        nozzle_row = pr1 if dr == 1 else pr0
        nozzle_cols = sorted(int(cc) for cc in range(pc0, pc1 + 1) if plus_mask[nozzle_row, cc])
        seed_cols = list(range(c0, c1 + 1))
        return seed_cols == nozzle_cols

    nozzle_col = pc1 if dc == 1 else pc0
    nozzle_rows = sorted(int(rr) for rr in range(pr0, pr1 + 1) if plus_mask[rr, nozzle_col])
    seed_rows = list(range(r0, r1 + 1))
    return seed_rows == nozzle_rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend large marker tiles away from shared two-color '+' anchors with alternating mirrors.

    Input structure:
    - The background is still the most common color.
    - Each anchor is a repeated two-color plus-like sprite: an interior color and a boundary / tip color.
    - Every plus has one adjacent large rectangular seed tile in other colors, and exactly one ray in the
      grid is already fully expanded as a clue.

    Rule:
    - Recover the pluses by finding the pair of colors whose union forms the repeated symmetric anchor
      components, then identify which of those two colors is the shared tip color.
    - For each non-plus component touching that tip color, infer the launch direction from the attachment
      side, treat the seed bbox as the full repeating period, and extend to the border one tile-unit at a
      time with original / mirrored / original / mirrored alternation along the ray axis.
    - Both plus colors are protected and may never be overwritten by the propagated bands.

    Key insight:
    - The puzzle is no longer solved by spotting one dominant color and doing plain periodic copy. The
      solver must read the anchor as a composite two-color object, use the tip-color adjacency to identify
      the correct marker, and copy the whole large rectangle with parity-dependent mirroring.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    plus_pair = detect_plus_color_pair(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_pair is None:
        return out

    plus_mask_all = np.isin(g, list(plus_pair))
    plus_lab, pn = label_mask(plus_mask_all, connectivity=4)
    if int(pn) == 0:
        return out

    tip_color = detect_tip_color(g, plus_lab, int(pn), plus_pair)
    if tip_color is None:
        return out

    plus_colors = {int(plus_pair[0]), int(plus_pair[1])}
    non_mask = (g != int(bg)) & (~np.isin(g, list(plus_colors)))
    mark_lab, _ = label_mask(non_mask, connectivity=4)

    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue

        tip_mask = plus_mask & (g == int(tip_color))
        adj_ids: set[int] = set()
        for r, c in np.argwhere(tip_mask):
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))

        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            if pts.size == 0:
                continue

            r0, c0, r1, c1 = bbox_coords(pts)
            direction_set: set[tuple[int, int]] = set()
            for pr, pc in pts:
                pr = int(pr)
                pc = int(pc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = pr + dr
                    cc = pc + dc
                    if 0 <= rr < H and 0 <= cc < W and bool(tip_mask[rr, cc]):
                        direction_set.add((-int(dr), -int(dc)))
            if len(direction_set) != 1:
                continue

            direction = next(iter(direction_set))
            if not _seed_matches_nozzle(plus_mask, plus_coords, (r0, c0, r1, c1), direction):
                continue

            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
            extend_alternating_mirror_tile(
                out,
                tile,
                int(r0),
                int(c0),
                direction,
                bg=int(bg),
                protect=plus_colors,
            )

    return out


# --------------- generator ---------------

_H_RANGE = (20, 33)
_W_RANGE = (20, 30)
_N_PLUS_RANGE = (2, 5)
_SEED_ALONG_CHOICES = [2, 3, 4]
_SEED_MAX_COLORS = 4
_SEED_MIN_COLORS = 2
_SEED_ALLOWED_CROSS = {2, 3}
_MIN_RAY_UNITS = 3
_PRE_EXT_IDEAL_UNITS = 4
_PLACEMENT_TRIES = 180
_GEN_RETRIES = 600

_PLUS_SHAPES = [
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=int,
    ),
]

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _plus_role_template(plus_shape: np.ndarray) -> np.ndarray:
    """
    Encode the multicolor sprite template.

    0 = empty, 1 = core color, 2 = boundary / arm-tip color.
    """
    mask = plus_shape.astype(bool)
    role = np.zeros_like(plus_shape, dtype=int)
    H, W = plus_shape.shape
    for r in range(H):
        for c in range(W):
            if not mask[r, c]:
                continue
            boundary = False
            for dr, dc in _DIRECTIONS:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < H and 0 <= cc < W) or not mask[rr, cc]:
                    boundary = True
                    break
            role[r, c] = 2 if boundary else 1
    return role


def _planned_directions(n_plus: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """
    Plan one primary direction per plus so the generator covers both axes consistently.

    With 2-3 pluses, ensure at least one vertical and one horizontal launch. With 4
    pluses, use all four cardinal directions once each in random order.
    """
    dirs = list(_DIRECTIONS)
    rng.shuffle(dirs)
    if n_plus >= len(dirs):
        return dirs[:n_plus]

    vertical = [d for d in dirs if d[1] == 0]
    horizontal = [d for d in dirs if d[0] == 0]
    plan = [vertical[0], horizontal[0]]
    for d in dirs:
        if len(plan) >= n_plus:
            break
        if d not in plan:
            plan.append(d)
    rng.shuffle(plan)
    return plan[:n_plus]


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive cross-axis span of the chosen nozzle."""
    dr, dc = direction
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])

    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _tile_has_rich_texture(tile: np.ndarray, direction: tuple[int, int]) -> bool:
    """Reject seeds that visually collapse into simple or mirror-invariant patterns."""
    if tile.size < 6:
        return False
    if len(set(map(int, tile.flatten()))) < _SEED_MIN_COLORS:
        return False
    if len({tuple(map(int, row)) for row in tile}) < 2:
        return False
    if len({tuple(map(int, tile[:, j])) for j in range(tile.shape[1])}) < 2:
        return False
    if np.array_equal(tile, _mirror_tile_for_direction(tile, direction)):
        return False
    return True


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build a large rectangular seed whose cross-axis matches the nozzle."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = e - s + 1
    if cross not in _SEED_ALLOWED_CROSS:
        return None

    for _ in range(40):
        along = int(rng.choice(_SEED_ALONG_CHOICES))
        dr, dc = direction
        th, tw = (along, cross) if dc == 0 else (cross, along)
        n_cells = th * tw
        max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
        if max_k < _SEED_MIN_COLORS:
            return None

        n_colors = int(rng.integers(_SEED_MIN_COLORS, max_k + 1))
        chosen = list(map(int, rng.choice(palette, size=n_colors, replace=False)))
        vals = chosen + list(map(int, rng.choice(chosen, size=n_cells - n_colors, replace=True)))
        rng.shuffle(vals)
        tile = np.array(vals, dtype=int).reshape(th, tw)
        if _tile_has_rich_texture(tile, direction):
            return tile

    return None


def _compute_ray_units(
    direction: tuple[int, int],
    tile: np.ndarray,
    plus_r0: int,
    plus_c0: int,
    ph: int,
    pw: int,
    H: int,
    W: int,
) -> int:
    """Count how many full tile units fit from the seed to the border, including the seed."""
    dr, dc = direction
    th, tw = tile.shape
    if dr == 1:
        return (H - (plus_r0 + ph)) // th
    if dr == -1:
        return plus_r0 // th
    if dc == 1:
        return (W - (plus_c0 + pw)) // tw
    return plus_c0 // tw


def _project_tile_coords(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int, int]:
    """Project the full alternating ray and return its painted cells plus the seed origin."""
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        c0 = pc0 + s
        r0 = pr1 + 1 if dr == 1 else pr0 - th
    else:
        r0 = pr0 + s
        c0 = pc1 + 1 if dc == 1 else pc0 - tw

    cells: list[tuple[int, int, int]] = []
    if dc == 0:
        r = r0
        step = th if dr == 1 else -th
        unit = 0
        while (r < H) if dr == 1 else (r + th > 0):
            cur = _tile_variant(tile, direction, unit)
            for i in range(th):
                rr = r + i
                if 0 <= rr < H:
                    for j in range(tw):
                        cells.append((rr, c0 + j, int(cur[i, j])))
            r += step
            unit += 1
    else:
        c = c0
        step = tw if dc == 1 else -tw
        unit = 0
        while (c < W) if dc == 1 else (c + tw > 0):
            cur = _tile_variant(tile, direction, unit)
            for i in range(th):
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((r0 + i, cc, int(cur[i, j])))
            c += step
            unit += 1

    return cells, int(r0), int(c0)


def _plus_touches_existing_ray(plus_cells: list[tuple[int, int]], occ: np.ndarray) -> bool:
    """Reject plus placements that would sit directly on or next to an existing ray."""
    H, W = occ.shape
    for r, c in plus_cells:
        for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and int(occ[rr, cc]) == 2:
                return True
    return False


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_roles: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    core_color: int,
    tip_color: int,
    preferred_directions: list[tuple[int, int]] | None = None,
) -> dict | None:
    """Place one two-color plus together with its full alternating output ray."""
    ph, pw = plus_roles.shape
    plus_mask = plus_roles > 0

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (r0 + dr, c0 + dc, int(plus_roles[dr, dc]))
            for dr in range(ph)
            for dc in range(pw)
            if plus_roles[dr, dc] > 0
        ]
        plus_xy = [(r, c) for r, c, _role in plus_cells]

        if any(occ[r, c] != 0 for r, c in plus_xy):
            continue
        if any(plus_occ[r, c] == 1 for r, c in plus_xy):
            continue
        if _plus_touches_existing_ray(plus_xy, occ):
            continue

        if preferred_directions is None:
            dirs = list(_DIRECTIONS)
            rng.shuffle(dirs)
        else:
            dirs = [tuple(map(int, d)) for d in preferred_directions]
        for direction in dirs:
            tile = _build_seed_tile(plus_mask, direction, palette, rng)
            if tile is None:
                continue

            units = _compute_ray_units(direction, tile, r0, c0, ph, pw, H, W)
            if units < _MIN_RAY_UNITS:
                continue

            cells, sr0, sc0 = _project_tile_coords(
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_mask,
                direction,
                tile,
            )
            if any(occ[r, c] != 0 for r, c, _ in cells if 0 <= r < H and 0 <= c < W):
                continue
            if any(plus_occ[r, c] == 1 for r, c, _ in cells if 0 <= r < H and 0 <= c < W):
                continue

            for r, c, role in plus_cells:
                occ[r, c] = 1
                out[r, c] = int(core_color if role == 1 else tip_color)
                for dr2, dc2 in _DIRECTIONS:
                    rr = r + dr2
                    cc = c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = int(v)

            return {
                "tile": tile,
                "cells": cells,
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "units": int(units),
                "direction": tuple(map(int, direction)),
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one long clue ray and collapse the others back to their compact large seeds."""
    pre_idx = max(range(len(pluses)), key=lambda i: pluses[i]["units"])
    if pluses[pre_idx]["units"] < _PRE_EXT_IDEAL_UNITS:
        candidates = [i for i, p in enumerate(pluses) if p["units"] >= _PRE_EXT_IDEAL_UNITS]
        if candidates:
            pre_idx = int(rng.choice(candidates))

    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        th, tw = p["tile"].shape
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            if (r, c) not in seed_set:
                inp[r, c] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    core_color, tip_color = map(int, rng.choice(others, size=2, replace=False))
    palette = [c for c in others if c not in {core_color, tip_color}]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    plus_roles = _plus_role_template(plus_shape)
    N = int(rng.integers(*_N_PLUS_RANGE))
    planned_dirs = _planned_directions(N, rng)

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for i in range(N):
        info = _place_plus_and_ray(
            rng=rng,
            plus_roles=plus_roles,
            palette=palette,
            H=H,
            W=W,
            occ=occ,
            plus_occ=plus_occ,
            out=out,
            core_color=core_color,
            tip_color=tip_color,
            preferred_directions=[planned_dirs[i]],
        )
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_output(out, pluses, bg, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
