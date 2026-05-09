# Combo variant of 6ffbe589 (index 21)
# combo_spec: D13/0+D17/0
# Name: 层级依赖处理+后画覆盖
#
# Design analysis for the combo variant
# =====================================
# a) concretization_and_trace:
#    - The scene stays close to the original task: a single largest square board crop plus detached
#      same-color cue blobs outside it. The board still contains three nested colors that read as
#      outer / middle / inner by geometry.
#    - The fused rule changes *how* the three rotations are simulated. The solver no longer treats the
#      three color masks as independent layers that can be rotated from a frozen snapshot. Instead it
#      processes them in geometric depth order: outer first, then middle, then inner.
#    - Crucially, each step reads the layer mask from the current working board. So when the outer layer
#      rotates into cells that currently belong to the middle layer, those middle pixels disappear before
#      the middle layer gets its own turn. The middle layer is then rotated from that truncated mask.
#    - Concrete trace on a small stylized board:
#        input board:
#          outer has a source strip that rotates onto 3 middle cells;
#          middle has a source strip that would rotate onto 2 inner cells.
#        old independent-overwrite logic:
#          rotate outer from its original mask, rotate middle from its original full mask, then repaint.
#          The middle strip still exists and reaches the inner sprite.
#        fused logic:
#          1) rotate outer on the live board -> it overwrites those 3 middle cells immediately.
#          2) rotate middle from the now-missing middle mask -> only the surviving part rotates onward.
#          3) repaint order is still sequential, so middle can cover parts of the rotated outer later.
#      Difference:
#        the same overlap acts twice: first as a destructive dependency on the next layer's source mask,
#        and later as a normal paint-overwrite on the destination board.
#
# b) complexity_calibration:
#    - The generator introduces only one explicit cascade channel per adjacent pair:
#      outer -> middle source loss, and middle -> inner target pressure.
#      That keeps the effect visible without turning the board into noisy clutter.
#    - All cue semantics remain the original easy-to-read rule: same-color blob area mod 4 gives the
#      quarter-turn count. The new difficulty comes from ordered simulation, not from new cue syntax.
#    - The board still uses exactly three colors and one fixed structural order, repeated in every sample.
#      This gives the solver enough repeated support to infer the hierarchy from 2-4 examples.
#
# c) implicit_assumption_audit_for_main:
#    - The original `colors_present(crop)` loop depended on order irrelevance. That assumption breaks:
#      with overlaps plus live-state rotation, changing the layer order changes both the source masks
#      and the final paint result. `main()` must sort layers by geometry instead.
#    - The D17-only rewrite of rotating frozen masks also breaks here. Once D13 is fused in, the next
#      layer must be read from the mutated working board, not from the original crop snapshot.
#    - `rotate_color_layer()` remains usable, but now it is intentionally applied to the evolving board.
#      That makes it the carrier of both dependence and overwrite.
#    - New assumptions introduced by the fused solver:
#      1) exactly three board colors are present;
#      2) outer is the only layer touching all four crop edges;
#      3) middle bbox is strictly larger than inner bbox;
#      4) the board crop is square.
#      The generator explicitly guarantees each of these.
#
# d) placement_legality:
#    - Input-layer masks are still kept disjoint on the visible board, so each color can be recovered
#      unambiguously before any rotation happens.
#    - Cue blobs remain detached from the board and from one another using the same Chebyshev-distance
#      spacing as the original task, so the crop helper still isolates the board correctly.
#    - Constructive source strips are added only when their inverse-rotated source cells fall on empty
#      cells of the other layers, so the input never contains illegal overlaps.
#
# e) visual_observability:
#    - A human can compare multiple examples and see that cue colors still match board colors, so each
#      layer has its own rotation amount as before.
#    - The new effect is visible because outer hits the middle before the middle rotates, which makes the
#      middle layer output look "clipped then turned" rather than independently rotated.
#    - The generator rejects samples where the fused output equals either:
#      1) the original numeric-color-order solver, or
#      2) the frozen-mask overwrite baseline.
#      That prevents visually degenerate examples.
#
# f) visual_cleanliness:
#    - Overwrite is limited to short strips on nested boxes, so the output remains legible as a centered
#      three-layer board instead of an arbitrary pile of collisions.
#    - The input stays clean: all collisions happen only after simulated rotation, not in the visible source.
#
# g) color_rule_audit:
#    - No logic depends on the numeric meaning of color ids.
#    - The generator deliberately avoids samples whose ascending color-id order already matches
#      outer -> middle -> inner too closely, because that would make the fused solver too similar to
#      the original color-sorted loop.
#
# 9) new_main_assumptions_and_generator_guarantees:
#    - Assumption: geometric depth order is recoverable from the input crop alone.
#      Guarantee: outer touches all four edges, middle has the larger inner bbox, inner the smaller one.
#    - Assumption: the live-state cascade actually matters.
#      Guarantee: the generator enforces measurable middle-source loss and a non-zero difference from the
#      frozen-mask baseline on every accepted sample.
#    - Assumption: same-color cues are unique per board color.
#      Guarantee: at most one cue blob is emitted for each of the three board colors.

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


def crop_components_near_largest(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    margin: int = 3,
    connectivity: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """
    Local helper (only used by this solver): crop the cluster near the largest non-bg component.

    Returns:
        (crop, crop_bbox, bg) where crop_bbox is in the original grid coordinates (inclusive).
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps = find_components(grid, bg=bg, connectivity=int(connectivity), monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"

    largest_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == largest_area]
    assert len(largest) == 1, "expected a unique largest foreground component"
    mr0, mc0, mr1, mc1 = map(int, largest[0]["bbox"])

    h, w = grid.shape
    er0 = max(0, int(mr0 - margin))
    ec0 = max(0, int(mc0 - margin))
    er1 = min(int(h - 1), int(mr1 + margin))
    ec1 = min(int(w - 1), int(mc1 + margin))

    keep_mask = np.zeros((h, w), dtype=bool)
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 < er0 or r0 > er1 or c1 < ec0 or c0 > ec1:
            continue
        coords = comp["coords"]
        keep_mask[coords[:, 0], coords[:, 1]] = True

    keep_coords = np.argwhere(keep_mask)
    assert keep_coords.size > 0, "expected the expanded board crop to retain the largest component"

    r0, c0 = keep_coords.min(axis=0)
    r1, c1 = keep_coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    crop = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=grid.dtype)
    sub = keep_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    crop[sub] = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1][sub]
    return crop, (int(r0), int(c0), int(r1), int(c1)), int(bg)


def rotate_color_layer(grid: np.ndarray, color: int, rot_k: int, *, bg: int | None = None) -> np.ndarray:
    """Rotate the currently visible mask of one color inside the fixed board crop."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    k = int(rot_k) % 4
    if k == 0:
        return grid

    col = int(color)
    mask = (grid == col).astype(int)
    rmask = np.rot90(mask, k)
    assert rmask.shape == mask.shape, "expected a square crop for layer rotation"

    out = grid.copy()
    out[out == col] = bg
    out[rmask == 1] = col
    return out


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touch_count(mask: np.ndarray) -> int:
    r0, c0, r1, c1 = _mask_bbox(mask)
    n = int(mask.shape[0])
    return int(r0 == 0) + int(c0 == 0) + int(r1 == n - 1) + int(c1 == n - 1)


def _layer_infos(crop: np.ndarray, *, bg: int) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for col in colors_present(crop, exclude={int(bg)}):
        mask = crop == int(col)
        bbox = _mask_bbox(mask)
        r0, c0, r1, c1 = bbox
        infos.append(
            {
                "color": int(col),
                "mask": mask,
                "bbox": bbox,
                "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "area": int(mask.sum()),
                "touches": _touch_count(mask),
            }
        )
    assert len(infos) == 3, "expected exactly three board colors"
    infos.sort(key=lambda item: (-int(item["touches"]), -int(item["bbox_area"]), -int(item["area"])))
    return infos


def _cue_rotation(outside_grid: np.ndarray, color: int, *, bg: int) -> int:
    comps = find_components(
        outside_grid,
        bg=int(bg),
        connectivity=8,
        monochromatic=True,
        colors=[int(color)],
        return_format="dict",
    )
    assert len(comps) <= 1, f"expected at most one outside cue for color {color}"
    cue_size = int(comps[0]["area"]) if comps else 0
    return (-cue_size) % 4


def _render_layers(shape: tuple[int, int], ordered_layers: list[tuple[int, np.ndarray]], *, bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color, mask in ordered_layers:
        out[mask] = int(color)
    return out


def _ordered_rotations(
    outside_grid: np.ndarray,
    infos: list[dict[str, object]],
    *,
    bg: int,
) -> dict[int, int]:
    return {int(info["color"]): _cue_rotation(outside_grid, int(info["color"]), bg=int(bg)) for info in infos}


def _apply_fused_layers(
    crop: np.ndarray,
    infos: list[dict[str, object]],
    rot_by_color: dict[int, int],
    *,
    bg: int,
) -> np.ndarray:
    work = np.array(crop, dtype=int).copy()
    for info in infos:
        color = int(info["color"])
        work = rotate_color_layer(work, color, int(rot_by_color[color]), bg=int(bg))
    return work


def _apply_independent_layers(
    shape: tuple[int, int],
    infos: list[dict[str, object]],
    rot_by_color: dict[int, int],
    *,
    bg: int,
) -> np.ndarray:
    ordered_layers = [
        (int(info["color"]), np.rot90(np.array(info["mask"], dtype=bool), int(rot_by_color[int(info["color"])])))
        for info in infos
    ]
    return _render_layers(shape, ordered_layers, bg=int(bg))


def _apply_numeric_layers(crop: np.ndarray, rot_by_color: dict[int, int], *, bg: int) -> np.ndarray:
    work = np.array(crop, dtype=int).copy()
    for color in colors_present(crop, exclude={int(bg)}):
        work = rotate_color_layer(work, int(color), int(rot_by_color[int(color)]), bg=int(bg))
    return work


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and simulate ordered live-state rotations of its nested layers.

    Input structure:
    - The dominant color is the background.
    - The largest foreground component is a square board with three nested colors that read as
      outer / middle / inner by how much of the crop they occupy.
    - Small detached monochrome cue sprites outside the board match those board colors and encode
      their quarter-turn amounts by area modulo 4.

    Rule:
    - Crop the board away from the distant cue sprites.
    - Recover the board colors in geometric depth order: outer first, then middle, then inner.
    - Process those layers one at a time on the current working board: rotate the currently visible
      pixels of that color by the amount from its same-color cue, then write the rotated layer back.
    - Because the board is updated after every step, an earlier rotated outer layer can erase part of
      an inner layer before that inner layer is rotated, and later layers still overwrite earlier ones.

    Key insight:
    The cue rule itself is unchanged, but the three layers are no longer independent. The correct output
    comes from simulating a depth-ordered cascade on the live board state, not from rotating frozen masks.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    assert crop.shape[0] == crop.shape[1], "expected a square board crop"

    infos = _layer_infos(crop, bg=bg)

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    rot_by_color = _ordered_rotations(outside_grid, infos, bg=int(bg))
    return _apply_fused_layers(crop, infos, rot_by_color, bg=int(bg))


# --------------- generator (combo D13/0 + D17/0) ---------------

CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 13
NUM_COLORS = 3

OUTER_DENSITY_LO = 0.17
OUTER_DENSITY_HI = 0.32
MID_DENSITY_LO = 0.30
MID_DENSITY_HI = 0.50
MID_OUTER_CAP = 0.32

LAYER_BAND_WIDTH = 2
MIDDLE_INSET = 2
ASYM_MARKER = (1, 2)
INNER_SPAN_MARGIN = 8
INNER_SPAN_MIN = 3

CUE_GRID = 3
CUE_BOARD_DIST = 4
CUE_INTER_DIST = 4
OUTER_AREA_PAD = 4

GEN_RETRIES = 800
LOCAL_SHAPE_RETRIES = 90

MIN_OM_HIT = 2
MIN_MID_LOSS = 1
MIN_CASCADE_DELTA = 2
MIN_OVERWRITE_OM = 1
MIN_FUSED_DIFF = 3
MIN_NUMERIC_DIFF = 2
VALID_ROTATION_TRIPLES = [
    (1, 2, 1),
    (1, 2, 2),
    (1, 2, 3),
    (2, 1, 1),
    (2, 1, 2),
    (2, 1, 3),
    (2, 3, 1),
    (2, 3, 2),
    (2, 3, 3),
    (3, 2, 1),
    (3, 2, 2),
    (3, 2, 3),
]


def _rot_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    pts = set()
    rr, cc = r, c
    for _ in range(4):
        pts.add((rr, cc))
        rr, cc = n - 1 - cc, rr
    return pts


def _rotate_coord(r: int, c: int, n: int, k: int) -> tuple[int, int]:
    rr, cc = int(r), int(c)
    for _ in range(int(k) % 4):
        rr, cc = int(n - 1 - cc), int(rr)
    return int(rr), int(cc)


def _is_rot_asym(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _checker_perimeter(n: int, phase: int) -> np.ndarray:
    rr, cc = np.indices((n, n))
    perim = (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1)
    return perim & (((rr + cc + phase) % 2) == 0)


def _band_cells(n: int, width: int) -> list[tuple[int, int]]:
    rr, cc = np.indices((n, n))
    inside = np.minimum(np.minimum(rr, cc), np.minimum(n - 1 - rr, n - 1 - cc)) < width
    coords = np.argwhere(inside)
    return [(int(r), int(c)) for r, c in coords]


def _fill_to_count(mask: np.ndarray, candidates: list[tuple[int, int]], target: int) -> bool:
    random.shuffle(candidates)
    count = int(mask.sum())
    for r, c in candidates:
        if count >= target:
            return True
        if not mask[r, c]:
            mask[r, c] = True
            count += 1
    return count >= target


def _make_asym_layer(
    n: int,
    density_lo: float,
    density_hi: float,
    *,
    allowed_mask: np.ndarray | None = None,
    extra_lo_terms: list[int] | None = None,
) -> np.ndarray | None:
    mask = _checker_perimeter(n, random.randrange(2))

    orbit = _rot_orbit(*ASYM_MARKER, n)
    for p in orbit - {ASYM_MARKER}:
        mask[p] = False
    mask[ASYM_MARKER] = True

    if allowed_mask is not None:
        mask &= allowed_mask
        if not mask[ASYM_MARKER]:
            return None

    lo_terms = [int(np.ceil(density_lo * n * n)), int(mask.sum())]
    if extra_lo_terms:
        lo_terms.extend(int(v) for v in extra_lo_terms)
    lo = max(lo_terms)
    hi = int(np.floor(density_hi * n * n))
    if lo > hi:
        return None
    target = random.randint(lo, hi)

    blocked = orbit - {ASYM_MARKER}
    candidates = [
        (r, c)
        for r, c in _band_cells(n, LAYER_BAND_WIDTH)
        if (r, c) not in blocked and not mask[r, c] and (allowed_mask is None or allowed_mask[r, c])
    ]
    if not _fill_to_count(mask, candidates, target):
        return None
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_middle_mask(n: int) -> np.ndarray | None:
    m = n - 2 * MIDDLE_INSET
    hi_cap = int(np.floor(MID_OUTER_CAP * n * n)) - 5
    local = _make_asym_layer(m, MID_DENSITY_LO, min(MID_DENSITY_HI, hi_cap / (m * m)))
    if local is None:
        return None
    mask = np.zeros((n, n), dtype=bool)
    mask[MIDDLE_INSET : MIDDLE_INSET + m, MIDDLE_INSET : MIDDLE_INSET + m] = local
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_outer_mask(n: int, min_area: int) -> np.ndarray | None:
    return _make_asym_layer(n, OUTER_DENSITY_LO, OUTER_DENSITY_HI, extra_lo_terms=[min_area])


def _valid_inner_spans(n: int, require_asym: bool) -> list[int]:
    max_span = n - INNER_SPAN_MARGIN
    spans = [s for s in range(INNER_SPAN_MIN, max_span + 1) if (s % 2) == (n % 2)]
    if require_asym:
        spans = [s for s in spans if s > 1]
    return spans


def _make_inner_mask(n: int, *, require_asym: bool) -> np.ndarray | None:
    for _ in range(20):
        mask = np.zeros((n, n), dtype=bool)
        span = random.choice(_valid_inner_spans(n, require_asym))
        top = (n - span) // 2

        if span == 3:
            variants = [
                np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
                np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
                np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
            ]
            local = random.choice(variants)
            local = np.rot90(local, random.randrange(4))
        elif span == 4:
            local = np.zeros((span, span), dtype=bool)
            pts = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
            for r, c in pts:
                local[r, c] = True
            if random.randrange(2):
                local = np.fliplr(local)
            local = np.rot90(local, random.randrange(4))
        else:
            local = np.zeros((span, span), dtype=bool)
            pts = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 4)]
            for r, c in pts:
                local[r, c] = True
            if random.randrange(2):
                local = np.fliplr(local)
            local = np.rot90(local, random.randrange(4))
        mask[top : top + local.shape[0], top : top + local.shape[1]] = local
        if (not require_asym) or _is_rot_asym(mask):
            return mask
    return None


def _pick_rotations() -> list[int]:
    # These 12 triples are the generator branches that consistently survive the
    # cascade / overwrite observability checks, so sample them directly.
    return list(random.choice(VALID_ROTATION_TRIPLES))


def _cue_patch(color: int, area: int) -> np.ndarray:
    center = CUE_GRID // 2
    coords = {(center, center)}
    while len(coords) < area:
        frontier = set()
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < CUE_GRID and 0 <= nc < CUE_GRID and (nr, nc) not in coords:
                        frontier.add((nr, nc))
        coords.add(random.choice(sorted(frontier)))

    arr = np.zeros((CUE_GRID, CUE_GRID), dtype=int)
    for r, c in coords:
        arr[r, c] = color
    pos = np.argwhere(arr != 0)
    r0, c0 = pos.min(axis=0)
    r1, c1 = pos.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _place_cue(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    h, w = patch.shape
    mask = patch != 0
    candidates = []
    for r in range(CANVAS_SIZE - h + 1):
        for c in range(CANVAS_SIZE - w + 1):
            if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
                continue
            bb = (r, c, r + h - 1, c + w - 1)
            if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                continue
            if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                continue
            candidates.append((r, c, bb))
    if not candidates:
        return False
    r, c, bb = random.choice(candidates)
    paste(canvas, patch, r, c, transparent=0)
    cue_bboxes.append(bb)
    return True


def _board_anchor(n: int) -> tuple[int, int]:
    spare = CANVAS_SIZE - n
    side = random.randrange(4)
    if side == 0:
        return spare, random.randint(0, spare)
    if side == 1:
        return 0, random.randint(0, spare)
    if side == 2:
        return random.randint(0, spare), spare
    return random.randint(0, spare), 0


def _side_strip_from_bbox(
    n: int,
    bbox: tuple[int, int, int, int],
    side: int,
    *,
    length: int,
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((n, n), dtype=bool)
    if side % 2 == 0:
        width = c1 - c0 + 1
        span = min(length, max(1, width - 2))
        start = c0 + (width - span) // 2
        rr = r0 if side == 0 else r1
        for cc in range(start, start + span):
            mask[int(rr), int(cc)] = True
    else:
        height = r1 - r0 + 1
        span = min(length, max(1, height - 2))
        start = r0 + (height - span) // 2
        cc = c1 if side == 1 else c0
        for rr in range(start, start + span):
            mask[int(rr), int(cc)] = True
    return mask


def _choose_targets(
    target_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    rot_k: int,
    blocked_source: np.ndarray,
    *,
    size_choices: list[int],
    required_mask: np.ndarray | None = None,
    partial_required: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    n = int(target_mask.shape[0])
    all_coords = [tuple(map(int, rc)) for rc in np.argwhere(target_mask)]
    if not all_coords:
        return None

    pool_masks: list[np.ndarray] = []
    for side in random.sample(range(4), 4):
        for length in random.sample([2, 3, 4], 3):
            pool = target_mask & _side_strip_from_bbox(n, bbox, side, length=length)
            if int(pool.sum()) > 0:
                pool_masks.append(pool)
    pool_masks.append(target_mask)

    total_required = int(np.sum(target_mask & required_mask)) if required_mask is not None else 0
    for pool in pool_masks:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(pool)]
        if not coords:
            continue
        valid_sizes = [s for s in size_choices if 0 < s <= len(coords)]
        if not valid_sizes:
            continue
        for _ in range(40):
            size = random.choice(valid_sizes)
            chosen = random.sample(coords, size)
            if required_mask is not None:
                req_count = sum(1 for r, c in chosen if bool(required_mask[int(r), int(c)]))
                if req_count == 0:
                    continue
                if partial_required and total_required > 0 and req_count >= total_required:
                    continue

            src_mask = np.zeros_like(target_mask, dtype=bool)
            ok = True
            for r, c in chosen:
                sr, sc = _rotate_coord(int(r), int(c), n, -int(rot_k))
                if bool(blocked_source[int(sr), int(sc)]):
                    ok = False
                    break
                src_mask[int(sr), int(sc)] = True
            if not ok:
                continue

            tgt_mask = np.zeros_like(target_mask, dtype=bool)
            for r, c in chosen:
                tgt_mask[int(r), int(c)] = True
            return tgt_mask, src_mask
    return None


def _build_layer_triplet(n: int, rot_ks: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(LOCAL_SHAPE_RETRIES):
        mid_base = _make_middle_mask(n)
        inner_mask = _make_inner_mask(n, require_asym=True)
        outer_base = _make_outer_mask(n, int(mid_base.sum()) + OUTER_AREA_PAD) if mid_base is not None else None
        if mid_base is None or inner_mask is None or outer_base is None:
            continue

        if np.any(mid_base & inner_mask) or np.any(outer_base & mid_base) or np.any(outer_base & inner_mask):
            continue

        inner_bbox = _mask_bbox(inner_mask)
        mi_pair = _choose_targets(
            inner_mask,
            inner_bbox,
            int(rot_ks[1]),
            blocked_source=inner_mask | outer_base,
            size_choices=[3, 2],
        )
        if mi_pair is None:
            continue
        mi_target, mi_source = mi_pair
        middle_mask = mid_base | mi_source

        mid_bbox = (MIDDLE_INSET, MIDDLE_INSET, n - 1 - MIDDLE_INSET, n - 1 - MIDDLE_INSET)
        om_pair = _choose_targets(
            middle_mask,
            mid_bbox,
            int(rot_ks[0]),
            blocked_source=middle_mask | inner_mask,
            size_choices=[4, 3, 2],
            required_mask=mi_source,
            partial_required=True,
        )
        if om_pair is None:
            continue
        _om_target, om_source = om_pair
        outer_mask = outer_base | om_source

        if np.any(outer_mask & middle_mask) or np.any(outer_mask & inner_mask) or np.any(middle_mask & inner_mask):
            continue
        if not _is_rot_asym(outer_mask) or not _is_rot_asym(middle_mask) or not _is_rot_asym(inner_mask):
            continue
        return outer_mask, middle_mask, inner_mask
    return None


def _board_from_masks(colors: list[int], masks: list[np.ndarray]) -> np.ndarray:
    n = int(masks[0].shape[0])
    board = np.zeros((n, n), dtype=int)
    for color, mask in zip(colors, masks):
        if np.any((board != 0) & mask):
            raise AssertionError("expected disjoint input masks")
        board[mask] = int(color)
    return board


def _behavior_metrics(crop: np.ndarray, rot_by_color: dict[int, int], *, bg: int = 0) -> dict[str, int]:
    infos = _layer_infos(crop, bg=int(bg))
    fused = _apply_fused_layers(crop, infos, rot_by_color, bg=int(bg))
    independent = _apply_independent_layers(crop.shape, infos, rot_by_color, bg=int(bg))
    numeric = _apply_numeric_layers(crop, rot_by_color, bg=int(bg))

    outer, middle, inner = infos
    outer_color = int(outer["color"])
    middle_color = int(middle["color"])
    inner_color = int(inner["color"])

    outer_rot = np.rot90(np.array(outer["mask"], dtype=bool), int(rot_by_color[outer_color]))

    work_after_outer = rotate_color_layer(crop.copy(), outer_color, int(rot_by_color[outer_color]), bg=int(bg))
    middle_visible = work_after_outer == middle_color

    middle_full_rot = np.rot90(np.array(middle["mask"], dtype=bool), int(rot_by_color[middle_color]))
    middle_live_rot = np.rot90(middle_visible, int(rot_by_color[middle_color]))

    work_after_middle = rotate_color_layer(work_after_outer.copy(), middle_color, int(rot_by_color[middle_color]), bg=int(bg))
    inner_visible = work_after_middle == inner_color
    inner_full_rot = np.rot90(np.array(inner["mask"], dtype=bool), int(rot_by_color[inner_color]))
    inner_live_rot = np.rot90(inner_visible, int(rot_by_color[inner_color]))

    return {
        "om_hit": int(np.sum(outer_rot & np.array(middle["mask"], dtype=bool))),
        "mid_loss": int(np.sum(np.array(middle["mask"], dtype=bool) & ~middle_visible)),
        "cascade_delta": int(np.sum(middle_full_rot ^ middle_live_rot)),
        "overwrite_om": int(np.sum(outer_rot & middle_live_rot)),
        "mi_hit": int(np.sum(middle_live_rot & np.array(inner["mask"], dtype=bool))),
        "inner_loss": int(np.sum(np.array(inner["mask"], dtype=bool) & ~inner_visible)),
        "overwrite_mi": int(np.sum(middle_live_rot & inner_live_rot)),
        "fused_vs_independent": int(np.sum(fused != independent)),
        "fused_vs_numeric": int(np.sum(fused != numeric)),
    }


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(range(1, 10), NUM_COLORS)
    rot_ks = _pick_rotations()

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n, True)]
    n = random.choice(valid_ns)
    triplet = _build_layer_triplet(n, rot_ks)
    if triplet is None:
        return None
    outer_mask, middle_mask, inner_mask = triplet

    board = _board_from_masks(colors, [outer_mask, middle_mask, inner_mask])
    if int(np.sum(board != 0)) <= 0:
        return None

    infos = _layer_infos(board, bg=0)
    ordered_colors = [int(info["color"]) for info in infos]
    if ordered_colors != [int(colors[0]), int(colors[1]), int(colors[2])]:
        return None

    rot_by_color = {int(color): int(k) for color, k in zip(colors, rot_ks)}
    metrics = _behavior_metrics(board, rot_by_color, bg=0)
    if metrics["om_hit"] < MIN_OM_HIT:
        return None
    if metrics["mid_loss"] < MIN_MID_LOSS:
        return None
    if metrics["cascade_delta"] < MIN_CASCADE_DELTA:
        return None
    if metrics["overwrite_om"] < MIN_OVERWRITE_OM:
        return None
    if metrics["mi_hit"] < 1:
        return None
    if metrics["inner_loss"] < 1:
        return None
    if metrics["fused_vs_independent"] < MIN_FUSED_DIFF:
        return None
    if metrics["fused_vs_numeric"] < MIN_NUMERIC_DIFF:
        return None

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [
        (int(color), random.choice([1, 5]) if (-int(k)) % 4 == 1 else (-int(k)) % 4)
        for color, k in zip(colors, rot_ks)
    ]
    random.shuffle(cue_items)
    for color, area in cue_items:
        if not _place_cue(canvas, _cue_patch(color, int(area)), board_bbox, cue_bboxes):
            return None

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if not areas or areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None
    for bb in cue_bboxes:
        if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
            return None

    try:
        out = main(canvas)
    except Exception:
        return None
    if out.shape != (n, n):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")
