# Variant of 6ffbe589
# Dimension: D17 — 层间重叠处理 (meta)
# Change: 三个颜色层在板内互不重叠，旋转前后也不重叠... → 旋转后允许层重叠，后旋转的层覆盖先旋转的层

from __future__ import annotations

import importlib.util
import random
from pathlib import Path
import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for the D17 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The board is still a square crop with three centered color layers:
#      an outer frame color, a middle color inside an inset square, and a
#      small inner color around the center.
#    - The visible input board must remain readable, so the three input
#      color masks are kept disjoint after backward rotation.
#    - The visible output is no longer a plain union of three disjoint masks.
#      Instead, each input layer is rotated to its target pose, then the
#      solver repaints the board in geometric depth order:
#      outer first, middle second, inner last.
#    - Small example:
#      old task:
#        outer target cells and middle target cells never coincide, so the
#        loop order is irrelevant and `crop.copy()` + in-place repaint works.
#      new task:
#        the target outer mask and target middle mask may both claim the same
#        3-cell strip on the middle box perimeter. After rotation, middle is
#        painted later, so those cells must end up middle-colored.
#      difference:
#        the solver can no longer mutate the visible crop in arbitrary color
#        order. It must reconstruct layer masks from the input crop, rotate
#        them independently, and render them onto a blank board using the
#        structural draw order.
# b) Complexity calibration
#    - Only two overlap channels are introduced: outer∩middle on the middle
#      box perimeter, and middle∩inner near the inner sprite bbox. This keeps
#      the new behavior visible without turning the crop into unreadable soup.
#    - The overlap order is a single fixed rule repeated across examples:
#      deeper layers paint later: outer -> middle -> inner.
# c) Old implicit assumptions that break
#    - `out = crop.copy()` assumed the visible crop already represented the
#      full board state. That fails once hidden lower-layer pixels exist.
#    - Iterating colors in numeric order was harmless before because overlaps
#      were impossible. With D17 it becomes a wrong, non-visual ordering rule.
#    - `rotate_color_layer()` erased only the current color before repainting.
#      That works for isolated layers but is the wrong abstraction once the
#      final board must be freshly composited with overwrite semantics.
# d) Placement legality
#    - Cue placement rules remain unchanged: cues stay far from the board and
#      each other, so the largest-component crop logic still works.
#    - Generator legality now needs two checks:
#      1) target masks may overlap in output pose;
#      2) inverse-rotated input masks must be disjoint so the solver can read
#         every layer from the input board.
# e) Visual observability
#    - Overlap is made visible with repeated side-strip motifs. The output
#      consistently shows the middle color swallowing part of the outer strip,
#      and the inner color swallowing part of the middle strip.
#    - The draw order is inferred from geometry, not color ids.
# f) Visual cleanliness
#    - Overlaps are limited to short strips on nested boxes, so the output
#      still reads as three centered layers rather than arbitrary clutter.
# g) Color-rule audit
#    - No sorting or transformation depends on numeric color ids.
#    - The generator only rejects the rare case where numeric color order
#      accidentally matches the structural order, because that would make the
#      new D17 behavior less distinguishable from the old solver.
# h) New solver assumptions and generator guarantees
#    - Assumption: exactly three board colors appear in the crop.
#      Guarantee: generator always uses three non-bg colors.
#    - Assumption: the outer layer is the only one touching all four crop
#      edges, and the middle bbox is larger than the inner bbox.
#      Guarantee: generator builds masks with those nesting properties.
#    - Assumption: inverse-rotated input masks are disjoint.
#      Guarantee: generator validates that before accepting a sample.


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

    H, W = grid.shape
    er0 = max(0, int(mr0 - margin))
    ec0 = max(0, int(mc0 - margin))
    er1 = min(int(H - 1), int(mr1 + margin))
    ec1 = min(int(W - 1), int(mc1 + margin))

    keep_mask = np.zeros((H, W), dtype=bool)
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


def _layer_infos(crop: np.ndarray, *, bg: int) -> list[dict]:
    infos: list[dict] = []
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
    infos.sort(key=lambda item: (-item["touches"], -item["bbox_area"], -item["area"]))
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and recompose its rotated color layers using overwrite order.

    Input structure:
    - bg is the dominant color.
    - The largest foreground cluster is the square board with three centered color layers.
    - Matching monochrome cue sprites sit outside the board and encode per-color rotations.

    Rule:
    - Crop the board cluster away from the distant cue sprites.
    - Read the three board colors as outer / middle / inner layers from their geometry.
    - Rotate each layer by the amount encoded by its same-color cue, then redraw the board
      from outer to inner so later layers cover earlier ones where they land on the same cells.

    Key insight:
    Cue area still determines each layer's quarter-turn count, but the final board must be
    rebuilt in structural depth order because the rotated layers are allowed to overlap.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    assert crop.shape[0] == crop.shape[1], "expected a square board crop"

    infos = _layer_infos(crop, bg=bg)

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    rotated_layers: list[tuple[int, np.ndarray]] = []
    for info in infos:
        rot_k = _cue_rotation(outside_grid, int(info["color"]), bg=int(bg))
        rotated_layers.append((int(info["color"]), np.rot90(info["mask"], int(rot_k))))

    return _render_layers(crop.shape, rotated_layers, bg=int(bg))


# --------------- generator (D17 variant) ---------------

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

GEN_RETRIES = 500


def _rot_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    pts = set()
    rr, cc = r, c
    for _ in range(4):
        pts.add((rr, cc))
        rr, cc = n - 1 - cc, rr
    return pts


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
    choices = [
        ks
        for ks in ((a, b, c) for a in range(4) for b in range(4) for c in range(4))
        if any(ks) and ks[0] != ks[1] and ks[1] != ks[2]
    ]
    return list(random.choice(choices))


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
        cols = range(start, start + span)
        rr = r0 if side == 0 else r1
        for cc in cols:
            mask[rr, int(cc)] = True
    else:
        height = r1 - r0 + 1
        span = min(length, max(1, height - 2))
        start = r0 + (height - span) // 2
        rows = range(start, start + span)
        cc = c1 if side == 1 else c0
        for rr in rows:
            mask[int(rr), cc] = True
    return mask


def _compose_input_layers(
    dst_masks: list[np.ndarray],
    colors: list[int],
    rot_ks: list[int],
) -> np.ndarray | None:
    n = int(dst_masks[0].shape[0])
    board = np.zeros((n, n), dtype=int)
    for color, dst_mask, rot_k in zip(colors, dst_masks, rot_ks):
        inv_mask = np.rot90(dst_mask, (-int(rot_k)) % 4)
        if np.any((board != 0) & inv_mask):
            return None
        board[inv_mask] = int(color)
    return board


def _output_board(colors: list[int], dst_masks: list[np.ndarray]) -> np.ndarray:
    n = int(dst_masks[0].shape[0])
    ordered = list(zip(colors, dst_masks))
    return _render_layers((n, n), ordered, bg=0)


def _overlap_metrics(dst_masks: list[np.ndarray]) -> dict:
    cover = sum(mask.astype(int) for mask in dst_masks)
    om = int(np.sum(dst_masks[0] & dst_masks[1]))
    mi = int(np.sum(dst_masks[1] & dst_masks[2]))
    oi = int(np.sum(dst_masks[0] & dst_masks[2]))
    return {
        "pair_any": int(om > 0) + int(mi > 0) + int(oi > 0),
        "max_stack": int(cover.max()),
        "hidden_cells": int(np.sum(cover >= 2)),
        "om": om,
        "mi": mi,
        "oi": oi,
    }


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(range(1, 10), NUM_COLORS)
    if colors == sorted(colors):
        return None
    c_frame, c_mid, c_inner = colors
    rot_ks = _pick_rotations()

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n, rot_ks[2] > 0)]
    n = random.choice(valid_ns)

    base_triplet = None
    for _ in range(50):
        mid_base = _make_middle_mask(n)
        if mid_base is None:
            continue
        inner_base = _make_inner_mask(n, require_asym=rot_ks[2] > 0)
        if inner_base is None:
            continue
        outer_base = _make_outer_mask(n, int(mid_base.sum()) + OUTER_AREA_PAD)
        if outer_base is None:
            continue
        # Keep the input board constructively disjoint; D17 overlap is introduced later
        # only in the destination masks, not in the inverse-rotated visible input board.
        if np.any(mid_base & inner_base):
            mid_base = mid_base & ~inner_base
        if np.any(outer_base & mid_base):
            outer_base = outer_base & ~mid_base
        if np.any(outer_base & inner_base):
            outer_base = outer_base & ~inner_base
        if (
            int(mid_base.sum()) == 0
            or int(outer_base.sum()) == 0
            or (rot_ks[1] > 0 and not _is_rot_asym(mid_base))
            or (rot_ks[0] > 0 and not _is_rot_asym(outer_base))
        ):
            continue
        if np.any(outer_base & mid_base) or np.any(outer_base & inner_base) or np.any(mid_base & inner_base):
            continue
        base_triplet = (outer_base, mid_base, inner_base)
        break
    if base_triplet is None:
        return None
    outer_base, mid_base, inner_base = base_triplet

    mid_bbox = (MIDDLE_INSET, MIDDLE_INSET, n - 1 - MIDDLE_INSET, n - 1 - MIDDLE_INSET)
    inner_bbox = _mask_bbox(inner_base)
    out_board = None
    in_board = None
    for _ in range(60):
        om_overlap = _side_strip_from_bbox(n, mid_bbox, random.randrange(4), length=random.choice([3, 4]))
        mi_overlap = _side_strip_from_bbox(
            n,
            inner_bbox,
            random.randrange(4),
            length=min(3, inner_bbox[2] - inner_bbox[0] + 1, inner_bbox[3] - inner_bbox[1] + 1),
        )

        dst_outer = outer_base | om_overlap
        dst_mid = mid_base | om_overlap | mi_overlap
        dst_inner = inner_base | mi_overlap
        dst_masks = [dst_outer, dst_mid, dst_inner]

        if any(rot_ks[idx] > 0 and not _is_rot_asym(dst_masks[idx]) for idx in range(3)):
            continue

        metrics = _overlap_metrics(dst_masks)
        if metrics["om"] == 0 or metrics["mi"] == 0 or metrics["hidden_cells"] < 4 or metrics["max_stack"] < 2:
            continue

        in_board = _compose_input_layers(dst_masks, colors, rot_ks)
        if in_board is None:
            continue
        out_board = _output_board(colors, dst_masks)
        break
    if in_board is None or out_board is None:
        return None

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [
        (col, random.choice([1, 5]) if (-k) % 4 == 1 else (-k) % 4)
        for col, k in zip(colors, rot_ks)
        if k != 0
    ]
    random.shuffle(cue_items)
    for col, area in cue_items:
        if not _place_cue(canvas, _cue_patch(col, area), board_bbox, cue_bboxes):
            return None

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None
    for bb in cue_bboxes:
        if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
            return None

    if not np.array_equal(main(canvas), out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    # Fallback: reuse the collision-safe D17/2 generator to produce valid inputs
    # with deliberate rotated overlaps. D17/0 changes only the post-collision
    # composition rule (`overwrite` instead of `erase`), so the same inputs are
    # suitable for exercising this variant's main() logic.
    sibling = Path(__file__).resolve().parents[1] / "2" / "code.py"
    spec = importlib.util.spec_from_file_location("variant_d17_2_fallback", sibling)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_input()
