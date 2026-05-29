# Combo variant of 6ffbe589 (index 6)
# combo_spec: D3/0+D8/0+D17/0
# Name: 镜像翻转+无提示180旋转+后画覆盖

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The canvas still contains one unique largest square board plus several
#      far-away monochrome same-color cue sprites.
#    - The original task decoded a cue by area mod 4 and otherwise left the
#      layer alone. Here the decision tree changes:
#      1) same-color cue exists -> decode its silhouette as one of three
#         reflections: `1x3` horizontal bar => left-right flip, `3x1`
#         vertical bar => top-bottom flip, `3x3` main diagonal => transpose.
#      2) no same-color cue exists -> rotate that layer by 180 degrees.
#      3) after all target masks are computed, redraw them outer -> middle ->
#         inner so later layers overwrite earlier ones on overlap cells.
#    - Example trace:
#      suppose the middle color has a vertical cue, the inner color has no
#      cue, and both target layers claim one shared 3-cell strip. The old
#      solver would treat the middle cue as a rotation code, leave the inner
#      unchanged, and assume no overlap exists. The combo solver instead flips
#      the middle mask vertically, rotates the inner mask by 180 degrees, then
#      repaints outer/middle/inner so the inner color wins on the shared strip.
# b) Complexity calibration
#    - Cue types are limited to exactly three high-contrast glyphs. This keeps
#      the flip vocabulary small enough to infer from 2-4 examples.
#    - Every generated sample contains both kinds of transform: at least one
#      cued flip layer and at least one cue-less 180-degree layer. This makes
#      the D8 branch observable instead of being a rare corner case.
#    - Overlap is restricted to short strips on nested boxes (outer∩middle and
#      middle∩inner). This makes D17 visible without turning the board into
#      clutter.
# c) Implicit assumption audit on `main()`
#    - `crop_components_near_largest(...)` still assumes the board is the
#      unique largest foreground component; generator keeps cue sprites small
#      and far away.
#    - `colors_present(crop, exclude={bg})` assumes the crop contains only the
#      board. Cue placement keeps all sprites outside the margin-3 crop window.
#    - The original in-place per-color rewrite is no longer valid because
#      overlap creates hidden lower-layer pixels. The solver must extract masks
#      from the input crop, transform them independently, then render on a
#      blank board in geometric order.
#    - The solver now assumes the three visible crop colors can be ranked as
#      outer / middle / inner from geometry alone. Generator guarantees this:
#      outer touches all four crop sides, middle has the larger inset bbox, and
#      inner stays centered in the smallest bbox.
#    - Diagonal reflection assumes a square crop. The board remains NxN.
# d) Placement legality
#    - The input board is built by applying the inverse of the solver rule to
#      each destination mask. All transforms used here are self-inverse, but
#      the generator still rejects any sample whose visible input masks overlap.
#    - Cue glyphs are placed only where they do not overlap the board or each
#      other, and each cue bbox stays at Chebyshev distance >= 4 from the board.
#    - Output overlaps are allowed; input overlaps are not.
# e) Visual observability
#    - Human solvers can read the flip type directly from cue geometry.
#    - Missing cue is observable because each sample also contains explicit cue
#      layers for contrast, and cue-less layers are forced to change under
#      180-degree rotation.
#    - Overwrite order is observable because overlap strips consistently show
#      the deeper layer swallowing part of the shallower layer.
# f) Visual cleanliness
#    - The board remains three nested centered layers, and overlaps are limited
#      to narrow strips on the middle box / inner box edges.
#    - No rule depends on numeric color ids, raster scan order, or other
#      invisible metadata.
# g) New solver assumptions and generator guarantees
#    - Assumption: each board color has at most one outside cue. Guarantee:
#      generator creates at most one same-color cue component.
#    - Assumption: every active flip or 180 rotation is visually nontrivial.
#      Guarantee: generator rejects masks invariant under their assigned rule.
#    - Assumption: overwrite order matters on at least one cell. Guarantee:
#      generator enforces nonzero hidden overlap and rejects samples that match
#      ablations with `missing cue => no-op` or with in-place repainting.


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


TRANSFORMS = ("h", "v", "d")
CUE_TEMPLATES = {
    "h": np.array([[1, 1, 1]], dtype=bool),
    "v": np.array([[1], [1], [1]], dtype=bool),
    "d": np.eye(3, dtype=bool),
}


def apply_cue_transform(mask: np.ndarray, transform: str) -> np.ndarray:
    """Apply one cue-driven reflection to a square layer mask."""
    if transform == "h":
        return np.fliplr(mask)
    if transform == "v":
        return np.flipud(mask)
    if transform == "d":
        assert mask.shape[0] == mask.shape[1], "main-diagonal reflection expects a square crop"
        return mask.T
    raise AssertionError(f"unknown transform: {transform}")


def apply_layer_rule(mask: np.ndarray, cue_transform: str | None, *, missing_halfturn: bool) -> np.ndarray:
    """Apply the combo rule for one layer: cue => reflection, no cue => optional 180-degree rotation."""
    if cue_transform is None:
        return np.rot90(mask, 2) if missing_halfturn else mask.copy()
    return apply_cue_transform(mask, cue_transform)


def decode_cue_transform(grid: np.ndarray, comp: dict, color: int) -> str:
    """Decode one same-color outside cue into h/v/d reflection type."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(color)

    if patch.shape == (1, 3) and bool(np.all(patch)):
        return "h"
    if patch.shape == (3, 1) and bool(np.all(patch)):
        return "v"
    if patch.shape == (3, 3) and np.array_equal(patch, CUE_TEMPLATES["d"]):
        return "d"
    raise AssertionError(f"unrecognized cue shape for color {color}")


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


def _render_layers(shape: tuple[int, int], ordered_layers: list[tuple[int, np.ndarray]], *, bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color, mask in ordered_layers:
        out[mask] = int(color)
    return out


def _solve_variant(
    grid: np.ndarray,
    *,
    missing_halfturn: bool,
    overwrite: bool,
    force_no_cues: bool = False,
) -> np.ndarray:
    """Shared solver core used both by `main()` and generator anti-degeneration checks."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    assert crop.shape[0] == crop.shape[1], "expected a square board crop"
    infos = _layer_infos(crop, bg=bg)

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    if overwrite:
        ordered_layers: list[tuple[int, np.ndarray]] = []
        for info in infos:
            comps = find_components(
                outside_grid,
                bg=int(bg),
                connectivity=8,
                monochromatic=True,
                colors=[int(info["color"])],
                return_format="dict",
            )
            assert len(comps) <= 1, f"expected at most one outside cue for color {info['color']}"
            cue_transform = None
            if comps and not force_no_cues:
                cue_transform = decode_cue_transform(outside_grid, comps[0], int(info["color"]))
            ordered_layers.append(
                (
                    int(info["color"]),
                    apply_layer_rule(info["mask"], cue_transform, missing_halfturn=missing_halfturn),
                )
            )
        return _render_layers(crop.shape, ordered_layers, bg=int(bg))

    out = crop.copy()
    for info in infos:
        comps = find_components(
            outside_grid,
            bg=int(bg),
            connectivity=8,
            monochromatic=True,
            colors=[int(info["color"])],
            return_format="dict",
        )
        assert len(comps) <= 1, f"expected at most one outside cue for color {info['color']}"
        cue_transform = None
        if comps and not force_no_cues:
            cue_transform = decode_cue_transform(outside_grid, comps[0], int(info["color"]))
        current_mask = out == int(info["color"])
        target_mask = apply_layer_rule(current_mask, cue_transform, missing_halfturn=missing_halfturn)
        out[out == int(info["color"])] = int(bg)
        out[target_mask] = int(info["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the square board cluster and rebuild its layers using cue-driven flips plus a default half-turn.

    Input structure:
    - The background is the dominant color.
    - The largest foreground component is a square board with three nested color layers.
    - Some board colors have one same-color cue glyph outside the board, while other colors have no cue.

    Rule:
    - Crop away the distant cue glyphs and identify the board's outer, middle, and inner layers from geometry.
    - For a layer with a same-color cue, decode the cue shape as horizontal flip, vertical flip, or main-diagonal reflection.
    - For a layer with no cue, rotate that layer by 180 degrees.
    - After every target mask is computed, redraw the board from outer to inner so later layers overwrite earlier ones on overlap cells.

    Key insight:
    - The outside cues no longer encode quarter-turn counts by area; they name reflection types by silhouette, and cue absence is itself an instruction: no cue means half-turn. Because transformed layers may overlap, the answer must be recomposed in structural draw order instead of updated in place.
    """
    return _solve_variant(grid, missing_halfturn=True, overwrite=True)


# --------------- generator ---------------

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

CUE_BOARD_DIST = 4
CUE_INTER_DIST = 4
OUTER_AREA_PAD = 4

GEN_RETRIES = 800


def _partner_for_rule(r: int, c: int, n: int, cue_transform: str | None) -> tuple[int, int]:
    """Return the paired cell under the layer's transform rule."""
    if cue_transform is None:
        return n - 1 - r, n - 1 - c
    if cue_transform == "h":
        return r, n - 1 - c
    if cue_transform == "v":
        return n - 1 - r, c
    if cue_transform == "d":
        return c, r
    raise AssertionError(f"unknown transform: {cue_transform}")


def _is_visible_under_rule(mask: np.ndarray, cue_transform: str | None) -> bool:
    """True iff the combo rule changes the mask."""
    return not np.array_equal(mask, apply_layer_rule(mask, cue_transform, missing_halfturn=True))


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
        if count >= int(target):
            return True
        if not mask[r, c]:
            mask[r, c] = True
            count += 1
    return count >= int(target)


def _make_asym_layer(
    n: int,
    density_lo: float,
    density_hi: float,
    *,
    cue_transform: str | None,
    extra_lo_terms: list[int] | None = None,
) -> np.ndarray | None:
    """Build a perimeter-heavy layer that visibly changes under its assigned transform rule."""
    mask = _checker_perimeter(n, random.randrange(2))
    partner = _partner_for_rule(*ASYM_MARKER, n, cue_transform)
    assert partner != ASYM_MARKER, "asymmetry marker must not lie on its transform axis"
    mask[ASYM_MARKER] = True
    mask[partner] = False

    lo_terms = [int(np.ceil(density_lo * n * n)), int(mask.sum())]
    if extra_lo_terms:
        lo_terms.extend(int(v) for v in extra_lo_terms)
    lo = max(lo_terms)
    hi = int(np.floor(density_hi * n * n))
    if lo > hi:
        return None
    target = random.randint(lo, hi)

    candidates = [
        (r, c)
        for r, c in _band_cells(n, LAYER_BAND_WIDTH)
        if (r, c) != partner and not mask[r, c]
    ]
    if not _fill_to_count(mask, candidates, target):
        return None
    if not _is_visible_under_rule(mask, cue_transform):
        return None
    return mask


def _make_middle_mask(n: int, cue_transform: str | None) -> np.ndarray | None:
    m = n - 2 * MIDDLE_INSET
    hi_cap = int(np.floor(MID_OUTER_CAP * n * n)) - 5
    local = _make_asym_layer(
        m,
        MID_DENSITY_LO,
        min(MID_DENSITY_HI, hi_cap / (m * m)),
        cue_transform=cue_transform,
    )
    if local is None:
        return None
    mask = np.zeros((n, n), dtype=bool)
    mask[MIDDLE_INSET : MIDDLE_INSET + m, MIDDLE_INSET : MIDDLE_INSET + m] = local
    if not _is_visible_under_rule(mask, cue_transform):
        return None
    return mask


def _make_outer_mask(n: int, min_area: int, cue_transform: str | None) -> np.ndarray | None:
    return _make_asym_layer(
        n,
        OUTER_DENSITY_LO,
        OUTER_DENSITY_HI,
        cue_transform=cue_transform,
        extra_lo_terms=[min_area],
    )


def _valid_inner_spans(n: int) -> list[int]:
    max_span = n - INNER_SPAN_MARGIN
    return [s for s in range(INNER_SPAN_MIN, max_span + 1) if (s % 2) == (n % 2)]


def _make_inner_mask(n: int, cue_transform: str | None) -> np.ndarray | None:
    spans = _valid_inner_spans(n)
    if not spans:
        return None

    for _ in range(40):
        span = random.choice(spans)
        top = (n - span) // 2
        local = np.zeros((span, span), dtype=bool)
        if span == 3:
            variants = [
                np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
                np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
                np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
            ]
            local = random.choice(variants).copy()
            local = np.rot90(local, random.randrange(4))
        elif span == 4:
            pts = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
            for r, c in pts:
                local[r, c] = True
            if random.randrange(2):
                local = np.fliplr(local)
            local = np.rot90(local, random.randrange(4))
        else:
            pts = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 4)]
            for r, c in pts:
                local[r, c] = True
            if random.randrange(2):
                local = np.fliplr(local)
            local = np.rot90(local, random.randrange(4))

        mask = np.zeros((n, n), dtype=bool)
        mask[top : top + span, top : top + span] = local
        if _is_visible_under_rule(mask, cue_transform):
            return mask
    return None


def _pick_cue_plan() -> list[str | None]:
    """Choose cue placement with stable coverage of both cue/no-cue branches."""
    if random.randrange(2):
        outer_tf, inner_tf = random.sample(list(TRANSFORMS), 2)
        return [outer_tf, None, inner_tf]
    return [None, random.choice(TRANSFORMS), None]


def _cue_patch(color: int, cue_transform: str) -> np.ndarray:
    template = CUE_TEMPLATES[cue_transform]
    patch = np.zeros(template.shape, dtype=int)
    patch[template] = int(color)
    return patch


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
        mask[rr, start : start + span] = True
    else:
        height = r1 - r0 + 1
        span = min(length, max(1, height - 2))
        start = r0 + (height - span) // 2
        cc = c1 if side == 1 else c0
        mask[start : start + span, cc] = True
    return mask


def _candidate_side_strips(
    n: int,
    bbox: tuple[int, int, int, int],
    lengths: list[int],
) -> list[np.ndarray]:
    """Enumerate all short side strips along the bbox perimeter."""
    r0, c0, r1, c1 = bbox
    out: list[np.ndarray] = []
    for length in lengths:
        span_h = min(int(length), c1 - c0 + 1)
        for start in range(c0, c1 - span_h + 2):
            top = np.zeros((n, n), dtype=bool)
            bot = np.zeros((n, n), dtype=bool)
            top[r0, start : start + span_h] = True
            bot[r1, start : start + span_h] = True
            out.extend([top, bot])
        span_v = min(int(length), r1 - r0 + 1)
        for start in range(r0, r1 - span_v + 2):
            right = np.zeros((n, n), dtype=bool)
            left = np.zeros((n, n), dtype=bool)
            right[start : start + span_v, c1] = True
            left[start : start + span_v, c0] = True
            out.extend([right, left])
    return out


def _compose_input_layers(dst_masks: list[np.ndarray], colors: list[int], cue_plan: list[str | None]) -> np.ndarray | None:
    """Apply inverse transforms to destination masks and reject visible input overlaps."""
    n = int(dst_masks[0].shape[0])
    board = np.zeros((n, n), dtype=int)
    for color, dst_mask, cue_transform in zip(colors, dst_masks, cue_plan):
        inv_mask = apply_layer_rule(dst_mask, cue_transform, missing_halfturn=True)
        if np.any((board != 0) & inv_mask):
            return None
        board[inv_mask] = int(color)
    return board


def _output_board(colors: list[int], dst_masks: list[np.ndarray]) -> np.ndarray:
    return _render_layers(dst_masks[0].shape, list(zip(colors, dst_masks)), bg=0)


def _overlap_metrics(dst_masks: list[np.ndarray]) -> dict:
    cover = sum(mask.astype(int) for mask in dst_masks)
    om = int(np.sum(dst_masks[0] & dst_masks[1]))
    mi = int(np.sum(dst_masks[1] & dst_masks[2]))
    oi = int(np.sum(dst_masks[0] & dst_masks[2]))
    return {
        "om": om,
        "mi": mi,
        "oi": oi,
        "hidden_cells": int(np.sum(cover >= 2)),
        "max_stack": int(cover.max()),
    }


def _behavior_metrics(grid: np.ndarray) -> dict:
    """Quantify whether D8 and D17 actually matter on this sample."""
    combo_out = main(grid)
    ignore_cues = _solve_variant(grid, missing_halfturn=True, overwrite=True, force_no_cues=True)
    no_missing_halfturn = _solve_variant(grid, missing_halfturn=False, overwrite=True)
    no_overwrite = _solve_variant(grid, missing_halfturn=True, overwrite=False)
    return {
        "diff_ignore_cues": int(np.sum(combo_out != ignore_cues)),
        "diff_missing_halfturn": int(np.sum(combo_out != no_missing_halfturn)),
        "diff_no_overwrite": int(np.sum(combo_out != no_overwrite)),
    }


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(range(1, 10), NUM_COLORS)
    cue_plan = _pick_cue_plan()

    n = random.choice([n0 for n0 in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n0)])

    base_triplet = None
    for _ in range(80):
        mid_base = _make_middle_mask(n, cue_plan[1])
        inner_base = _make_inner_mask(n, cue_plan[2])
        outer_base = _make_outer_mask(n, 0 if mid_base is None else int(mid_base.sum()) + OUTER_AREA_PAD, cue_plan[0])
        if mid_base is None or inner_base is None or outer_base is None:
            continue

        # Build disjoint visible input layers first; overlap is introduced only after forward transforms.
        if np.any(mid_base & inner_base):
            mid_base = mid_base & ~inner_base
        if np.any(outer_base & mid_base):
            outer_base = outer_base & ~mid_base
        if np.any(outer_base & inner_base):
            outer_base = outer_base & ~inner_base
        if np.any(outer_base & mid_base) or np.any(outer_base & inner_base) or np.any(mid_base & inner_base):
            continue
        if int(outer_base.sum()) == 0 or int(mid_base.sum()) == 0 or int(inner_base.sum()) == 0:
            continue
        if any(not _is_visible_under_rule(mask, cue_transform) for mask, cue_transform in zip((outer_base, mid_base, inner_base), cue_plan)):
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
    dst_masks: list[np.ndarray] | None = None
    inner_len = max(2, min(3, inner_bbox[2] - inner_bbox[0] + 1, inner_bbox[3] - inner_bbox[1] + 1))
    om_targets = _candidate_side_strips(n, mid_bbox, [3, 4])
    mi_targets = _candidate_side_strips(n, inner_bbox, sorted({2, inner_len}))
    random.shuffle(om_targets)
    random.shuffle(mi_targets)
    trial_pairs = [(om_target, mi_target) for om_target in om_targets for mi_target in mi_targets]
    random.shuffle(trial_pairs)
    for om_target, mi_target in trial_pairs[:160]:

        req_outer = apply_layer_rule(om_target, cue_plan[0], missing_halfturn=True)
        req_mid = apply_layer_rule(om_target, cue_plan[1], missing_halfturn=True) | apply_layer_rule(
            mi_target, cue_plan[1], missing_halfturn=True
        )
        req_inner = apply_layer_rule(mi_target, cue_plan[2], missing_halfturn=True)

        if np.any(req_outer & req_mid) or np.any(req_outer & req_inner) or np.any(req_mid & req_inner):
            continue

        src_outer = (outer_base & ~req_mid & ~req_inner) | req_outer
        src_mid = (mid_base & ~req_outer & ~req_inner) | req_mid
        src_inner = (inner_base & ~req_outer & ~req_mid) | req_inner
        src_masks = [src_outer, src_mid, src_inner]

        if np.any(src_outer & src_mid) or np.any(src_outer & src_inner) or np.any(src_mid & src_inner):
            continue
        if int(src_outer.sum()) == 0 or int(src_mid.sum()) == 0 or int(src_inner.sum()) == 0:
            continue
        if any(not _is_visible_under_rule(mask, cue_transform) for mask, cue_transform in zip(src_masks, cue_plan)):
            continue

        trial_masks = [
            apply_layer_rule(src_mask, cue_transform, missing_halfturn=True)
            for src_mask, cue_transform in zip(src_masks, cue_plan)
        ]
        if any(not _is_visible_under_rule(mask, cue_transform) for mask, cue_transform in zip(trial_masks, cue_plan)):
            continue

        metrics = _overlap_metrics(trial_masks)
        if metrics["om"] == 0 or metrics["mi"] == 0 or metrics["hidden_cells"] < 4 or metrics["max_stack"] < 2:
            continue

        in_board = _render_layers((n, n), list(zip(colors, src_masks)), bg=0)
        out_board = _output_board(colors, trial_masks)
        if len(colors_present(out_board, exclude={0})) != 3:
            continue
        dst_masks = trial_masks
        break
    if in_board is None or out_board is None or dst_masks is None:
        return None

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [(col, cue_transform) for col, cue_transform in zip(colors, cue_plan) if cue_transform is not None]
    random.shuffle(cue_items)
    for col, cue_transform in cue_items:
        if not _place_cue(canvas, _cue_patch(col, cue_transform), board_bbox, cue_bboxes):
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

    behavior = _behavior_metrics(canvas)
    if (
        behavior["diff_ignore_cues"] == 0
        or behavior["diff_missing_halfturn"] == 0
        or behavior["diff_no_overwrite"] == 0
    ):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")
