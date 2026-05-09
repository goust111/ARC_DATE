# Variant of 247ef758
# Dimension: D1 — 匹配条件：边缘颜色匹配 (match_cond)
# Change: 精灵颜色 == 框架左/右边缘行标记颜色 且 == 上/下边缘列标记颜色（双对边镜像匹配），只有两对对边同时存在该颜色时... → 形状嵌合匹配

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import label_mask


# ---------------------------------------------------------------------------
# Design analysis for the D1 variant
# ---------------------------------------------------------------------------
#
# variant_spec:
#   kept_from_original:
#     - layout stays "left sprite strip | divider column | framed area on right"
#     - movers are still erased from the left and stamped inside the frame interior
#     - each mover still replicates to the Cartesian product of its row marks and col marks
#     - overlap rule stays "bottom sprites are stamped first, upper sprites cover later"
#   d1_change_core:
#     - old row/col marks were encoded by mover colors copied onto opposite frame edges
#     - new row/col marks are encoded by shape-matching edge notches
#     - each mover shape appears as bg cavities on left/right edges at its target rows
#     - the same mover shape appears as bg cavities on top/bottom edges at its target cols
#     - solver must identify a mover by geometry, not by any edge color
#   concrete_visualization:
#     - the frame is now a thicker monochrome ring rather than a 1-cell outline
#     - every notch occupies the mover's tight bbox inside that ring
#     - inside that bbox, the mover silhouette is carved out as bg; the remaining cells stay frame-colored
#     - this means the local edge patch is the bbox-complement of the mover, while the notch void itself
#       is the mover silhouette humans can match directly
#     - the interior remains a clean bg rectangle, so the final stamping area is unchanged conceptually
#   behavior_trace:
#     original_small_example:
#       - a red sprite has red on both left/right at row 5 and red on both top/bottom at col 9
#       - solver reads color red on edges, infers target (5, 9), then stamps the red sprite there
#     variant_small_example:
#       - the same sprite is no longer referenced by red edge pixels
#       - instead, its exact silhouette is cut out on both side bands centered at row 5 and on both
#         top/bottom bands centered at col 9
#       - solver extracts left-strip sprite masks, extracts edge-notch masks, matches equal shapes,
#         intersects opposite-edge row positions and col positions, then stamps the sprite at (5, 9)
#     key_difference:
#       - the old pipeline grouped targets by color value
#       - the new pipeline groups targets by normalized shape signature
#   complexity_calibration:
#     - using full free-form random blobs up to 5x5 would force a much thicker frame and cause clutter
#     - to keep the notch cue visible and the grid within the original 10-15 x 14-20 envelope, mover
#       templates are restricted to distinctive 2x2 asymmetric silhouettes
#     - movers use unique silhouettes so shape matching is not ambiguous
#     - at least one mover gets two row marks or two col marks, so the replication behavior remains visible
#   implicit_assumption_audit:
#     original_main_dependencies:
#       - frame thickness was hard-coded as 1 via fr0=ir0-1 etc
#       - edge markers were assumed to be color-valued single cells on the frame boundary
#       - target extraction relied on "same color on both opposite edges"
#     why_that_breaks_now:
#       - the new frame ring is thicker, so fr0/fc0 cannot be inferred as one cell outside interior
#       - edge cues are bg cavities spread over several cells, so per-cell color reads no longer work
#       - mover colors never appear on the frame edges, so the old row_marks/col_marks logic would infer no targets
#     new_solver_guarantees_needed:
#       - largest non-bg monochrome component must be the frame ring
#       - largest enclosed bg component must still be the frame interior
#       - every notch must be a single 8-connected bg component touching exactly one side of the frame bbox
#       - no two movers may share the same normalized shape signature
#       - row notches and col notches must stay out of the corner bands so different edge cues do not overlap
#   placement_legality:
#     - notch bboxes are placed wholly inside the frame bands
#     - row targets are chosen so side notches never enter top/bottom bands
#     - col targets are chosen so top/bottom notches never enter left/right bands
#     - target centers are chosen with the same centering arithmetic used by stamping, so every mover fits fully
#     - mark positions are globally spaced so notch bboxes from different movers do not overlap
#   observability_and_cleanliness:
#     - the edges now visibly contain repeated shape cutouts instead of color dots, making the D1 change obvious
#     - the frame itself stays one solid color, so sprites and notches are easy to parse
#     - because mover colors do not appear on the frame at all, the new mechanism cannot be mistaken for
#       the original color-matching rule
#   generator_checks:
#     - main(input) must run and produce a changed output
#     - bg must remain the most frequent color
#     - old color-based solver would not move the movers on these inputs because their colors are absent
#       from all frame-edge solid pixels; this is used later as a non-degeneracy check


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected True component that does not touch the grid border."""
    h, w = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(h - 1)).any() or (cs == 0).any() or (cs == int(w - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _dominant_nonbg_color_bbox(grid: np.ndarray, bg: int) -> dict[str, object]:
    counts: dict[int, int] = {}
    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        counts[int(col)] = int(np.sum(grid == int(col)))
    assert counts
    frame_color = max(counts.items(), key=lambda kv: kv[1])[0]
    coords = np.argwhere(grid == int(frame_color))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return {
        "color": int(frame_color),
        "area": int(coords.shape[0]),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    arr = np.array(mask, dtype=int)
    return (int(arr.shape[0]), int(arr.shape[1])), tuple(map(int, arr.flatten()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move left-strip sprites into the frame interior.

    D1 variant:
    - Targets are no longer read from edge colors.
    - Instead, each mover's shape is carved as notches on opposite frame edges.
    - Matching opposite notches of the same normalized shape yields row marks and col marks.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = 0

    # The frame base color dominates the right-side ring; using its full-color bbox is robust even if
    # large notches split the ring into several components.
    frame = _dominant_nonbg_color_bbox(g, bg)
    fr0, fc0, fr1, fc1 = map(int, frame["bbox"])

    # The interior is still the largest enclosed bg component.
    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)

    # Divider detection stays close to the original task.
    divider_mask = np.zeros((h, w), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(h - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(w)[None, :] < int(fc0)

    # Extract left-strip sprites as in the original task.
    sprites = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask)
        if not bool(mask.any()):
            continue
        slab, sn = label_mask(mask, connectivity=8)
        for sid in range(1, int(sn) + 1):
            pts = np.argwhere(slab == int(sid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": m,
                    "sig": _mask_signature(m),
                }
            )

    # Read bg notch components that touch exactly one frame side.
    notch_window = g[int(fr0) : int(fr1) + 1, int(fc0) : int(fc1) + 1] == int(bg)
    nlab, nn = label_mask(notch_window, connectivity=8)
    row_side_marks: dict[tuple[tuple[int, int], tuple[int, ...]], dict[str, set[int]]] = defaultdict(
        lambda: {"left": set(), "right": set()}
    )
    col_side_marks: dict[tuple[tuple[int, int], tuple[int, ...]], dict[str, set[int]]] = defaultdict(
        lambda: {"top": set(), "bottom": set()}
    )
    for nid in range(1, int(nn) + 1):
        pts = np.argwhere(nlab == int(nid))
        if pts.size == 0:
            continue
        rs = pts[:, 0]
        cs = pts[:, 1]
        sides = []
        if bool((rs == 0).any()):
            sides.append("top")
        if bool((rs == int(notch_window.shape[0] - 1)).any()):
            sides.append("bottom")
        if bool((cs == 0).any()):
            sides.append("left")
        if bool((cs == int(notch_window.shape[1] - 1)).any()):
            sides.append("right")
        if len(sides) != 1:
            continue
        side = sides[0]
        nr0, nc0 = pts.min(axis=0)
        nr1, nc1 = pts.max(axis=0)
        mask = nlab[int(nr0) : int(nr1) + 1, int(nc0) : int(nc1) + 1] == int(nid)
        sig = _mask_signature(mask)
        gr0 = int(fr0 + nr0)
        gc0 = int(fc0 + nc0)
        gr1 = int(fr0 + nr1)
        gc1 = int(fc0 + nc1)
        if side in ("left", "right"):
            row_side_marks[sig][side].add((int(gr0) + int(gr1)) // 2)
        else:
            col_side_marks[sig][side].add((int(gc0) + int(gc1)) // 2)

    row_marks = {
        sig: sorted(v["left"] & v["right"])
        for sig, v in row_side_marks.items()
        if v["left"] and v["right"] and (v["left"] & v["right"])
    }
    col_marks = {
        sig: sorted(v["top"] & v["bottom"])
        for sig, v in col_side_marks.items()
        if v["top"] and v["bottom"] and (v["top"] & v["bottom"])
    }

    targets: dict[int, list[tuple[int, int]]] = {}
    for s in sprites:
        sig = s["sig"]
        if sig not in row_marks or sig not in col_marks:
            continue
        pts = {(int(r), int(c)) for r in row_marks[sig] for c in col_marks[sig]}
        if pts:
            targets[int(s["color"])] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]
        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


# ---------------------------------------------------------------------------
# generate_input
# ---------------------------------------------------------------------------

_SPRITE_TEMPLATES = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
]


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask.astype(bool), int(color), 0).astype(int)


def _placeable_center_range(lo: int, hi: int, span: int) -> tuple[int, int]:
    left = int(lo + span // 2)
    right = int(hi - (span - 1 - span // 2))
    return left, right


def _pick_mark_centers(
    low: int,
    high: int,
    count: int,
    span: int,
    occupied: list[tuple[int, int]],
) -> list[int] | None:
    candidates = list(range(int(low), int(high) + 1))
    random.shuffle(candidates)
    from itertools import combinations

    for combo in combinations(candidates, int(count)):
        picked = sorted(map(int, combo))
        ok = True
        for i in range(len(picked)):
            for j in range(i + 1, len(picked)):
                if abs(int(picked[i]) - int(picked[j])) <= int(span):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for ov, ospan in occupied:
            if any(abs(int(v) - int(ov)) <= int(max(span, ospan)) for v in picked):
                ok = False
                break
        if ok:
            return picked
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    sprite_area_w = 4
    n_movers = 2
    n_distractors = 0

    if n_movers + n_distractors > len(_SPRITE_TEMPLATES):
        return None

    template_idxs = list(range(n_movers + n_distractors))
    random.shuffle(template_idxs)
    mover_templates = [_SPRITE_TEMPLATES[i] for i in template_idxs[:n_movers]]
    distractor_templates = [_SPRITE_TEMPLATES[i] for i in template_idxs[n_movers:]]

    # Fixed roomy geometry is more reliable than resampling many incompatible sizes.
    band = 3
    interior_h = 9
    interior_w = 9
    h = interior_h + 2 * band
    frame_w = interior_w + 2 * band
    w = sprite_area_w + 1 + frame_w
    if h != 15 or w != 20:
        return None

    div_col = sprite_area_w
    fr0, fr1 = 0, h - 1
    fc0, fc1 = sprite_area_w + 1, w - 1
    ir0, ic0 = fr0 + band, fc0 + band
    ir1, ic1 = fr1 - band, fc1 - band

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    sprites = [_colorize(mask, color) for mask, color in zip(mover_templates, mover_colors)]
    distractors = [_colorize(mask, color) for mask, color in zip(distractor_templates, distractor_colors)]

    mover_targets: dict[int, tuple[list[int], list[int], np.ndarray]] = {}
    occupied_rows: list[tuple[int, int]] = []
    occupied_cols: list[tuple[int, int]] = []

    featured = random.randrange(n_movers)
    featured_axis = "rows"

    order = sorted(range(n_movers), key=lambda i: -max(sprites[i].shape))
    for mi in order:
        sp = sprites[mi]
        color = mover_colors[mi]
        sh, sw = map(int, sp.shape)

        n_rows = 2 if mi == featured and featured_axis == "rows" else 1
        n_cols = 2 if mi == featured and featured_axis == "cols" else 1

        min_row, max_row = _placeable_center_range(ir0, ir1, sh)
        min_col, max_col = _placeable_center_range(ic0, ic1, sw)
        if min_row > max_row or min_col > max_col:
            return None

        rows = _pick_mark_centers(min_row, max_row, n_rows, sh, occupied_rows)
        cols = _pick_mark_centers(min_col, max_col, n_cols, sw, occupied_cols)
        if rows is None or cols is None:
            return None

        occupied_rows.extend((r, sh) for r in rows)
        occupied_cols.extend((c, sw) for c in cols)
        mover_targets[int(color)] = (rows, cols, sp)

    inp = np.full((h, w), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    inp[int(fr0) : int(fr1) + 1, int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[int(ir0) : int(ir1) + 1, int(ic0) : int(ic1) + 1] = int(bg)

    # Carve shape notches on opposite edges.
    for color, (row_marks, col_marks, sp) in mover_targets.items():
        sh, sw = map(int, sp.shape)
        mask = sp != int(bg)

        for tr in row_marks:
            top = int(tr - sh // 2)
            left_c0 = int(fc0)
            right_c0 = int(fc1 - sw + 1)
            inp[int(top) : int(top) + sh, int(left_c0) : int(left_c0) + sw][mask] = int(bg)
            inp[int(top) : int(top) + sh, int(right_c0) : int(right_c0) + sw][mask] = int(bg)

        for tc in col_marks:
            left = int(tc - sw // 2)
            top_r0 = int(fr0)
            bottom_r0 = int(fr1 - sh + 1)
            inp[int(top_r0) : int(top_r0) + sh, int(left) : int(left) + sw][mask] = int(bg)
            inp[int(bottom_r0) : int(bottom_r0) + sh, int(left) : int(left) + sw][mask] = int(bg)

    cur_row = 0
    all_items: list[np.ndarray] = sprites + distractors
    for sp in all_items:
        sh, sw = map(int, sp.shape)
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        patch = sp != int(bg)
        inp[int(cur_row) : int(cur_row) + sh, int(c_off) : int(c_off) + sw][patch] = sp[patch]
        cur_row += sh + random.choice([0, 0, 1])
        if cur_row > h:
            return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(1200):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 1200 attempts")
