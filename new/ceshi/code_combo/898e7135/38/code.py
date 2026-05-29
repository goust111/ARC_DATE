# Combo variant of 898e7135 (index 38)
# combo_spec: D7/0+D13/0
# Name: 分层拼合填充

from __future__ import annotations

from collections import Counter
from math import gcd
import random

import numpy as np

from arc_utils import bbox, crop_bbox, find_components, get_bg, paste, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
# -----------------------------------------------------------------------------
#
# spec_variant:
#   preserved_from_original:
#     - input still contains one monochrome frame on bg=0, scattered kxk-block
#       pieces outside the frame, and singleton distractor pixels in one fresh
#       color
#     - output is still the frame's inner layout, tight-cropped, scaled by k,
#       with frame color used as the output background
#     - pieces may still arrive under arbitrary dihedral transforms
#   fused_change:
#     - D7/0: hole components are no longer in strict 1:1 correspondence with
#       outside pieces; one visible hole can be the union of several final slots
#     - D13/0: pieces must be inserted in descending visible area order; each
#       insertion removes cells from the current hole map and can reveal the next
#       exact slot inside the same large cavity
#   concrete_pixel_behavior:
#     - the generator builds two composite cavities, not one:
#       1. a 4x5 rectangle split into a 3x4 rectangle (area 12) plus an L-strip
#          (area 8)
#       2. a 4x4 rectangle split into a 3x3 square (area 9) plus an L-strip
#          (area 7)
#     - inside the input frame only the two large rectangles are directly visible
#       as hole components; the two smaller strip pieces are hidden because their
#       future zero-patterns have not been carved out yet
#     - after placing the 12-cell piece, the first 4x5 cavity becomes the 8-cell
#       L exactly; after placing the 9-cell piece, the second 4x4 cavity becomes
#       the 7-cell L exactly
#   small_trace:
#     - original logic would enumerate connected hole components and expect four
#       components for four pieces; here it sees only two components, so strict
#       slot-by-slot matching fails immediately
#     - combo logic on the current hole map:
#       1. area-12 rectangle is matched at the topmost-leftmost exact 3x4 block
#          window and filled
#       2. area-9 square is matched inside the second composite rectangle
#       3. the first cavity has now changed shape, exposing the area-8 L-strip
#       4. the second cavity has changed shape, exposing the area-7 L-strip
#   complexity_calibration:
#     - there are exactly four pieces with unique areas 12, 9, 8, 7 at cell
#       scale, so ordering is human-visible and deterministic without introducing
#       too many object types
#     - two separate many:1 cavities make the D7 change visually obvious; this
#       does not collapse into the weaker "one special cluster plus unrelated
#       singleton holes" pattern
#     - the smaller pieces are non-rectangular, preventing accidental direct
#       matches inside the untouched full rectangles
#   implicit_assumptions_review:
#     - old assumption: `connected holes == final slots`. Broken. A current hole
#       component can contain multiple future slots.
#     - old assumption: matching can compare each piece against one precomputed
#       slot mask. Broken. Matching must search exact submasks in an evolving
#       layout.
#     - old assumption: k can be inferred from a unit square piece. Broken. This
#       combo has no unit slot, so k is inferred from the gcd of piece extents
#       and run lengths.
#     - new solver assumption: for each area in descending order there is at
#       least one exact placement on the current layout, and the intended one is
#       the topmost-leftmost valid hit. The generator enforces this by fixing the
#       two large composite cavities at distinct positions and by choosing small
#       pieces whose zero-patterns cannot appear before their parent large piece
#       is removed.
#   placement_legality:
#     - the final slot masks are disjoint by construction; their unions form two
#       rectangular cavities separated by frame-color walls
#     - the layout touches all four borders of its tight crop, so the solver's
#       crop-to-hole-bbox step recovers the intended layout exactly
#     - the input canvas places frame and pieces with >= 1-cell 8-neighborhood
#       separation and adds only singleton distractors
#   visual_observability:
#     - a human can count four outside pieces but only two large holes inside the
#       frame, so many:1 composition is immediately visible
#     - the area ordering cue is directly visible because the pieces have unique
#       sizes and the smaller L-pieces cannot be matched to the initial full
#       rectangles until the larger blocks are conceptually inserted first
#     - no hidden ordering by color id or generation-time metadata is used
#   neatness:
#     - the scene uses only the original ingredients: frame, pieces, and
#       singleton noise; no auxiliary markers or overlaid guide colors are added
# -----------------------------------------------------------------------------


K_CHOICES = [2, 3]
CANVAS_DIM_RANGE = (24, 30)
AREA_BUFFER_MULT = 2
AREA_BUFFER_CONST = 28
DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8
GENERATE_MAX_RETRIES = 300


def _run_lengths(mask: np.ndarray) -> list[int]:
    vals: list[int] = []
    for row in mask.astype(int):
        run = 0
        for v in row:
            if int(v) == 1:
                run += 1
            elif run:
                vals.append(int(run))
                run = 0
        if run:
            vals.append(int(run))
    for col in mask.T.astype(int):
        run = 0
        for v in col:
            if int(v) == 1:
                run += 1
            elif run:
                vals.append(int(run))
                run = 0
        if run:
            vals.append(int(run))
    return vals


def _infer_k(pieces: list[dict]) -> int:
    g = 0
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        g = gcd(int(g), int(h))
        g = gcd(int(g), int(w))
        for run in _run_lengths(p["mask"]):
            g = gcd(int(g), int(run))
    assert g >= 1
    return int(g)


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    if hh % k != 0 or ww % k != 0:
        return None
    out = np.zeros((hh // k, ww // k), dtype=bool)
    for r in range(hh // k):
        for c in range(ww // k):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        p = np.rot90(patch, kk)
        m = np.rot90(mask, kk)
        base.append((p, m))
        base.append((np.fliplr(p), np.fliplr(m)))
    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for p, m in base:
        key = (m.tobytes(), tuple(map(int, m.shape)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m))
    return uniq


def _find_exact_matches(layout: np.ndarray, need: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, need.shape)
    H, W = map(int, layout.shape)
    hits: list[tuple[int, int]] = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            if np.array_equal(layout[r : r + h, c : c + w], need):
                hits.append((int(r), int(c)))
    return hits


def _tighten(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), int(r0), int(c0)


def _make_cluster(a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    big = np.zeros((a + 1, b + 1), dtype=bool)
    small = np.zeros((a + 1, b + 1), dtype=bool)
    big[:a, :b] = True
    small[:a, b] = True
    small[a, :] = True
    return big, small


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a scaled framed layout by carving composite cavities into final slots.

    Input structure:
    - The most common color is the background. One monochrome frame surrounds a
      cropped inner hole-layout, but some visible holes are larger unions of
      several final destinations rather than one destination each.
    - The outside sprites are monochrome kxk-block pieces under arbitrary
      rotation/reflection. Singleton noise pixels in another color are distractors.

    Rule:
    - Infer k from the outside pieces, decode every dihedral block-level shape,
      and sort pieces by decreasing cell area.
    - On the current hole map, find the topmost-leftmost exact match for the next
      piece's downsampled mask, place it there, and delete those hole cells before
      processing the next piece.
    - Emit the tight framed layout scaled by k, with the frame color as the
      background and all matched pieces pasted into their recovered slots.

    Key insight:
    - A current hole component may represent several future slots fused together.
      The smaller zero-patterns are not present yet; they only appear after larger
      earlier pieces turn part of the cavity into new walls, so matching must be
      sequential over an evolving layout instead of one-shot over connected holes.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "must contain frame and pieces"

    frame_candidates = []
    for comp in comps:
        cr0, cc0, cr1, cc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (cr0, cc0, cr1, cc1))
        hole = patch == int(bg)
        if not bool(hole.any()):
            continue
        if bool(hole[0, :].any() or hole[-1, :].any() or hole[:, 0].any() or hole[:, -1].any()):
            continue
        frame_candidates.append(comp)
    assert frame_candidates, "must contain a framed hole layout"
    frame_comp = max(frame_candidates, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose holes"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1].copy()

    pieces: list[dict] = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros_like(patch, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "must contain non-frame pieces"

    k = _infer_k(pieces)

    decoded: list[dict] = []
    for pi, p in enumerate(pieces):
        variants: list[dict] = []
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            variants.append({"um": um, "vp": vp, "vm": vm})
        assert variants, "every piece must be k-block-scaled"
        decoded.append(
            {
                "pi": int(pi),
                "cell_area": int(p["area"] // (k * k)),
                "variants": variants,
            }
        )

    decoded.sort(
        key=lambda d: (
            -int(d["cell_area"]),
            -max(int(v["um"].shape[0] * v["um"].shape[1]) for v in d["variants"]),
            int(d["pi"]),
        )
    )

    placements: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    current = layout.copy()
    for piece in decoded:
        options: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
        seen: set[tuple[int, int, bytes, tuple[int, int]]] = set()
        for var in piece["variants"]:
            um = var["um"]
            for r0, c0 in _find_exact_matches(current, um):
                key = (int(r0), int(c0), um.tobytes(), tuple(map(int, um.shape)))
                if key in seen:
                    continue
                seen.add(key)
                options.append((int(r0), int(c0), var["vp"], var["vm"], um))
        assert options, "every piece must have a sequential placement"
        options.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[4].shape[0]), int(x[4].shape[1])))
        r0, c0, vp, vm, um = options[0]
        placements.append((int(r0), int(c0), vp, vm))
        h, w = map(int, um.shape)
        current[r0 : r0 + h, c0 : c0 + w][um] = False

    assert not bool(current.any()), "all hole cells must be consumed"

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)
    for r0, c0, vp, vm in placements:
        paste(out, vp, (int(r0) + 1) * int(k), (int(c0) + 1) * int(k), mask=vm, overwrite=True)
    return out


def _mask_to_patch(mask: np.ndarray, color: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    km = np.kron(mask, np.ones((k, k), dtype=bool))
    return np.where(km, color, 0).astype(int), km


def _build_layout_and_specs(colors: list[int]) -> tuple[np.ndarray, list[dict]]:
    layout = np.zeros((9, 8), dtype=bool)
    specs: list[dict] = []
    color_iter = iter(colors)

    big_a_full, small_a_full = _make_cluster(3, 4)
    big_a_mask, ba_r, ba_c = _tighten(big_a_full)
    small_a_mask, sa_r, sa_c = _tighten(small_a_full)
    cluster_a_r, cluster_a_c = 0, 0
    layout[cluster_a_r : cluster_a_r + 4, cluster_a_c : cluster_a_c + 5] = True
    specs.append(
        {"mask": big_a_mask, "r": int(cluster_a_r + ba_r), "c": int(cluster_a_c + ba_c), "area": int(big_a_mask.sum()), "color": int(next(color_iter))}
    )
    specs.append(
        {"mask": small_a_mask, "r": int(cluster_a_r + sa_r), "c": int(cluster_a_c + sa_c), "area": int(small_a_mask.sum()), "color": int(next(color_iter))}
    )

    big_b_full, small_b_full = _make_cluster(3, 3)
    big_b_mask, bb_r, bb_c = _tighten(big_b_full)
    small_b_mask, sb_r, sb_c = _tighten(small_b_full)
    cluster_b_r, cluster_b_c = 5, 4
    layout[cluster_b_r : cluster_b_r + 4, cluster_b_c : cluster_b_c + 4] = True
    specs.append(
        {"mask": big_b_mask, "r": int(cluster_b_r + bb_r), "c": int(cluster_b_c + bb_c), "area": int(big_b_mask.sum()), "color": int(next(color_iter))}
    )
    specs.append(
        {"mask": small_b_mask, "r": int(cluster_b_r + sb_r), "c": int(cluster_b_c + sb_c), "area": int(small_b_mask.sum()), "color": int(next(color_iter))}
    )

    return layout, specs


def _build_expected_output(piece_specs: list[dict], layout_shape: tuple[int, int], k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_shape[0] + 2) * k, (layout_shape[1] + 2) * k), frame_color, dtype=int)
    for p in piece_specs:
        patch, mask = _mask_to_patch(p["mask"], int(p["color"]), int(k))
        paste(out, patch, (int(p["r"]) + 1) * int(k), (int(p["c"]) + 1) * int(k), mask=mask, overwrite=True)
    return out


def _build_input_pieces(piece_specs: list[dict], k: int) -> list[dict]:
    pieces: list[dict] = []
    for p in piece_specs:
        out_patch, out_mask = _mask_to_patch(p["mask"], int(p["color"]), int(k))
        rot = random.randint(0, 3)
        in_patch = np.rot90(out_patch, rot)
        in_mask = np.rot90(out_mask, rot)
        if random.random() < 0.5:
            in_patch = np.fliplr(in_patch)
            in_mask = np.fliplr(in_mask)
        pieces.append(
            {
                "area": int(p["area"]),
                "out_patch": out_patch,
                "out_mask": out_mask,
                "in_patch": in_patch,
                "in_mask": in_mask,
            }
        )
    return pieces


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    layout_h, layout_w = map(int, layout_mask.shape)
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
    total_piece_area = sum(int(p["in_mask"].sum()) for p in pieces)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 2)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 2)
    canvas_h = random.randint(min_h, CANVAS_DIM_RANGE[1])
    canvas_w = random.randint(min_w, CANVAS_DIM_RANGE[1])
    while canvas_h * canvas_w < needed_cells and (canvas_h < CANVAS_DIM_RANGE[1] or canvas_w < CANVAS_DIM_RANGE[1]):
        if canvas_h <= canvas_w and canvas_h < CANVAS_DIM_RANGE[1]:
            canvas_h += 1
        elif canvas_w < CANVAS_DIM_RANGE[1]:
            canvas_w += 1
        else:
            break
    if canvas_h * canvas_w < needed_cells:
        return None

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    frame_img = np.full((frame_h, frame_w), frame_color, dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, frame_color)

    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for p in sorted(pieces, key=lambda x: -int(x["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                canvas,
                p["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, p["in_patch"], pr, pc, mask=p["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if canvas[rr, cc] != 0:
            continue
        r0, r1 = max(0, rr - 1), min(canvas_h, rr + 2)
        c0, c1 = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r0:r1, c0:c1] != 0):
            continue
        canvas[rr, cc] = distractor_color
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _initial_matchability(layout_mask: np.ndarray, piece_specs: list[dict]) -> list[bool]:
    flags: list[bool] = []
    for p in sorted(piece_specs, key=lambda d: -int(d["area"])):
        flags.append(bool(_find_exact_matches(layout_mask, p["mask"])))
    return flags


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    piece_colors = [palette.pop() for _ in range(4)]

    layout_mask, piece_specs = _build_layout_and_specs(piece_colors)
    sorted_specs = sorted(piece_specs, key=lambda d: -int(d["area"]))

    initial_hits = _initial_matchability(layout_mask.copy(), sorted_specs)
    if initial_hits != [True, True, False, False]:
        return None

    expected_out = _build_expected_output(sorted_specs, layout_mask.shape, k, frame_color)
    pieces = _build_input_pieces(sorted_specs, k)
    canvas = _build_input_canvas(pieces, layout_mask, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
