# Variant of 898e7135
# Dimension: D13 — 多步处理与排序 (multi_step)
# Change: 单步处理 — 所有精灵同时匹配并填入，无顺序依赖... → 分层顺序填充

from __future__ import annotations

from collections import Counter
from math import gcd
import random

import numpy as np

from arc_utils import bbox, crop_bbox, find_components, get_bg, paste, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the D13 variant
# -----------------------------------------------------------------------------
#
# spec_variant:
#   preserved_from_original:
#     - input still contains one monochrome frame, bg=0, scattered outside pieces,
#       and singleton distractor noise in a dedicated color
#     - each piece is still a kxk-block-scaled sprite, and input-side orientation
#       may be any dihedral transform
#     - output is still the tight frame layout scaled by k, with frame color used
#       as the background
#   core_change:
#     - original: every hole-region inside the frame is already a final slot, so
#       all pieces can be matched independently in one pass
#     - variant: some holes are composite cavities. A large rectangle-like piece
#       occupies one part of such a cavity; only after it is placed do the
#       remaining cells become a clean exact-bbox match for a smaller strip/L
#       piece. Thus later matching depends on earlier fills.
#   concrete_pixel_behavior:
#     - inside the frame, at least one pair of pieces forms a dependency cluster:
#       the union of the two final slots is a solid rectangle.
#     - the larger piece sits in one corner of that rectangle.
#     - the smaller piece is the remaining right/bottom strip.
#     - in the input, the whole rectangle is empty bg, so the smaller strip is
#       not directly visible as a slot: its bbox currently contains the larger
#       piece's cells too.
#     - after filling the larger piece, those filled cells become new walls for
#       later steps, and the strip becomes an exact match.
#   small_trace:
#     - original logic on a 3x4 rectangle + surrounding 4x5 strip cluster:
#       the frame contains one 4x5 hole component, but no piece has shape 4x5,
#       so one-shot component matching fails.
#     - variant logic:
#       1. sort pieces by area: 12-cell rectangle first, 8-cell strip later.
#       2. scan current hole layout for an exact bbox match of the 12-cell piece.
#          Inside a full rectangle several shifts can be locally valid, so the
#          solver uses a stable topmost-then-leftmost choice; generator always
#          places the intended large piece at that earliest location.
#       3. remove those hole cells from the layout.
#       4. now the remaining cells inside that 4x5 bbox equal the 8-cell strip,
#          so the later piece becomes matchable.
#   complexity_calibration:
#     - only 4 pieces are generated, with unique areas 12, 9, 8, 7 at block
#       scale. This keeps ordering obvious while still forcing at least two
#       delayed matches in most samples.
#     - dependency structure is limited to 2 clusters, so the logic is visible
#       in 2-3 examples and does not require large search depth.
#   implicit_assumptions_review:
#     - old `slots = connected components(layout)` assumed each final slot was
#       already an isolated hole. This is false here because a later slot can be
#       merged into a larger rectangle with an earlier slot.
#     - old loop `for s in slots: find one piece with cand["um"] == s["mask"]`
#       assumed matching against full components. Now matching must search exact
#       subpatches in the evolving layout.
#     - old `k` inference via smallest solid square piece assumed a unit slot or
#       solid-square piece existed. The variant may not contain one, so `k` is
#       inferred from the gcd of piece bbox sizes and run lengths.
#   placement_legality:
#     - generator builds final slot masks first, then unions them into the input
#       hole layout, so no overlaps exist.
#     - cluster unions are separated by at least one wall cell; outside pieces
#       are placed with random_free_location(..., padding=1, connectivity=8).
#     - oracle check `main(input) == expected_output` is kept.
#   visual_observability:
#     - the key visible cue is that some frame holes are plain rectangles, while
#       one of the outside pieces is only an L/strip that could not have matched
#       that rectangle until another piece turned part of it into wall.
#     - sorting uses visible area only; no color-value ordering is used.
#   neatness:
#     - only frame, pieces, and singleton distractors remain; no extra marker
#       systems or hidden codes are introduced.
# -----------------------------------------------------------------------------


# --- tunables ----------------------------------------------------------------

K_CHOICES = [2, 3]
CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 2
AREA_BUFFER_CONST = 20
DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8
GENERATE_MAX_RETRIES = 300


def _run_lengths(mask: np.ndarray) -> list[int]:
    vals: list[int] = []
    for row in mask.astype(int):
        cnt = 0
        for v in row:
            if int(v) == 1:
                cnt += 1
            elif cnt:
                vals.append(int(cnt))
                cnt = 0
        if cnt:
            vals.append(int(cnt))
    for col in mask.T.astype(int):
        cnt = 0
        for v in col:
            if int(v) == 1:
                cnt += 1
            elif cnt:
                vals.append(int(cnt))
                cnt = 0
        if cnt:
            vals.append(int(cnt))
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


def _find_exact_match(layout: np.ndarray, need: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, need.shape)
    H, W = map(int, layout.shape)
    hits: list[tuple[int, int]] = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            if np.array_equal(layout[r : r + h, c : c + w], need):
                hits.append((int(r), int(c)))
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sequentially extract large-to-small piece shapes from a frame layout.

    Variant rule:
    - Pieces are still kxk-block-scaled and may be rotated/flipped.
    - The frame's holes are not all final slots up front: some are composite
      cavities whose later slots only become exact matches after earlier fills.
    - Process pieces by descending area. At each step, find an exact-bbox
      placement of that piece in the current hole layout, choose the topmost
      then leftmost valid placement, fill it, and continue.
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
        usable: list[dict] = []
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            usable.append({"um": um, "vp": vp, "vm": vm})
        assert usable, "every piece must be k-block-scaled"
        decoded.append(
            {
                "pi": int(pi),
                "pixel_area": int(p["area"]),
                "cell_area": int(p["area"] // (k * k)),
                "variants": usable,
            }
        )

    decoded.sort(key=lambda d: (-int(d["cell_area"]), -max(v["um"].shape[0] * v["um"].shape[1] for v in d["variants"]), int(d["pi"])))

    placements: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    current = layout.copy()
    for piece in decoded:
        options: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
        seen_exact: set[tuple[int, int, bytes, tuple[int, int]]] = set()
        for var in piece["variants"]:
            um = var["um"]
            for r, c in _find_exact_match(current, um):
                key = (int(r), int(c), um.tobytes(), tuple(map(int, um.shape)))
                if key in seen_exact:
                    continue
                seen_exact.add(key)
                options.append((int(r), int(c), var["vp"], var["vm"], um))
        assert options, "every piece must have a sequential placement"
        options.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[4].shape[0]), int(x[4].shape[1])))
        r0, c0, vp, vm, um = options[0]
        placements.append((int(r0), int(c0), vp, vm))
        h, w = map(int, um.shape)
        current[r0 : r0 + h, c0 : c0 + w][um] = False

    assert not bool(current.any()), "all holes must be exhausted after sequential filling"

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)
    for r0, c0, vp, vm in placements:
        paste(out, vp, (int(r0) + 1) * int(k), (int(c0) + 1) * int(k), mask=vm, overwrite=True)
    return out


# -----------------------------------------------------------------------------
# Generator
# -----------------------------------------------------------------------------


def _apply_dihedral(mask: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    out = np.rot90(mask, int(rot))
    if flip:
        out = np.fliplr(out)
    return out


def _mask_to_patch(mask: np.ndarray, color: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    km = np.kron(mask, np.ones((k, k), dtype=bool))
    return np.where(km, color, 0).astype(int), km


def _make_cluster(a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    big = np.zeros((a + 1, b + 1), dtype=bool)
    small = np.zeros((a + 1, b + 1), dtype=bool)
    big[:a, :b] = True
    small[:a, b] = True
    small[a, :] = True
    return big, small


def _tighten(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), int(r0), int(c0)


def _build_piece_specs(k: int, colors: list[int]) -> tuple[np.ndarray, list[dict]]:
    color_iter = iter(colors)
    layout = np.zeros((12, 8), dtype=bool)
    piece_specs: list[dict] = []

    # Dependency cluster: 12-cell rectangle first, then 8-cell strip becomes visible.
    big_full, small_full = _make_cluster(3, 4)
    big_mask, br0, bc0 = _tighten(big_full)
    small_mask, sr0, sc0 = _tighten(small_full)
    cluster_r = 0
    cluster_c = 0
    layout[cluster_r : cluster_r + 4, cluster_c : cluster_c + 5] = True
    piece_specs.append(
        {"mask": big_mask, "r": int(cluster_r + br0), "c": int(cluster_c + bc0), "area": int(big_mask.sum()), "color": int(next(color_iter))}
    )
    piece_specs.append(
        {"mask": small_mask, "r": int(cluster_r + sr0), "c": int(cluster_c + sc0), "area": int(small_mask.sum()), "color": int(next(color_iter))}
    )

    # Independent 9-cell piece: non-rectangular, so it cannot be spuriously found inside the full rectangle cluster.
    mask9 = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    )
    # Independent 7-cell piece.
    mask7 = np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    )

    indep9_pos = (5, 4)
    indep7_pos = (9, 0)
    r9, c9 = indep9_pos
    r7, c7 = indep7_pos
    h9, w9 = map(int, mask9.shape)
    h7, w7 = map(int, mask7.shape)
    if r9 + h9 > layout.shape[0] or c9 + w9 > layout.shape[1]:
        return np.zeros((1, 1), dtype=bool), []
    if r7 + h7 > layout.shape[0] or c7 + w7 > layout.shape[1]:
        return np.zeros((1, 1), dtype=bool), []

    if bool(layout[r9 : r9 + h9, c9 : c9 + w9].any()):
        return np.zeros((1, 1), dtype=bool), []
    if bool(layout[r7 : r7 + h7, c7 : c7 + w7].any()):
        return np.zeros((1, 1), dtype=bool), []

    layout[r9 : r9 + h9, c9 : c9 + w9] |= mask9
    layout[r7 : r7 + h7, c7 : c7 + w7] |= mask7

    piece_specs.append({"mask": mask9, "r": int(r9), "c": int(c9), "area": int(mask9.sum()), "color": int(next(color_iter))})
    piece_specs.append({"mask": mask7, "r": int(r7), "c": int(c7), "area": int(mask7.sum()), "color": int(next(color_iter))})

    hole_box = bbox(layout)
    assert hole_box is not None
    r0, c0, r1, c1 = map(int, hole_box)
    tight = layout[r0 : r1 + 1, c0 : c1 + 1].copy()
    for p in piece_specs:
        p["r"] -= int(r0)
        p["c"] -= int(c0)
    return tight, piece_specs


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
                "color": int(p["color"]),
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
    canvas_h = random.randint(max(min_h, 24), CANVAS_DIM_RANGE[1])
    canvas_w = random.randint(max(min_w, 24), CANVAS_DIM_RANGE[1])
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

    for p in sorted(pieces, key=lambda p: -int(p["in_mask"].sum())):
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
        rr, cc = random.randint(0, canvas_h - 1), random.randint(0, canvas_w - 1)
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
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
        hits = _find_exact_match(layout_mask, p["mask"])
        flags.append(bool(hits))
    return flags


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    piece_colors = [palette.pop() for _ in range(4)]

    layout_mask, piece_specs = _build_piece_specs(k, piece_colors)
    if not (4 <= layout_mask.shape[0] <= 12 and 4 <= layout_mask.shape[1] <= 8):
        return None

    sorted_specs = sorted(piece_specs, key=lambda d: -int(d["area"]))
    # At least one later piece must be hidden initially; usually both strip pieces are.
    initial_hits = _initial_matchability(layout_mask.copy(), sorted_specs)
    if sum(1 for hit in initial_hits[1:] if not hit) < 1:
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
