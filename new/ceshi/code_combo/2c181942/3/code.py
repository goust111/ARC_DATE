# Combo variant of 2c181942 (index 3)
# combo_spec: D1/0+D7/2
# Name: 形状匹配+非对称唯一旋转

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for this combo variant
# ---------------------------------------------------------------------------
#
# fusion_summary:
#   - D1 removes color pairing entirely: hub and all sprites share one foreground
#     color, so side assignment must come from geometry alone.
#   - D7 removes mirror-symmetry disambiguation: each real sprite is made fully
#     asymmetric, and the solver must test all 4 rotations and keep the unique
#     one whose contact mouth complements one hub edge.
#   - The two changes only become observable together if the hub itself exposes
#     visible side-specific shape codes. Therefore this combo keeps the D1 dock:
#     a 6x6 hollow core plus a 2-cell-thick outer docking strip on each side.
#     The outer dock row/col contains a 4-bit notch pattern; the sprite exposes
#     the matching 4-bit protrusion row/col backed by a solid support band.
#
# concrete_pixel_geometry:
#   - Core hub: 6x6 hollow square, corners/bg interior = bg, all 16 edge cells
#     use one shared non-bg color.
#   - Each side has a 2-cell-thick dock, still in the same color:
#       top dock  : rows r-2,r-1 over cols c+1..c+4
#       bot dock  : rows r+6,r+7 over cols c+1..c+4
#       left dock : cols c-2,c-1 over rows r+1..r+4
#       right dock: cols c+6,c+7 over rows r+1..r+4
#   - The outermost dock lane is encoded by four fixed notch patterns:
#       top   = 1000
#       bot   = 1100
#       left  = 1010
#       right = 1110
#     Here 1 means "hub leaves bg notch here", so the matching sprite must have
#     a filled protrusion at that slot. These patterns are pairwise distinct and
#     not reverses of one another, which prevents accidental side confusion.
#
# tiny_behavior_trace:
#   - Original task:
#       1. find central hollow square
#       2. read edge colors
#       3. for each color-matching sprite, rotate until symmetry + contact length
#          fit, then attach
#   - This combo:
#       1. find the central 6x6 hollow core and the surrounding same-color dock
#       2. read the four hub notch patterns from the dock outer lane
#       3. collect every non-hub component, because color no longer separates them
#       4. for each side, try each candidate component in all 4 rotations
#       5. accept only a rotation where:
#            - the support band is a full run of length 4 on the contact side
#            - the adjacent protrusion lane equals that side's notch pattern
#            - that component has exactly one valid rotation for that side
#       6. paste the matched component flush to the dock
#   - Example:
#       if the top dock outer row is 1000, then a top sprite must present a
#       bottom contact mouth whose inner support row is 1111 and whose outer lip
#       is 1000. Because the sprite body is lopsided, only one rotation yields
#       that mouth on the bottom edge.
#
# complexity_calibration:
#   - If we only unified colors, the four hub sides would become indistinguishable
#     and the task would collapse. The dock is therefore necessary, not optional.
#   - If we kept symmetric sprites, rotation would still be easy to short-circuit.
#     Requiring full D4 asymmetry keeps the D7 change visually present.
#   - Pattern vocabulary is intentionally tiny: 4 fixed 4-bit shapes reused in
#     every sample, so humans can infer "notch <-> protrusion complement" from
#     2-4 examples instead of memorizing arbitrary codes.
#
# implicit_assumption_audit:
#   - Original `m = (g == col) & (~hub_mask)` assumed one side <-> one color.
#     This is invalid now; solver must inspect all non-hub components instead.
#   - Original symmetry check (`flipud` / `fliplr`) is invalid now; solver must
#     count valid rotations by direct contact-mouth geometry.
#   - Original distractors could be ignored by using a fresh color. That also
#     breaks here, so distractors use the same color but are generated to have
#     zero valid `(rotation, side)` matches.
#   - New solver assumption: the hub is the unique central same-color component
#     containing the 6x6 hollow core and its dock. Generator enforces margins so
#     scattered sprites never touch the hub.
#   - New solver assumption: for each real sprite there is exactly one valid
#     `(rotation, side)` pair globally. Generator enforces this by checking
#     `_count_valid_matches(mask) == [(0, side)]` before scattering.
#
# placement_legality / cleanliness:
#   - Output is built first; every attached placement is bounds-checked and must
#     avoid foreground overlap. The only adjacencies are the intended dock joins.
#   - Input scattering uses `random_free_location(..., padding=2)` so sprites are
#     visually separated from the hub and from one another.
#   - All foreground elements share one color, so visual decoding relies purely
#     on contour. To keep that readable, object count stays low (2-4 real sprites,
#     0-1 distractor) and the dock uses only one extra geometric layer.
#
# observability / non-degeneracy target:
#   - Every valid sample should satisfy:
#       1. exactly one non-bg color in the whole grid
#       2. hub exposes 4 distinct notch patterns
#       3. every real sprite is fully asymmetric under the 7 non-identity D4
#          transforms
#       4. every real sprite has exactly one valid `(rotation, side)` match and
#          the required solving rotation is non-zero in the scattered input
#   - The metric helper near the bottom reports these quantities so we can check
#     that generation is not silently collapsing back to an easier regime.


HUB_N = 6
EDGE_LEN = HUB_N - 2
DOCK_THICKNESS = 2
BG_COLOR = 8
SIDES = ("top", "bot", "left", "right")
PATTERNS = {
    "top": (1, 0, 0, 0),
    "bot": (1, 1, 0, 0),
    "left": (1, 0, 1, 0),
    "right": (1, 1, 1, 0),
}


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    dummy = np.zeros(mask.shape, dtype=int)
    _, cropped = _crop_to_mask(dummy, mask)
    return cropped


def _find_core_hub(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    H, W = g.shape
    best = None
    for r in range(DOCK_THICKNESS, int(H - HUB_N - DOCK_THICKNESS)):
        for c in range(DOCK_THICKNESS, int(W - HUB_N - DOCK_THICKNESS)):
            if (
                int(g[int(r), int(c)]) != int(bg)
                or int(g[int(r), int(c + HUB_N - 1)]) != int(bg)
                or int(g[int(r + HUB_N - 1), int(c)]) != int(bg)
                or int(g[int(r + HUB_N - 1), int(c + HUB_N - 1)]) != int(bg)
            ):
                continue

            inner = g[int(r + 1) : int(r + HUB_N - 1), int(c + 1) : int(c + HUB_N - 1)]
            if not bool((inner == int(bg)).all()):
                continue

            top = g[int(r), int(c + 1) : int(c + HUB_N - 1)]
            bot = g[int(r + HUB_N - 1), int(c + 1) : int(c + HUB_N - 1)]
            left = g[int(r + 1) : int(r + HUB_N - 1), int(c)]
            right = g[int(r + 1) : int(r + HUB_N - 1), int(c + HUB_N - 1)]
            if (
                bool((top == int(bg)).any())
                or bool((bot == int(bg)).any())
                or bool((left == int(bg)).any())
                or bool((right == int(bg)).any())
            ):
                continue

            colors = set(map(int, np.concatenate([top, bot, left, right])))
            if len(colors) != 1:
                continue
            color = int(next(iter(colors)))
            if color == int(bg):
                continue

            cr = float(r + (HUB_N - 1) / 2)
            cc = float(c + (HUB_N - 1) / 2)
            score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
            cand = (float(score), int(r), int(c), int(color))
            if best is None or cand[0] < best[0]:
                best = cand

    assert best is not None
    _, r, c, color = best
    return int(r), int(c), HUB_N, int(color)


def _side_span(r: int, c: int, side: str) -> tuple[int, int]:
    if side in ("top", "bot"):
        return int(c + 1), int(c + HUB_N - 2)
    return int(r + 1), int(r + HUB_N - 2)


def _dock_outer_coord(r: int, c: int, side: str) -> int:
    if side == "top":
        return int(r - 2)
    if side == "bot":
        return int(r + HUB_N + 1)
    if side == "left":
        return int(c - 2)
    return int(c + HUB_N + 1)


def _read_hub_patterns(g: np.ndarray, r: int, c: int, bg: int) -> dict[str, tuple[int, ...]]:
    patterns: dict[str, tuple[int, ...]] = {}
    for side in SIDES:
        start, end = _side_span(r, c, side)
        outer = _dock_outer_coord(r, c, side)
        vals = []
        if side in ("top", "bot"):
            for cc in range(int(start), int(end) + 1):
                vals.append(1 if int(g[int(outer), int(cc)]) == int(bg) else 0)
        else:
            for rr in range(int(start), int(end) + 1):
                vals.append(1 if int(g[int(rr), int(outer)]) == int(bg) else 0)
        patterns[side] = tuple(vals)
    return patterns


def _contact_band(mask: np.ndarray, side: str, pattern: tuple[int, ...]) -> int | None:
    h, w = mask.shape
    L = len(pattern)
    hits: list[int] = []

    if side == "top":
        if h < 2 or w < L:
            return None
        for j in range(int(w - L + 1)):
            if bool(mask[h - 2, :j].any()) or bool(mask[h - 2, j + L :].any()):
                continue
            if bool(mask[h - 1, :j].any()) or bool(mask[h - 1, j + L :].any()):
                continue
            if not bool(mask[h - 2, j : j + L].all()):
                continue
            if tuple(int(v) for v in mask[h - 1, j : j + L]) != pattern:
                continue
            hits.append(int(j))
    elif side == "bot":
        if h < 2 or w < L:
            return None
        for j in range(int(w - L + 1)):
            if bool(mask[1, :j].any()) or bool(mask[1, j + L :].any()):
                continue
            if bool(mask[0, :j].any()) or bool(mask[0, j + L :].any()):
                continue
            if not bool(mask[1, j : j + L].all()):
                continue
            if tuple(int(v) for v in mask[0, j : j + L]) != pattern:
                continue
            hits.append(int(j))
    elif side == "left":
        if w < 2 or h < L:
            return None
        for j in range(int(h - L + 1)):
            if bool(mask[:j, w - 2].any()) or bool(mask[j + L :, w - 2].any()):
                continue
            if bool(mask[:j, w - 1].any()) or bool(mask[j + L :, w - 1].any()):
                continue
            if not bool(mask[j : j + L, w - 2].all()):
                continue
            if tuple(int(v) for v in mask[j : j + L, w - 1]) != pattern:
                continue
            hits.append(int(j))
    else:
        if w < 2 or h < L:
            return None
        for j in range(int(h - L + 1)):
            if bool(mask[:j, 1].any()) or bool(mask[j + L :, 1].any()):
                continue
            if bool(mask[:j, 0].any()) or bool(mask[j + L :, 0].any()):
                continue
            if not bool(mask[j : j + L, 1].all()):
                continue
            if tuple(int(v) for v in mask[j : j + L, 0]) != pattern:
                continue
            hits.append(int(j))

    if len(hits) != 1:
        return None
    return int(hits[0])


def _count_valid_matches(mask: np.ndarray) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for krot in range(4):
        rot = np.rot90(mask, int(krot))
        rot = _crop_mask(rot)
        for side, pattern in PATTERNS.items():
            if _contact_band(rot, side, pattern) is not None:
                matches.append((int(krot), str(side)))
    return matches


def _is_fully_asymmetric(mask: np.ndarray) -> bool:
    variants = [
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        np.fliplr(mask),
        np.flipud(mask),
        mask.T,
        np.fliplr(mask.T),
    ]
    return all(not np.array_equal(mask, other) for other in variants)


def _placeable(
    out: np.ndarray,
    arr: np.ndarray,
    mask: np.ndarray,
    dr: int,
    dc: int,
    bg: int,
) -> bool:
    H, W = out.shape
    for rr, cc in np.argwhere(mask):
        R = int(rr + dr)
        C = int(cc + dc)
        if not (0 <= R < H and 0 <= C < W):
            return False
        val = int(arr[int(rr), int(cc)])
        if val == int(bg):
            continue
        if int(out[int(R), int(C)]) != int(bg) and int(out[int(R), int(C)]) != val:
            return False
    return True


def _attachment_shift(mask: np.ndarray, side: str, pattern: tuple[int, ...], hub_r: int, hub_c: int) -> tuple[int, int] | None:
    pos = _contact_band(mask, side, pattern)
    if pos is None:
        return None

    span_start, _span_end = _side_span(hub_r, hub_c, side)
    outer = _dock_outer_coord(hub_r, hub_c, side)
    h, w = mask.shape

    if side == "top":
        return int(outer - (h - 1)), int(span_start - pos)
    if side == "bot":
        return int(outer), int(span_start - pos)
    if side == "left":
        return int(span_start - pos), int(outer - (w - 1))
    return int(span_start - pos), int(outer)


def _match_component_to_side(
    comp_arr: np.ndarray,
    comp_mask: np.ndarray,
    side: str,
    pattern: tuple[int, ...],
    hub_r: int,
    hub_c: int,
    out: np.ndarray,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    hits = []
    for krot in range(4):
        arr = np.rot90(comp_arr, int(krot))
        mask = np.rot90(comp_mask, int(krot))
        arr, mask = _crop_to_mask(arr, mask)
        shift = _attachment_shift(mask, side, pattern, hub_r, hub_c)
        if shift is None:
            continue
        dr, dc = shift
        if not _placeable(out, arr, mask, dr, dc, bg):
            continue
        hits.append((arr, mask, int(dr), int(dc)))
    if len(hits) != 1:
        return None
    return hits[0]


def _draw_hub(canvas: np.ndarray, hub_r: int, hub_c: int, color: int, bg: int) -> None:
    for cc in range(1, HUB_N - 1):
        canvas[int(hub_r), int(hub_c + cc)] = int(color)
        canvas[int(hub_r + HUB_N - 1), int(hub_c + cc)] = int(color)
    for rr in range(1, HUB_N - 1):
        canvas[int(hub_r + rr), int(hub_c)] = int(color)
        canvas[int(hub_r + rr), int(hub_c + HUB_N - 1)] = int(color)

    span_c0 = int(hub_c + 1)
    span_r0 = int(hub_r + 1)
    canvas[int(hub_r - 1), span_c0 : span_c0 + EDGE_LEN] = int(color)
    canvas[int(hub_r + HUB_N), span_c0 : span_c0 + EDGE_LEN] = int(color)
    canvas[span_r0 : span_r0 + EDGE_LEN, int(hub_c - 1)] = int(color)
    canvas[span_r0 : span_r0 + EDGE_LEN, int(hub_c + HUB_N)] = int(color)

    for i, v in enumerate(PATTERNS["top"]):
        if not v:
            canvas[int(hub_r - 2), int(span_c0 + i)] = int(color)
    for i, v in enumerate(PATTERNS["bot"]):
        if not v:
            canvas[int(hub_r + HUB_N + 1), int(span_c0 + i)] = int(color)
    for i, v in enumerate(PATTERNS["left"]):
        if not v:
            canvas[int(span_r0 + i), int(hub_c - 2)] = int(color)
    for i, v in enumerate(PATTERNS["right"]):
        if not v:
            canvas[int(span_r0 + i), int(hub_c + HUB_N + 1)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub_r, hub_c, hub_n, _hub_color = _find_core_hub(g, bg)
    assert hub_n == HUB_N

    lab, _nc = label_mask(g != bg, connectivity=8)
    core_seed = (int(hub_r), int(hub_c + 1))
    hub_id = int(lab[core_seed])
    assert hub_id > 0
    hub_mask = lab == hub_id

    out = np.full((H, W), bg, dtype=int)
    out[hub_mask] = g[hub_mask]
    hub_patterns = _read_hub_patterns(g, hub_r, hub_c, bg)

    components = []
    for cid in range(1, int(lab.max()) + 1):
        if cid == hub_id:
            continue
        mask = lab == cid
        arr = np.where(mask, g, bg)
        arr, mask = _crop_to_mask(arr, mask)
        components.append((int(cid), arr, mask, int(mask.sum())))
    components.sort(key=lambda x: -x[3])

    used: set[int] = set()
    for side in SIDES:
        pattern = hub_patterns[side]
        found = None
        found_id = None
        for cid, arr, mask, _sz in components:
            if cid in used:
                continue
            match = _match_component_to_side(arr, mask, side, pattern, hub_r, hub_c, out, bg)
            if match is None:
                continue
            found = match
            found_id = int(cid)
            break
        if found is None:
            continue
        arr, mask, dr, dc = found
        paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=True)
        used.add(int(found_id))

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _add_body_noise(mask: np.ndarray, side: str, band_start: int, rng: random.Random) -> None:
    h, w = mask.shape

    if side == "top":
        if h >= 3:
            mask[h - 3, max(0, band_start - 1)] = True
        for rr in range(0, max(0, h - 3)):
            for cc in range(w):
                if rng.random() < (0.24 if (rr + 2 * cc) % 3 else 0.36):
                    mask[rr, cc] = True
        if h >= 4:
            mask[0, 0] = True
    elif side == "bot":
        if h >= 3:
            mask[2, min(w - 1, band_start + EDGE_LEN)] = True
        for rr in range(3, h):
            for cc in range(w):
                if rng.random() < (0.24 if (2 * rr + cc) % 3 else 0.36):
                    mask[rr, cc] = True
        if h >= 4:
            mask[h - 1, w - 1] = True
    elif side == "left":
        if w >= 3:
            mask[max(0, band_start - 1), w - 3] = True
        for rr in range(h):
            for cc in range(0, max(0, w - 3)):
                if rng.random() < (0.24 if (rr + 2 * cc) % 3 else 0.36):
                    mask[rr, cc] = True
        if w >= 4:
            mask[0, 0] = True
    else:
        if w >= 3:
            mask[min(h - 1, band_start + EDGE_LEN), 2] = True
        for rr in range(h):
            for cc in range(3, w):
                if rng.random() < (0.24 if (2 * rr + cc) % 3 else 0.36):
                    mask[rr, cc] = True
        if w >= 4:
            mask[h - 1, w - 1] = True

    # Cut a few easy accidental symmetries away from the contact mouth itself.
    if side in ("top", "bot"):
        for cc in range(w):
            if int(mask[:, cc].sum()) == h and h >= 4:
                cut_r = 0 if side == "top" else h - 1
                if cut_r not in (h - 1, h - 2, 0, 1):
                    mask[cut_r, cc] = False
    else:
        for rr in range(h):
            if int(mask[rr, :].sum()) == w and w >= 4:
                cut_c = 0 if side == "left" else w - 1
                if cut_c not in (w - 1, w - 2, 0, 1):
                    mask[rr, cut_c] = False


def _make_sprite_mask(pattern: tuple[int, ...], side: str, rng: random.Random) -> np.ndarray | None:
    L = len(pattern)
    for _ in range(1200):
        h = rng.randint(4, 7)
        w = rng.randint(4, 7)
        if side in ("top", "bot") and w < L:
            w = L
        if side in ("left", "right") and h < L:
            h = L

        mask = np.zeros((h, w), dtype=bool)
        band_start = rng.randint(0, w - L) if side in ("top", "bot") else rng.randint(0, h - L)

        if side == "top":
            mask[h - 2, band_start : band_start + L] = True
            for i, v in enumerate(pattern):
                if v:
                    mask[h - 1, band_start + i] = True
        elif side == "bot":
            mask[1, band_start : band_start + L] = True
            for i, v in enumerate(pattern):
                if v:
                    mask[0, band_start + i] = True
        elif side == "left":
            mask[band_start : band_start + L, w - 2] = True
            for i, v in enumerate(pattern):
                if v:
                    mask[band_start + i, w - 1] = True
        else:
            mask[band_start : band_start + L, 1] = True
            for i, v in enumerate(pattern):
                if v:
                    mask[band_start + i, 0] = True

        _add_body_noise(mask, side, band_start, rng)

        if not bool(mask.any()):
            continue
        mask = _crop_mask(mask)
        px = int(mask.sum())
        if px < 6 or px > 16:
            continue

        _lab, nc = label_mask(mask, connectivity=8)
        if int(nc) != 1:
            continue
        if not _is_fully_asymmetric(mask):
            continue

        matches = _count_valid_matches(mask)
        if matches != [(0, side)]:
            continue
        return mask

    return None


def _make_distractor_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(1200):
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        mask = np.zeros((h, w), dtype=bool)
        k = rng.randint(4, min(10, h * w))
        for rr, cc in rng.sample([(r, c) for r in range(h) for c in range(w)], k):
            mask[int(rr), int(cc)] = True
        if not bool(mask.any()):
            continue
        mask = _crop_mask(mask)
        if int(mask.sum()) < 4:
            continue
        _lab, nc = label_mask(mask, connectivity=8)
        if int(nc) != 1:
            continue
        if _count_valid_matches(mask):
            continue
        return mask
    return None


def _attach_sprite(
    out: np.ndarray,
    mask: np.ndarray,
    side: str,
    hub_r: int,
    hub_c: int,
    color: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    arr = np.where(mask, color, bg).astype(int)
    shift = _attachment_shift(mask, side, PATTERNS[side], hub_r, hub_c)
    if shift is None:
        return None
    dr, dc = shift
    if not _placeable(out, arr, mask, dr, dc, bg):
        return None
    paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=False)
    return arr, mask, int(dr), int(dc)


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    non_bg_colors = sorted(set(map(int, g[g != bg].flatten())))

    hub_r, hub_c, _n, _color = _find_core_hub(g, bg)
    hub_patterns = _read_hub_patterns(g, hub_r, hub_c, bg)

    lab, nc = label_mask(g != bg, connectivity=8)
    hub_id = int(lab[int(hub_r), int(hub_c + 1)])
    match_records = []
    for cid in range(1, int(nc) + 1):
        if cid == hub_id:
            continue
        mask = lab == cid
        mask = _crop_mask(mask)
        matches = _count_valid_matches(mask)
        match_records.append(
            {
                "cid": int(cid),
                "pixels": int(mask.sum()),
                "asymmetric": bool(_is_fully_asymmetric(mask)),
                "matches": [(int(k), str(side)) for k, side in matches],
            }
        )

    true_like = [
        rec for rec in match_records if rec["asymmetric"] and len(rec["matches"]) == 1
    ]
    required_rotations = [int(rec["matches"][0][0]) for rec in true_like]
    return {
        "non_bg_colors": len(non_bg_colors),
        "hub_patterns": hub_patterns,
        "distinct_hub_patterns": len(set(hub_patterns.values())),
        "sprite_components": len(match_records),
        "true_like_components": len(true_like),
        "required_rotations": required_rotations,
        "match_records": match_records,
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG_COLOR
    color = int(rng.choice([c for c in range(10) if c != bg]))

    H = rng.randint(20, 28)
    W = rng.randint(22, 28)
    K = rng.randint(2, 4)
    sprite_sides = rng.sample(list(SIDES), K)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(PATTERNS[side], side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    max_extent = max(max_sprite_h, max_sprite_w)
    margin = DOCK_THICKNESS + max_extent + 2

    hub_r_min = margin
    hub_r_max = H - HUB_N - margin
    hub_c_min = margin
    hub_c_max = W - HUB_N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    _draw_hub(out, hub_r, hub_c, color, bg)

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        placed = _attach_sprite(out, sprite_masks[side], side, hub_r, hub_c, color, bg)
        if placed is None:
            return None
        attached[side] = placed

    inp = np.full((H, W), bg, dtype=int)
    _draw_hub(inp, hub_r, hub_c, color, bg)
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask, _sr, _sc = attached[side]
        rot_k = rng.choice([1, 2, 3])
        rot_arr = np.rot90(arr, int(rot_k))
        rot_mask = np.rot90(mask, int(rot_k))
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        patch = np.where(rot_mask, rot_arr, bg).astype(int)
        try:
            pr, pc = random_free_location(
                canvas,
                patch,
                transparent=bg,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    if rng.random() < 0.35:
        dist_mask = _make_distractor_mask(rng)
        if dist_mask is not None:
            dist_arr = np.where(dist_mask, color, bg).astype(int)
            try:
                pr, pc = random_free_location(
                    canvas,
                    dist_arr,
                    transparent=bg,
                    padding=2,
                    padding_connectivity=8,
                )
                paste(canvas, dist_arr, pr, pc, transparent=bg)
                paste(inp, dist_arr, pr, pc, transparent=bg)
            except ValueError:
                pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    metrics = _analyze_input(inp)
    if int(metrics["non_bg_colors"]) != 1:
        return None
    if int(metrics["distinct_hub_patterns"]) != 4:
        return None
    if int(metrics["true_like_components"]) != K:
        return None
    required_rotations = [int(v) for v in metrics["required_rotations"]]
    if len(required_rotations) != K or any(rot == 0 for rot in required_rotations):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
