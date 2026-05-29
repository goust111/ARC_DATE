# Combo variant of 2c181942 (index 4)
# combo_spec: D1/1+D5/0
# Name: 位置匹配+矩形hub

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage as _ndimage
from scipy.ndimage import label as _label

from arc_utils import bbox_coords, get_bg, label_mask, paste

# ---------------------------------------------------------------------------
# Design analysis for the fused variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace
# - Original task:
#   - detect a central hollow 4x4 square hub;
#   - each detached monochrome sprite is paired to one side by color equality;
#   - every side has the same contact length 2.
# - Fused task:
#   - the hub is a hollow 4x6 rectangle, so top/bottom contact length is 4 and
#     left/right contact length is 2;
#   - detached sprites are paired by hub-relative position, not by their color:
#     above -> top, below -> bottom, left -> left, right -> right;
#   - once attached, the sprite is recolored to the destination edge color.
# - Small trace:
#   - suppose a detached orange sprite sits above the hub and another orange
#     sprite sits left of the hub. The top hub edge may be blue and the left hub
#     edge may be red.
#   - original solver would fail because it looks for a blue detached component
#     for the top and a red one for the left.
#   - fused solver instead classifies the above component as "top", rotates it so
#     its bottom contact edge has length 4, then repaints it blue; it classifies
#     the left component as "left", rotates it so its right contact edge has
#     length 2, then repaints it red.
# - Main visual difference:
#   - color no longer explains which sprite goes to which side;
#   - the rectangular hub forces humans to notice that horizontal and vertical
#     sides demand different attachment lengths.
#
# b) Complexity calibration
# - The task still requires nontrivial rotation reasoning because position alone
#   does not determine orientation; the solver must test symmetry + contact edge.
# - All true sprites share one source color distinct from every hub edge color.
#   This makes the D1 change visible and prevents accidental fallback to the
#   original color-matching explanation.
# - The generator requires at least one horizontal-side sprite and one vertical-
#   side sprite in every sample, so the D5 rectangular-length change is always
#   visible in the examples rather than only appearing occasionally.
# - Hub size is fixed at 4x6 instead of allowing many rectangles; this keeps the
#   concept learnable from 2-4 samples while still being clearly different from
#   the base task.
#
# c) Implicit assumption review against the original solver
# - Original `_find_hub()` assumes an NxN square. That is invalid here, so hub
#   search is rewritten for a fixed 4x6 rectangle.
# - Original `target = n - 2` assumes one shared attachment length. Invalid here;
#   the fused solver uses `HORIZ_TARGET=4` for top/bottom and `VERT_TARGET=2`
#   for left/right.
# - Original `m = (g == col) & (~hub_mask)` assumes matching by destination color.
#   Invalid here because true sprites all share one source color unrelated to the
#   hub. The fused solver labels all outside non-bg components, then classifies
#   them into side corridors by geometry.
# - Original paste keeps the sprite's input color. Invalid here because the side
#   color is now the output color. The fused solver paints with the destination
#   edge color after choosing the correct rotation.
# - New fused assumption:
#   each intended sprite lies in exactly one side corridor and overlaps the hub's
#   inner span on the orthogonal axis. The generator guarantees this.
#
# d) Placement legality
# - Generation stays output-first:
#   1. build the true output with hub + correctly attached sprites,
#   2. detach and scatter rotated copies into side-specific corridors,
#   3. optionally add a distractor into a corner corridor that matches no side.
# - Every placement checks bounds and uses a dilated occupancy mask with padding
#   so sprites do not overlap or touch the hub/each other.
#
# e) Visual observability
# - Because all true sprites share one detached source color, humans can see that
#   color cannot be the matching key.
# - Because every sample includes both a horizontal and a vertical attachment,
#   humans can also see that the rectangular hub expects two different contact
#   lengths.
# - The only consistent cross-example explanation is:
#   spatial side selects destination side, then the shape rotates and adopts that
#   side's color.
#
# f) Visual cleanliness
# - Hub is placed near the center with a margin larger than every sprite extent.
# - True sprites live only in side corridors; distractors live only in corners.
#   This keeps components visually separable and side classification readable.
#
# g) Color-change inferability
# - Relevant colors are limited: four hub edge colors, one shared source-sprite
#   color, plus at most one distractor color.
# - The recoloring rule has a strong structural cue: the output sprite color is
#   always the color of the hub edge it touches.
#
# h) Quantitative anti-degeneracy checks intended after implementation
# - For true sprites, the count of input sprites whose color matches their target
#   edge color should be exactly 0 in essentially every sample.
# - Every generated sample should contain both a horizontal-side and a vertical-
#   side sprite, proving the rectangular length asymmetry is exercised.
# - Running the original color-based square-hub solver on these inputs should not
#   reproduce the fused outputs except by accident.
#
# i) New solver assumptions and generator guarantees
# - Solver assumption: there is a unique central 4x6 hollow rectangle with four
#   solid non-bg edges of distinct colors. Generator creates exactly one.
# - Solver assumption: each participating side has at most one dominant component
#   in its corridor. Generator places exactly one true sprite per active side and
#   corner distractors outside those corridors.
# - Solver assumption: for each sprite, exactly one rotation satisfies the side's
#   symmetry + contact-length constraints. Generator uses validated templates that
#   enforce this before scattering.

HUB_H = 4
HUB_W = 6
HORIZ_TARGET = HUB_W - 2
VERT_TARGET = HUB_H - 2
SIDES = ("top", "bot", "left", "right")

_TOP_CANONICAL = [
    np.array(
        [
            [0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
]

_LEFT_CANONICAL = [
    np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    ),
]


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _edge_segment(mask: np.ndarray, side: str):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        rr = int(pts[:, 0].min())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", rr, cols)
    if side == "bot":
        rr = int(pts[:, 0].max())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", rr, cols)
    if side == "left":
        cc = int(pts[:, 1].min())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", cc, rows)
    if side == "right":
        cc = int(pts[:, 1].max())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", cc, rows)
    return None


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _target_len_for_side(side: str) -> int:
    return HORIZ_TARGET if side in ("top", "bot") else VERT_TARGET


def _matches_attachment(mask: np.ndarray, need_edge: str, target_len: int) -> bool:
    if need_edge in ("left", "right"):
        if not bool(np.array_equal(mask, np.flipud(mask))):
            return False
    else:
        if not bool(np.array_equal(mask, np.fliplr(mask))):
            return False

    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return False

    if need_edge in ("left", "right"):
        _typ, _edge_c, rows = seg
        return len(rows) == int(target_len) and _is_contig(rows)

    _typ, _edge_r, cols = seg
    return len(cols) == int(target_len) and _is_contig(cols)


def _count_valid_rotations(mask: np.ndarray, need_edge: str, target_len: int) -> int:
    count = 0
    arr = mask.astype(int)
    for krot in range(4):
        rot_arr = np.rot90(arr, int(krot))
        rot_mask = np.rot90(mask, int(krot))
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        if _matches_attachment(rot_mask, need_edge, int(target_len)):
            count += 1
    return count


def _make_sprite_mask(side: str, rng: random.Random) -> np.ndarray:
    if side in ("top", "bot"):
        base = rng.choice(_TOP_CANONICAL).copy()
        mask = base if side == "top" else np.flipud(base)
        need_edge = "bot" if side == "top" else "top"
    else:
        base = rng.choice(_LEFT_CANONICAL).copy()
        mask = base if side == "left" else np.fliplr(base)
        need_edge = "right" if side == "left" else "left"

    assert _count_valid_rotations(mask, need_edge, _target_len_for_side(side)) == 1
    labeled, num = _label(mask, structure=np.ones((3, 3), dtype=int))
    assert int(num) == 1
    return mask


def _build_hub(out: np.ndarray, hub_r: int, hub_c: int, side_colors: dict[str, int], bg: int) -> None:
    out[hub_r : hub_r + HUB_H, hub_c : hub_c + HUB_W] = int(bg)
    out[hub_r, hub_c + 1 : hub_c + HUB_W - 1] = int(side_colors["top"])
    out[hub_r + HUB_H - 1, hub_c + 1 : hub_c + HUB_W - 1] = int(side_colors["bot"])
    out[hub_r + 1 : hub_r + HUB_H - 1, hub_c] = int(side_colors["left"])
    out[hub_r + 1 : hub_r + HUB_H - 1, hub_c + HUB_W - 1] = int(side_colors["right"])


def _attachment_offset(mask: np.ndarray, side: str, hub_r: int, hub_c: int) -> tuple[int, int]:
    if side == "top":
        _typ, edge_r, cols = _edge_segment(mask, "bot")
        return int(hub_r - edge_r - 1), int(hub_c + 1 - cols[0])
    if side == "bot":
        _typ, edge_r, cols = _edge_segment(mask, "top")
        return int(hub_r + HUB_H - edge_r), int(hub_c + 1 - cols[0])
    if side == "left":
        _typ, edge_c, rows = _edge_segment(mask, "right")
        return int(hub_r + 1 - rows[0]), int(hub_c - edge_c - 1)
    _typ, edge_c, rows = _edge_segment(mask, "left")
    return int(hub_r + 1 - rows[0]), int(hub_c + HUB_W - edge_c)


def _crop_patch(arr: np.ndarray, bg: int) -> np.ndarray:
    mask = arr != int(bg)
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _candidate_positions(
    canvas: np.ndarray,
    patch: np.ndarray,
    region: str,
    hub_r: int,
    hub_c: int,
    bg: int,
    padding: int = 2,
) -> list[tuple[int, int]]:
    H, W = canvas.shape
    ph, pw = patch.shape
    occ = canvas != int(bg)
    if int(padding) > 0:
        occ = _ndimage.binary_dilation(occ, structure=np.ones((3, 3), dtype=bool), iterations=int(padding))
    mask = patch != int(bg)

    inner_r0, inner_r1 = int(hub_r + 1), int(hub_r + HUB_H - 2)
    inner_c0, inner_c1 = int(hub_c + 1), int(hub_c + HUB_W - 2)
    candidates: list[tuple[int, int]] = []

    for pr in range(0, H - ph + 1):
        for pc in range(0, W - pw + 1):
            if bool((occ[pr : pr + ph, pc : pc + pw] & mask).any()):
                continue

            rr0, rr1 = int(pr), int(pr + ph - 1)
            cc0, cc1 = int(pc), int(pc + pw - 1)
            if region == "top":
                if rr1 >= int(hub_r):
                    continue
                if not _intervals_overlap(cc0, cc1, inner_c0, inner_c1):
                    continue
            elif region == "bot":
                if rr0 <= int(hub_r + HUB_H - 1):
                    continue
                if not _intervals_overlap(cc0, cc1, inner_c0, inner_c1):
                    continue
            elif region == "left":
                if cc1 >= int(hub_c):
                    continue
                if not _intervals_overlap(rr0, rr1, inner_r0, inner_r1):
                    continue
            elif region == "right":
                if cc0 <= int(hub_c + HUB_W - 1):
                    continue
                if not _intervals_overlap(rr0, rr1, inner_r0, inner_r1):
                    continue
            elif region == "tl":
                if rr1 >= int(hub_r) or cc1 >= int(hub_c):
                    continue
            elif region == "tr":
                if rr1 >= int(hub_r) or cc0 <= int(hub_c + HUB_W - 1):
                    continue
            elif region == "bl":
                if rr0 <= int(hub_r + HUB_H - 1) or cc1 >= int(hub_c):
                    continue
            elif region == "br":
                if rr0 <= int(hub_r + HUB_H - 1) or cc0 <= int(hub_c + HUB_W - 1):
                    continue
            else:
                raise ValueError(f"unknown region: {region}")

            candidates.append((int(pr), int(pc)))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach detached sprites to a central hollow rectangle by hub-relative position.

    Fused rule:
    - The hub is a hollow 4x6 rectangle.
    - Matching is by location relative to the hub: above/below/left/right.
    - Top/bottom attachments need contact length 4; left/right need contact length 2.
    - Attached sprites are recolored to the destination edge color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _find_hub():
        best = None  # (score, (r,c), (top,bot,left,right))
        for r in range(0, H - HUB_H + 1):
            for c in range(0, W - HUB_W + 1):
                if (
                    int(g[r, c]) != int(bg)
                    or int(g[r, c + HUB_W - 1]) != int(bg)
                    or int(g[r + HUB_H - 1, c]) != int(bg)
                    or int(g[r + HUB_H - 1, c + HUB_W - 1]) != int(bg)
                ):
                    continue

                inner = g[r + 1 : r + HUB_H - 1, c + 1 : c + HUB_W - 1]
                if not bool((inner == int(bg)).all()):
                    continue

                top = g[r, c + 1 : c + HUB_W - 1]
                bot = g[r + HUB_H - 1, c + 1 : c + HUB_W - 1]
                left = g[r + 1 : r + HUB_H - 1, c]
                right = g[r + 1 : r + HUB_H - 1, c + HUB_W - 1]
                if (
                    bool((top == int(bg)).any())
                    or bool((bot == int(bg)).any())
                    or bool((left == int(bg)).any())
                    or bool((right == int(bg)).any())
                ):
                    continue
                if (
                    len(set(map(int, top))) != 1
                    or len(set(map(int, bot))) != 1
                    or len(set(map(int, left))) != 1
                    or len(set(map(int, right))) != 1
                ):
                    continue

                side_cols = (int(top[0]), int(bot[0]), int(left[0]), int(right[0]))
                if len(set(side_cols)) != 4:
                    continue

                cr = float(r + (HUB_H - 1) / 2)
                cc = float(c + (HUB_W - 1) / 2)
                score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
                cand = (float(score), (int(r), int(c)), side_cols)
                if best is None or float(cand[0]) < float(best[0]):
                    best = cand
        return best

    hub = _find_hub()
    assert hub is not None

    _score, (r, c), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[r : r + HUB_H, c : c + HUB_W] = True

    out = np.full_like(g, int(bg))
    out[r : r + HUB_H, c : c + HUB_W] = g[r : r + HUB_H, c : c + HUB_W]

    side_specs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + HUB_W - 1))), HORIZ_TARGET)),
        "bot": (int(bot_col), ("top", int(r + HUB_H), list(range(int(c + 1), int(c + HUB_W - 1))), HORIZ_TARGET)),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + HUB_H - 1))), VERT_TARGET)),
        "right": (int(right_col), ("left", int(c + HUB_W), list(range(int(r + 1), int(r + HUB_H - 1))), VERT_TARGET)),
    }

    inner_r0, inner_r1 = int(r + 1), int(r + HUB_H - 2)
    inner_c0, inner_c1 = int(c + 1), int(c + HUB_W - 2)

    outside = (g != int(bg)) & (~hub_mask)
    lab, nc = label_mask(outside, connectivity=8)
    by_side: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {side: [] for side in SIDES}

    for cid in range(1, int(nc) + 1):
        comp_mask = lab == int(cid)
        pts = np.argwhere(comp_mask)
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)

        candidates: list[str] = []
        if int(r1) < int(r) and _intervals_overlap(int(c0), int(c1), inner_c0, inner_c1):
            candidates.append("top")
        if int(r0) > int(r + HUB_H - 1) and _intervals_overlap(int(c0), int(c1), inner_c0, inner_c1):
            candidates.append("bot")
        if int(c1) < int(c) and _intervals_overlap(int(r0), int(r1), inner_r0, inner_r1):
            candidates.append("left")
        if int(c0) > int(c + HUB_W - 1) and _intervals_overlap(int(r0), int(r1), inner_r0, inner_r1):
            candidates.append("right")
        if len(candidates) != 1:
            continue

        side = candidates[0]
        comp_arr = g.copy()
        comp_arr[~comp_mask] = int(bg)
        comp_arr, comp_mask = _crop_to_mask(comp_arr, comp_mask)
        by_side[side].append((int(comp_mask.sum()), comp_arr, comp_mask))

    for side, (paint_col, (need_edge, fixed, span, target)) in side_specs.items():
        if not by_side[side]:
            continue
        _area, spr_arr, spr_mask = max(by_side[side], key=lambda item: int(item[0]))

        found = None  # (mask, dr, dc)
        for krot in range(4):
            a = np.rot90(spr_arr, int(krot))
            msk = np.rot90(spr_mask, int(krot))
            a, msk = _crop_to_mask(a, msk)

            if not _matches_attachment(msk, need_edge, int(target)):
                continue

            seg = _edge_segment(msk, need_edge)
            if seg is None:
                continue

            if need_edge in ("left", "right"):
                _typ, edge_c, rows = seg
                dr = int(span[0] - rows[0])
                dc = int(fixed - edge_c)
            else:
                _typ, edge_r, cols = seg
                dr = int(fixed - edge_r)
                dc = int(span[0] - cols[0])

            placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
            if any(not (0 <= rr < H and 0 <= cc2 < W) for rr, cc2 in placed):
                continue

            ok = True
            for rr, cc2 in placed:
                if int(out[rr, cc2]) != int(bg) and int(out[rr, cc2]) != int(paint_col):
                    ok = False
                    break
            if not ok:
                continue

            found = (msk, int(dr), int(dc))
            break

        if found is None:
            continue

        msk, dr, dc = found
        paint = np.where(msk, int(paint_col), int(bg)).astype(int)
        paste(out, paint, int(dr), int(dc), mask=msk, overwrite=True, clip=True)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    remaining = [c for c in range(10) if c != bg and c not in side_colors.values()]
    if not remaining:
        return None
    src_col = int(rng.choice(remaining))

    while True:
        k = rng.randint(2, 4)
        sprite_sides = rng.sample(list(SIDES), k)
        if any(side in ("top", "bot") for side in sprite_sides) and any(side in ("left", "right") for side in sprite_sides):
            break

    sprite_masks = {side: _make_sprite_mask(side, rng) for side in sprite_sides}

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    margin = max(max_sprite_h, max_sprite_w) + 2

    hub_r_min = margin
    hub_r_max = H - HUB_H - margin
    hub_c_min = margin
    hub_c_max = W - HUB_W - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), int(bg), dtype=int)
    _build_hub(out, hub_r, hub_c, side_colors, bg)

    for side in sprite_sides:
        mask = sprite_masks[side]
        arr = np.where(mask, int(side_colors[side]), int(bg)).astype(int)
        sr, sc = _attachment_offset(mask, side, hub_r, hub_c)
        sh, sw = mask.shape

        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        if bool((out[sr : sr + sh, sc : sc + sw] != int(bg))[mask].any()):
            return None
        paste(out, arr, int(sr), int(sc), transparent=int(bg))

    inp = np.full((H, W), int(bg), dtype=int)
    _build_hub(inp, hub_r, hub_c, side_colors, bg)
    canvas = inp.copy()

    for side in sprite_sides:
        mask = sprite_masks[side]
        patch = np.where(mask, int(src_col), int(bg)).astype(int)
        patch = _crop_patch(np.rot90(patch, rng.choice([1, 2, 3])), bg)

        candidates = _candidate_positions(canvas, patch, side, hub_r, hub_c, bg, padding=2)
        if not candidates:
            return None
        pr, pc = rng.choice(candidates)
        paste(canvas, patch, int(pr), int(pc), transparent=int(bg))
        paste(inp, patch, int(pr), int(pc), transparent=int(bg))

    if rng.random() < 0.3:
        distractor_colors = [c for c in remaining if int(c) != int(src_col)]
        if distractor_colors:
            dcol = int(rng.choice(distractor_colors))
            dside = rng.choice(list(SIDES))
            dpatch = np.where(_make_sprite_mask(dside, rng), dcol, bg).astype(int)
            dpatch = _crop_patch(np.rot90(dpatch, rng.choice([0, 1, 2, 3])), bg)
            region = rng.choice(["tl", "tr", "bl", "br"])
            candidates = _candidate_positions(canvas, dpatch, region, hub_r, hub_c, bg, padding=2)
            if candidates:
                pr, pc = rng.choice(candidates)
                paste(canvas, dpatch, int(pr), int(pc), transparent=int(bg))
                paste(inp, dpatch, int(pr), int(pc), transparent=int(bg))

    result = main(inp)
    if not np.array_equal(result, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
