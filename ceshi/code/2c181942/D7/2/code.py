# Variant of 2c181942
# Dimension: D7 — 精灵对称性约束 (match_cond)
# Change: 贴附方向的精灵必须沿接触轴镜像对称（上下贴附需fliplr对称，左右贴附需flipud对称），用于消歧旋转... → 非对称精灵+唯一旋转保证
#
# Design analysis for this variant
# --------------------------------
# spec_variant:
#   unchanged_core:
#     - grid still uses bg=8 and size 18-26 x 20-26
#     - a central hollow 4x4 hub still has four differently-colored sides
#     - 2-4 monochrome sprites are still paired to hub sides by color
#     - output is still the hub plus the correctly attached sprites only
#   d7_change:
#     old_rule:
#       - in the attached orientation, top/bottom sprites had to be fliplr-symmetric
#       - left/right sprites had to be flipud-symmetric
#       - solver used that symmetry as the main rotation disambiguator
#     new_rule:
#       - sprites are fully asymmetric as silhouettes: no mirror symmetry and no
#         rotational self-match among the 7 non-identity D4 transforms
#       - instead of checking symmetry, the solver must test all 4 rotations and
#         keep the unique one whose contact edge is on the required side and has
#         exactly N-2 contiguous cells
#       - generator guarantees that, for the target side fixed by color, exactly
#         one rotation satisfies that contact-edge constraint
#   behavior_trace:
#     - original flow for a left-side sprite:
#       1. isolate the color-matching component
#       2. rotate it
#       3. accept the first rotation that is flipud-symmetric and whose right edge
#          has length 2
#     - new flow for a left-side sprite:
#       1. isolate the color-matching component
#       2. try all 4 rotations
#       3. for each rotation, inspect the would-be contact side only
#       4. keep the rotation iff that side exposes exactly 2 contiguous edge cells
#       5. require the set of such rotations to have size exactly 1
#     - concrete difference:
#       the old solver relied on a global symmetry property of the whole sprite;
#       the new solver relies on a local edge-geometry search over rotations
#   complexity_calibration:
#     - the hub remains simple, so the visible novelty must come from the sprites
#     - sprites are kept monochrome, but their silhouettes are intentionally lopsided
#       so the change is visible in 2-3 examples
#     - the key free variable is limited to 4 rotations; there is no extra color
#       mapping or multi-branch rule beyond the original side-color pairing
#   hidden_assumptions_review:
#     - original code assumed symmetry was always present; that assumption is removed
#     - original code broke at the first valid rotation; in this variant that would be
#       unsafe if multiple rotations happened to fit, so main() now counts valid
#       rotations and only accepts a unique one
#     - output placement still assumes the matching sprite is the largest component of
#       that color outside the hub; generator preserves that by not creating same-color
#       distractors
#   placement_legality:
#     - generator still builds the final output first, then scatters rotated sprites
#     - attached placement is checked for bounds and overlap before accepting a sample
#     - scattered placement uses random_free_location(..., padding=2, connectivity=8),
#       preserving visual separation from the hub and from other sprites
#   visual_observability:
#     - because sprites are asymmetric and are placed in a random non-identity input
#       rotation, the solver must visibly "search" for the unique attachment
#     - the visual cue is not a hidden numeric color order; it is the side-color match
#       plus the fact that only one orientation presents the required 2-cell contact edge
#   neatness:
#     - no extra overlays or markers are added
#     - objects stay separated by padding, so each sprite silhouette is easy to read

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


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


def _edge_segment(mask: np.ndarray, side: str):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        rr = int(pts[:, 0].min())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", int(rr), cols)
    if side == "bot":
        rr = int(pts[:, 0].max())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", int(rr), cols)
    if side == "left":
        cc = int(pts[:, 1].min())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", int(cc), rows)
    if side == "right":
        cc = int(pts[:, 1].max())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", int(cc), rows)
    return None


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


def _rotation_attachment(
    arr: np.ndarray,
    mask: np.ndarray,
    *,
    need_edge: str,
    fixed: int,
    span: list[int],
    target: int,
):
    arr, mask = _crop_to_mask(arr, mask)
    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return None

    if need_edge in ("left", "right"):
        _typ, edge_c, rows = seg
        if len(rows) != int(target) or not _is_contig(rows):
            return None
        dr = int(span[0] - rows[0])
        dc = int(fixed - edge_c)
    else:
        _typ, edge_r, cols = seg
        if len(cols) != int(target) or not _is_contig(cols):
            return None
        dr = int(fixed - edge_r)
        dc = int(span[0] - cols[0])

    return arr, mask, int(dr), int(dc)


def _count_valid_rotations(mask: np.ndarray, need_edge: str, target: int) -> int:
    count = 0
    dummy = np.where(mask, 1, 0).astype(int)
    for krot in range(4):
        cand = _rotation_attachment(
            np.rot90(dummy, int(krot)),
            np.rot90(mask, int(krot)),
            need_edge=need_edge,
            fixed=0,
            span=list(range(int(target))),
            target=int(target),
        )
        if cand is not None:
            count += 1
    return int(count)


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


def _contact_side_for_hub_side(side: str) -> str:
    return {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]


def _rotate_canonical_to_side(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return np.rot90(mask, -1)
    return np.rot90(mask, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach monochrome sprites to the matching sides of a central hollow square.

    D7 variant:
    - sprites are intentionally asymmetric
    - rotation is disambiguated by trying all 4 rotations and requiring exactly
      one rotation whose contact edge matches the target side geometry
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _find_hub():
        best = None  # (score, (r,c,n), (top,bot,left,right))
        for n in range(3, 7):
            for r in range(0, int(H - n + 1)):
                for c in range(0, int(W - n + 1)):
                    if (
                        int(g[int(r), int(c)]) != int(bg)
                        or int(g[int(r), int(c + n - 1)]) != int(bg)
                        or int(g[int(r + n - 1), int(c)]) != int(bg)
                        or int(g[int(r + n - 1), int(c + n - 1)]) != int(bg)
                    ):
                        continue
                    inner = g[int(r + 1) : int(r + n - 1), int(c + 1) : int(c + n - 1)]
                    if not bool((inner == int(bg)).all()):
                        continue

                    top = g[int(r), int(c + 1) : int(c + n - 1)]
                    bot = g[int(r + n - 1), int(c + 1) : int(c + n - 1)]
                    left = g[int(r + 1) : int(r + n - 1), int(c)]
                    right = g[int(r + 1) : int(r + n - 1), int(c + n - 1)]

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

                    cr = float(r + (n - 1) / 2)
                    cc = float(c + (n - 1) / 2)
                    score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
                    cand = (
                        float(score),
                        (int(r), int(c), int(n)),
                        (int(top[0]), int(bot[0]), int(left[0]), int(right[0])),
                    )
                    if best is None or float(cand[0]) < float(best[0]):
                        best = cand
        return best

    hub = _find_hub()
    assert hub is not None

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    out = np.full_like(g, int(bg))
    out[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]

    target = int(n - 2)
    segs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }

    for _side, (col, (need_edge, fixed, span)) in segs.items():
        m = (g == int(col)) & (~hub_mask)
        if not bool(m.any()):
            continue

        labc, nc = label_mask(m, connectivity=8)
        best_id = None
        best_sz = -1
        for cid in range(1, int(nc) + 1):
            sz = int((labc == int(cid)).sum())
            if sz > best_sz:
                best_sz = sz
                best_id = int(cid)
        if best_id is None or best_sz <= 0:
            continue

        spr_mask = labc == int(best_id)
        spr_arr = g.copy()
        spr_arr[~spr_mask] = int(bg)
        spr_arr, spr_mask = _crop_to_mask(spr_arr, spr_mask)

        candidates = []
        for krot in range(4):
            cand = _rotation_attachment(
                np.rot90(spr_arr, int(krot)),
                np.rot90(spr_mask, int(krot)),
                need_edge=need_edge,
                fixed=int(fixed),
                span=span,
                target=int(target),
            )
            if cand is not None:
                candidates.append(cand)

        if len(candidates) != 1:
            continue

        a, msk, dr, dc = candidates[0]
        placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
        if any(not (0 <= rr < H and 0 <= cc2 < W) for rr, cc2 in placed):
            continue

        ok = True
        for rr, cc2 in placed:
            val = int(a[int(rr - dr), int(cc2 - dc)])
            if val == int(bg):
                continue
            if int(out[int(rr), int(cc2)]) != int(bg) and int(out[int(rr), int(cc2)]) != val:
                ok = False
                break
        if not ok:
            continue

        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)

    return out


def _make_canonical_mask(edge_len: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(600):
        h = rng.choice([4, 5])
        w = rng.choice([3, 4, 5])
        run_start = rng.randint(0, int(h - edge_len))
        target_pixels = rng.randint(5, min(12, h * w - max(0, h - edge_len - 1)))

        mask = np.zeros((h, w), dtype=bool)
        mask[int(run_start) : int(run_start + edge_len), int(w - 1)] = True
        anchor_row = rng.choice(list(range(int(run_start), int(run_start + edge_len))))
        mask[int(anchor_row), int(w - 2)] = True

        while int(mask.sum()) < int(target_pixels):
            candidates = set()
            for rr, cc in np.argwhere(mask):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = int(rr + dr)
                        nc = int(cc + dc)
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        if bool(mask[int(nr), int(nc)]):
                            continue
                        if nc == int(w - 1) and not (int(run_start) <= nr < int(run_start + edge_len)):
                            continue
                        candidates.add((int(nr), int(nc)))
            if not candidates:
                break

            cand_list = list(candidates)
            weights = []
            for nr, nc in cand_list:
                weight = 1 + (int(w - 1) - int(nc))
                if int(nc) <= int(w - 3):
                    weight += 2
                if not (int(run_start) <= int(nr) < int(run_start + edge_len)):
                    weight += 1
                weights.append(float(weight))
            nr, nc = rng.choices(cand_list, weights=weights, k=1)[0]
            mask[int(nr), int(nc)] = True

        mask = _crop_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < 5 or int(mask.sum()) > 12:
            continue

        lab, nc = label_mask(mask, connectivity=8)
        if int(nc) != 1:
            continue
        if not _is_fully_asymmetric(mask):
            continue
        if _count_valid_rotations(mask, "right", edge_len) != 1:
            continue

        return mask

    return None


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    need_edge = _contact_side_for_hub_side(side)
    for _ in range(200):
        canonical = _make_canonical_mask(edge_len, rng)
        if canonical is None:
            continue
        mask = _rotate_canonical_to_side(canonical, side)
        if _count_valid_rotations(mask, need_edge, edge_len) != 1:
            continue
        if not _is_fully_asymmetric(mask):
            continue
        return mask
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    n = 4
    edge_len = n - 2

    h = rng.randint(18, 26)
    w = rng.randint(20, 26)

    all_colors = [col for col in range(10) if col != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]

    sides = ["top", "bot", "left", "right"]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    k = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, k)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    margin = max(max_sprite_h, max_sprite_w) + 2

    hub_r_min = margin
    hub_r_max = h - n - margin
    hub_c_min = margin
    hub_c_max = w - n - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((h, w), bg, dtype=int)
    for ci in range(1, n - 1):
        out[hub_r, hub_c + ci] = int(top_col)
        out[hub_r + n - 1, hub_c + ci] = int(bot_col)
    for ri in range(1, n - 1):
        out[hub_r + ri, hub_c] = int(left_col)
        out[hub_r + ri, hub_c + n - 1] = int(right_col)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        sh, sw = mask.shape
        color = int(side_colors[side])
        arr = np.where(mask, color, bg).astype(int)

        if side == "top":
            filled = np.where(mask[-1])[0]
            sc = hub_c + 1 - int(filled[0])
            sr = hub_r - sh
        elif side == "bot":
            filled = np.where(mask[0])[0]
            sc = hub_c + 1 - int(filled[0])
            sr = hub_r + n
        elif side == "left":
            filled = np.where(mask[:, -1])[0]
            sr = hub_r + 1 - int(filled[0])
            sc = hub_c - sw
        else:
            filled = np.where(mask[:, 0])[0]
            sr = hub_r + 1 - int(filled[0])
            sc = hub_c + n

        if sr < 0 or sc < 0 or sr + sh > h or sc + sw > w:
            return None

        for rr in range(sh):
            for cc in range(sw):
                if bool(mask[int(rr), int(cc)]) and int(out[int(sr + rr), int(sc + cc)]) != int(bg):
                    return None

        paste(out, arr, sr, sc, transparent=bg)
        attached_sprites[side] = (arr, mask, int(sr), int(sc))

    inp = np.full((h, w), bg, dtype=int)
    inp[hub_r : hub_r + n, hub_c : hub_c + n] = out[hub_r : hub_r + n, hub_c : hub_c + n]
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        rot_k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, int(rot_k))
        rotated_mask = np.rot90(mask, int(rot_k))
        cropped_arr, cropped_mask = _crop_to_mask(rotated_arr, rotated_mask)
        patch = np.where(cropped_mask, cropped_arr, bg).astype(int)

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

    if rng.random() < 0.3:
        used_colors = set(side_colors.values())
        distractor_colors = [col for col in range(10) if col != bg and col not in used_colors]
        if distractor_colors:
            dcol = int(rng.choice(distractor_colors))
            dist_mask = _make_sprite_mask(edge_len, rng.choice(sides), rng)
            if dist_mask is not None:
                dist_arr = np.where(dist_mask, dcol, bg).astype(int)
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

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
