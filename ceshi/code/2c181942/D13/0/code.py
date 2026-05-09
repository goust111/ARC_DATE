# Variant of 2c181942
# Dimension: D13 — 轨迹/源保留策略 (trajectory)
# Change: 默认消失——精灵从输入中的散布位置移除，输出中仅在贴附位置出现
#         → 保留精灵原始位置；输出同时显示散布源精灵和贴附后的副本

from __future__ import annotations

"""
Design analysis for the D13 keep_source variant
==============================================

variant:
  core_change:
    original: matched sprite is effectively moved from its scattered source position to the hub edge
    new: matched sprite is copied; source stays visible and a second copy is attached to the hub
  invariant_parts:
    - hub detection is unchanged
    - color-to-side matching is unchanged
    - rotation disambiguation by symmetry/contact-edge length is unchanged
    - distractors are still ignored and do not survive into the output

a_concretization_and_trace:
  scene:
    - input still contains one hollow 4x4 hub near center and 2-4 scattered monochrome sprites
    - each matched sprite is outside the hub, rotated away from its final attached orientation
    - output now contains: hub + attached copies + the original scattered matched sprites
  local_trace:
    - original flow:
      1. detect hub
      2. find red sprite matching red hub edge
      3. rotate/translate red sprite to hub
      4. build output from blank background, so the scattered red source disappears
    - variant flow:
      1. detect hub
      2. find the same red source sprite
      3. preserve that source component in the output at its original coordinates
      4. also rotate/translate a second copy to the red hub edge
    - visible difference:
      the same colored shape appears twice in the output, once scattered and once attached

b_complexity_calibration:
  - only D13 changes; there is no new matching attribute or new object type
  - the original asymmetry constraints on sprites are kept so rotation remains inferable
  - keeping source copies is visually obvious because every matched color gains one extra instance
  - distractors remain removable so the rule is not diluted into "copy every object"

c_implicit_assumption_audit_for_main:
  - original `out = bg + hub only` depended on the old semantics that movers vanish from source.
    This is the main line that must change.
  - color extraction `(g == col) & (~hub_mask)` still works because input format is unchanged:
    only scattered source sprites are outside the hub, not already-attached copies.
  - choosing the largest same-color 8-connected component still works because generator keeps
    matched colors unique and any distractor uses a fresh non-hub color.
  - conflict testing against `out` now sees preserved source copies too, so generator must keep
    scattered sprites away from hub/attached zones. The original padding already guarantees this.
  - we must not start from `out = g.copy()` because that would preserve distractors, which is
    unrelated to D13 and would introduce another dimension change.

d_placement_legality:
  - attached copies use the original exact geometry checks, so they remain in-bounds and flush
    to the hub edge with no conflicting overlap
  - scattered source sprites are placed by `random_free_location(..., padding=2)` on a canvas
    containing hub and previously placed sprites, so they stay disjoint from each other and from
    future attached regions
  - expected output is built as `hub + attached copies + recorded matched source placements`
    while distractors are omitted

e_visual_observability:
  - in 2-4 examples, the repeated appearance of each matched sprite is directly visible
  - because colors already identify the destination edge, the added copy is easy to track by eye
  - the rule remains single-branch: matched sprites duplicate, distractors do not

f_visual_cleanliness:
  - no overlay is added on top of the source positions; the extra copies live at the hub
  - spacing remains the same as the original generator, so the doubled sprites do not clutter
    the central structure

g_color_inference:
  - colors keep their original role only: sprite color matches one hub side color
  - no new color mapping is introduced, so the variant stays human-inferable from few examples

post-change_main_assumptions_checked_against_generator:
  - every matched color has exactly one scattered component outside the hub -> guaranteed
  - preserved source components do not overlap attached placements -> guaranteed by placement padding
  - distractors never reuse hub-edge colors -> guaranteed by fresh distractor color sampling
"""

import random
import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location
from scipy.ndimage import label as _label


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach monochrome sprites to the matching sides of a central hollow square,
    but preserve the original scattered sprite as well.
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

    def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.argwhere(mask)
        r0, c0, r1, c1 = bbox_coords(pts)
        return (
            arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
            mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        )

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

    segs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }

    for _side, (col, (need_edge, fixed, span)) in segs.items():
        col = int(col)
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

        spr_mask_full = labc == int(best_id)
        pts = np.argwhere(spr_mask_full)
        src_r0, src_c0, _src_r1, _src_c1 = bbox_coords(pts)
        spr_arr = g.copy()
        spr_arr[~spr_mask_full] = int(bg)
        spr_arr, spr_mask = _crop_to_mask(spr_arr, spr_mask_full)

        found = None  # (a, msk, dr, dc)
        for krot in range(4):
            a = np.rot90(spr_arr, int(krot))
            msk = np.rot90(spr_mask, int(krot))
            a, msk = _crop_to_mask(a, msk)

            if need_edge in ("left", "right"):
                if not bool(np.array_equal(msk, np.flipud(msk))):
                    continue
            else:
                if not bool(np.array_equal(msk, np.fliplr(msk))):
                    continue

            seg = _edge_segment(msk, need_edge)
            if seg is None:
                continue

            if need_edge in ("left", "right"):
                _typ, edge_c, rows = seg
                if len(rows) != int(target) or not _is_contig(rows):
                    continue
                dr = int(span[0] - rows[0])
                dc = int(fixed - edge_c)
            else:
                _typ, edge_r, cols = seg
                if len(cols) != int(target) or not _is_contig(cols):
                    continue
                dr = int(fixed - edge_r)
                dc = int(span[0] - cols[0])

            placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
            if any(not (0 <= rr2 < H and 0 <= cc2 < W) for rr2, cc2 in placed):
                continue

            ok = True
            for rr2, cc2 in placed:
                val = int(a[int(rr2 - dr), int(cc2 - dc)])
                if val == int(bg):
                    continue
                if int(out[int(rr2), int(cc2)]) != int(bg) and int(out[int(rr2), int(cc2)]) != val:
                    ok = False
                    break
            if not ok:
                continue

            found = (a, msk, int(dr), int(dc))
            break

        if found is None:
            continue

        # D13 change: preserve the identified source sprite, but still only for the matched
        # component we are actually attaching. Distractors remain absent from the output.
        paste(out, spr_arr, int(src_r0), int(src_c0), mask=spr_mask, overwrite=True, clip=True)

        a, msk, dr, dc = found
        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)

    return out


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Constructively build a sprite mask in the attached orientation for `side`.
    The mask is bool, 8-connected, has the correct symmetry and contact-edge length,
    and exactly 1 valid rotation for the intended attachment logic.
    """
    e = edge_len
    h = rng.choice([3, 4])
    w = rng.randint(2, 4)

    half_h = (h + 1) // 2

    for _attempt in range(100):
        grid = np.zeros((h, w), dtype=bool)

        for ri in range(half_h):
            mirror_ri = h - 1 - ri
            for ci in range(w):
                val = rng.random() < 0.5
                grid[ri, ci] = val
                grid[mirror_ri, ci] = val

        right_col = grid[:, -1]
        runs = []
        count = 0
        for v in right_col:
            if v:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        if len(runs) != 1 or runs[0] != e:
            continue

        rows_any = grid.any(axis=1)
        cols_any = grid.any(axis=0)
        if not rows_any.any():
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        grid = grid[r0 : r1 + 1, c0 : c1 + 1]

        if grid.shape[0] < 2 or grid.shape[1] < 2:
            continue
        if not np.array_equal(grid, np.flipud(grid)):
            continue

        labeled, num = _label(grid, structure=np.ones((3, 3), dtype=int))
        if num != 1:
            continue
        if np.array_equal(grid, np.fliplr(grid)):
            continue

        px = int(grid.sum())
        if px < 4 or px > 12:
            continue

        valid_for_left = 0
        for k in range(4):
            rot = np.rot90(grid, k)
            ra = rot.any(axis=1)
            ca = rot.any(axis=0)
            rr0, rr1 = np.where(ra)[0][[0, -1]]
            cc0, cc1 = np.where(ca)[0][[0, -1]]
            cr = rot[rr0 : rr1 + 1, cc0 : cc1 + 1]
            if not np.array_equal(cr, np.flipud(cr)):
                continue
            rc = cr[:, -1]
            runs2 = []
            cnt2 = 0
            for v in rc:
                if v:
                    cnt2 += 1
                else:
                    if cnt2 > 0:
                        runs2.append(cnt2)
                    cnt2 = 0
            if cnt2 > 0:
                runs2.append(cnt2)
            if len(runs2) == 1 and runs2[0] == e:
                valid_for_left += 1

        if valid_for_left != 1:
            continue

        if side == "left":
            mask = grid.copy()
        elif side == "right":
            mask = np.fliplr(grid)
        elif side == "top":
            mask = np.rot90(grid, -1)
        else:
            mask = np.rot90(grid, 1)
        return mask

    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    N = 4
    edge_len = N - 2

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]

    sides = ["top", "bot", "left", "right"]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)

    sprite_masks = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(m.shape[0] for m in sprite_masks.values())
    max_sprite_w = max(m.shape[1] for m in sprite_masks.values())
    max_extent = max(max_sprite_h, max_sprite_w)
    margin = max_extent + 2

    hub_r_min = margin
    hub_r_max = H - N - margin
    hub_c_min = margin
    hub_c_max = W - N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)

    for ci in range(1, N - 1):
        out[hub_r, hub_c + ci] = top_col
        out[hub_r + N - 1, hub_c + ci] = bot_col
    for ri in range(1, N - 1):
        out[hub_r + ri, hub_c] = left_col
        out[hub_r + ri, hub_c + N - 1] = right_col

    attached_sprites = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        sh, sw = mask.shape
        color = side_colors[side]
        arr = np.where(mask, color, bg).astype(int)

        if side == "top":
            filled = np.where(mask[-1])[0]
            sc = hub_c + 1 - int(filled[0])
            sr = hub_r - sh
        elif side == "bot":
            filled = np.where(mask[0])[0]
            sc = hub_c + 1 - int(filled[0])
            sr = hub_r + N
        elif side == "left":
            filled = np.where(mask[:, -1])[0]
            sr = hub_r + 1 - int(filled[0])
            sc = hub_c - sw
        else:
            filled = np.where(mask[:, 0])[0]
            sr = hub_r + 1 - int(filled[0])
            sc = hub_c + N

        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None

        for rr in range(sh):
            for cc in range(sw):
                if mask[rr, cc] and out[sr + rr, sc + cc] != bg:
                    return None

        paste(out, arr, sr, sc, transparent=bg)
        attached_sprites[side] = (arr, mask, sr, sc)

    inp = np.full((H, W), bg, dtype=int)
    inp[hub_r : hub_r + N, hub_c : hub_c + N] = out[hub_r : hub_r + N, hub_c : hub_c + N]

    canvas = inp.copy()
    expected = out.copy()

    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, k)
        rotated_mask = np.rot90(mask, k)

        rows_any = rotated_mask.any(axis=1)
        cols_any = rotated_mask.any(axis=0)
        r0_idx, r1_idx = np.where(rows_any)[0][[0, -1]]
        c0_idx, c1_idx = np.where(cols_any)[0][[0, -1]]
        cropped_arr = rotated_arr[r0_idx : r1_idx + 1, c0_idx : c1_idx + 1]
        cropped_mask = rotated_mask[r0_idx : r1_idx + 1, c0_idx : c1_idx + 1]
        patch = np.where(cropped_mask, cropped_arr, bg).astype(int)

        try:
            pr, pc = random_free_location(canvas, patch, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)
        paste(expected, patch, pr, pc, transparent=bg)

    if rng.random() < 0.3:
        used_colors = set(side_colors.values())
        distractor_colors = [c for c in range(10) if c != bg and c not in used_colors]
        if distractor_colors:
            dcol = rng.choice(distractor_colors)
            dist_mask = _make_sprite_mask(edge_len, rng.choice(sides), rng)
            if dist_mask is not None:
                dist_arr = np.where(dist_mask, dcol, bg).astype(int)
                try:
                    pr, pc = random_free_location(canvas, dist_arr, transparent=bg, padding=2, padding_connectivity=8)
                    paste(canvas, dist_arr, pr, pc, transparent=bg)
                    paste(inp, dist_arr, pr, pc, transparent=bg)
                except ValueError:
                    pass

    result = main(inp)
    if not np.array_equal(result, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
