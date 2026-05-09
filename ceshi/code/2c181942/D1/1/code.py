# Variant of 2c181942
# Dimension: D1 — 匹配条件：颜色配对机制 (match_cond)
# Change: s.head.color == d.edge.color — 精灵颜色与hub边缘颜色相同即配对，颜色唯一决定精灵应贴附... → 行列对齐匹配

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage as _ndimage
from scipy.ndimage import label as _label

from arc_utils import bbox_coords, get_bg, label_mask, paste

# ---------------------------------------------------------------------------
# Design analysis for the D1 variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace
# - The hub stays the same: a central hollow 4x4 square whose four sides are solid
#   colors and whose corners/interior are background.
# - The sprite geometry also stays the same: 2-4 detached monochrome 8-connected
#   shapes, each having exactly one valid attachment rotation because of contact-edge
#   length + symmetry constraints.
# - The changed behavior is only how a detached sprite is paired to a hub side.
#   Original: find sprite by color equality with the side color.
#   Variant: find sprite by its spatial side relative to the hub. A sprite entirely
#   above the hub and overlapping the hub's interior columns belongs to the top edge;
#   entirely left and overlapping the hub's interior rows belongs to the left edge;
#   analogous for bottom/right. Color no longer selects the side.
# - Small trace:
#   original input step: top edge is blue, detached top sprite is also blue, so the
#   solver searches "all blue cells outside the hub", rotates that blue component,
#   then pastes it onto the top.
#   variant step: top edge may be blue but the detached top sprite may be orange like
#   every other sprite; the solver instead finds "the component above the hub whose
#   bbox overlaps the hub's inner columns", rotates that component, recolors it blue,
#   and pastes it onto the top.
#   difference: matching changes from color partitioning to spatial partitioning, and
#   attachment color is now taken from the destination side rather than the source.
#
# b) Complexity calibration
# - I keep the same hub detection, same rotation search, same symmetry/contact tests,
#   and same 2-4 sprite count so the task is not simplified into trivial move-by-side.
# - To make the D1 change clearly visible, all true sprites share one non-hub source
#   color in the generator. This prevents accidental fallback to the original color
#   rule while keeping the visual vocabulary small and easy to parse.
# - I still allow 2-4 participating sides and an optional distractor so examples do
#   not collapse to a single repeated layout.
#
# c) Original assumptions that no longer hold
# - `m = (g == col) & (~hub_mask)` assumed side color uniquely isolates one sprite.
#   This fails once input sprite colors are decoupled from hub colors.
# - `labc = label_mask(m, connectivity=8)` therefore also depended on the same color
#   prefilter. In the variant, components must be found from all non-bg cells outside
#   the hub, then classified by position.
# - `val = a[...]` and `paste(out, a, ...)` implicitly assumed the attached sprite keeps
#   its input color. In the variant the placed sprite must instead be recolored to the
#   hub side color, so conflict checks and paste use a recolored patch.
# - The rest of the rotation logic remains valid because it only depends on sprite mask
#   geometry, not on how the sprite was matched.
#
# d) Placement legality
# - True sprites are scattered only in side-specific corridors:
#   top/bottom sprites must be entirely above/below the hub and overlap the hub's inner
#   columns; left/right sprites must be entirely left/right and overlap the hub's inner
#   rows. This makes classification unambiguous.
# - Padding >= 2 from hub and other objects prevents accidental touching/merging.
# - Optional distractors are placed in corner regions (above+left, above+right, etc.),
#   so they do not satisfy any side corridor test and are safely ignored by the solver.
#
# e) Visual observability
# - Because all true sprites share one source color that does not appear on the hub,
#   a human cannot explain the output by color matching.
# - The repeated cross-example pattern is visible: the sprite above moves to the top,
#   the sprite left moves to the left, and each attached sprite adopts the side color.
#
# f) Visual cleanliness
# - Hub stays near center with margins.
# - Side corridors and corner distractor regions keep objects visually separated.
# - No overlapping annotations or extra markers are introduced.
#
# g) Color-change inferability
# - Only five relevant colors are visible in a typical sample: four hub edge colors plus
#   one shared source-sprite color. This is small enough that the recoloring rule is easy
#   to infer from examples: attached output color is determined by destination side.
#
# h) Variant-specific quantitative check used after implementation
# - Every true sprite is generated with a source color different from all hub edge colors.
#   Therefore the number of color-matched true sprites should be 0 in every sample.
# - Running the original solver on variant inputs should almost never reproduce the new
#   output, because the old solver relies on missing color matches.
#
# i) New solver assumptions and generator guarantees
# - Solver assumption: intended sprites lie in exactly one of the four side corridors.
#   Generator guarantee: side-specific placement helper enforces that geometry.
# - Solver assumption: ignored components either fall in no corridor or are smaller than
#   the intended component for that side. Generator guarantee: distractors are only in
#   corner regions, so they fall in no corridor.
# - Solver assumption: attachment geometry still has one valid rotation. Generator keeps
#   the original mask-construction logic and validates every sample with `main(inp)`.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach detached sprites to the central hollow square by hub-relative position.

    Variant rule:
    - A component above the hub and aligned with the hub's interior columns attaches
      to the top edge; below -> bottom; left -> left; right -> right.
    - The sprite is rotated exactly as in the original task so its contact edge has
      length N-2 and the required mirror symmetry.
    - When attached, the sprite is recolored to the destination edge color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _find_hub():
        best = None  # (score, (r,c,n), (top,bot,left,right))
        for n in range(3, 7):
            for r in range(0, H - n + 1):
                for c in range(0, W - n + 1):
                    if (
                        int(g[r, c]) != bg
                        or int(g[r, c + n - 1]) != bg
                        or int(g[r + n - 1, c]) != bg
                        or int(g[r + n - 1, c + n - 1]) != bg
                    ):
                        continue
                    inner = g[r + 1 : r + n - 1, c + 1 : c + n - 1]
                    if not bool((inner == bg).all()):
                        continue

                    top = g[r, c + 1 : c + n - 1]
                    bot = g[r + n - 1, c + 1 : c + n - 1]
                    left = g[r + 1 : r + n - 1, c]
                    right = g[r + 1 : r + n - 1, c + n - 1]
                    if (
                        bool((top == bg).any())
                        or bool((bot == bg).any())
                        or bool((left == bg).any())
                        or bool((right == bg).any())
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
                    score = abs(cr - (H - 1) / 2) + abs(cc - (W - 1) / 2)
                    cand = (
                        float(score),
                        (int(r), int(c), int(n)),
                        (int(top[0]), int(bot[0]), int(left[0]), int(right[0])),
                    )
                    if best is None or float(cand[0]) < float(best[0]):
                        best = cand
        return best

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

    hub = _find_hub()
    assert hub is not None

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[r : r + n, c : c + n] = True

    out = np.full_like(g, bg)
    out[r : r + n, c : c + n] = g[r : r + n, c : c + n]

    target = int(n - 2)
    side_specs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }

    inner_r0, inner_r1 = int(r + 1), int(r + n - 2)
    inner_c0, inner_c1 = int(c + 1), int(c + n - 2)

    outside = (g != bg) & (~hub_mask)
    lab, nc = label_mask(outside, connectivity=8)
    by_side: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {side: [] for side in side_specs}

    for cid in range(1, int(nc) + 1):
        comp_mask = lab == int(cid)
        pts = np.argwhere(comp_mask)
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)

        candidates: list[str] = []
        if int(r1) < int(r) and _intervals_overlap(int(c0), int(c1), inner_c0, inner_c1):
            candidates.append("top")
        if int(r0) > int(r + n - 1) and _intervals_overlap(int(c0), int(c1), inner_c0, inner_c1):
            candidates.append("bot")
        if int(c1) < int(c) and _intervals_overlap(int(r0), int(r1), inner_r0, inner_r1):
            candidates.append("left")
        if int(c0) > int(c + n - 1) and _intervals_overlap(int(r0), int(r1), inner_r0, inner_r1):
            candidates.append("right")
        if len(candidates) != 1:
            continue

        side = candidates[0]
        comp_arr = g.copy()
        comp_arr[~comp_mask] = bg
        comp_arr, comp_mask = _crop_to_mask(comp_arr, comp_mask)
        by_side[side].append((int(comp_mask.sum()), comp_arr, comp_mask))

    for side, (paint_col, (need_edge, fixed, span)) in side_specs.items():
        if not by_side[side]:
            continue
        _area, spr_arr, spr_mask = max(by_side[side], key=lambda item: int(item[0]))

        found = None  # (mask, dr, dc)
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
                if len(rows) != target or not _is_contig(rows):
                    continue
                dr = int(span[0] - rows[0])
                dc = int(fixed - edge_c)
            else:
                _typ, edge_r, cols = seg
                if len(cols) != target or not _is_contig(cols):
                    continue
                dr = int(fixed - edge_r)
                dc = int(span[0] - cols[0])

            placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
            if any(not (0 <= rr < H and 0 <= cc2 < W) for rr, cc2 in placed):
                continue

            ok = True
            for rr, cc2 in placed:
                if int(out[rr, cc2]) != bg and int(out[rr, cc2]) != int(paint_col):
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


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Build a sprite mask in the attached orientation for `side`.

    This is the original geometry recipe: 8-connected, correct symmetry, contact-edge
    length = edge_len, and exactly one valid rotation for the intended attachment side.
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
        if int(num) != 1:
            continue
        if np.array_equal(grid, np.fliplr(grid)):
            continue

        px = int(grid.sum())
        if px < 4 or px > 12:
            continue

        valid_for_left = 0
        for k in range(4):
            rot = np.rot90(grid, int(k))
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
        if int(valid_for_left) != 1:
            continue

        if side == "left":
            return grid.copy()
        if side == "right":
            return np.fliplr(grid)
        if side == "top":
            return np.rot90(grid, -1)
        return np.rot90(grid, 1)

    return None


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


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
    n: int,
    bg: int,
    padding: int = 2,
) -> list[tuple[int, int]]:
    H, W = canvas.shape
    ph, pw = patch.shape
    occ = canvas != int(bg)
    if int(padding) > 0:
        occ = _ndimage.binary_dilation(occ, structure=np.ones((3, 3), dtype=bool), iterations=int(padding))
    mask = patch != int(bg)

    inner_r0, inner_r1 = int(hub_r + 1), int(hub_r + n - 2)
    inner_c0, inner_c1 = int(hub_c + 1), int(hub_c + n - 2)
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
                if rr0 <= int(hub_r + n - 1):
                    continue
                if not _intervals_overlap(cc0, cc1, inner_c0, inner_c1):
                    continue
            elif region == "left":
                if cc1 >= int(hub_c):
                    continue
                if not _intervals_overlap(rr0, rr1, inner_r0, inner_r1):
                    continue
            elif region == "right":
                if cc0 <= int(hub_c + n - 1):
                    continue
                if not _intervals_overlap(rr0, rr1, inner_r0, inner_r1):
                    continue
            elif region == "tl":
                if rr1 >= int(hub_r) or cc1 >= int(hub_c):
                    continue
            elif region == "tr":
                if rr1 >= int(hub_r) or cc0 <= int(hub_c + n - 1):
                    continue
            elif region == "bl":
                if rr0 <= int(hub_r + n - 1) or cc1 >= int(hub_c):
                    continue
            elif region == "br":
                if rr0 <= int(hub_r + n - 1) or cc0 <= int(hub_c + n - 1):
                    continue
            else:
                raise ValueError(f"unknown region: {region}")

            candidates.append((int(pr), int(pc)))
    return candidates


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
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    remaining = [c for c in range(10) if c != bg and c not in side_colors.values()]
    if not remaining:
        return None
    src_col = int(rng.choice(remaining))

    sides = ["top", "bot", "left", "right"]
    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_extent = max(max(m.shape) for m in sprite_masks.values())
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
        out[hub_r, hub_c + ci] = int(top_col)
        out[hub_r + N - 1, hub_c + ci] = int(bot_col)
    for ri in range(1, N - 1):
        out[hub_r + ri, hub_c] = int(left_col)
        out[hub_r + ri, hub_c + N - 1] = int(right_col)

    for side in sprite_sides:
        mask = sprite_masks[side]
        sh, sw = mask.shape
        color = int(side_colors[side])
        arr = np.where(mask, color, bg).astype(int)

        if side == "top":
            filled = np.where(mask[-1])[0]
            sc = int(hub_c + 1 - int(filled[0]))
            sr = int(hub_r - sh)
        elif side == "bot":
            filled = np.where(mask[0])[0]
            sc = int(hub_c + 1 - int(filled[0]))
            sr = int(hub_r + N)
        elif side == "left":
            filled = np.where(mask[:, -1])[0]
            sr = int(hub_r + 1 - int(filled[0]))
            sc = int(hub_c - sw)
        else:
            filled = np.where(mask[:, 0])[0]
            sr = int(hub_r + 1 - int(filled[0]))
            sc = int(hub_c + N)

        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        if bool((out[sr : sr + sh, sc : sc + sw] != bg)[mask].any()):
            return None
        paste(out, arr, sr, sc, transparent=bg)

    inp = np.full((H, W), bg, dtype=int)
    inp[hub_r : hub_r + N, hub_c : hub_c + N] = out[hub_r : hub_r + N, hub_c : hub_c + N]
    canvas = inp.copy()

    for side in sprite_sides:
        mask = sprite_masks[side]
        patch = np.where(mask, int(src_col), int(bg)).astype(int)
        k = rng.choice([1, 2, 3])
        patch = _crop_patch(np.rot90(patch, int(k)), bg)

        candidates = _candidate_positions(canvas, patch, side, hub_r, hub_c, N, bg, padding=2)
        if not candidates:
            return None
        pr, pc = rng.choice(candidates)
        paste(canvas, patch, int(pr), int(pc), transparent=bg)
        paste(inp, patch, int(pr), int(pc), transparent=bg)

    if rng.random() < 0.3:
        distractor_colors = [c for c in remaining if int(c) != int(src_col)]
        if distractor_colors:
            dcol = int(rng.choice(distractor_colors))
            dist_mask = _make_sprite_mask(edge_len, rng.choice(sides), rng)
            if dist_mask is not None:
                dist_patch = np.where(dist_mask, dcol, bg).astype(int)
                dist_patch = _crop_patch(np.rot90(dist_patch, rng.choice([0, 1, 2, 3])), bg)
                corner = rng.choice(["tl", "tr", "bl", "br"])
                candidates = _candidate_positions(canvas, dist_patch, corner, hub_r, hub_c, N, bg, padding=2)
                if candidates:
                    pr, pc = rng.choice(candidates)
                    paste(canvas, dist_patch, int(pr), int(pc), transparent=bg)
                    paste(inp, dist_patch, int(pr), int(pc), transparent=bg)

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
