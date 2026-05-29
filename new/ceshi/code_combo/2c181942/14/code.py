# Combo variant of 2c181942 (index 14)
# combo_spec: D8/0+D12/1
# Name: 紧凑裁剪+缺边消色

from __future__ import annotations

import random

import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location
from scipy.ndimage import label as _label

# Design analysis for the fused variant
# a) Concrete behavior / trace:
#    The input stays close to the source task: a central hollow 4x4 hub shows all four
#    edge colors, and only 2-3 of those colors also appear as scattered monochrome sprites.
#    Example: if only the top and right colors have matching sprites, the source solver would
#    copy the whole hub, attach those two sprites, and leave the left/bottom hub edges colored.
#    D12 changes that hub rendering step: left/bottom are erased to bg because those colors
#    never appear outside the hub. D8 then crops the result tightly, but crucially it crops
#    the already-erased scene, not the original full hub. This coupling matters because a
#    missing hub side can make an entire outer hub row/column become bg, so the crop can bite
#    into the hub footprint itself instead of merely removing distant blank margins.
# b) Complexity calibration:
#    The generator always leaves 1-2 sides missing (K in {2,3}) so the D12 effect is visible
#    in every sample. Colors remain one-to-one with sides, and there are still only 2-3 real
#    sprites plus at most one fresh-color distractor, so matching stays learnable instead of
#    collapsing into clutter or ambiguous many-color bookkeeping.
# c) Implicit-assumption review of the original solver:
#    - `_find_hub()` assumes the input hub still shows all four colored sides. We keep that.
#    - The original `out[hub] = g[hub]` is no longer valid because it hardcodes "missing sides
#      remain colored". The combo must start from bg and repaint only sides with matched sprites.
#    - `m = (g == col) & (~hub_mask)` being empty now has semantic weight: it means both
#      "no sprite to attach" and "erase this hub side before cropping".
#    - Cropping can no longer be a blind final step on the original hub copy; it must happen
#      after conditional side painting, otherwise erased sides would still reserve space.
# d) Placement legality:
#    The generator still uses output-first assembly. It places the hub with margin based on the
#    maximum sprite extent, checks each attached sprite for bounds and overlap, then scatters
#    rotated sprites with `random_free_location` so hub, scattered sprites, and distractors stay
#    disjoint. The oracle target is the tight crop of the assembled combo output.
# e) Visual observability:
#    Across a few examples, humans can see two stable clues:
#    1. output is smaller than input because of tight cropping,
#    2. some hub sides disappear exactly when their color lacks a scattered sprite.
#    Because missing sides are always present, the combo does not degenerate into crop-only data.
# f) Visual cleanliness:
#    The scene remains one clean hub plus detached single-color sprites. No layered markings or
#    numeric color ordering is introduced. The crop simplifies the output further by stripping
#    the large blank frame around the assembled object.
# g) Color-rule inferability:
#    Colors only encode the side-to-sprite pairing already present in the source task. The new
#    disappearance rule is structural: a hub edge color survives only if that same color exists
#    as a sprite. That is observable from presence/absence, not hidden color-id arithmetic.
# h) New solver assumptions introduced by the combo, and generator guarantees:
#    - `main()` now assumes that every side which has an external same-color component has at
#      least one valid rotation satisfying the source symmetry/contact constraints. The generator
#      preserves this by constructing sprites in attached orientation first.
#    - `main()` also assumes absent colors really are absent outside the hub, otherwise an edge
#      would be kept. The generator never places distractors in hub-edge colors.
#    - Because cropping may trim away erased hub rows/cols, we rely on at least one kept side or
#      sprite existing. The generator uses K in {2,3}, so the output is never empty.


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(grid != int(bg))
    if pts.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(pts)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


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


def _paint_hub_side(out: np.ndarray, side: str, r: int, c: int, n: int, col: int) -> None:
    if side == "top":
        out[int(r), int(c + 1) : int(c + n - 1)] = int(col)
    elif side == "bot":
        out[int(r + n - 1), int(c + 1) : int(c + n - 1)] = int(col)
    elif side == "left":
        out[int(r + 1) : int(r + n - 1), int(c)] = int(col)
    else:
        out[int(r + 1) : int(r + n - 1), int(c + n - 1)] = int(col)


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Match each scattered sprite to the hub side with the same color.
    2. Rotate and attach it exactly as in the source task.
    3. If a hub side has no matching sprite, erase that hub side to bg.
    4. Tight-crop the final assembled scene.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    out = np.full_like(g, int(bg))
    target = int(n - 2)
    segs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }

    for side, (col, (need_edge, fixed, span)) in segs.items():
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

        found = None
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

            found = (a, msk, int(dr), int(dc))
            break

        if found is None:
            continue

        _paint_hub_side(out, side, r, c, n, col)
        a, msk, dr, dc = found
        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)

    return _tight_crop_non_bg(out, bg)


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Build a sprite whose attached orientation is uniquely identifiable for one side.
    Canonical frame: attaches on the left side of the hub, so the sprite contact edge
    is on its rightmost column and the mask is flipud-symmetric.
    """
    e = int(edge_len)
    h = 4
    w = rng.randint(2, 4)
    half_h = (h + 1) // 2

    for _attempt in range(100):
        grid = np.zeros((h, w), dtype=bool)
        grid[1, -1] = True
        grid[2, -1] = True

        for ri in range(half_h):
            mirror_ri = h - 1 - ri
            for ci in range(w - 1):
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
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        grid = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

        if grid.shape[0] < 2 or grid.shape[1] < 2:
            continue
        if not np.array_equal(grid, np.flipud(grid)):
            continue

        _labeled, num = _label(grid, structure=np.ones((3, 3), dtype=int))
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
            cr = rot[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1]
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
            return grid.copy()
        if side == "right":
            return np.fliplr(grid)
        if side == "top":
            return np.rot90(grid, -1)
        return np.rot90(grid, 1)

    return None


def _place_attached_sprite(
    canvas: np.ndarray,
    side: str,
    mask: np.ndarray,
    color: int,
    hub_r: int,
    hub_c: int,
    n: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    sh, sw = mask.shape
    arr = np.where(mask, int(color), int(bg)).astype(int)

    if side == "top":
        filled = np.where(mask[-1])[0]
        sc = int(hub_c + 1 - int(filled[0]))
        sr = int(hub_r - sh)
    elif side == "bot":
        filled = np.where(mask[0])[0]
        sc = int(hub_c + 1 - int(filled[0]))
        sr = int(hub_r + n)
    elif side == "left":
        filled = np.where(mask[:, -1])[0]
        sr = int(hub_r + 1 - int(filled[0]))
        sc = int(hub_c - sw)
    else:
        filled = np.where(mask[:, 0])[0]
        sr = int(hub_r + 1 - int(filled[0]))
        sc = int(hub_c + n)

    if sr < 0 or sc < 0 or sr + sh > canvas.shape[0] or sc + sw > canvas.shape[1]:
        return None

    for rr in range(sh):
        for cc2 in range(sw):
            if bool(mask[rr, cc2]) and int(canvas[int(sr + rr), int(sc + cc2)]) != int(bg):
                return None

    paste(canvas, arr, int(sr), int(sc), transparent=int(bg))
    return arr, mask, int(sr), int(sc)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    n = 4
    edge_len = n - 2

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]

    sides = ["top", "bot", "left", "right"]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    # Force 1-2 missing sides so both the edge-erasure rule and the crop-through-hub effect
    # are visible instead of degenerating into the source task or a crop-only variant.
    K = rng.choice([2, 3])
    sprite_sides = rng.sample(sides, int(K))

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(int(m.shape[0]) for m in sprite_masks.values())
    max_sprite_w = max(int(m.shape[1]) for m in sprite_masks.values())
    max_extent = max(max_sprite_h, max_sprite_w)
    margin = int(max_extent + 2)

    hub_r_min = margin
    hub_r_max = H - n - margin
    hub_c_min = margin
    hub_c_max = W - n - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(int(hub_r_min), int(hub_r_max))
    hub_c = rng.randint(int(hub_c_min), int(hub_c_max))

    assembled = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)

    # Input hub still exposes all four side colors; output hub only keeps sides that actually
    # have matching sprites, so missing sides can disappear before the tight crop.
    for side, color in side_colors.items():
        _paint_hub_side(inp, side, hub_r, hub_c, n, int(color))
        if side in sprite_sides:
            _paint_hub_side(assembled, side, hub_r, hub_c, n, int(color))

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        placed = _place_attached_sprite(
            assembled,
            side,
            sprite_masks[side],
            side_colors[side],
            hub_r,
            hub_c,
            n,
            bg,
        )
        if placed is None:
            return None
        attached_sprites[side] = placed

    expected = _tight_crop_non_bg(assembled, bg)

    canvas = inp.copy()
    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, int(k))
        rotated_mask = np.rot90(mask, int(k))

        rows_any = rotated_mask.any(axis=1)
        cols_any = rotated_mask.any(axis=0)
        r0_idx, r1_idx = np.where(rows_any)[0][[0, -1]]
        c0_idx, c1_idx = np.where(cols_any)[0][[0, -1]]
        cropped_arr = rotated_arr[int(r0_idx) : int(r1_idx) + 1, int(c0_idx) : int(c1_idx) + 1]
        cropped_mask = rotated_mask[int(r0_idx) : int(r1_idx) + 1, int(c0_idx) : int(c1_idx) + 1]
        patch = np.where(cropped_mask, cropped_arr, int(bg)).astype(int)

        try:
            pr, pc = random_free_location(canvas, patch, transparent=int(bg), padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, int(pr), int(pc), transparent=int(bg))
        paste(inp, patch, int(pr), int(pc), transparent=int(bg))

    if rng.random() < 0.3:
        used_colors = set(int(v) for v in side_colors.values())
        distractor_colors = [c for c in range(10) if c != bg and c not in used_colors]
        if distractor_colors:
            dcol = rng.choice(distractor_colors)
            dist_mask = _make_sprite_mask(edge_len, rng.choice(sides), rng)
            if dist_mask is not None:
                dist_arr = np.where(dist_mask, int(dcol), int(bg)).astype(int)
                try:
                    pr, pc = random_free_location(canvas, dist_arr, transparent=int(bg), padding=2, padding_connectivity=8)
                    paste(canvas, dist_arr, int(pr), int(pc), transparent=int(bg))
                    paste(inp, dist_arr, int(pr), int(pc), transparent=int(bg))
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
