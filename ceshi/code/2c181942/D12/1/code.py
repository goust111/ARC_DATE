# Variant of 2c181942
# Dimension: D12 — 缺失边处理 (meta)
# Change: 0-2条hub边可以没有对应精灵（输出中该边仅保留hub边缘颜色，不贴附任何精灵）... → 缺失边在输出中移除边缘颜色

from __future__ import annotations

import random

import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location
from scipy.ndimage import label as _label

# Design analysis for the D12 variant
# a) Concrete behavior:
#    Input keeps the original hub: a hollow 4x4 square whose four sides are all colored.
#    Only 2-3 of those side colors appear again as scattered monochrome sprites.
#    Output attaches only the existing sprites, and the hub edges for missing colors are
#    erased back to the background. Example: if only top/right sprites exist, the output
#    keeps only the top/right hub edge colors; left/bottom hub edges become bg.
#    This changes the original pipeline exactly at hub rendering time:
#    - original: copy the whole hub into output first, so absent sides stay visible.
#    - variant: start from bg and repaint only the hub sides whose matching sprite exists.
# b) Complexity calibration:
#    To make the change visible, generation always leaves 1-2 sides missing (2 or 3 sprites,
#    never 4). Colors remain unique per side, and no distractor reuses a hub color, so the
#    side-presence signal is clean and visually inferable.
# c) Implicit-assumption review of the original solver:
#    - _find_hub() assumes all four input hub edges are present, uniform, and non-bg.
#      This still holds because only the output rule changes.
#    - The old `out[hub] = g[hub]` relied on missing edges remaining colored. That is the
#      exact broken assumption for D12 and must be removed.
#    - `m = (g == col) & (~hub_mask)` returning empty now means "erase that hub edge".
#      Generator guarantees missing sides truly have no matching sprite outside the hub.
#    - Placement/conflict logic remains valid because attached sprites sit adjacent to the
#      hub; they never overlap the hub edge cells that are now conditionally repainted.
# d) Placement legality:
#    Generator still uses output-first construction, checks bounds/overlap for each attached
#    sprite, scatters rotated sprites with random_free_location, and oracle-checks main(inp).
# e) Observability / cleanliness:
#    Every generated sample contains at least one erased hub edge in the output, so the new
#    rule is visible from a few examples. The scene stays uncluttered: one central hub,
#    2-3 matching sprites, optional one-color distractor in a fresh color.


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


def _paint_hub_side(out: np.ndarray, side: str, r: int, c: int, n: int, col: int) -> None:
    if side == "top":
        out[r, c + 1 : c + n - 1] = int(col)
    elif side == "bot":
        out[r + n - 1, c + 1 : c + n - 1] = int(col)
    elif side == "left":
        out[r + 1 : r + n - 1, c] = int(col)
    else:
        out[r + 1 : r + n - 1, c + n - 1] = int(col)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Match and attach sprites exactly as in the source task.
    - But a hub side is kept in the output only if its matching sprite exists in input.
      Missing sides are erased to bg instead of remaining colored.
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

    hub = _find_hub()
    assert hub is not None

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[r : r + n, c : c + n] = True

    out = np.full_like(g, bg)
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
        spr_arr[~spr_mask] = bg
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
                val = int(a[rr - dr, cc2 - dc])
                if val == bg:
                    continue
                if int(out[rr, cc2]) != bg and int(out[rr, cc2]) != val:
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

    return out


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Build a sprite whose attached orientation is uniquely identifiable for one side.
    Canonical frame: attaches on the left side of the hub, so the sprite contact edge
    is on its rightmost column and the mask is flipud-symmetric.
    """
    e = edge_len
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
    n = 4
    edge_len = n - 2

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]

    sides = ["top", "bot", "left", "right"]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    # Always leave 1-2 sides missing so the D12 behavior is visible.
    K = rng.choice([2, 3])
    sprite_sides = rng.sample(sides, K)
    sprite_masks: dict[str, np.ndarray] = {}
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
    hub_r_max = H - n - margin
    hub_c_min = margin
    hub_c_max = W - n - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    inp = np.full((H, W), bg, dtype=int)

    # Input hub always shows all four edge colors; output hub only keeps sides
    # whose matching sprite exists.
    for side, color in side_colors.items():
        _paint_hub_side(inp, side, hub_r, hub_c, n, color)
        if side in sprite_sides:
            _paint_hub_side(out, side, hub_r, hub_c, n, color)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
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
            sr = hub_r + n
        elif side == "left":
            filled = np.where(mask[:, -1])[0]
            sr = hub_r + 1 - int(filled[0])
            sc = hub_c - sw
        else:
            filled = np.where(mask[:, 0])[0]
            sr = hub_r + 1 - int(filled[0])
            sc = hub_c + n

        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None

        for rr in range(sh):
            for cc2 in range(sw):
                if mask[rr, cc2] and out[sr + rr, sc + cc2] != bg:
                    return None

        paste(out, arr, sr, sc, transparent=bg)
        attached_sprites[side] = (arr, mask, sr, sc)

    canvas = inp.copy()
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
    if not np.array_equal(result, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
