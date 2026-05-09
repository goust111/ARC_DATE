# Variant of 2c181942
# Dimension: D6 — 干扰项处理策略 (src)
# Change: 0-1个干扰精灵，使用非hub边缘的颜色，solver自然忽略（颜色不匹配任何边）... → 同色干扰精灵（形状不匹配）

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


# Design analysis for the D6 variant
# ----------------------------------
# a) Concrete behavior trace
#    - The hub is unchanged: a centered hollow 4x4 square whose four sides have four
#      distinct colors, with bg corners and bg interior.
#    - The valid sprites are unchanged in meaning: one monochrome 8-connected sprite
#      for 2-4 hub sides, scattered away from the hub, each requiring rotation before
#      attachment. In attached orientation, the sprite must have a contact edge of
#      length N-2 (=2 here) and the same mirror symmetry as in the original task.
#    - The variant adds a new visible source-side phenomenon: sometimes one extra
#      sprite uses exactly the same color as one real sprite, but its geometry is
#      invalid for that hub side. Concretely, it is larger than the real sprite and
#      fails the attachment test in every rotation because it lacks the required
#      symmetry and/or the required contact-edge length.
#    - Small execution trace:
#      original logic for a color c:
#        1. take all non-hub cells of color c
#        2. label components
#        3. keep only the largest component
#        4. try rotations until one satisfies symmetry + edge-length constraints
#      new logic for the same color c:
#        1. take all non-hub cells of color c
#        2. label components
#        3. test every component independently against the same geometry constraints
#        4. among components with exactly one valid rotation and a legal placement,
#           attach the largest valid one
#      difference:
#        in the variant, a same-color distractor can be larger than the real sprite,
#        so "largest component first" is no longer safe. Geometry must participate in
#        component selection, not just in rotation selection.
# b) Complexity calibration
#    - The distractor count stays 0-1, so the task remains close to the original.
#    - The distractor shares color with a real sprite and is usually larger, making the
#      D6 change both visible and behaviorally necessary.
#    - True sprites still come from a small family of asymmetric masks with exactly one
#      valid rotation for their side, so matching remains learnable from 2-4 examples.
# c) Implicit assumption audit of the original main()
#    - `m = (g == col) & (~hub_mask)` still holds: the side color still identifies the
#      pool of candidates for that side.
#    - `label_mask(..., connectivity=8)` still holds: both valid sprites and distractors
#      are still monochrome 8-connected objects.
#    - The original "pick the largest component" assumption no longer holds. A larger
#      same-color distractor would incorrectly suppress the real sprite. This is the
#      precise line that must change.
#    - The symmetry checks and contact-edge checks remain valid; they become the
#      discriminative mechanism that rejects the same-color distractor.
#    - Placement conflict checks remain valid because output semantics are unchanged:
#      hub plus the attached real sprites only.
# d) Placement legality
#    - Output is still built first by placing the hub and attaching only valid sprites.
#    - Input placement still uses `random_free_location(..., padding=2, connectivity=8)`
#      so hub, real sprites, and distractor never merge into one component.
#    - The distractor is generated as one 8-connected component, rotated arbitrarily,
#      and placed with the same spacing guarantees as the true sprites.
# e) Visual observability
#    - A human can now see two objects of the same color in some inputs, but only one
#      of them has the "right kind of side" to plug into the hub. This is the core
#      observable difference from the original task.
#    - The distractor is not filtered by hidden numeric properties; it is rejected by
#      visible geometry: wrong silhouette, wrong symmetry, or wrong edge length.
# f) Visual tidiness
#    - There is no overdraw: all objects are isolated with a Chebyshev gap of at least 3.
#    - The distractor is monochrome like the real sprite family, so the scene stays clean
#      rather than adding extra markers or annotations.
# g) Color inference
#    - Hub-edge colors still map one-to-one to side destinations.
#    - The only color change in the task is the new reuse of one existing sprite color by
#      the distractor. Since both objects are visible simultaneously, the solver cannot
#      rely on color alone, but the color rule itself is still easy to infer.


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    dummy = np.where(mask, 1, 0).astype(int)
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


def _need_edge_for_side(side: str) -> str:
    return {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]


def _valid_rotations_for_side(mask: np.ndarray, need_edge: str, target: int) -> list[tuple[np.ndarray, tuple]]:
    valid: list[tuple[np.ndarray, tuple]] = []
    for krot in range(4):
        rot = _crop_mask(np.rot90(mask, int(krot)))
        if need_edge in ("left", "right"):
            if not bool(np.array_equal(rot, np.flipud(rot))):
                continue
        else:
            if not bool(np.array_equal(rot, np.fliplr(rot))):
                continue
        seg = _edge_segment(rot, need_edge)
        if seg is None:
            continue
        if need_edge in ("left", "right"):
            _typ, _edge_c, rows = seg
            if len(rows) != int(target) or not _is_contig(rows):
                continue
        else:
            _typ, _edge_r, cols = seg
            if len(cols) != int(target) or not _is_contig(cols):
                continue
        valid.append((rot, seg))
    return valid


def _aligned_translation(mask: np.ndarray, need_edge: str, fixed: int, span: list[int], target: int) -> tuple[int, int] | None:
    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return None
    if need_edge in ("left", "right"):
        _typ, edge_c, rows = seg
        if len(rows) != int(target) or not _is_contig(rows):
            return None
        return int(span[0] - rows[0]), int(fixed - edge_c)
    _typ, edge_r, cols = seg
    if len(cols) != int(target) or not _is_contig(cols):
        return None
    return int(fixed - edge_r), int(span[0] - cols[0])


def _attach_specs(r: int, c: int, n: int, top_col: int, bot_col: int, left_col: int, right_col: int) -> dict[str, tuple[int, tuple[str, int, list[int]]]]:
    return {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    D6 variant: same-color distractors are possible, so the solver must search every
    monochrome component of a hub-edge color and keep only the geometrically valid one.
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

                    if bool((top == int(bg)).any()) or bool((bot == int(bg)).any()) or bool((left == int(bg)).any()) or bool((right == int(bg)).any()):
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
    segs = _attach_specs(r, c, n, top_col, bot_col, left_col, right_col)

    for side in ("top", "bot", "left", "right"):
        col, (need_edge, fixed, span) = segs[side]
        m = (g == int(col)) & (~hub_mask)
        if not bool(m.any()):
            continue

        labc, nc = label_mask(m, connectivity=8)
        candidates = []
        for cid in range(1, int(nc) + 1):
            comp_mask = labc == int(cid)
            comp_arr = np.where(comp_mask, int(col), int(bg)).astype(int)
            comp_arr, comp_mask = _crop_to_mask(comp_arr, comp_mask)

            valid_rots = _valid_rotations_for_side(comp_mask, need_edge, target)
            if len(valid_rots) != 1:
                continue

            placed_mask, _seg = valid_rots[0]
            shift = _aligned_translation(placed_mask, need_edge, fixed, span, target)
            if shift is None:
                continue
            dr, dc = shift

            placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(placed_mask)}
            if any(not (0 <= R < H and 0 <= C < W) for R, C in placed):
                continue

            ok = True
            for R, C in placed:
                if int(out[int(R), int(C)]) != int(bg) and int(out[int(R), int(C)]) != int(col):
                    ok = False
                    break
            if not ok:
                continue

            candidates.append((int(placed_mask.sum()), np.where(placed_mask, int(col), int(bg)).astype(int), placed_mask, int(dr), int(dc)))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)
        _area, arr, mask, dr, dc = candidates[0]
        paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=True)

    return out


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Build a true sprite from a small family of canonical left-attach masks.

    Canonical frame:
    - height 4
    - contact edge on the right
    - flipud symmetric
    - right edge has exactly two contiguous cells
    - not fliplr symmetric
    - exactly one valid rotation for the left-side test
    """
    if int(edge_len) != 2:
        return None

    left_attach_caps = [
        np.array([1, 0], dtype=bool),
        np.array([1, 0, 0], dtype=bool),
        np.array([1, 1, 0], dtype=bool),
        np.array([1, 0, 0, 0], dtype=bool),
    ]
    rng.shuffle(left_attach_caps)

    for cap in left_attach_caps:
        w = int(cap.size)
        grid = np.zeros((4, w), dtype=bool)
        grid[0] = cap
        grid[1] = True
        grid[2] = True
        grid[3] = cap
        grid = _crop_mask(grid)
        if int(grid.sum()) > 10:
            continue
        if len(_valid_rotations_for_side(grid, "right", edge_len)) != 1:
            continue

        if side == "left":
            return grid.copy()
        if side == "right":
            return np.fliplr(grid)
        if side == "top":
            return np.rot90(grid, -1)
        return np.rot90(grid, 1)

    return None


def _grow_connected_mask(h: int, w: int, area: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    sr = rng.randrange(h)
    sc = rng.randrange(w)
    mask[sr, sc] = True

    while int(mask.sum()) < int(area):
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = rng.choice(list(frontier))
        mask[int(nr), int(nc)] = True

    return _crop_mask(mask)


def _make_same_color_distractor_mask(edge_len: int, side: str, min_area: int, rng: random.Random) -> np.ndarray | None:
    """
    Generate a larger same-color component that is invalid for the chosen side in all
    four rotations. This makes color-only or largest-component-only selection fail.
    """
    need_edge = _need_edge_for_side(side)
    lo = max(int(min_area), 11)
    hi = 12
    if lo > hi:
        lo = hi

    for _ in range(250):
        h = rng.randint(4, 5)
        w = rng.randint(4, 5)
        area = rng.randint(lo, hi)
        if area >= h * w:
            area = h * w - 1
        mask = _grow_connected_mask(h, w, area, rng)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if int(mask.sum()) < int(lo):
            continue
        _lab, ncomp = label_mask(mask, connectivity=8)
        if int(ncomp) != 1:
            continue
        if _valid_rotations_for_side(mask, need_edge, edge_len):
            continue
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

    K = rng.randint(2, 4)
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
    for ci in range(1, n - 1):
        out[hub_r, hub_c + ci] = top_col
        out[hub_r + n - 1, hub_c + ci] = bot_col
    for ri in range(1, n - 1):
        out[hub_r + ri, hub_c] = left_col
        out[hub_r + ri, hub_c + n - 1] = right_col

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    segs = _attach_specs(hub_r, hub_c, n, top_col, bot_col, left_col, right_col)
    for side in sprite_sides:
        mask = sprite_masks[side]
        color = side_colors[side]
        arr = np.where(mask, color, bg).astype(int)
        _col, (need_edge, fixed, span) = segs[side]
        shift = _aligned_translation(mask, need_edge, fixed, span, edge_len)
        if shift is None:
            return None
        sr, sc = shift
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr, cc in np.argwhere(mask):
            if int(out[int(sr + rr), int(sc + cc)]) != int(bg):
                return None
        paste(out, arr, sr, sc, transparent=bg)
        attached_sprites[side] = (arr, mask)

    inp = np.full((H, W), bg, dtype=int)
    inp[hub_r : hub_r + n, hub_c : hub_c + n] = out[hub_r : hub_r + n, hub_c : hub_c + n]
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, k)
        rotated_mask = _crop_mask(np.rot90(mask, k))
        rotated_arr = np.where(rotated_mask, side_colors[side], bg).astype(int)
        try:
            pr, pc = random_free_location(canvas, rotated_arr, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, rotated_arr, pr, pc, transparent=bg)
        paste(inp, rotated_arr, pr, pc, transparent=bg)

    if rng.random() < 0.75:
        dside = rng.choice(sprite_sides)
        real_area = int(sprite_masks[dside].sum())
        dist_mask = _make_same_color_distractor_mask(edge_len, dside, real_area + 1, rng)
        if dist_mask is None:
            return None
        dist_mask = np.rot90(dist_mask, rng.choice([0, 1, 2, 3]))
        dcol = side_colors[dside]
        dist_arr = np.where(dist_mask, dcol, bg).astype(int)
        try:
            pr, pc = random_free_location(canvas, dist_arr, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, dist_arr, pr, pc, transparent=bg)
        paste(inp, dist_arr, pr, pc, transparent=bg)

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
