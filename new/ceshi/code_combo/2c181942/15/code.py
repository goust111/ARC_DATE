# Combo variant of 2c181942 (index 15)
# combo_spec: D12/1+D14/0
# Name: 缺边消色+颜色序贴附

from __future__ import annotations

import random

import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location

# Design analysis for the combo variant
# a) Concrete behavior / trace:
#    The source task copies the whole colored hub to the output first, then independently
#    attaches every matching sprite. Here that breaks in two places.
#    Example: suppose the four hub edge colors are {top=1, bot=2, left=3, right=4}, but
#    only colors 1/3/4 exist as scattered sprites. The old output would still keep the
#    bottom edge color 2; this combo must erase that edge to bg because its sprite is
#    missing (D12). Also, the present sprites are not commutative anymore: they are
#    attached in color order 1 -> 3 -> 4, and later ones overwrite earlier ones on the
#    one-cell corner overlaps produced by the generator (D14). So the final corner color is
#    determined jointly by sprite existence and color order.
# b) Complexity calibration:
#    To keep the rule visible, generation always omits 1-2 hub sides and always chooses a
#    side subset containing at least one adjacent pair, so attached sprites necessarily
#    overlap at an outer hub corner. The color pool is fixed to {1,2,3,4}; side assignment
#    is shuffled per sample so precedence is tied to color rather than side.
# c) Implicit-assumption review:
#    - `_find_hub()` still assumes the input hub itself shows all four colored edges.
#      Generator preserves that; only the output hub is selective.
#    - The original `out[hub] = g[hub]` assumed absent sides stay colored. That assumption
#      is invalid now, so rendering starts from bg and paints only present sides.
#    - The original conflict rejection against previously pasted sprites assumed order
#      should not matter. That is the exact D14 assumption we must remove. We now collect
#      all valid attachments first, then sort by color and paste with overwrite enabled.
#    - Matching still assumes one largest component per hub color outside the hub. Generator
#      guarantees one sprite per present color and no distractors reusing hub colors.
# d) Placement legality:
#    Output is still built first. Each attached sprite placement is deterministic from its
#    contact edge, checked for bounds, and allowed to overlap only with other sprites.
#    Input sprites are then detached, rotated, and scattered with `random_free_location`
#    so they do not overlap each other or the hub.
# e) Visual observability:
#    Missing sides are always present in every sample, and overlap pixels are always
#    present too. Because the same visible palette {1,2,3,4} repeats across samples while
#    side assignment changes, a human can observe that the covering priority follows color.
# f) Visual cleanliness:
#    The generator uses 2-3 sprites only, no distractors, and controlled one-cell corner
#    overlaps. The scene remains a single central hub with a small number of large, clean
#    monochrome sprites.
# g) Color-order inferability:
#    Ordering by raw color id would be opaque if colors varied wildly. Restricting the hub
#    palette to a fixed repeated set avoids that: the same few visible colors repeatedly
#    win or lose on overlaps across examples, making the precedence relation observable.

from scipy.ndimage import label as _label


BG = 8
HUB_N = 4
EDGE_LEN = HUB_N - 2
SIDES = ("top", "bot", "left", "right")
ORDERED_COLORS = (1, 2, 3, 4)
SIDE_SETS = (
    ("top", "left"),
    ("top", "right"),
    ("bot", "left"),
    ("bot", "right"),
    ("top", "left", "right"),
    ("bot", "left", "right"),
    ("top", "bot", "left"),
    ("top", "bot", "right"),
)
CANONICAL_LEFT_TEMPLATES = (
    np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
        ],
        dtype=bool,
    ),
)


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


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


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
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


def _locate_attachment(
    g: np.ndarray,
    hub_mask: np.ndarray,
    out_shape: tuple[int, int],
    bg: int,
    color: int,
    need_edge: str,
    fixed: int,
    span: list[int],
    target: int,
):
    H, W = out_shape
    m = (g == int(color)) & (~hub_mask)
    if not bool(m.any()):
        return None

    labc, nc = label_mask(m, connectivity=8)
    best_id = None
    best_sz = -1
    for cid in range(1, int(nc) + 1):
        sz = int((labc == int(cid)).sum())
        if sz > best_sz:
            best_sz = sz
            best_id = int(cid)
    if best_id is None or best_sz <= 0:
        return None

    spr_mask = labc == int(best_id)
    spr_arr = g.copy()
    spr_arr[~spr_mask] = int(bg)
    spr_arr, spr_mask = _crop_to_mask(spr_arr, spr_mask)

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
        return a, msk, int(dr), int(dc)

    return None


def _collect_attachments(
    g: np.ndarray,
    bg: int,
    hub,
) -> tuple[int, int, int, list[tuple[int, str, np.ndarray, np.ndarray, int, int]]]:
    H, W = g.shape
    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[r : r + n, c : c + n] = True
    target = int(n - 2)
    segs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }

    attachments = []
    for side, (col, (need_edge, fixed, span)) in segs.items():
        found = _locate_attachment(g, hub_mask, (H, W), bg, int(col), need_edge, fixed, span, target)
        if found is None:
            continue
        a, msk, dr, dc = found
        attachments.append((int(col), side, a, msk, int(dr), int(dc)))
    return int(r), int(c), int(n), attachments


def _render_ordered(
    hub_geom: tuple[int, int, int],
    attachments: list[tuple[int, str, np.ndarray, np.ndarray, int, int]],
    out_shape: tuple[int, int],
    bg: int,
    *,
    reverse: bool = False,
) -> np.ndarray:
    r, c, n = hub_geom
    out = np.full(out_shape, int(bg), dtype=int)
    ordered = sorted(attachments, key=lambda item: int(item[0]), reverse=bool(reverse))
    for col, side, a, msk, dr, dc in ordered:
        _paint_hub_side(out, side, int(r), int(c), int(n), int(col))
        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Missing sides are erased: a hub edge survives only if its matching sprite exists.
    - Present sprites are attached in ascending color order, and later colors overwrite
      earlier ones where attached sprites overlap.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    hub = _find_hub(g, bg)
    assert hub is not None
    r, c, n, attachments = _collect_attachments(g, bg, hub)
    return _render_ordered((r, c, n), attachments, g.shape, bg, reverse=False)


def _orient_from_left(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return np.rot90(mask, -1)
    return np.rot90(mask, 1)


def _valid_attached_orientation(mask: np.ndarray, side: str) -> bool:
    if side in ("top", "bot"):
        if not bool(np.array_equal(mask, np.fliplr(mask))):
            return False
        edge = mask[-1] if side == "top" else mask[0]
        filled = np.where(edge)[0]
    else:
        if not bool(np.array_equal(mask, np.flipud(mask))):
            return False
        edge = mask[:, -1] if side == "left" else mask[:, 0]
        filled = np.where(edge)[0]
    return len(filled) == EDGE_LEN and _is_contig([int(x) for x in filled])


def _unique_rotation_for_side(mask: np.ndarray, side: str) -> bool:
    valid = 0
    for k in range(4):
        rot = _crop_mask(np.rot90(mask, k))
        if _valid_attached_orientation(rot, side):
            valid += 1
    return valid == 1


def _place_attached_mask(mask: np.ndarray, side: str, hub_r: int, hub_c: int, n: int) -> tuple[int, int]:
    sh, sw = mask.shape
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
    return int(sr), int(sc)


def _make_sprite_masks(rng: random.Random) -> dict[str, np.ndarray]:
    base = rng.choice(CANONICAL_LEFT_TEMPLATES)
    masks = {side: _orient_from_left(base, side) for side in SIDES}
    for side, mask in masks.items():
        assert _unique_rotation_for_side(mask, side)
    return masks


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    H = rng.randint(24, 26)
    W = rng.randint(24, 26)

    colors = list(ORDERED_COLORS)
    rng.shuffle(colors)
    side_colors = dict(zip(SIDES, colors))
    sprite_sides = list(rng.choice(SIDE_SETS))
    sprite_masks = _make_sprite_masks(rng)

    max_sprite_h = max(int(m.shape[0]) for m in sprite_masks.values())
    max_sprite_w = max(int(m.shape[1]) for m in sprite_masks.values())
    margin = max(max_sprite_h, max_sprite_w) + 2
    hub_r_min = margin
    hub_r_max = H - HUB_N - margin
    hub_c_min = margin
    hub_c_max = W - HUB_N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None
    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    for side, color in side_colors.items():
        _paint_hub_side(inp, side, hub_r, hub_c, HUB_N, color)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    cover_counts = np.zeros((H, W), dtype=int)

    for side in sprite_sides:
        mask = sprite_masks[side]
        color = int(side_colors[side])
        arr = np.where(mask, color, BG).astype(int)
        sr, sc = _place_attached_mask(mask, side, hub_r, hub_c, HUB_N)
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        attached_sprites[side] = (arr, mask, sr, sc)
        for rr, cc2 in np.argwhere(mask):
            cover_counts[sr + int(rr), sc + int(cc2)] += 1

    overlap_pixels = int((cover_counts > 1).sum())
    if overlap_pixels <= 0:
        return None

    ordered_sides = sorted(sprite_sides, key=lambda side: int(side_colors[side]))
    for side in ordered_sides:
        color = int(side_colors[side])
        _paint_hub_side(out, side, hub_r, hub_c, HUB_N, color)
        arr, mask, sr, sc = attached_sprites[side]
        paste(out, arr, sr, sc, mask=mask, overwrite=True, clip=True)

    canvas = inp.copy()
    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, int(k))
        rotated_mask = np.rot90(mask, int(k))
        rotated_arr, rotated_mask = _crop_to_mask(rotated_arr, rotated_mask)
        patch = np.where(rotated_mask, rotated_arr, BG).astype(int)
        try:
            pr, pc = random_free_location(canvas, patch, transparent=BG, padding=1, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=BG)
        paste(inp, patch, pr, pc, transparent=BG)

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    missing_count = 4 - len(sprite_sides)
    if missing_count <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
