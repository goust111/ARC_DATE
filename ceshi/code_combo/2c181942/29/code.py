# Combo variant of 2c181942 (index 29)
# combo_spec: D6/0+D15/1
# Name: 多干扰精灵+多背景区域
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    Original task assumes a single global bg color. A color mask such as "all red cells"
#    is therefore enough to isolate the red sprite, and clearing the input means painting
#    everything back to one bg color.
#    In this combo, the canvas is split into 3 vertical solid-color regions. Example:
#    left band = 8, middle band = 5, right band = 0. The hub sits inside the middle band,
#    while valid sprites and 2-3 distractors are scattered across all bands. A red sprite in
#    the left band must be extracted against bg=8, while a green distractor in the right band
#    must be extracted against bg=0. Output keeps the banded background and only reattaches
#    the hub-matching sprites to the hub.
#    Difference from original: extraction is now local-to-region, and "clear everything else"
#    means restoring each object's own regional background rather than one global color.
# b) Complexity calibration:
#    The visual change must be obvious, so the regions are large full-height vertical bands
#    with distinct colors, not tiny patches. Distractors are 2-3 sprite-like objects built
#    from the same mask family as valid sprites, so the input is visibly noisier but still
#    interpretable because only hub-edge colors are actionable.
# c) Broken implicit assumptions in the original solver:
#    - `get_bg(grid)` is no longer valid because there is no single background.
#    - "largest component of the matching color" is no longer the right abstraction; the
#      solver must first enumerate all small monochrome components outside the hub.
#    - Clearing to a single bg color would destroy the region layout.
#    The new solver therefore finds the hub by local square structure, extracts all
#    sprite-sized monochrome 8-components, infers each component's local background from the
#    1-cell ring around its bbox, erases candidates back to that local bg, and only then
#    reattaches components whose color matches a hub edge and whose rotated mask satisfies
#    the same contact-edge/symmetry constraints as the original task.
# d) Placement legality:
#    The generator uses output-first construction, keeps the hub and all attached output
#    sprites inside the middle band, scatters input objects with padding, and keeps every
#    scattered object's bbox at least 1 cell away from region borders. This guarantees
#    no overlap, no out-of-bounds placement, and an unambiguous local background ring.
# e) Visual observability:
#    A human sees three clean background bands plus several detached monochrome objects.
#    Across examples, the only sprites that ever reattach are those whose color appears on a
#    hub edge; the others disappear. The local-background restoration is also visible because
#    removed distractors leave behind their own band's color rather than a universal fill.
# f) Visual cleanliness:
#    Regions are simple stripes, hub output stays inside the middle stripe, and objects are
#    spaced apart by padding. This avoids boundary clutter and makes each component legible.
# g) Inferability of changes:
#    No hidden ordering uses numeric color ids. Region identity is conveyed spatially by
#    large solid bands, while valid-vs-distractor is conveyed structurally by color matching
#    to the hub edges. The mapping is therefore recoverable from a few IO pairs.
# h) New solver assumptions introduced by the combo:
#    - Sprite-like components are small monochrome 8-components with a mostly uniform outer
#      ring; the generator enforces this with region-border margin and inter-object padding.
#    - Hub corners/interior share one local bg color; the generator keeps the hub fully inside
#      the middle band.
#    - Hub-edge colors are distinct from region colors and distractor colors; the generator
#      samples colors accordingly.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label as _label

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, label_mask, paste


SIDES = ("top", "bot", "left", "right")
CANONICAL_LEFT_MASKS = (
    np.array([[1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0]], dtype=bool),
)


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


def _count_runs(bits: np.ndarray) -> list[int]:
    runs: list[int] = []
    cur = 0
    for v in bits.astype(bool).tolist():
        if v:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


def _rotation_valid_count(mask: np.ndarray, need_edge: str, edge_len: int) -> int:
    count = 0
    for krot in range(4):
        rot = np.rot90(mask, krot)
        _, rot = _crop_to_mask(rot.astype(int), rot)
        if need_edge in ("left", "right"):
            if not np.array_equal(rot, np.flipud(rot)):
                continue
        else:
            if not np.array_equal(rot, np.fliplr(rot)):
                continue
        seg = _edge_segment(rot, need_edge)
        if seg is None:
            continue
        if need_edge in ("left", "right"):
            _typ, _edge_c, rows = seg
            if len(rows) == edge_len and _is_contig(rows):
                count += 1
        else:
            _typ, _edge_r, cols = seg
            if len(cols) == edge_len and _is_contig(cols):
                count += 1
    return count


def _find_hub(g: np.ndarray):
    H, W = g.shape
    best = None
    for n in range(3, 7):
        for r in range(H - n + 1):
            for c in range(W - n + 1):
                bg = int(g[r, c])
                if (
                    int(g[r, c + n - 1]) != bg
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
                side_cols = (int(top[0]), int(bot[0]), int(left[0]), int(right[0]))
                if (
                    len(set(map(int, top))) != 1
                    or len(set(map(int, bot))) != 1
                    or len(set(map(int, left))) != 1
                    or len(set(map(int, right))) != 1
                    or len(set(side_cols)) != 4
                ):
                    continue
                cr = r + (n - 1) / 2
                cc = c + (n - 1) / 2
                score = abs(cr - (H - 1) / 2) + abs(cc - (W - 1) / 2)
                cand = (
                    float(score),
                    int(r),
                    int(c),
                    int(n),
                    int(bg),
                    side_cols,
                )
                if best is None or cand < best:
                    best = cand
    if best is None:
        return None
    _score, r, c, n, bg, side_cols = best
    return {
        "r": r,
        "c": c,
        "n": n,
        "bg": bg,
        "top_col": int(side_cols[0]),
        "bot_col": int(side_cols[1]),
        "left_col": int(side_cols[2]),
        "right_col": int(side_cols[3]),
    }


def _component_local_bg(grid: np.ndarray, coords: np.ndarray) -> tuple[int, float] | None:
    H, W = grid.shape
    r0, c0, r1, c1 = bbox_coords(coords)
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(H - 1, r1 + 1)
    ec1 = min(W - 1, c1 + 1)

    ring: list[int] = []
    for rr in range(er0, er1 + 1):
        for cc in range(ec0, ec1 + 1):
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            ring.append(int(grid[rr, cc]))
    if not ring:
        return None
    counts = Counter(ring)
    bg, cnt = counts.most_common(1)[0]
    return int(bg), float(cnt) / float(len(ring))


def _extract_candidates(g: np.ndarray, hub_mask: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    for color in sorted(set(map(int, g.flatten()))):
        mask = g == int(color)
        mask[hub_mask] = False
        if not bool(mask.any()):
            continue
        lab, nc = label_mask(mask, connectivity=8)
        for cid in range(1, nc + 1):
            comp = lab == cid
            area = int(comp.sum())
            if area < 4 or area > 16:
                continue
            coords = np.argwhere(comp)
            local = _component_local_bg(g, coords)
            if local is None:
                continue
            local_bg, dominance = local
            if local_bg == int(color) or dominance < 0.8:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            submask = comp[r0 : r1 + 1, c0 : c1 + 1]
            candidates.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": submask,
                    "local_bg": int(local_bg),
                }
            )
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    hub = _find_hub(g)
    assert hub is not None

    r = int(hub["r"])
    c = int(hub["c"])
    n = int(hub["n"])
    target = int(n - 2)

    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[r : r + n, c : c + n] = True

    candidates = _extract_candidates(g, hub_mask)

    out = g.copy()
    for cand in candidates:
        coords = cand["coords"]
        out[coords[:, 0], coords[:, 1]] = int(cand["local_bg"])

    occupied = np.zeros((H, W), dtype=bool)
    occupied[r : r + n, c : c + n] = True

    segs = {
        "top": (int(hub["top_col"]), ("bot", int(r - 1), list(range(c + 1, c + n - 1)))),
        "bot": (int(hub["bot_col"]), ("top", int(r + n), list(range(c + 1, c + n - 1)))),
        "left": (int(hub["left_col"]), ("right", int(c - 1), list(range(r + 1, r + n - 1)))),
        "right": (int(hub["right_col"]), ("left", int(c + n), list(range(r + 1, r + n - 1)))),
    }

    for _side, (col, (need_edge, fixed, span)) in segs.items():
        matches = [cand for cand in candidates if int(cand["color"]) == int(col)]
        found = None
        for cand in matches:
            for krot in range(4):
                rot_mask = np.rot90(cand["mask"], krot)
                rot_arr = np.full(rot_mask.shape, int(col), dtype=int)
                rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)

                if need_edge in ("left", "right"):
                    if not np.array_equal(rot_mask, np.flipud(rot_mask)):
                        continue
                else:
                    if not np.array_equal(rot_mask, np.fliplr(rot_mask)):
                        continue

                seg = _edge_segment(rot_mask, need_edge)
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

                placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(rot_mask)}
                if any(not (0 <= rr < H and 0 <= cc2 < W) for rr, cc2 in placed):
                    continue
                if any(bool(occupied[rr, cc2]) for rr, cc2 in placed):
                    continue

                found = (rot_arr, rot_mask, int(dr), int(dc), placed)
                break
            if found is not None:
                break

        if found is None:
            continue

        arr, mask, dr, dc, placed = found
        paste(out, arr, dr, dc, mask=mask, overwrite=True, clip=False)
        for rr, cc2 in placed:
            occupied[rr, cc2] = True

    return out


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    if edge_len != 2:
        return None
    options = [m for m in CANONICAL_LEFT_MASKS if _rotation_valid_count(m, "right", edge_len) == 1]
    if not options:
        return None
    base = rng.choice(options).copy()
    if side == "left":
        return base
    if side == "right":
        return np.fliplr(base)
    if side == "top":
        return np.rot90(base, -1)
    if side == "bot":
        return np.rot90(base, 1)
    return None


def _attached_top_left(mask: np.ndarray, side: str, hub_r: int, hub_c: int, n: int) -> tuple[int, int]:
    sh, sw = mask.shape
    if side == "top":
        filled = np.where(mask[-1])[0]
        return int(hub_r - sh), int(hub_c + 1 - filled[0])
    if side == "bot":
        filled = np.where(mask[0])[0]
        return int(hub_r + n), int(hub_c + 1 - filled[0])
    if side == "left":
        filled = np.where(mask[:, -1])[0]
        return int(hub_r + 1 - filled[0]), int(hub_c - sw)
    filled = np.where(mask[:, 0])[0]
    return int(hub_r + 1 - filled[0]), int(hub_c + n)


def _place_mask_patch(
    grid: np.ndarray,
    mask: np.ndarray,
    color: int,
    region_c0: int,
    region_c1: int,
    blocked: np.ndarray,
    rng: random.Random,
    *,
    padding: int,
) -> tuple[int, int] | None:
    H, W = grid.shape
    h, w = mask.shape
    r_min = 1
    r_max = H - h - 1
    c_min = region_c0 + 1
    c_max = region_c1 - w
    if r_min > r_max or c_min > c_max:
        return None

    options: list[tuple[int, int]] = []
    for rr in range(r_min, r_max + 1):
        for cc in range(c_min, c_max + 1):
            if bool(blocked[rr : rr + h, cc : cc + w].any()):
                continue
            options.append((int(rr), int(cc)))
    if not options:
        return None

    rr, cc = rng.choice(options)
    patch = np.full(mask.shape, int(color), dtype=int)
    paste(grid, patch, rr, cc, mask=mask, overwrite=True, clip=False)
    br0 = max(0, rr - padding)
    bc0 = max(0, cc - padding)
    br1 = min(H, rr + h + padding)
    bc1 = min(W, cc + w + padding)
    blocked[br0:br1, bc0:bc1] = True
    return int(rr), int(cc)


def _build_region_grid(H: int, W: int, colors: tuple[int, int, int], rng: random.Random):
    for _ in range(80):
        left_w = rng.randint(6, 8)
        right_w = rng.randint(6, 8)
        mid_w = W - left_w - right_w
        if 10 <= mid_w <= 14:
            break
    else:
        return None

    c0 = left_w - 1
    c1 = left_w + mid_w - 1
    grid = np.empty((H, W), dtype=int)
    grid[:, : left_w] = int(colors[0])
    grid[:, left_w : left_w + mid_w] = int(colors[1])
    grid[:, left_w + mid_w :] = int(colors[2])
    regions = [
        (0, c0, int(colors[0])),
        (left_w, c1, int(colors[1])),
        (left_w + mid_w, W - 1, int(colors[2])),
    ]
    return grid, regions


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = 4
    edge_len = N - 2

    H = rng.randint(20, 26)
    W = rng.randint(24, 26)

    all_colors = list(range(10))
    rng.shuffle(all_colors)
    bg_cols = tuple(all_colors[:3])
    hub_cols = tuple(all_colors[3:7])
    dist_cols = list(all_colors[7:])
    if len(dist_cols) < 2:
        return None

    built = _build_region_grid(H, W, bg_cols, rng)
    if built is None:
        return None
    base, regions = built
    mid_c0, mid_c1, mid_bg = regions[1]

    side_colors = {
        "top": int(hub_cols[0]),
        "bot": int(hub_cols[1]),
        "left": int(hub_cols[2]),
        "right": int(hub_cols[3]),
    }

    k_valid = rng.randint(2, 4)
    sprite_sides = rng.sample(list(SIDES), k_valid)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_h = max(mask.shape[0] for mask in sprite_masks.values())
    hub_r_min = max_h + 1
    hub_r_max = H - N - max_h - 1
    if hub_r_min > hub_r_max:
        return None

    candidates: list[tuple[int, int]] = []
    for hub_r in range(hub_r_min, hub_r_max + 1):
        for hub_c in range(mid_c0 + 1, mid_c1 - N + 1):
            ok = True
            occupied = np.zeros((H, W), dtype=bool)
            occupied[hub_r : hub_r + N, hub_c : hub_c + N] = True
            for side in sprite_sides:
                mask = sprite_masks[side]
                sr, sc = _attached_top_left(mask, side, hub_r, hub_c, N)
                sh, sw = mask.shape
                if sr < 0 or sc < mid_c0 or sr + sh > H or sc + sw - 1 > mid_c1:
                    ok = False
                    break
                for rr, cc in np.argwhere(mask):
                    if occupied[sr + rr, sc + cc]:
                        ok = False
                        break
                    occupied[sr + rr, sc + cc] = True
                if not ok:
                    break
            if ok:
                candidates.append((int(hub_r), int(hub_c)))
    if not candidates:
        return None

    hub_r, hub_c = rng.choice(candidates)

    out = base.copy()
    for ci in range(1, N - 1):
        out[hub_r, hub_c + ci] = int(side_colors["top"])
        out[hub_r + N - 1, hub_c + ci] = int(side_colors["bot"])
    for ri in range(1, N - 1):
        out[hub_r + ri, hub_c] = int(side_colors["left"])
        out[hub_r + ri, hub_c + N - 1] = int(side_colors["right"])

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        color = int(side_colors[side])
        arr = np.where(mask, color, mid_bg).astype(int)
        sr, sc = _attached_top_left(mask, side, hub_r, hub_c, N)
        paste(out, arr, sr, sc, mask=mask, overwrite=True, clip=False)
        attached_sprites[side] = (arr, mask, sr, sc)

    inp = base.copy()
    inp[hub_r : hub_r + N, hub_c : hub_c + N] = out[hub_r : hub_r + N, hub_c : hub_c + N]

    blocked = np.zeros((H, W), dtype=bool)
    blocked[max(0, hub_r - 2) : min(H, hub_r + N + 2), max(0, hub_c - 2) : min(W, hub_c + N + 2)] = True

    objects: list[tuple[np.ndarray, int]] = []
    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        krot = rng.choice([1, 2, 3])
        rot_arr = np.rot90(arr, krot)
        rot_mask = np.rot90(mask, krot)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        objects.append((rot_mask, int(side_colors[side])))

    d_count = rng.randint(2, 3)
    for i in range(d_count):
        side = rng.choice(list(SIDES))
        dmask = _make_sprite_mask(edge_len, side, rng)
        if dmask is None:
            return None
        dcolor = int(dist_cols[i % len(dist_cols)])
        krot = rng.choice([0, 1, 2, 3])
        dmask = np.rot90(dmask, krot)
        _, dmask = _crop_to_mask(dmask.astype(int), dmask)
        objects.append((dmask, dcolor))

    region_order = [0, 2, 1]
    while len(region_order) < len(objects):
        region_order.append(rng.choice([0, 1, 2]))
    rng.shuffle(region_order)

    for (mask, color), region_idx in zip(objects, region_order):
        region_c0, region_c1, _bg = regions[region_idx]
        placed = _place_mask_patch(inp, mask, color, region_c0, region_c1, blocked, rng, padding=2)
        if placed is None:
            return None

    result = main(inp)
    if not np.array_equal(result, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
