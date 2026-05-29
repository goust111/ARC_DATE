# Combo variant of 2c181942 (index 24)
# combo_spec: D9/0+D14/0
# Name: 空间排列顺序+颜色序处理

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste

"""
Design analysis for the combo:

a) Concrete behavior trace:
   Original task: edge color selects the sprite, then each sprite is attached independently.
   This combo removes that coupling. In the input, the 4 sprites are stacked from top to
   bottom and mean [top, right, bottom, left]. Example: if the highest sprite is yellow and
   the hub's top edge is maroon, yellow still goes to the top because row order, not color,
   determines the side. After side assignment, sprites are pasted in ascending sprite-color
   order, so later colors overwrite earlier ones at overlap cells near hub corners.

b) Complexity calibration:
   The variant needs both changes to be visible. To avoid degeneration:
   - hub colors and sprite colors come from disjoint palettes, so color matching is always wrong;
   - all 4 sides are always present, so the top/right/bottom/left row code has repeated support;
   - attached sprites overlap at exactly the 4 exterior corner cells, making overwrite order visible
     without turning the grid into a cluttered pile-up.

c) Hidden-assumption audit of the original solver:
   - "for each hub-edge color, find matching sprite": no longer valid; replaced with ordered external
     components sorted by vertical position.
   - "order of pasting is irrelevant": no longer valid; overlap is intentional, so placements are
     computed first and only then composited by ascending sprite color.
   - "reject overlap with already-drawn sprites": no longer valid; only overlap with the hub itself is
     forbidden, sprite-sprite overlap is allowed and required.

d) Placement legality:
   The generator is output-first. It places the hub, computes each attached sprite location exactly,
   verifies bounds and non-overlap with the hub interior/frame, then builds the input by detaching the
   same sprites into 4 non-overlapping row slots on the left side.

e) Visual observability:
   The position code is human-visible because the sprites form a clear top->bottom ordered column.
   The overwrite rule is human-visible because the same sprite colors repeat across samples from the
   fixed palette [1,2,3,4], and corner ownership changes according to those repeated color precedences.

f) Visual cleanliness:
   Overlap is limited to one cell per hub corner. This is enough to make order matter while keeping
   each sprite's body readable.

g) Inference support:
   Color ordering alone would be too implicit if the palette varied arbitrarily. The generator therefore
   always uses the same 4 sprite colors, merely permuted across sides, so repeated examples reveal a
   stable precedence relation between visible colors.

New solver assumptions introduced by this combo, and how the generator guarantees them:
 - Exactly 4 external monochrome sprites exist: generator always creates 4 and no distractors.
 - Their vertical ordering is strict and readable: generator places them into 4 separated row slots.
 - Sprite colors are distinct: generator assigns a permutation of [1,2,3,4].
 - Attached placements may overlap each other but never the hub: generator and solver use the same
   attachment geometry and only permit overlaps outside the hub.
"""


SIDE_ORDER = ["top", "right", "bot", "left"]
SPRITE_COLORS = [1, 2, 3, 4]
HUB_COLORS = [5, 6, 7, 9]
BG = 8
HUB_N = 4

# Canonical "left-attach" mask:
# - flipud symmetric
# - right edge has exactly 2 contiguous filled cells
# - not fliplr symmetric
# - large enough that adjacent attached sprites overlap at exterior hub corners
BASE_LEFT_MASK = np.array(
    [
        [1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0],
    ],
    dtype=bool,
)


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _find_hub(g: np.ndarray, bg: int) -> tuple[float, tuple[int, int, int], tuple[int, int, int, int]] | None:
    H, W = g.shape
    best = None
    for n in range(3, 7):
        for r in range(H - n + 1):
            for c in range(W - n + 1):
                if g[r, c] != bg or g[r, c + n - 1] != bg or g[r + n - 1, c] != bg or g[r + n - 1, c + n - 1] != bg:
                    continue
                inner = g[r + 1 : r + n - 1, c + 1 : c + n - 1]
                if not bool((inner == bg).all()):
                    continue
                top = g[r, c + 1 : c + n - 1]
                bot = g[r + n - 1, c + 1 : c + n - 1]
                left = g[r + 1 : r + n - 1, c]
                right = g[r + 1 : r + n - 1, c + n - 1]
                if bool((top == bg).any()) or bool((bot == bg).any()) or bool((left == bg).any()) or bool((right == bg).any()):
                    continue
                if len(set(map(int, top))) != 1 or len(set(map(int, bot))) != 1 or len(set(map(int, left))) != 1 or len(set(map(int, right))) != 1:
                    continue
                cr = r + (n - 1) / 2
                cc = c + (n - 1) / 2
                score = abs(cr - (H - 1) / 2) + abs(cc - (W - 1) / 2)
                cand = (float(score), (int(r), int(c), int(n)), (int(top[0]), int(bot[0]), int(left[0]), int(right[0])))
                if best is None or cand[0] < best[0]:
                    best = cand
    return best


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


def _mask_for_side(side: str) -> np.ndarray:
    if side == "left":
        return BASE_LEFT_MASK.copy()
    if side == "right":
        return np.fliplr(BASE_LEFT_MASK)
    if side == "top":
        return np.rot90(BASE_LEFT_MASK, -1)
    if side == "bot":
        return np.rot90(BASE_LEFT_MASK, 1)
    raise ValueError(f"unknown side: {side}")


def _attached_sprite(color: int, side: str, hub_r: int, hub_c: int, n: int, bg: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    mask = _mask_for_side(side)
    arr = np.where(mask, int(color), int(bg)).astype(int)
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
    return arr, mask, int(sr), int(sc)


def _extract_external_sprites(g: np.ndarray, bg: int, hub_mask: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    colors = sorted(set(map(int, g[~hub_mask].flatten())) - {int(bg)})
    for col in colors:
        m = (g == int(col)) & (~hub_mask)
        if not bool(m.any()):
            continue
        lab, nc = label_mask(m, connectivity=8)
        for cid in range(1, int(nc) + 1):
            comp_mask = lab == int(cid)
            if int(comp_mask.sum()) <= 0:
                continue
            pts = np.argwhere(comp_mask)
            r0, c0, r1, c1 = bbox_coords(pts)
            arr = np.full_like(g, int(bg))
            arr[comp_mask] = int(col)
            arr, comp_mask = _crop_to_mask(arr, comp_mask)
            sprites.append(
                {
                    "color": int(col),
                    "area": int(pts.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "center_r": float(r0 + r1) / 2.0,
                    "arr": arr,
                    "mask": comp_mask,
                }
            )
    return sprites


def _solve_attachment(
    arr: np.ndarray,
    mask: np.ndarray,
    side: str,
    hub_r: int,
    hub_c: int,
    n: int,
    hub_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    target = int(n - 2)
    specs = {
        "top": ("bot", int(hub_r - 1), list(range(int(hub_c + 1), int(hub_c + n - 1)))),
        "bot": ("top", int(hub_r + n), list(range(int(hub_c + 1), int(hub_c + n - 1)))),
        "left": ("right", int(hub_c - 1), list(range(int(hub_r + 1), int(hub_r + n - 1)))),
        "right": ("left", int(hub_c + n), list(range(int(hub_r + 1), int(hub_r + n - 1)))),
    }
    need_edge, fixed, span = specs[side]
    H, W = hub_mask.shape
    for krot in range(4):
        a = np.rot90(arr, int(krot))
        msk = np.rot90(mask, int(krot))
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
        placed = {(int(rr + dr), int(cc + dc)) for rr, cc in np.argwhere(msk)}
        if any(not (0 <= pr < H and 0 <= pc < W) for pr, pc in placed):
            continue
        if any(bool(hub_mask[pr, pc]) for pr, pc in placed):
            continue
        return a, msk, int(dr), int(dc)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    hub = _find_hub(g, bg)
    assert hub is not None

    _score, (hub_r, hub_c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[hub_r : hub_r + n, hub_c : hub_c + n] = True

    out = np.full_like(g, int(bg))
    out[hub_r : hub_r + n, hub_c : hub_c + n] = g[hub_r : hub_r + n, hub_c : hub_c + n]

    sprites = _extract_external_sprites(g, bg, hub_mask)
    sprites = sorted(sprites, key=lambda d: (float(d["center_r"]), int(d["bbox"][1]), -int(d["area"])))
    if len(sprites) < 4:
        return out
    sprites = sprites[:4]

    planned = []
    for side, spr in zip(SIDE_ORDER, sprites):
        found = _solve_attachment(spr["arr"], spr["mask"], side, hub_r, hub_c, n, hub_mask)
        if found is None:
            continue
        a, msk, dr, dc = found
        planned.append({"color": int(spr["color"]), "arr": a, "mask": msk, "dr": int(dr), "dc": int(dc)})

    for item in sorted(planned, key=lambda d: int(d["color"])):
        paste(out, item["arr"], item["dr"], item["dc"], mask=item["mask"], overwrite=True, clip=True)

    return out


def _rotated_input_patch(arr: np.ndarray, mask: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    # Chosen so every input sprite keeps height 6, preserving the row-order code.
    if side in ("left", "right"):
        k = random.choice([1, 3])
    else:
        k = 2
    rot_arr = np.rot90(arr, int(k))
    rot_mask = np.rot90(mask, int(k))
    return _crop_to_mask(rot_arr, rot_mask)


def _build_output_and_layers(
    H: int,
    W: int,
    hub_r: int,
    hub_c: int,
    side_to_sprite_color: dict[str, int],
    side_to_hub_color: dict[str, int],
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray, int, int]]]:
    out = np.full((H, W), BG, dtype=int)
    for ci in range(1, HUB_N - 1):
        out[hub_r, hub_c + ci] = int(side_to_hub_color["top"])
        out[hub_r + HUB_N - 1, hub_c + ci] = int(side_to_hub_color["bot"])
    for ri in range(1, HUB_N - 1):
        out[hub_r + ri, hub_c] = int(side_to_hub_color["left"])
        out[hub_r + ri, hub_c + HUB_N - 1] = int(side_to_hub_color["right"])

    layers: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in SIDE_ORDER:
        arr, mask, sr, sc = _attached_sprite(int(side_to_sprite_color[side]), side, hub_r, hub_c, HUB_N, BG)
        sh, sw = mask.shape
        assert 0 <= sr and 0 <= sc and sr + sh <= H and sc + sw <= W
        hub_occ = np.zeros((H, W), dtype=bool)
        hub_occ[hub_r : hub_r + HUB_N, hub_c : hub_c + HUB_N] = True
        for rr, cc in np.argwhere(mask):
            assert not bool(hub_occ[sr + rr, sc + cc])
        layers[side] = (arr, mask, sr, sc)

    for side in sorted(SIDE_ORDER, key=lambda s: int(side_to_sprite_color[s])):
        arr, mask, sr, sc = layers[side]
        paste(out, arr, sr, sc, mask=mask, overwrite=True, clip=False)

    return out, layers


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    rng = random.Random()
    H = rng.randint(29, 33)
    W = rng.randint(24, 28)

    side_to_hub_color = dict(zip(SIDE_ORDER, rng.sample(HUB_COLORS, len(HUB_COLORS))))
    side_perm = rng.sample(SIDE_ORDER, len(SIDE_ORDER))
    side_to_sprite_color = {side: col for side, col in zip(side_perm, SPRITE_COLORS)}

    hub_r = H // 2 - 2
    hub_c = W // 2 + 2
    if hub_r < 6 or hub_c < 8:
        return None

    out, layers = _build_output_and_layers(H, W, hub_r, hub_c, side_to_sprite_color, side_to_hub_color)

    inp = np.full((H, W), BG, dtype=int)
    inp[hub_r : hub_r + HUB_N, hub_c : hub_c + HUB_N] = out[hub_r : hub_r + HUB_N, hub_c : hub_c + HUB_N]

    row_gap = 1
    slot_h = 6
    total_h = slot_h * 4 + row_gap * 3
    top_margin = rng.randint(1, max(1, H - total_h - 1))
    strip_c = rng.randint(1, 2)

    input_boxes = []
    for idx, side in enumerate(SIDE_ORDER):
        arr, mask, _sr, _sc = layers[side]
        patch_arr, patch_mask = _rotated_input_patch(arr, mask, side)
        ph, pw = patch_mask.shape
        if ph != slot_h:
            return None
        pr = top_margin + idx * (slot_h + row_gap)
        pc = strip_c + rng.randint(0, 1)
        if pr < 0 or pc < 0 or pr + ph > H or pc + pw > W:
            return None
        if pc + pw + 2 >= hub_c:
            return None
        for r0, c0, r1, c1 in input_boxes:
            if not (pr + ph - 1 < r0 or r1 < pr or pc + pw - 1 < c0 or c1 < pc):
                return None
        paste(inp, patch_arr, pr, pc, mask=patch_mask, overwrite=True, clip=False)
        input_boxes.append((pr, pc, pr + ph - 1, pc + pw - 1))

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    overlap_counts = {}
    occupancy: dict[tuple[int, int], list[str]] = {}
    for side in SIDE_ORDER:
        _arr, mask, sr, sc = layers[side]
        for rr, cc in np.argwhere(mask):
            occupancy.setdefault((sr + rr, sc + cc), []).append(side)
    overlap_cells = {pos: sides for pos, sides in occupancy.items() if len(sides) > 1}
    for pos, sides in overlap_cells.items():
        overlap_counts[pos] = tuple(sorted(int(side_to_sprite_color[s]) for s in sides))

    meta = {
        "side_to_sprite_color": {k: int(v) for k, v in side_to_sprite_color.items()},
        "side_to_hub_color": {k: int(v) for k, v in side_to_hub_color.items()},
        "overlap_cells": overlap_counts,
    }
    return inp, meta


def generate_input() -> np.ndarray:
    for _ in range(200):
        made = _gen_attempt()
        if made is not None:
            inp, _meta = made
            return inp
    raise RuntimeError("generate_input: failed after 200 attempts")

