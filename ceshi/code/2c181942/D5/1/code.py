# Variant of 2c181942
# Dimension: D5 — 目标结构：hub的形状与设计 (dst)
# Change: NxN空心正方形(N=4)，角为背景色，内部为背景色，四条边各为一种不同颜色，位于网格中心附近... → 十字形hub
#
# Design analysis
# ---------------
# spec:
#   grid: 18-26 rows x 20-26 cols, bg=8
#   dst:
#     old_hub: 4x4 hollow square with 4 colored sides, contact length = 2
#     new_hub: 6x6 thick cross made from five 2x2 blocks:
#       "..TT.."
#       "..TT.."
#       "LLCCRR"
#       "LLCCRR"
#       "..BB.."
#       "..BB.."
#     semantics:
#       - T/B/L/R are four distinct arm colors; these still drive sprite matching
#       - C is a fifth neutral hub-center color used only to keep the cross 4-connected
#       - the four 2-cell outer faces of the arms are the new attachment sites
#   src:
#     unchanged conceptually: 2-4 monochrome 8-connected sprites + optional distractor
#     each matching sprite keeps the original constraints:
#       - contact edge length exactly 2
#       - unique valid rotation for its intended side
#       - mirror symmetry across the contact axis
#   output:
#     same as original pipeline, except sprites attach flush to the arm tips instead of square sides
#
# Behavior trace:
# - Original top side attachment used the hollow square's segment at row r, cols c+1..c+2.
# - New top attachment uses the top arm tip at row r, cols c+2..c+3.
# - The sprite rule stays visible: find the same-colored scattered sprite, rotate until its 2-cell
#   contact edge aligns with that tip, then translate it to touch the hub. The difference is purely
#   structural/positional: square perimeter -> radial cross arms.
#
# Complexity calibration:
# - Contact length stays 2 so the original sprite disambiguation logic still matters.
# - The neutral 2x2 center is necessary; without it the four arm colors cannot form one connected cross.
# - Only one new degree of freedom is added: hub silhouette. Matching colors, sprite count, and scatter
#   strategy stay in the original range, so the task is still learnable from few examples.
#
# Old implicit assumptions that break under D5:
# - `_find_hub()` assumed bg corners + bg interior inside one square bbox.
# - `target = n - 2` assumed contact sites were the side segments of that square.
# - `segs[...]` assumed attachment rows/cols were just outside a square perimeter.
# These are replaced by an explicit 6x6 cross detector and four arm-tip contact specs.
#
# Placement legality / observability:
# - Generator is still output-first: build final attached scene, then scatter rotated sprites.
# - Hub margin is based on the largest sprite extent so attached output never clips.
# - Scattered sprites use `random_free_location(..., padding=2)` to stay separated from hub/others.
# - The cross occupies 20 cells inside a 6x6 bbox and has four empty 2x2 corners, so it is visually
#   distinct from the original hollow square in every sample.
#
# New solver assumptions and generator guarantees:
# - Exactly one near-center 6x6 thick cross exists.
# - Its four arm blocks are 2x2, uniform, pairwise distinct, and non-bg.
# - The center block is a uniform non-bg color distinct from the arm colors.
# - Matching sprites still satisfy the original symmetry/contact-edge constraints.
# - Distractors never reuse arm colors.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label as _label

for _parent in Path(__file__).resolve().parents:
    _cand = _parent / "solvers" / "arc_utils.py"
    if _cand.exists():
        sys.path.append(str(_cand.parent))
        break

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


HUB_SIZE = 6
BLOCK = 2
ARM_FACE = 2
SIDES = ("top", "bot", "left", "right")


def _trim_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


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


def _hub_cells(r: int, c: int) -> dict[str, tuple[slice, slice]]:
    r = int(r)
    c = int(c)
    return {
        "top": (slice(r, r + 2), slice(c + 2, c + 4)),
        "left": (slice(r + 2, r + 4), slice(c, c + 2)),
        "center": (slice(r + 2, r + 4), slice(c + 2, c + 4)),
        "right": (slice(r + 2, r + 4), slice(c + 4, c + 6)),
        "bot": (slice(r + 4, r + 6), slice(c + 2, c + 4)),
    }


def _hub_mask(shape: tuple[int, int], r: int, c: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rs, cs in _hub_cells(r, c).values():
        mask[rs, cs] = True
    return mask


def _contact_specs(r: int, c: int, top_col: int, bot_col: int, left_col: int, right_col: int):
    return {
        "top": (int(top_col), ("bot", int(r - 1), [int(c + 2), int(c + 3)])),
        "bot": (int(bot_col), ("top", int(r + HUB_SIZE), [int(c + 2), int(c + 3)])),
        "left": (int(left_col), ("right", int(c - 1), [int(r + 2), int(r + 3)])),
        "right": (int(right_col), ("left", int(c + HUB_SIZE), [int(r + 2), int(r + 3)])),
    }


def _find_cross_hub(g: np.ndarray, bg: int):
    H, W = g.shape
    best = None
    for r in range(0, int(H - HUB_SIZE + 1)):
        for c in range(0, int(W - HUB_SIZE + 1)):
            # The four 2x2 corners must be background so the 6x6 bbox visibly reads as a cross.
            if not bool((g[r : r + 2, c : c + 2] == int(bg)).all()):
                continue
            if not bool((g[r : r + 2, c + 4 : c + 6] == int(bg)).all()):
                continue
            if not bool((g[r + 4 : r + 6, c : c + 2] == int(bg)).all()):
                continue
            if not bool((g[r + 4 : r + 6, c + 4 : c + 6] == int(bg)).all()):
                continue

            cells = _hub_cells(r, c)
            block_colors: dict[str, int] = {}
            ok = True
            for name, (rs, cs) in cells.items():
                block = g[rs, cs]
                vals = set(map(int, block.flatten()))
                if len(vals) != 1:
                    ok = False
                    break
                val = int(next(iter(vals)))
                if val == int(bg):
                    ok = False
                    break
                block_colors[name] = val
            if not ok:
                continue

            arm_cols = [block_colors["top"], block_colors["bot"], block_colors["left"], block_colors["right"]]
            if len(set(arm_cols)) != 4:
                continue
            if block_colors["center"] in set(arm_cols):
                continue

            cr = float(r + (HUB_SIZE - 1) / 2)
            cc = float(c + (HUB_SIZE - 1) / 2)
            score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
            cand = (
                float(score),
                (int(r), int(c)),
                (
                    int(block_colors["top"]),
                    int(block_colors["bot"]),
                    int(block_colors["left"]),
                    int(block_colors["right"]),
                    int(block_colors["center"]),
                ),
            )
            if best is None or float(cand[0]) < float(best[0]):
                best = cand
    return best


def _rotation_valid_for_side(mask: np.ndarray, need_edge: str) -> bool:
    trimmed_mask = _trim_to_mask(mask.astype(int), mask.astype(bool))[1]
    if need_edge in ("left", "right"):
        if not bool(np.array_equal(trimmed_mask, np.flipud(trimmed_mask))):
            return False
    else:
        if not bool(np.array_equal(trimmed_mask, np.fliplr(trimmed_mask))):
            return False
    seg = _edge_segment(trimmed_mask, need_edge)
    if seg is None:
        return False
    if need_edge in ("left", "right"):
        _typ, _edge_c, rows = seg
        return len(rows) == ARM_FACE and _is_contig(rows)
    _typ, _edge_r, cols = seg
    return len(cols) == ARM_FACE and _is_contig(cols)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_cross_hub(g, bg)
    assert hub is not None

    _score, (r, c), (top_col, bot_col, left_col, right_col, _center_col) = hub
    hub_mask = _hub_mask((H, W), r, c)

    out = np.full_like(g, int(bg))
    out[hub_mask] = g[hub_mask]

    segs = _contact_specs(r, c, top_col, bot_col, left_col, right_col)

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
        spr_arr, spr_mask = _trim_to_mask(spr_arr, spr_mask)

        found = None
        for krot in range(4):
            a = np.rot90(spr_arr, int(krot))
            msk = np.rot90(spr_mask, int(krot))
            a, msk = _trim_to_mask(a, msk)

            if not _rotation_valid_for_side(msk, need_edge):
                continue

            seg = _edge_segment(msk, need_edge)
            assert seg is not None
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

        a, msk, dr, dc = found
        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)

    return out


def _run_lengths(bits: np.ndarray) -> list[int]:
    runs: list[int] = []
    count = 0
    for v in bits:
        if bool(v):
            count += 1
        elif count > 0:
            runs.append(int(count))
            count = 0
    if count > 0:
        runs.append(int(count))
    return runs


def _canonical_left_masks() -> list[np.ndarray]:
    raw = [
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
        ],
        [
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        [
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        ],
        [
            [1, 0, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 0],
        ],
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
    ]
    out: list[np.ndarray] = []
    for grid in raw:
        mask = np.array(grid, dtype=bool)
        px = int(mask.sum())
        if px < 4 or px > 12:
            continue
        if not bool(np.array_equal(mask, np.flipud(mask))):
            continue
        if bool(np.array_equal(mask, np.fliplr(mask))):
            continue
        labeled, num = _label(mask, structure=np.ones((3, 3), dtype=int))
        if int(num) != 1:
            continue
        if _run_lengths(mask[:, -1]) != [ARM_FACE]:
            continue
        valid_rots = [k for k in range(4) if _rotation_valid_for_side(np.rot90(mask, k), "right")]
        if valid_rots != [0]:
            continue
        out.append(mask)
    assert out
    return out


def _orient_from_left(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return np.rot90(mask, -1)
    return np.rot90(mask, 1)


def _make_sprite_mask(side: str, rng: random.Random) -> np.ndarray:
    base = rng.choice(_canonical_left_masks())
    return _orient_from_left(base, side)


def _attach_coords(mask: np.ndarray, side: str, hub_r: int, hub_c: int) -> tuple[int, int]:
    sh, sw = mask.shape
    if side == "top":
        filled = np.where(mask[-1])[0]
        return int(hub_r - sh), int(hub_c + 2 - int(filled[0]))
    if side == "bot":
        filled = np.where(mask[0])[0]
        return int(hub_r + HUB_SIZE), int(hub_c + 2 - int(filled[0]))
    if side == "left":
        filled = np.where(mask[:, -1])[0]
        return int(hub_r + 2 - int(filled[0])), int(hub_c - sw)
    filled = np.where(mask[:, 0])[0]
    return int(hub_r + 2 - int(filled[0])), int(hub_c + HUB_SIZE)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col, center_col = all_colors[:5]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    K = rng.randint(2, 4)
    sprite_sides = rng.sample(list(SIDES), K)
    sprite_masks = {side: _make_sprite_mask(side, rng) for side in sprite_sides}

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    max_extent = max(max_sprite_h, max_sprite_w)
    margin = max_extent + 2

    hub_r_min = margin
    hub_r_max = H - HUB_SIZE - margin
    hub_c_min = margin
    hub_c_max = W - HUB_SIZE - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    cells = _hub_cells(hub_r, hub_c)
    for side, col in side_colors.items():
        rs, cs = cells[side]
        out[rs, cs] = int(col)
    rs, cs = cells["center"]
    out[rs, cs] = int(center_col)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        color = int(side_colors[side])
        arr = np.where(mask, color, bg).astype(int)
        sr, sc = _attach_coords(mask, side, hub_r, hub_c)
        sh, sw = mask.shape

        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr in range(sh):
            for cc in range(sw):
                if mask[rr, cc] and out[sr + rr, sc + cc] != bg:
                    return None

        paste(out, arr, sr, sc, transparent=bg)
        attached_sprites[side] = (arr, mask, sr, sc)

    inp = np.full((H, W), bg, dtype=int)
    hub_mask = _hub_mask((H, W), hub_r, hub_c)
    inp[hub_mask] = out[hub_mask]
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, k)
        rotated_mask = np.rot90(mask, k)
        cropped_arr, cropped_mask = _trim_to_mask(rotated_arr, rotated_mask)
        patch = np.where(cropped_mask, cropped_arr, bg).astype(int)

        try:
            pr, pc = random_free_location(canvas, patch, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    if rng.random() < 0.3:
        used_colors = set(side_colors.values()) | {int(center_col)}
        distractor_colors = [c for c in range(10) if c != bg and c not in used_colors]
        if distractor_colors:
            dcol = int(rng.choice(distractor_colors))
            dmask = _make_sprite_mask(rng.choice(list(SIDES)), rng)
            darr = np.where(dmask, dcol, bg).astype(int)
            try:
                pr, pc = random_free_location(canvas, darr, transparent=bg, padding=2, padding_connectivity=8)
                paste(canvas, darr, pr, pc, transparent=bg)
                paste(inp, darr, pr, pc, transparent=bg)
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
