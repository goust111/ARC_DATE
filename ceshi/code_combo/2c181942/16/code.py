# Combo variant of 2c181942 (index 16)
# combo_spec: D13/0+D14/0
# Name: 保留源位置+颜色序处理

from __future__ import annotations

import random

import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste
from scipy.ndimage import label as _label

"""
Design analysis for combo D13/0 + D14/0
=======================================

a) Concretization and traced behavior
-------------------------------------
- Base task:
  one scattered sprite of a hub-edge color is rotated and attached to the matching
  side; the source position disappears.
- D13 alone:
  the same matched sprite is copied instead of moved, so the output contains both
  the attached copy and the original scattered copy.
- This combo adds D14 sequential overwrite:
  matched sprites are processed in ascending color order; each step keeps the
  source copy and adds the attached copy, and later steps overwrite any earlier
  non-hub pixels they land on.
- Concrete trace:
  assume the sorted matched colors are blue(1), red(2), green(3), yellow(4).
  The red source is deliberately scattered on top of part of the blue attachment
  zone. The green source is scattered on top of part of the red attachment zone.
  Then:
    1. process blue: keep blue source, attach blue to hub
    2. process red: keep red source; where it overlaps the already-drawn blue
       attached copy, red overwrites blue at those source cells; then attach red
    3. process green: same idea over part of the red attached zone
    4. process yellow likewise
  Difference from the source task:
  output is no longer just "hub + attached sprites", and it is not even the plain
  D13 union. The order of matched colors changes visible pixels because preserved
  source copies are allowed to cover earlier attached copies.

b) Complexity calibration
-------------------------
- To keep the D14 effect visible, the generator always uses all 4 hub colors and
  fixed palette colors {1,2,3,4}. This makes the global precedence relation
  repeat across examples instead of depending on arbitrary fresh colors.
- The overlap is only between a later sprite's preserved source and an earlier
  sprite's attached copy. Input sprites themselves never overlap, so the source
  shapes remain readable and recoverable.
- Only 3 sequential conflicts are introduced at most, one per later color, which
  is enough to make order matter without turning the board into a dense pileup.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- Original assumption: `out` starts as hub-only because sources disappear.
  Broken by D13; now matched sources must be pasted back into the output.
- Original assumption: attached placements must avoid all existing non-bg pixels
  in `out`. Broken by this combo; later steps are supposed to overwrite earlier
  non-hub pixels, so conflicts with previous attached/source copies are now legal.
  Only overlap with the hub itself must remain illegal.
- Original assumption: side iteration order is irrelevant.
  Broken by D14; the loop must be sorted by color, otherwise overlaps are resolved
  inconsistently.
- Original assumption: largest same-color component is enough.
  Still valid because the generator keeps exactly one matched source component per
  hub color, and any distractor uses a fresh non-hub color.

d) Placement legality
---------------------
- The generator is still output-first with separately recorded attached placements.
- Input sources are scattered one by one. They must stay in-bounds, avoid the hub,
  and avoid each other in the input.
- For D14 visibility, later-color sources are searched near earlier attached zones
  until they overlap those zones in the eventual output by at least one pixel, but
  not completely; this keeps both objects partially visible.
- Attached copies remain the original legal hub-flush placements, so the solver's
  rotation/contact logic still applies unchanged.

e) Visual observability
-----------------------
- Humans can see three stable facts from a few examples:
  1. every matched sprite survives at its source position,
  2. every matched sprite also appears attached to the hub,
  3. some source copies sit on top of earlier attached copies, so later colors
     visibly dominate in those overlap pockets.
- The precedence is learnable because the same small palette subset {1,2,3,4} is
  reused repeatedly; the examples consistently show which hue covers which.

f) Visual cleanliness
---------------------
- The generator never overlaps source sprites in the input.
- Overwrite interactions are localized to thin pockets around the hub-adjacent
  attachment zones rather than across the entire board.
- Distractors are omitted to avoid diluting the only new signal, which is
  "duplicate the sprite and resolve overlaps by ordered processing".

g) New solver assumptions and generator guarantees
--------------------------------------------------
- `main()` now assumes at least one non-hub overlap may exist; generator enforces
  that ascending-order output differs from descending-order output.
- `main()` assumes matched source components never cover hub cells in the input;
  generator checks this during source placement.
- `main()` assumes the hub colors come from a repeated small set whose precedence
  is stable across examples; generator fixes that set to {1,2,3,4}.
"""


ORDERED_COLORS = [1, 2, 3, 4]


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


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
    best = None
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


def _sorted_side_specs(r: int, c: int, n: int, colors: tuple[int, int, int, int]):
    top_col, bot_col, left_col, right_col = colors
    specs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }
    return sorted(specs.items(), key=lambda kv: int(kv[1][0]))


def _extract_largest_component(
    g: np.ndarray, hub_mask: np.ndarray, bg: int, color: int
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
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

    full_mask = labc == int(best_id)
    pts = np.argwhere(full_mask)
    src_r0, src_c0, _src_r1, _src_c1 = bbox_coords(pts)
    arr = g.copy()
    arr[~full_mask] = int(bg)
    arr, mask = _crop_to_mask(arr, full_mask)
    return arr, mask, int(src_r0), int(src_c0)


def _find_attachment(
    arr0: np.ndarray,
    mask0: np.ndarray,
    *,
    need_edge: str,
    fixed: int,
    span: list[int],
    target: int,
    H: int,
    W: int,
    hub_mask: np.ndarray,
    bg: int,
):
    for krot in range(4):
        a = np.rot90(arr0, int(krot))
        msk = np.rot90(mask0, int(krot))
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

        placed = [(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)]
        if any(not (0 <= rr2 < H and 0 <= cc2 < W) for rr2, cc2 in placed):
            continue
        if any(bool(hub_mask[int(rr2), int(cc2)]) for rr2, cc2 in placed):
            continue
        return a, msk, int(dr), int(dc)
    return None


def _render_with_order(grid: np.ndarray, *, ascending: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None
    _score, (r, c, n), colors = hub

    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    ordered_specs = _sorted_side_specs(int(r), int(c), int(n), colors)
    if not ascending:
        ordered_specs = list(reversed(ordered_specs))

    out = np.full_like(g, int(bg))
    out[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]
    target = int(n - 2)

    for _side, (col, (need_edge, fixed, span)) in ordered_specs:
        src = _extract_largest_component(g, hub_mask, bg, int(col))
        if src is None:
            continue
        spr_arr, spr_mask, src_r0, src_c0 = src
        found = _find_attachment(
            spr_arr,
            spr_mask,
            need_edge=str(need_edge),
            fixed=int(fixed),
            span=list(span),
            target=int(target),
            H=int(H),
            W=int(W),
            hub_mask=hub_mask,
            bg=int(bg),
        )
        if found is None:
            continue

        # D13: preserve the matched source component at its original position.
        paste(out, spr_arr, int(src_r0), int(src_c0), mask=spr_mask, overwrite=True, clip=True)

        # D14: because we process in color order, later paste operations are allowed
        # to overwrite earlier non-hub pixels.
        a, msk, dr, dc = found
        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _render_with_order(grid, ascending=True)


def _mask_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + rr), int(c0 + cc)) for rr, cc in np.argwhere(mask)}


def _mask_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    return np.where(mask, int(color), int(bg)).astype(int)


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    e = edge_len
    h = 4
    w = rng.randint(2, 4)
    half_h = (h + 1) // 2

    for _attempt in range(120):
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
        grid = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

        if grid.shape[0] < 2 or grid.shape[1] < 2:
            continue
        if not np.array_equal(grid, np.flipud(grid)):
            continue
        if np.array_equal(grid, np.fliplr(grid)):
            continue

        labeled, num = _label(grid, structure=np.ones((3, 3), dtype=int))
        if int(num) != 1:
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


def _attached_origin(mask: np.ndarray, side: str, hub_r: int, hub_c: int, N: int) -> tuple[int, int]:
    sh, sw = mask.shape
    if side == "top":
        filled = np.where(mask[-1])[0]
        return int(hub_r - sh), int(hub_c + 1 - int(filled[0]))
    if side == "bot":
        filled = np.where(mask[0])[0]
        return int(hub_r + N), int(hub_c + 1 - int(filled[0]))
    if side == "left":
        filled = np.where(mask[:, -1])[0]
        return int(hub_r + 1 - int(filled[0])), int(hub_c - sw)
    filled = np.where(mask[:, 0])[0]
    return int(hub_r + 1 - int(filled[0])), int(hub_c + N)


def _random_rotated_patch(
    mask: np.ndarray, color: int, bg: int, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    arr = _mask_patch(mask, int(color), int(bg))
    k = rng.choice([1, 2, 3])
    rot_arr = np.rot90(arr, int(k))
    rot_mask = np.rot90(mask, int(k))
    rows_any = rot_mask.any(axis=1)
    cols_any = rot_mask.any(axis=0)
    r0_idx, r1_idx = np.where(rows_any)[0][[0, -1]]
    c0_idx, c1_idx = np.where(cols_any)[0][[0, -1]]
    return (
        rot_arr[int(r0_idx) : int(r1_idx) + 1, int(c0_idx) : int(c1_idx) + 1],
        rot_mask[int(r0_idx) : int(r1_idx) + 1, int(c0_idx) : int(c1_idx) + 1],
    )


def _can_place_input_patch(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    r0: int,
    c0: int,
    blocked_cells: set[tuple[int, int]],
) -> bool:
    H, W = canvas.shape
    bg = int(get_bg(canvas))
    for rr, cc in np.argwhere(patch_mask):
        gr = int(r0 + rr)
        gc = int(c0 + cc)
        if not (0 <= gr < H and 0 <= gc < W):
            return False
        if (gr, gc) in blocked_cells:
            return False
        if int(canvas[gr, gc]) != int(bg):
            return False
    return True


def _find_random_source_location(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    blocked_cells: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    H, W = canvas.shape
    ph, pw = patch_mask.shape
    candidates = []
    for r0 in range(0, int(H - ph + 1)):
        for c0 in range(0, int(W - pw + 1)):
            if not _can_place_input_patch(canvas, patch_mask, int(r0), int(c0), blocked_cells):
                continue
            candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    return rng.choice(candidates)


def _find_overlap_source_location(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    anchor_cells: set[tuple[int, int]],
    anchor_r0: int,
    anchor_c0: int,
    blocked_cells: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int, int] | None:
    H, W = canvas.shape
    ph, pw = patch_mask.shape
    candidates = []
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            r0 = int(anchor_r0 + dr)
            c0 = int(anchor_c0 + dc)
            if r0 < 0 or c0 < 0 or r0 + ph > H or c0 + pw > W:
                continue
            if not _can_place_input_patch(canvas, patch_mask, int(r0), int(c0), blocked_cells):
                continue
            placed = _mask_cells(patch_mask, int(r0), int(c0))
            overlap = int(len(placed & anchor_cells))
            if 1 <= overlap < int(len(placed)) and overlap < int(len(anchor_cells)):
                dist = abs(int(dr)) + abs(int(dc))
                candidates.append((overlap, -dist, int(r0), int(c0)))
    if not candidates:
        return None
    best_overlap = max(int(item[0]) for item in candidates)
    pool = [item for item in candidates if int(item[0]) == int(best_overlap)]
    _ov, _dist, r0, c0 = rng.choice(pool)
    return int(r0), int(c0), int(best_overlap)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    N = 4
    edge_len = N - 2

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    shuffled = ORDERED_COLORS.copy()
    rng.shuffle(shuffled)
    top_col, bot_col, left_col, right_col = shuffled
    sides = ["top", "bot", "left", "right"]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(int(m.shape[0]) for m in sprite_masks.values())
    max_sprite_w = max(int(m.shape[1]) for m in sprite_masks.values())
    margin = max(max_sprite_h, max_sprite_w) + 3

    hub_r_min = margin
    hub_r_max = H - N - margin
    hub_c_min = margin
    hub_c_max = W - N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    hub_only = np.full((H, W), bg, dtype=int)
    for ci in range(1, N - 1):
        hub_only[int(hub_r), int(hub_c + ci)] = int(top_col)
        hub_only[int(hub_r + N - 1), int(hub_c + ci)] = int(bot_col)
    for ri in range(1, N - 1):
        hub_only[int(hub_r + ri), int(hub_c)] = int(left_col)
        hub_only[int(hub_r + ri), int(hub_c + N - 1)] = int(right_col)
    blocked_cells = {
        (int(rr), int(cc))
        for rr in range(int(hub_r), int(hub_r + N))
        for cc in range(int(hub_c), int(hub_c + N))
    }

    attached_sprites: dict[str, dict[str, object]] = {}
    scene = hub_only.copy()
    for side in sides:
        mask = sprite_masks[side]
        color = int(side_colors[side])
        arr = _mask_patch(mask, color, bg)
        sr, sc = _attached_origin(mask, side, int(hub_r), int(hub_c), int(N))
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr, cc in np.argwhere(mask):
            if int(scene[int(sr + rr), int(sc + cc)]) != int(bg):
                return None
        paste(scene, arr, int(sr), int(sc), transparent=bg)
        attached_sprites[side] = {
            "arr": arr,
            "mask": mask,
            "sr": int(sr),
            "sc": int(sc),
            "cells": _mask_cells(mask, int(sr), int(sc)),
        }

    inp = hub_only.copy()
    canvas = hub_only.copy()

    ordered_sides = [side for side, _info in sorted(side_colors.items(), key=lambda kv: int(kv[1]))]
    overlap_counts: list[int] = []

    for idx, side in enumerate(ordered_sides):
        color = int(side_colors[side])
        patch, patch_mask = _random_rotated_patch(sprite_masks[side], color, bg, rng)

        if idx == 0:
            loc = _find_random_source_location(canvas, patch_mask, blocked_cells, rng)
            if loc is None:
                return None
            pr, pc = loc
        else:
            anchor_side = ordered_sides[idx - 1]
            anchor = attached_sprites[anchor_side]
            loc = _find_overlap_source_location(
                canvas,
                patch_mask,
                anchor["cells"],
                int(anchor["sr"]),
                int(anchor["sc"]),
                blocked_cells,
                rng,
            )
            if loc is None:
                return None
            pr, pc, overlap = loc
            overlap_counts.append(int(overlap))

        paste(canvas, patch, int(pr), int(pc), transparent=bg)
        paste(inp, patch, int(pr), int(pc), transparent=bg)

    combo = _render_with_order(inp, ascending=True)
    reverse = _render_with_order(inp, ascending=False)
    if np.array_equal(combo, reverse):
        return None
    if len(overlap_counts) != 3 or any(int(v) <= 0 for v in overlap_counts):
        return None
    if not np.array_equal(main(inp), combo):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
