# Combo variant of 2c181942 (index 25)
# combo_spec: D8/2+D13/2
# Name: 仅精灵输出+原位轮廓印记
#
# Design analysis
# ----------------
# a) Concrete behavior trace
#    Original task: detect the central hollow hub, match each scattered monochrome sprite
#    to the hub side with the same color, rotate the sprite into its unique valid attached
#    orientation, and output "hub + attached sprites" on a full-size canvas.
#    This combo changes two stages at once:
#    - D8/2: the hub itself must disappear from the output.
#    - D13/2: when a matched sprite leaves its scattered source position, it leaves behind
#      the 1-pixel outline of its source bounding rectangle, in the same color.
#    Example trace:
#    - Suppose a red sprite originally sits at rows 2..4, cols 10..12.
#    - Old output: the red sprite moves next to the red hub edge; the source area becomes bg.
#    - New output: the red sprite still appears in the attached position, but the source area
#      shows only a hollow red 3x3 rectangle. The hub is absent entirely.
#    - Final canvas is then tight-cropped around the union of attached sprites and outline marks.
#
# b) Complexity calibration
#    - The color-to-side pairing remains unchanged, so the task still requires understanding
#      the original sprite/hub correspondence.
#    - The output no longer contains the hub, which prevents the combo from being a trivial
#      "copy original output then add frames" variant.
#    - Outline marks are constrained to source bounding boxes, so they are visually regular and
#      distinguishable from the attached sprite bodies.
#    - Generator keeps sources near the central assembly and reserves the future attached region,
#      so the cropped output is visibly smaller than the input while staying readable.
#
# c) Implicit-assumption audit of the original solver
#    - `_find_hub()` assumes the input still contains an intact hub. This remains true because
#      only the output changes, not the input structure.
#    - The original `out` started with the hub already painted. That assumption is invalid here:
#      the output must omit the hub, so `out` must start as pure background.
#    - Component selection by `largest same-color 8-connected component outside hub` still works
#      because generator keeps exactly one matched sprite per hub color and distractors use fresh colors.
#    - Conflict checks previously used the hub as an occupied zone. We preserve that internal
#      occupancy canvas so attached sprites are still validated against hub geometry even though
#      the hub is not drawn into the final result.
#    - Source preservation is no longer "keep the whole sprite"; only the source bbox outline is kept.
#      Therefore generator must ensure source bboxes do not overlap attached placements, otherwise
#      the frame/body distinction becomes unreadable.
#
# d) Placement legality
#    - Attached sprites reuse the original exact side-contact geometry and in-bounds checks.
#    - Scattered source sprites are placed against a reservation canvas that already contains the
#      hub and all future attached sprites, preventing output-time overlaps between outline marks
#      and attached bodies.
#    - Tight crop is taken only after all output objects are painted, so no object is truncated.
#
# e) Visual observability
#    - In 2-4 examples, the solver can visibly infer: colors still determine the destination side,
#      the hub vanishes, and each moved sprite leaves a regular rectangular outline where it came from.
#    - The rectangular outline is a human-visible cue; it does not depend on numeric color ordering
#      or any hidden convention.
#
# f) Visual cleanliness
#    - The generator biases source placement to a moderate window around the hub so the crop does
#      not balloon back to near-input size.
#    - Reserving attached zones prevents overlapping layers of source outlines and attached sprites.
#
# g) Color inference
#    - No new recoloring rule is introduced. Colors still only express sprite-to-hub matching.
#    - The outline uses the same sprite color, which is structurally supported by the shared bbox.
#
# Post-change assumptions that generator must guarantee
#    - Every hub-edge color that is meant to match has exactly one scattered component outside the hub.
#    - The scattered source bbox of each matched sprite is disjoint from the future attached area.
#    - The output crop is genuinely active in most samples; otherwise D8/2 would visually degenerate.

from __future__ import annotations

import random

import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste
from scipy import ndimage as _ndi
from scipy.ndimage import label as _label


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(grid != int(bg))
    if pts.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(pts)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


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


def _draw_bbox_outline(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[int(r0), int(c0) : int(c1) + 1] = int(color)
    out[int(r1), int(c0) : int(c1) + 1] = int(color)
    out[int(r0) : int(r1) + 1, int(c0)] = int(color)
    out[int(r0) : int(r1) + 1, int(c1)] = int(color)


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


def _collect_attachment_records(g: np.ndarray):
    g = np.array(g, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    target = int(n - 2)
    occupancy = np.full_like(g, int(bg))
    occupancy[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]

    segs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }

    records = []
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

        spr_mask_full = labc == int(best_id)
        src_bbox = bbox_coords(np.argwhere(spr_mask_full))
        spr_arr = g.copy()
        spr_arr[~spr_mask_full] = int(bg)
        spr_arr, spr_mask = _crop_to_mask(spr_arr, spr_mask_full)

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
            if any(not (0 <= rr2 < H and 0 <= cc2 < W) for rr2, cc2 in placed):
                continue

            ok = True
            for rr2, cc2 in placed:
                val = int(a[int(rr2 - dr), int(cc2 - dc)])
                if val == int(bg):
                    continue
                if int(occupancy[int(rr2), int(cc2)]) != int(bg) and int(occupancy[int(rr2), int(cc2)]) != val:
                    ok = False
                    break
            if not ok:
                continue

            found = (a, msk, int(dr), int(dc))
            break

        if found is None:
            continue

        a, msk, dr, dc = found
        paste(occupancy, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)
        records.append(
            {
                "side": side,
                "color": int(col),
                "source_bbox": src_bbox,
                "attached_arr": a,
                "attached_mask": msk,
                "dr": int(dr),
                "dc": int(dc),
            }
        )

    return {
        "bg": int(bg),
        "hub": (int(r), int(c), int(n)),
        "records": records,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered sprites to hub-edge colors, rotate them into the attached positions,
    erase the hub from the output, and leave only source bbox outlines at their old locations.
    """
    solved = _collect_attachment_records(grid)
    bg = int(solved["bg"])
    g = np.array(grid, dtype=int)
    out = np.full_like(g, int(bg))

    for rec in solved["records"]:
        _draw_bbox_outline(out, rec["source_bbox"], rec["color"])
        paste(out, rec["attached_arr"], rec["dr"], rec["dc"], mask=rec["attached_mask"], overwrite=True, clip=True)

    return _tight_crop_non_bg(out, bg)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Build an 8-connected monochrome sprite in the attached orientation for `side`.
    It must satisfy:
    - correct symmetry for the intended contact axis
    - exactly `edge_len` contiguous contact pixels on the attachment side
    - exactly one valid rotation for the intended left-attach test
    - bbox not equal to 2x2, so the outline imprint remains visibly frame-like
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
        if grid.shape == (2, 2):
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


def _place_patch_in_window(
    canvas: np.ndarray,
    patch: np.ndarray,
    bg: int,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    rng: random.Random,
    *,
    padding: int = 0,
) -> tuple[int, int]:
    H, W = canvas.shape
    h, w = patch.shape
    row_min = max(0, int(row_min))
    col_min = max(0, int(col_min))
    row_max = min(int(row_max), int(H - h))
    col_max = min(int(col_max), int(W - w))
    if row_max < row_min or col_max < col_min:
        raise ValueError("no search window")

    occ = canvas != int(bg)
    if int(padding) > 0:
        occ = _ndi.binary_dilation(occ, structure=np.ones((3, 3), dtype=bool), iterations=int(padding))
    m = patch != int(bg)

    candidates: list[tuple[int, int]] = []
    for r0 in range(int(row_min), int(row_max) + 1):
        for c0 in range(int(col_min), int(col_max) + 1):
            if not bool((occ[int(r0) : int(r0 + h), int(c0) : int(c0 + w)] & m).any()):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        raise ValueError("no free location in window")
    return rng.choice(candidates)


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

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    max_extent = max(max_sprite_h, max_sprite_w)
    margin = max_extent + 3

    hub_r_min = margin
    hub_r_max = H - N - margin
    hub_c_min = margin
    hub_c_max = W - N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    hub_patch = np.full((N, N), bg, dtype=int)
    for ci in range(1, N - 1):
        hub_patch[0, ci] = top_col
        hub_patch[N - 1, ci] = bot_col
    for ri in range(1, N - 1):
        hub_patch[ri, 0] = left_col
        hub_patch[ri, N - 1] = right_col

    attached_only = np.full((H, W), bg, dtype=int)
    reservation = np.full((H, W), bg, dtype=int)
    paste(reservation, hub_patch, hub_r, hub_c, transparent=bg)

    attached_sprites = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        sh, sw = mask.shape
        color = int(side_colors[side])
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
                if mask[rr, cc] and reservation[int(sr + rr), int(sc + cc)] != bg:
                    return None

        paste(attached_only, arr, sr, sc, transparent=bg)
        paste(reservation, arr, sr, sc, transparent=bg)
        attached_sprites[side] = (arr, mask, int(sr), int(sc))

    inp = np.full((H, W), bg, dtype=int)
    paste(inp, hub_patch, hub_r, hub_c, transparent=bg)

    full_out = attached_only.copy()
    placement_canvas = reservation.copy()
    window_pad = max_extent + 6

    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, int(k))
        rotated_mask = np.rot90(mask, int(k))

        rows_any = rotated_mask.any(axis=1)
        cols_any = rotated_mask.any(axis=0)
        r0_idx, r1_idx = np.where(rows_any)[0][[0, -1]]
        c0_idx, c1_idx = np.where(cols_any)[0][[0, -1]]
        patch = np.where(
            rotated_mask[int(r0_idx) : int(r1_idx) + 1, int(c0_idx) : int(c1_idx) + 1],
            rotated_arr[int(r0_idx) : int(r1_idx) + 1, int(c0_idx) : int(c1_idx) + 1],
            bg,
        ).astype(int)

        ph, pw = patch.shape
        row_min = hub_r - window_pad - ph
        row_max = hub_r + N + window_pad
        col_min = hub_c - window_pad - pw
        col_max = hub_c + N + window_pad
        try:
            pr, pc = _place_patch_in_window(
                placement_canvas,
                patch,
                bg,
                row_min,
                row_max,
                col_min,
                col_max,
                rng,
                padding=2,
            )
        except ValueError:
            return None

        paste(placement_canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

        source_mask = patch != bg
        src_bbox = bbox_coords(np.argwhere(source_mask) + np.array([[int(pr), int(pc)]]))
        _draw_bbox_outline(full_out, src_bbox, side_colors[side])

    if rng.random() < 0.3:
        used_colors = set(side_colors.values())
        distractor_colors = [c for c in range(10) if c != bg and c not in used_colors]
        if distractor_colors:
            dcol = int(rng.choice(distractor_colors))
            dist_mask = _make_sprite_mask(edge_len, rng.choice(sides), rng)
            if dist_mask is not None:
                dist_arr = np.where(dist_mask, dcol, bg).astype(int)
                try:
                    pr, pc = _place_patch_in_window(
                        placement_canvas,
                        dist_arr,
                        bg,
                        0,
                        H - dist_arr.shape[0],
                        0,
                        W - dist_arr.shape[1],
                        rng,
                        padding=2,
                    )
                    paste(placement_canvas, dist_arr, pr, pc, transparent=bg)
                    paste(inp, dist_arr, pr, pc, transparent=bg)
                except ValueError:
                    pass

    out = _tight_crop_non_bg(full_out, bg)
    if out.size == 0:
        return None

    # Keep D8/2 visually active: the crop should usually be smaller than the input canvas.
    if out.shape == inp.shape:
        return None
    if out.shape[0] * out.shape[1] > int(0.9 * H * W):
        return None

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
