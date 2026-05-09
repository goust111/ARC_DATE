# Combo variant of 2c181942 (index 37)
# combo_spec: D6/0+D12/1+D14/0
# Name: 多干扰+缺边消色+颜色序

from __future__ import annotations

import random

import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location
from scipy.ndimage import label as _label

# Design analysis for the combo variant
# a) Concrete behavior:
#    Start from the source task: a 4x4 hollow hub with four colored sides, plus scattered
#    monochrome sprites that can be rotated and attached to the matching side color.
#    This combo changes three stages at once:
#    - D6: input now always includes 2-3 extra distractor sprites in fresh non-hub colors.
#      They are built from the same sprite family, so the scene contains many plausible
#      candidates, but only sprites whose color repeats a hub edge are usable.
#    - D12: output no longer preserves unmatched hub edges. If a hub-side color never
#      appears outside the hub, that entire side is erased to bg in the output.
#    - D14: attached sprites are rendered in ascending color order. Because attached
#      sprites are intentionally chosen to overlap near outer hub corners, later colors
#      visibly overwrite earlier ones.
#    Example trace: suppose hub colors are blue/red/green/yellow, and only blue/green/yellow
#    have real sprites while red is missing. The input also contains 2-3 distractors in
#    grey/orange/maroon. The output paints only blue/green/yellow hub sides; the red hub
#    side disappears. Then the blue sprite is attached first, green second, yellow last.
#    If yellow and green overlap at a corner, yellow remains visible there.
# b) Complexity calibration:
#    The generator fixes exactly 3 real sprites and 2-3 distractors. That keeps the
#    D12 change visible in every sample (exactly one missing side) while still giving
#    enough real sprites for D14 overlaps and enough noise for D6 to matter.
#    To make the ordering inferable to humans, the hub palette is always drawn from the
#    same visible color set [1,2,3,4]. The repeated examples therefore expose a stable
#    visible precedence relation among hues instead of hiding the rule in arbitrary ids.
# c) Implicit-assumption review:
#    - `_find_hub()` still assumes the input hub itself has all four colored sides. The
#      combo keeps that true; only the output erases missing sides.
#    - The source solver copied the whole hub into `out` before matching sprites. That is
#      invalid here because D12 requires missing edges to vanish. We instead start from bg
#      and repaint only sides whose matching sprite was found.
#    - The source solver also assumed side processing order did not matter, and therefore
#      rejected overlaps with already-drawn sprites. D14 breaks that assumption. The new
#      solver first collects all legal side attachments, sorts them by color, and only then
#      renders with overwrite enabled.
#    - The source solver picked the largest same-color component before rotation checks.
#      With more clutter we instead scan all same-color components and keep the largest one
#      that actually has a legal attachment geometry.
# d) Placement legality:
#    Generation is still output-first. We first choose attached sprites, verify at least one
#    overlap among them, render the ordered output, then scatter rotated copies into free
#    input locations with padding. Every placement is bounds-checked, and the final sample
#    is oracle-checked by `main(inp) == out`.
# e) Visual observability:
#    The rule is human-visible because examples repeatedly show three cues together:
#    one hub side disappears, several non-matching distractors are ignored, and one visible
#    hue consistently ends up on top when corner overlaps occur. No step depends on bg=0 or
#    on an otherwise invisible total order that changes between examples.
# f) Cleanliness:
#    Overlap is restricted to attached output sprites around the hub corners. Input sprites
#    remain scattered with padding, so the noisy search stage is readable and the overlap
#    effect is localized instead of turning the entire grid into clutter.
# g) New solver assumptions introduced here:
#    - If a hub-edge color appears outside the hub, at least one component of that color has
#      a unique valid attachment rotation. The generator guarantees this by using the same
#      constructive sprite family for both real sprites and distractors.
#    - At least one pair of real attached sprites overlaps, so the ordered rendering is not
#      degenerate. The generator explicitly measures and enforces that condition.

SIDES = ("top", "bot", "left", "right")
ORDERED_HUB_COLORS = (1, 2, 3, 4)


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
    best = None
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


def _attached_origin(mask: np.ndarray, side: str, hub_r: int, hub_c: int, n: int) -> tuple[int, int]:
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


def _orient_canonical_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return np.rot90(mask, -1)
    return np.rot90(mask, 1)


def _attachment_cells(mask: np.ndarray, dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(rr + dr), int(cc + dc)) for rr, cc in np.argwhere(mask)}


def _find_side_attachment(
    g: np.ndarray,
    bg: int,
    hub_mask: np.ndarray,
    target: int,
    side: str,
    col: int,
    need_edge: str,
    fixed: int,
    span: list[int],
):
    H, W = g.shape
    m = (g == int(col)) & (~hub_mask)
    if not bool(m.any()):
        return None

    labc, nc = label_mask(m, connectivity=8)
    best = None
    for cid in range(1, int(nc) + 1):
        spr_mask = labc == int(cid)
        if not bool(spr_mask.any()):
            continue
        spr_arr = g.copy()
        spr_arr[~spr_mask] = int(bg)
        spr_arr, spr_mask = _crop_to_mask(spr_arr, spr_mask)

        for krot in range(4):
            arr = np.rot90(spr_arr, int(krot))
            mask = np.rot90(spr_mask, int(krot))
            arr, mask = _crop_to_mask(arr, mask)

            if need_edge in ("left", "right"):
                if not bool(np.array_equal(mask, np.flipud(mask))):
                    continue
            else:
                if not bool(np.array_equal(mask, np.fliplr(mask))):
                    continue

            seg = _edge_segment(mask, need_edge)
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

            cells = _attachment_cells(mask, dr, dc)
            if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in cells):
                continue

            cand = {
                "side": side,
                "color": int(col),
                "arr": arr,
                "mask": mask,
                "dr": int(dr),
                "dc": int(dc),
                "cells": cells,
                "size": int(mask.sum()),
            }
            if best is None or int(cand["size"]) > int(best["size"]):
                best = cand
            break
    return best


def _collect_attachments(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    hub = _find_hub(g, bg)
    assert hub is not None

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
    for side in SIDES:
        col, (need_edge, fixed, span) = segs[side]
        found = _find_side_attachment(g, bg, hub_mask, target, side, int(col), need_edge, int(fixed), span)
        if found is not None:
            attachments.append(found)
    return g, bg, (int(r), int(c), int(n)), attachments


def _render_output(shape: tuple[int, int], bg: int, hub: tuple[int, int, int], attachments: list[dict], color_order: list[int]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    r, c, n = hub
    by_color = {int(att["color"]): att for att in attachments}
    for col in color_order:
        att = by_color.get(int(col))
        if att is None:
            continue
        _paint_hub_side(out, str(att["side"]), int(r), int(c), int(n), int(att["color"]))
        paste(out, att["arr"], int(att["dr"]), int(att["dc"]), mask=att["mask"], overwrite=True, clip=True)
    return out


def _pairwise_overlap(attachments: list[dict]) -> int:
    total = 0
    for i in range(len(attachments)):
        for j in range(i + 1, len(attachments)):
            total += len(set(attachments[i]["cells"]) & set(attachments[j]["cells"]))
    return int(total)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Keep only hub sides that have a matching sprite somewhere outside the hub.
    - Ignore extra distractors whose colors do not match a hub side.
    - Attach every matched sprite, but render them in ascending color order so later
      colors overwrite earlier ones where attached sprites overlap.
    """
    g, bg, hub, attachments = _collect_attachments(grid)
    order = sorted(int(att["color"]) for att in attachments)
    return _render_output(g.shape, int(bg), hub, attachments, order)


def _is_unique_left_attachment(mask: np.ndarray, edge_len: int) -> bool:
    valid = 0
    for k in range(4):
        rot = np.rot90(mask, int(k))
        if not bool(np.array_equal(rot, np.flipud(rot))):
            continue
        rc = rot[:, -1]
        runs = []
        count = 0
        for v in rc:
            if v:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        if runs == [int(edge_len)]:
            valid += 1
    return valid == 1


def _make_canonical_mask(edge_len: int, rng: random.Random) -> np.ndarray | None:
    h = 8
    start = (h - edge_len) // 2
    half_h = (h + 1) // 2

    for _attempt in range(300):
        w = rng.randint(4, 6)
        grid = np.zeros((h, w), dtype=bool)
        grid[start : start + edge_len, -1] = True

        for ri in range(half_h):
            mirror_ri = h - 1 - ri
            for ci in range(w - 1):
                bias = 0.48 if ci < 2 else 0.34
                val = rng.random() < bias
                grid[ri, ci] = val
                grid[mirror_ri, ci] = val

        rows_any = grid.any(axis=1)
        cols_any = grid.any(axis=0)
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        grid = grid[r0 : r1 + 1, c0 : c1 + 1]

        if grid.shape[0] < 6 or grid.shape[1] < 3:
            continue
        if not bool(np.array_equal(grid, np.flipud(grid))):
            continue
        if bool(np.array_equal(grid, np.fliplr(grid))):
            continue

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
        if runs != [int(edge_len)]:
            continue

        labeled, num = _label(grid, structure=np.ones((3, 3), dtype=int))
        if int(num) != 1:
            continue

        px = int(grid.sum())
        if px < 8 or px > 22:
            continue
        if not _is_unique_left_attachment(grid, edge_len):
            continue
        return grid
    return None


def _choose_sprite_set(edge_len: int, sprite_sides: list[str], rng: random.Random):
    masks: dict[str, np.ndarray] = {}
    for _attempt in range(250):
        masks.clear()
        for side in sprite_sides:
            base = _make_canonical_mask(edge_len, rng)
            if base is None:
                break
            masks[side] = _orient_canonical_mask(base, side)
        if len(masks) != len(sprite_sides):
            continue

        attachments = []
        for side in sprite_sides:
            mask = masks[side]
            sr, sc = _attached_origin(mask, side, 20, 20, 4)
            attachments.append({"side": side, "cells": _attachment_cells(mask, sr, sc)})
        if _pairwise_overlap(attachments) >= 2:
            return {side: masks[side].copy() for side in sprite_sides}
    return None


def _choose_hub_position(H: int, W: int, n: int, sprite_masks: dict[str, np.ndarray]):
    candidates = []
    for hub_r in range(0, H - n + 1):
        for hub_c in range(0, W - n + 1):
            ok = True
            for side, mask in sprite_masks.items():
                sr, sc = _attached_origin(mask, side, hub_r, hub_c, n)
                sh, sw = mask.shape
                if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
                    ok = False
                    break
            if not ok:
                continue
            score = abs((hub_r + (n - 1) / 2) - (H - 1) / 2) + abs((hub_c + (n - 1) / 2) - (W - 1) / 2)
            candidates.append((float(score), int(hub_r), int(hub_c)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    _score, hub_r, hub_c = random.choice(best)
    return int(hub_r), int(hub_c)


def _scatter_rotated_patch(
    canvas: np.ndarray,
    arr: np.ndarray,
    mask: np.ndarray,
    bg: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    k = rng.choice([1, 2, 3])
    rot_arr = np.rot90(arr, int(k))
    rot_mask = np.rot90(mask, int(k))
    rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
    patch = np.where(rot_mask, rot_arr, int(bg)).astype(int)
    pr, pc = random_free_location(canvas, patch, transparent=int(bg), padding=2, padding_connectivity=8)
    return patch, np.array([int(pr), int(pc)], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    n = 4
    edge_len = n - 2

    H = rng.randint(24, 30)
    W = rng.randint(24, 30)

    colors = list(ORDERED_HUB_COLORS)
    rng.shuffle(colors)
    side_colors = dict(zip(SIDES, colors))

    sprite_sides = rng.sample(list(SIDES), 3)
    sprite_masks = _choose_sprite_set(edge_len, sprite_sides, rng)
    if sprite_masks is None:
        return None

    hub_pos = _choose_hub_position(H, W, n, sprite_masks)
    if hub_pos is None:
        return None
    hub_r, hub_c = hub_pos

    out = np.full((H, W), int(bg), dtype=int)
    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    attachments_for_metric = []

    for side in sorted(sprite_sides, key=lambda s: int(side_colors[s])):
        mask = sprite_masks[side]
        color = int(side_colors[side])
        arr = np.where(mask, color, int(bg)).astype(int)
        sr, sc = _attached_origin(mask, side, hub_r, hub_c, n)
        attachments_for_metric.append(
            {
                "side": side,
                "color": color,
                "arr": arr,
                "mask": mask,
                "dr": sr,
                "dc": sc,
                "cells": _attachment_cells(mask, sr, sc),
            }
        )
        attached[side] = (arr, mask, int(sr), int(sc))

    if _pairwise_overlap(attachments_for_metric) < 2:
        return None

    order = sorted(int(side_colors[side]) for side in sprite_sides)
    out = _render_output((H, W), int(bg), (int(hub_r), int(hub_c), int(n)), attachments_for_metric, order)

    inp = np.full((H, W), int(bg), dtype=int)
    for side, color in side_colors.items():
        _paint_hub_side(inp, side, int(hub_r), int(hub_c), int(n), int(color))

    canvas = inp.copy()
    for side in sprite_sides:
        arr, mask, _sr, _sc = attached[side]
        try:
            patch, pos = _scatter_rotated_patch(canvas, arr, mask, int(bg), rng)
        except ValueError:
            return None
        paste(canvas, patch, int(pos[0]), int(pos[1]), transparent=int(bg))
        paste(inp, patch, int(pos[0]), int(pos[1]), transparent=int(bg))

    distractor_count = rng.choice([2, 3])
    used_colors = set(int(v) for v in side_colors.values())
    distractor_colors = [c for c in range(10) if c != int(bg) and c not in used_colors]
    rng.shuffle(distractor_colors)
    if len(distractor_colors) < distractor_count:
        return None

    for dcol in distractor_colors[:distractor_count]:
        base = _make_canonical_mask(edge_len, rng)
        if base is None:
            return None
        mask = _orient_canonical_mask(base, rng.choice(list(SIDES)))
        arr = np.where(mask, int(dcol), int(bg)).astype(int)
        try:
            patch, pos = _scatter_rotated_patch(canvas, arr, mask, int(bg), rng)
        except ValueError:
            return None
        paste(canvas, patch, int(pos[0]), int(pos[1]), transparent=int(bg))
        paste(inp, patch, int(pos[0]), int(pos[1]), transparent=int(bg))

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
