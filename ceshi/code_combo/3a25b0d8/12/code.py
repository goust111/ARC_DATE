# Combo variant of 3a25b0d8 (index 12)
# combo_spec: D3/1+D5/3
# Name: 多dst框架+打包输出
#
# Design analysis for the combo variant
# a) Concrete behavior trace:
#    - Original task: the canvas has exactly one src object and one dst object. `main()`
#      finds the two largest 8-connected objects, matches all src colored cavities to all
#      dst holes by centroid-rank order, fills one dst, then tight-crops that single result.
#    - Combo task: the canvas has one src plus 2-3 separate dst frames. Each dst corresponds
#      to one subset of src cavities rather than the whole src. After solving each dst
#      independently, the filled dsts are packed into a compact 2-column output grid with
#      1-cell background separator lines.
#    - Small trace: if src contains three structural groups A/B/C, and the input has dst
#      objects for groups C and A, the solver must:
#         1. identify the unique colored src object,
#         2. enumerate every empty dst object,
#         3. match each dst to the correct src subset by local centroid-rank signature,
#         4. fill each dst,
#         5. emit the filled dsts in dst bbox reading order as a packed grid.
#      The new behavior is therefore not "solve one object then crop", but "solve several
#      subset matches, then assemble a multi-tile summary image".
# b) Complexity calibration:
#    - A naive multi-dst design would be ambiguous if two dsts shared the same local rank
#      pattern. The generator therefore uses a small bank of distinct subgroup signatures:
#      horizontal pair, vertical pair, horizontal trio, pair-over-center, center-over-pair.
#      Each sample uses distinct signatures, so the one-to-many assignment is human-visible.
#    - The number of dsts is capped at 2 or 3, and packing uses a fixed 2-column rule. That
#      is enough to make the combo obvious without turning the scene into clutter.
# c) Implicit-assumption audit against the old solver:
#    - "There are exactly two relevant objects" is broken; now there are 3-4.
#    - "The dst is the one with more hole area" is broken; there are multiple empty-hole
#      objects and only one colored src.
#    - "All src cavities map to one dst" is broken; now src cavities must be partitioned
#      into disjoint subsets, one per dst.
#    - "The output is a tight crop of one object" is broken; the new output is a packed
#      collage of several individually solved dsts.
#    - New solver assumption: every dst signature is unique within a sample. The generator
#      guarantees this by never repeating subgroup templates inside one scene.
# d) Placement legality:
#    - Generator places src and every dst with `random_free_location(..., border=1, padding=1)`,
#      so patches stay in-bounds and do not overlap.
#    - dst ordering in the packed output is derived only from their visible bbox reading order
#      on the input canvas; there are no hidden numeric tie-breakers.
# e) Visual observability:
#    - Across examples, a human can see that one source object contains several internal
#      cavity groups, that multiple dst frames each mirror one of those groups, and that the
#      output is a neat packed set of filled frames rather than a single crop.
#    - No invisible properties such as color ids are used for ordering; only geometry and
#      visible input positions matter.
# f) Visual cleanliness:
#    - The input contains only isolated outline objects on background. No labels, overlays,
#      or extra marker pixels are introduced.
#    - The packed output uses only full background separator rows/columns, so tiles do not
#      visually merge.
# g) Color-change inferability:
#    - Colors are not remapped arbitrarily. Each dst simply inherits the colors of the src
#      subgroup that matches its hole arrangement. Repeated colors are allowed, but the
#      assignment evidence comes from subgroup structure and dst shape, not from hidden rules.

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
import math
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


CANVAS_SIZE_RANGE = (24, 30)
PACK_COLS = 2
PACK_GAP = 1
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
MAX_GEN_ATTEMPTS = 400

NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40

CENTERED_SLOT_WIDTHS = [3, 3, 5]
PAIR_SLOT_WIDTHS = [2, 2, 3, 3]
SIDE_SLOT_WIDTHS = [2, 2, 3]
MID_SLOT_WIDTHS = [1, 3]
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]
TRIO_MID_SIDE_GAP = 1
ROW_HEIGHT_CHOICES = [1, 2, 2, 3]
GROUP_FILL_GAP = 1

PATTERN_COUNTS = {
    "hpair": 2,
    "vpair": 2,
    "trio": 3,
    "pair_center": 3,
    "center_pair": 3,
}


def _components(mask: np.ndarray, *, connectivity: int) -> list[dict]:
    lab, n = label_mask(mask, connectivity=connectivity)
    comps: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        comps.append(
            {
                "cid": int(cid),
                "pts": pts,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "size": int(len(pts)),
            }
        )
    comps.sort(key=lambda d: (int(d["bbox"][0]), int(d["bbox"][1]), -int(d["size"])))
    return comps


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objs: list[dict] = []
    for comp in _components(grid != int(bg), connectivity=8):
        r0, c0, r1, c1 = comp["bbox"]
        objs.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "sub": crop_bbox(grid, (int(r0), int(c0), int(r1), int(c1))),
                "size": int(comp["size"]),
            }
        )
    return objs


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    mask = sub == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    H, W = sub.shape
    holes: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if bool(r0 == 0 or c0 == 0 or r1 == int(H - 1) or c1 == int(W - 1)):
            continue
        cent = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        holes.append(
            {
                "pts": pts,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cent": cent,
                "size": int(len(pts)),
            }
        )
    holes.sort(key=lambda h: (float(h["cent"][0]), float(h["cent"][1])))
    return holes


def _outline_color(sub: np.ndarray, bg: int) -> int:
    cnt = Counter(map(int, sub.flatten()))
    opts = [(color, count) for color, count in cnt.items() if int(color) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


def _colored_components(sub: np.ndarray, bg: int) -> list[dict]:
    outline = _outline_color(sub, bg)
    mask = (sub != int(bg)) & (sub != int(outline))
    lab, n = label_mask(mask, connectivity=4)
    out: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        color = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        out.append(
            {
                "pts": pts,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                "color": int(color),
            }
        )
    out.sort(key=lambda d: (float(d["cent"][0]), float(d["cent"][1])))
    return out


def _rank_signature(points: list[tuple[float, float]]) -> tuple[tuple[int, int], ...]:
    rows = sorted({round(float(r), 3) for r, _ in points})
    cols = sorted({round(float(c), 3) for _, c in points})
    rrank = {r: i for i, r in enumerate(rows)}
    crank = {c: i for i, c in enumerate(cols)}
    sig = [(int(rrank[round(float(r), 3)]), int(crank[round(float(c), 3)])) for r, c in points]
    return tuple(sorted(sig))


def _sorted_by_local_rank(items: list[dict], key_name: str) -> list[dict]:
    pts = [tuple(map(float, item[key_name])) for item in items]
    rows = sorted({round(r, 3) for r, _ in pts})
    cols = sorted({round(c, 3) for _, c in pts})
    rrank = {r: i for i, r in enumerate(rows)}
    crank = {c: i for i, c in enumerate(cols)}
    ranked: list[tuple[tuple[int, int], float, float, dict]] = []
    for item in items:
        r, c = map(float, item[key_name])
        key = (int(rrank[round(r, 3)]), int(crank[round(c, 3)]))
        ranked.append((key, float(r), float(c), item))
    ranked.sort(key=lambda t: (t[0][0], t[0][1], t[1], t[2]))
    return [item for _key, _r, _c, item in ranked]


def _choose_partition(candidate_masks: list[list[int]], total_items: int) -> list[int] | None:
    full_mask = (1 << int(total_items)) - 1
    order = sorted(range(len(candidate_masks)), key=lambda i: len(candidate_masks[i]))
    chosen = [-1] * len(candidate_masks)

    def backtrack(pos: int, used: int) -> bool:
        if pos == len(order):
            return used == full_mask
        dst_idx = order[pos]
        for mask in candidate_masks[dst_idx]:
            if int(mask) & int(used):
                continue
            chosen[dst_idx] = int(mask)
            if backtrack(pos + 1, int(used | mask)):
                return True
            chosen[dst_idx] = -1
        return False

    if backtrack(0, 0):
        return chosen
    return None


def _tight_crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    b = bbox(arr != int(bg))
    if b is None:
        return np.full((1, 1), int(bg), dtype=int)
    return crop_bbox(arr, b)


def _pack_tiles(tiles: list[np.ndarray], bg: int, cols: int = PACK_COLS, gap: int = PACK_GAP) -> np.ndarray:
    if not tiles:
        return np.full((1, 1), int(bg), dtype=int)
    cols = max(1, min(int(cols), len(tiles)))
    rows = int(math.ceil(len(tiles) / cols))
    row_heights = [0] * rows
    col_widths = [0] * cols
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        row_heights[r] = max(row_heights[r], int(tile.shape[0]))
        col_widths[c] = max(col_widths[c], int(tile.shape[1]))
    out_h = sum(row_heights) + int(gap) * max(0, rows - 1)
    out_w = sum(col_widths) + int(gap) * max(0, cols - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    y = 0
    for r in range(rows):
        x = 0
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(tiles):
                break
            paste(out, tiles[idx], int(y), int(x), transparent=bg)
            x += int(col_widths[c]) + int(gap)
        y += int(row_heights[r]) + int(gap)
    return out


def _fill_one_dst(dst: np.ndarray, holes: list[dict], src_group: list[dict], bg: int) -> np.ndarray:
    out = dst.copy()
    ordered_holes = _sorted_by_local_rank(holes, "cent")
    ordered_src = _sorted_by_local_rank(src_group, "cent")
    for src_item, hole in zip(ordered_src, ordered_holes):
        for rr, cc in hole["pts"]:
            out[int(rr), int(cc)] = int(src_item["color"])
    return _tight_crop_non_bg(out, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the multi-dst packed-output variant of 3a25b0d8.

    Steps:
    1. Find all 8-connected non-bg objects.
    2. The unique object with colored inner components is src.
    3. Every other object with enclosed bg holes is a dst frame.
    4. Match each dst's local centroid-rank signature to a disjoint subset of src cavities.
    5. Fill each dst and pack the solved dsts in input bbox reading order with bg separators.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    objects = _extract_objects(g, bg)

    src_info: dict | None = None
    dst_infos: list[dict] = []
    for obj in objects:
        sub = obj["sub"]
        holes = _enclosed_bg_components(sub, bg)
        colored = _colored_components(sub, bg)
        if colored:
            if src_info is None or len(colored) > len(src_info["colored"]):
                src_info = {"obj": obj, "colored": colored}
        elif holes:
            dst_infos.append({"obj": obj, "holes": holes})

    if src_info is None or len(dst_infos) < 1:
        non_bg = _tight_crop_non_bg(g, bg)
        return non_bg

    dst_infos.sort(key=lambda d: (int(d["obj"]["bbox"][0]), int(d["obj"]["bbox"][1])))
    src_subs = list(src_info["colored"])
    if not src_subs:
        return _pack_tiles([_tight_crop_non_bg(d["obj"]["sub"], bg) for d in dst_infos], bg)

    candidate_masks: list[list[int]] = []
    for dst in dst_infos:
        holes = dst["holes"]
        signature = _rank_signature([tuple(h["cent"]) for h in holes])
        masks: list[int] = []
        for idxs in combinations(range(len(src_subs)), len(holes)):
            pts = [tuple(src_subs[i]["cent"]) for i in idxs]
            if _rank_signature(pts) != signature:
                continue
            mask = 0
            for idx in idxs:
                mask |= 1 << int(idx)
            masks.append(int(mask))
        candidate_masks.append(sorted(set(masks)))

    chosen = _choose_partition(candidate_masks, len(src_subs))
    if chosen is None:
        filled = []
        for dst in dst_infos:
            filled.append(_tight_crop_non_bg(dst["obj"]["sub"], bg))
        return _pack_tiles(filled, bg)

    tiles: list[np.ndarray] = []
    for dst_idx, dst in enumerate(dst_infos):
        subset = [src_subs[i] for i in range(len(src_subs)) if int(chosen[dst_idx]) & (1 << int(i))]
        tiles.append(_fill_one_dst(dst["obj"]["sub"], dst["holes"], subset, bg))
    return _pack_tiles(tiles, bg)


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int, symmetric: bool) -> np.ndarray:
    out = mask.copy()
    H, W = out.shape
    n_remove = random.randint(1, max(1, int(n_remove_max)))
    removed = 0
    tried = 0
    while removed < n_remove and tried < MAX_MUTATE_ATTEMPTS:
        tried += 1
        cand: list[tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
                if not out[r, c]:
                    continue
                nb = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and out[nr, nc]:
                        nb += 1
                if nb <= 2:
                    cand.append((int(r), int(c)))
        if not cand:
            break
        r, c = random.choice(cand)
        remove = [(int(r), int(c))]
        if symmetric:
            mc = int(W - 1 - c)
            if mc != int(c):
                remove.append((int(r), int(mc)))
        trial = out.copy()
        for rr, cc in remove:
            trial[int(rr), int(cc)] = False
        if not bool(trial.any()):
            continue
        _, n = label_mask(trial, connectivity=4)
        if int(n) != 1:
            continue
        out = trial
        removed += len(remove)
    return out if bool(out.any()) else mask.copy()


def _gen_shape(h: int, w: int, *, symmetric: bool) -> np.ndarray:
    mask = _random_rect_shape(h, w)
    if random.random() < NONRECT_PROBABILITY and h * w >= 3:
        mask = _mutate_shape(mask, n_remove_max=max(1, (h * w) // 2), symmetric=symmetric)
    return mask


def _mirror_shape(mask: np.ndarray) -> np.ndarray:
    return mask[:, ::-1].copy()


def _gen_dst_src_pair(h: int, w: int, *, symmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    return _gen_shape(h, w, symmetric=symmetric), _gen_shape(h, w, symmetric=symmetric)


def _add_centered_slot(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w = random.choice(CENTERED_SLOT_WIDTHS)
    dst, src = _gen_dst_src_pair(row_h, w, symmetric=True)
    slots.append(
        {
            "r0": int(cur_r),
            "col_offset": int(-(w // 2)),
            "shape_dst": dst,
            "shape_src": src,
            "pair_id": int(pair_id),
        }
    )


def _add_mirror_pair_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w = random.choice(PAIR_SLOT_WIDTHS)
    dst_l, src_l = _gen_dst_src_pair(row_h, w, symmetric=False)
    dst_r = _mirror_shape(dst_l)
    src_r = _mirror_shape(src_l)
    axis_gap = random.choice(PAIR_AXIS_GAP_CHOICES)
    if axis_gap % 2 == 0:
        axis_gap += 1
    half = axis_gap // 2
    slots.append(
        {
            "r0": int(cur_r),
            "col_offset": int(-(half + w)),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": int(pair_id),
        }
    )
    slots.append(
        {
            "r0": int(cur_r),
            "col_offset": int(half + 1),
            "shape_dst": dst_r,
            "shape_src": src_r,
            "pair_id": int(pair_id),
        }
    )


def _add_trio_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w_side = random.choice(SIDE_SLOT_WIDTHS)
    w_mid = random.choice(MID_SLOT_WIDTHS)
    dst_l, src_l = _gen_dst_src_pair(row_h, w_side, symmetric=False)
    dst_r = _mirror_shape(dst_l)
    src_r = _mirror_shape(src_l)
    dst_m, src_m = _gen_dst_src_pair(row_h, w_mid, symmetric=True)
    mid_half = w_mid // 2
    slots.append(
        {
            "r0": int(cur_r),
            "col_offset": int(-(mid_half + w_side + TRIO_MID_SIDE_GAP)),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": int(pair_id),
        }
    )
    slots.append(
        {
            "r0": int(cur_r),
            "col_offset": int(-mid_half),
            "shape_dst": dst_m,
            "shape_src": src_m,
            "pair_id": int(pair_id + 1),
        }
    )
    slots.append(
        {
            "r0": int(cur_r),
            "col_offset": int(mid_half + TRIO_MID_SIDE_GAP + 1),
            "shape_dst": dst_r,
            "shape_src": src_r,
            "pair_id": int(pair_id),
        }
    )


def _pattern_layout(name: str, pair_id_start: int) -> tuple[list[dict], int]:
    slots: list[dict] = []
    if name == "hpair":
        _add_mirror_pair_slots(slots, 0, pair_id_start, random.choice(ROW_HEIGHT_CHOICES))
        return slots, int(pair_id_start + 1)
    if name == "vpair":
        h1 = random.choice(ROW_HEIGHT_CHOICES)
        h2 = random.choice(ROW_HEIGHT_CHOICES)
        _add_centered_slot(slots, 0, pair_id_start, h1)
        _add_centered_slot(slots, int(h1 + 1), pair_id_start + 1, h2)
        return slots, int(pair_id_start + 2)
    if name == "trio":
        _add_trio_slots(slots, 0, pair_id_start, random.choice(ROW_HEIGHT_CHOICES))
        return slots, int(pair_id_start + 2)
    if name == "pair_center":
        h1 = random.choice(ROW_HEIGHT_CHOICES)
        h2 = random.choice(ROW_HEIGHT_CHOICES)
        _add_mirror_pair_slots(slots, 0, pair_id_start, h1)
        _add_centered_slot(slots, int(h1 + 1), pair_id_start + 1, h2)
        return slots, int(pair_id_start + 2)
    if name == "center_pair":
        h1 = random.choice(ROW_HEIGHT_CHOICES)
        h2 = random.choice(ROW_HEIGHT_CHOICES)
        _add_centered_slot(slots, 0, pair_id_start, h1)
        _add_mirror_pair_slots(slots, int(h1 + 1), pair_id_start + 1, h2)
        return slots, int(pair_id_start + 2)
    raise ValueError(f"unknown pattern {name}")


def _layout_bbox(layout: list[dict]) -> tuple[int, int, int, int]:
    rows: list[int] = []
    cols: list[int] = []
    for slot in layout:
        sh = slot["shape_dst"]
        rr, cc = np.where(sh)
        rows.extend((slot["r0"] + rr).tolist())
        cols.extend((slot["col_offset"] + cc).tolist())
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _shift_layout(layout: list[dict], row_shift: int) -> list[dict]:
    shifted: list[dict] = []
    for slot in layout:
        shifted.append(
            {
                "r0": int(slot["r0"] + row_shift),
                "col_offset": int(slot["col_offset"]),
                "shape_dst": slot["shape_dst"],
                "shape_src": slot["shape_src"],
                "pair_id": int(slot["pair_id"]),
            }
        )
    return shifted


def _pick_group_patterns(n_groups: int) -> list[str]:
    if n_groups == 2:
        if random.random() < 0.65:
            out = [random.choice(["hpair", "vpair"]), random.choice(["trio", "pair_center", "center_pair"])]
        else:
            out = random.sample(["trio", "pair_center", "center_pair"], 2)
        random.shuffle(out)
        return out
    if random.random() < 0.5:
        out = ["hpair", "vpair", random.choice(["trio", "pair_center", "center_pair"])]
    else:
        out = [random.choice(["hpair", "vpair"]), *random.sample(["trio", "pair_center", "center_pair"], 2)]
    random.shuffle(out)
    return out


def _assign_pair_colors(layout: list[dict], fill_pool: list[int]) -> list[int]:
    pair_ids: list[int] = []
    for slot in layout:
        if int(slot["pair_id"]) not in pair_ids:
            pair_ids.append(int(slot["pair_id"]))
    pair_colors = list(fill_pool)
    while len(pair_colors) < len(pair_ids):
        pair_colors.append(int(random.choice(fill_pool)))
    random.shuffle(pair_colors)
    color_map = {int(pid): int(col) for pid, col in zip(pair_ids, pair_colors)}
    return [int(color_map[int(slot["pair_id"])]) for slot in layout]


def _render_object(
    layout: list[dict],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    use_src_shapes: bool,
    filled: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    placed = []
    key = "shape_src" if use_src_shapes else "shape_dst"
    for idx, slot in enumerate(layout):
        sh = slot[key]
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed.append((idx, slot["r0"] + rr, slot["col_offset"] + cc))
    if not placed:
        return None, None
    all_rows = np.concatenate([rows for _idx, rows, _cols in placed])
    all_cols = np.concatenate([cols for _idx, _rows, cols in placed])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())
    pad = 2
    H = int(r_max - r_min + 1 + 2 * pad)
    W = int(c_max - c_min + 1 + 2 * pad)
    canvas = np.full((H, W), int(bg), dtype=int)
    fill_mask = np.zeros((H, W), dtype=bool)
    for idx, rows, cols in placed:
        rr = rows - int(r_min) + int(pad)
        cc = cols - int(c_min) + int(pad)
        canvas[rr, cc] = int(colors[idx])
        fill_mask[rr, cc] = True
    outline_mask = np.zeros_like(fill_mask)
    outline_mask[1:, :] |= fill_mask[:-1, :]
    outline_mask[:-1, :] |= fill_mask[1:, :]
    outline_mask[:, 1:] |= fill_mask[:, :-1]
    outline_mask[:, :-1] |= fill_mask[:, 1:]
    outline_mask &= ~fill_mask
    canvas[outline_mask] = int(outline_col)
    filled_canvas = canvas.copy()
    if not filled:
        canvas[fill_mask] = int(bg)
    b = bbox(canvas != int(bg))
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    fr0, fc0, fr1, fc1 = b
    return cropped, filled_canvas[int(fr0) : int(fr1) + 1, int(fc0) : int(fc1) + 1].copy()


def _is_8connected(arr: np.ndarray, bg: int) -> bool:
    mask = arr != int(bg)
    if not bool(mask.any()):
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _count_holes(arr: np.ndarray, bg: int) -> int:
    return len(_enclosed_bg_components(arr, bg))


def _build_scene_objects() -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], int] | None:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != int(bg)]
    random.shuffle(palette)
    outline_col = int(palette[0])
    n_groups = random.choice([2, 3, 3])
    patterns = _pick_group_patterns(int(n_groups))

    groups: list[list[dict]] = []
    master: list[dict] = []
    pair_id = 0
    cur_r = 0
    for name in patterns:
        local, pair_id = _pattern_layout(name, pair_id)
        r0, _c0, _r1, _c1 = _layout_bbox(local)
        shifted = _shift_layout(local, int(cur_r - r0))
        groups.append(shifted)
        master.extend(shifted)
        _gr0, _gc0, gr1, _gc1 = _layout_bbox(shifted)
        cur_r = int(gr1 + GROUP_FILL_GAP + 1)

    n_slots = len(master)
    fill_pool = palette[1 : 1 + min(max(3, n_slots), len(palette) - 1)]
    colors = _assign_pair_colors(master, fill_pool)

    src, _src_filled = _render_object(master, colors, bg, outline_col, use_src_shapes=True, filled=True)
    if src is None or not _is_8connected(src, bg):
        return None
    if len(_colored_components(src, bg)) != n_slots:
        return None

    group_color_lists: list[list[int]] = []
    start = 0
    for group in groups:
        count = len(group)
        group_color_lists.append(colors[start : start + count])
        start += count

    dsts: list[np.ndarray] = []
    filled_dsts: list[np.ndarray] = []
    for group, group_colors in zip(groups, group_color_lists):
        dst, filled = _render_object(group, group_colors, bg, outline_col, use_src_shapes=False, filled=False)
        if dst is None or filled is None:
            return None
        if not _is_8connected(dst, bg):
            return None
        if _count_holes(dst, bg) != len(group):
            return None
        dsts.append(dst)
        filled_dsts.append(filled)

    return src, dsts, filled_dsts, bg


def _place_scene(src: np.ndarray, dsts: list[np.ndarray], bg: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    H = random.randint(*CANVAS_SIZE_RANGE)
    W = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((H, W), int(bg), dtype=int)
    bboxes: list[tuple[int, int, int, int]] = [(-1, -1, -1, -1)] * len(dsts)

    try:
        sr, sc = random_free_location(
            canvas,
            src,
            transparent=bg,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
    except ValueError:
        return None
    paste(canvas, src, int(sr), int(sc), transparent=bg)

    order = sorted(range(len(dsts)), key=lambda i: int(dsts[i].shape[0] * dsts[i].shape[1]), reverse=True)
    for idx in order:
        patch = dsts[idx]
        try:
            r, c = random_free_location(
                canvas,
                patch,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
            )
        except ValueError:
            return None
        paste(canvas, patch, int(r), int(c), transparent=bg)
        bboxes[idx] = (int(r), int(c), int(r + patch.shape[0] - 1), int(c + patch.shape[1] - 1))
    return canvas, bboxes


def _expected_packed_output(filled_dsts: list[np.ndarray], dst_bboxes: list[tuple[int, int, int, int]], bg: int) -> np.ndarray:
    order = sorted(range(len(filled_dsts)), key=lambda i: (int(dst_bboxes[i][0]), int(dst_bboxes[i][1])))
    tiles = [_tight_crop_non_bg(filled_dsts[i], bg) for i in order]
    return _pack_tiles(tiles, bg)


def _combo_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(inp))
    objs = _extract_objects(inp, bg)
    src_count = 0
    dst_count = 0
    for obj in objs:
        colored = len(_colored_components(obj["sub"], bg))
        holes = len(_enclosed_bg_components(obj["sub"], bg))
        if colored > 0:
            src_count += 1
        elif holes > 0:
            dst_count += 1
    sep_rows = sum(1 for r in range(out.shape[0]) if bool((out[r, :] == int(bg)).all()))
    sep_cols = sum(1 for c in range(out.shape[1]) if bool((out[:, c] == int(bg)).all()))
    return {
        "src_count": int(src_count),
        "dst_count": int(dst_count),
        "sep_rows": int(sep_rows),
        "sep_cols": int(sep_cols),
    }


def _gen_attempt() -> np.ndarray | None:
    built = _build_scene_objects()
    if built is None:
        return None
    src, dsts, filled_dsts, bg = built

    placed = _place_scene(src, dsts, bg)
    if placed is None:
        return None
    canvas, dst_bboxes = placed
    expected = _expected_packed_output(filled_dsts, dst_bboxes, bg)

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not isinstance(actual, np.ndarray):
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None

    metrics = _combo_metrics(canvas, expected)
    if metrics["src_count"] != 1:
        return None
    if metrics["dst_count"] < 2:
        return None
    if metrics["dst_count"] >= 2 and metrics["sep_cols"] < 1:
        return None
    if metrics["dst_count"] >= 3 and metrics["sep_rows"] < 1:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
