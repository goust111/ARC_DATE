# Combo variant of 3a25b0d8 (index 23)
# combo_spec: D7/3+D18/0
# Name: 层级填充+嵌套空腔
#
# Design analysis for the fused variant
# -------------------------------------
# 1. Concrete behavior:
#    - Original task: dst has one layer of empty holes; src has one layer of colored
#      components. The solver can collect every hole/component once, rank centroids,
#      and fill everything in a single pass.
#    - New variant: some large outer cavities contain a 1-cell gap around a small
#      inner colored component. After outline rendering, dst shows an outer empty ring,
#      a small outline island, and then an inner empty hole inside that island.
#      The intended reasoning is:
#        outer src component  <-> outer dst hole
#        inner src component  <-> inner dst hole inside the matched parent
#      So filling is executed parent-first, then child.
#    - Example sketch:
#        src slot:  colored ring  / outline island / inner colored dot
#        dst slot:  empty ring    / outline island / inner empty dot
#      The final output equals dst with both regions colored, but the nested relation is
#      now visible and the input object is no longer a single raw 8-connected component.
#
# 2. Complexity calibration:
#    - We keep only two levels (outer + one inner child) so the pattern stays readable.
#    - Each puzzle has 3-4 outer cavities and 1-2 nested ones, giving 4-6 colored/hole
#      regions overall. This is enough repetition for humans to notice hierarchy without
#      turning the object into noise.
#
# 3. Broken implicit assumptions from the original solver:
#    - "Each logical object is a single 8-connected component" no longer holds for dst,
#      because the inner outline islands are disconnected from the outer shell when the
#      cavities are empty.
#    - "All holes live at one flat level" no longer holds. Inner holes must be attached
#      to parent outer holes through the intermediate outline island.
#    - "Counting/filling all holes at once is sufficient" is replaced by a tree walk.
#    - The new solver therefore first groups logical objects via a 1-step 8-neighborhood
#      dilation, then reconstructs outer-vs-inner relations from adjacency to outline
#      components, and only then performs ordered filling.
#
# 4. Placement legality:
#    - The generator places src and dst as separate cropped patches on a larger canvas
#      with extra padding. This prevents the dilation-based logical grouping from merging
#      the two objects, while still allowing dst itself to contain multiple raw components.
#
# 5. Visual observability / neatness:
#    - Nested cavities are rendered as large ring-like holes with a small inner hole,
#      which is easy to see directly in 2-4 examples.
#    - The inner child always sits inside its parent, so the hierarchy is spatially
#      obvious rather than encoded by invisible metadata or color ids.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


def _dilate8(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    out = mask.astype(bool).copy()
    H, W = out.shape
    for _ in range(int(steps)):
        p = np.pad(out, 1, constant_values=False)
        nxt = np.zeros_like(out)
        for dr in range(3):
            for dc in range(3):
                nxt |= p[dr : dr + H, dc : dc + W]
        out = nxt
    return out


def _component_records(mask: np.ndarray, *, connectivity: int) -> tuple[np.ndarray, list[dict]]:
    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    recs: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        recs.append(
            {
                "id": int(cid),
                "pts": pts,
                "size": int(len(pts)),
                "cent": (
                    int(round(float(pts[:, 0].mean()))),
                    int(round(float(pts[:, 1].mean()))),
                ),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    return lab, recs


def _adjacent_labels(pts: np.ndarray, lab: np.ndarray) -> set[int]:
    H, W = lab.shape
    out: set[int] = set()
    for rr, cc in pts:
        r = int(rr)
        c = int(cc)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < H and 0 <= nc < W:
                v = int(lab[nr, nc])
                if v > 0:
                    out.add(v)
    return out


def _enclosed_bg_records(sub: np.ndarray, bg: int) -> list[dict]:
    _, recs = _component_records(sub == int(bg), connectivity=4)
    H, W = sub.shape
    holes = []
    for rec in recs:
        r0, c0, r1, c1 = rec["bbox"]
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
            continue
        holes.append(rec)
    return holes


def _logical_groups(grid: np.ndarray, bg: int) -> list[dict]:
    raw = grid != int(bg)
    grown = _dilate8(raw, steps=1)
    glab, gn = label_mask(grown, connectivity=8)
    groups = []
    for gid in range(1, int(gn) + 1):
        pts = np.argwhere(raw & (glab == int(gid)))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        groups.append(
            {
                "id": int(gid),
                "pts": pts,
                "size": int(len(pts)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    groups.sort(key=lambda g: (g["bbox"][1], g["bbox"][0], -g["size"]))
    return groups


def _extract_group_subgrid(grid: np.ndarray, group: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = group["bbox"]
    sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for rr, cc in group["pts"]:
        sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
    return sub


def _rank_keys(pts: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[int, int], dict[int, int]]:
    rows = sorted(set(int(r) for r, _c in pts))
    cols = sorted(set(int(c) for _r, c in pts))
    rrank = {int(r): i for i, r in enumerate(rows)}
    crank = {int(c): i for i, c in enumerate(cols)}
    keys = [(int(rrank[int(r)]), int(crank[int(c)])) for r, c in pts]
    return keys, rrank, crank


def _transform_point(
    p: tuple[int, int], H0: int, W0: int, k: int, flip: bool
) -> tuple[int, int]:
    r, c = int(p[0]), int(p[1])
    if int(k) == 0:
        rr, cc = int(r), int(c)
        H2, W2 = int(H0), int(W0)
    elif int(k) == 1:
        rr, cc = int(W0 - 1 - c), int(r)
        H2, W2 = int(W0), int(H0)
    elif int(k) == 2:
        rr, cc = int(H0 - 1 - r), int(W0 - 1 - c)
        H2, W2 = int(H0), int(W0)
    else:
        rr, cc = int(c), int(H0 - 1 - r)
        H2, W2 = int(W0), int(H0)
    if bool(flip):
        cc = int(W2 - 1 - cc)
    return int(rr), int(cc)


def _detect_outline_color(src: np.ndarray, dst: np.ndarray, bg: int) -> int:
    src_cols = set(map(int, src.flatten())) - {int(bg)}
    dst_cols = set(map(int, dst.flatten())) - {int(bg)}
    common = sorted(src_cols & dst_cols)
    if common:
        return int(common[0])
    cnt = Counter(map(int, dst.flatten()))
    opts = [(col, n) for col, n in cnt.items() if int(col) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0])


def _build_dest_tree(sub: np.ndarray, bg: int, outline: int) -> list[dict]:
    outline_lab, outline_recs = _component_records(sub == int(outline), connectivity=4)
    if not outline_recs:
        return []
    bg_lab, bg_recs = _component_records(sub == int(bg), connectivity=4)
    outside_bg_ids = {
        int(rec["id"])
        for rec in bg_recs
        if rec["bbox"][0] == 0
        or rec["bbox"][1] == 0
        or rec["bbox"][2] == sub.shape[0] - 1
        or rec["bbox"][3] == sub.shape[1] - 1
    }
    exterior_outline_ids = {
        int(rec["id"])
        for rec in outline_recs
        if _adjacent_labels(rec["pts"], bg_lab) & outside_bg_ids
    }
    holes = _enclosed_bg_records(sub, int(bg))

    outer_nodes: list[dict] = []
    island_to_outer: dict[int, dict] = {}

    for hole in holes:
        adj = _adjacent_labels(hole["pts"], outline_lab)
        exterior_adj = adj & exterior_outline_ids
        if exterior_adj:
            node = {
                "id": int(hole["id"]),
                "pts": hole["pts"],
                "cent": hole["cent"],
                "size": hole["size"],
                "bbox": hole["bbox"],
                "child": None,
            }
            outer_nodes.append(node)
            for island_id in sorted(adj - exterior_outline_ids):
                island_to_outer[int(island_id)] = node

    for hole in holes:
        adj = _adjacent_labels(hole["pts"], outline_lab)
        if adj & exterior_outline_ids:
            continue
        if len(adj) != 1:
            continue
        island_id = int(next(iter(adj)))
        parent = island_to_outer.get(int(island_id))
        if parent is None:
            continue
        parent["child"] = {
            "pts": hole["pts"],
            "cent": hole["cent"],
            "size": hole["size"],
            "bbox": hole["bbox"],
        }

    outer_nodes.sort(key=lambda n: (n["cent"][0], n["cent"][1]))
    return outer_nodes


def _build_src_tree(sub: np.ndarray, bg: int, outline: int) -> list[dict]:
    outline_lab, outline_recs = _component_records(sub == int(outline), connectivity=4)
    if not outline_recs:
        return []
    bg_lab, bg_recs = _component_records(sub == int(bg), connectivity=4)
    outside_bg_ids = {
        int(rec["id"])
        for rec in bg_recs
        if rec["bbox"][0] == 0
        or rec["bbox"][1] == 0
        or rec["bbox"][2] == sub.shape[0] - 1
        or rec["bbox"][3] == sub.shape[1] - 1
    }
    exterior_outline_ids = {
        int(rec["id"])
        for rec in outline_recs
        if _adjacent_labels(rec["pts"], bg_lab) & outside_bg_ids
    }

    fill_mask = (sub != int(bg)) & (sub != int(outline))
    fill_lab, fill_recs = _component_records(fill_mask, connectivity=4)

    outer_nodes: list[dict] = []
    island_to_outer: dict[int, dict] = {}

    for rec in fill_recs:
        adj = _adjacent_labels(rec["pts"], outline_lab)
        col = int(Counter(map(int, sub[fill_lab == int(rec["id"])].flatten())).most_common(1)[0][0])
        if adj & exterior_outline_ids:
            node = {
                "id": int(rec["id"]),
                "pts": rec["pts"],
                "cent": rec["cent"],
                "size": rec["size"],
                "bbox": rec["bbox"],
                "color": int(col),
                "child": None,
            }
            outer_nodes.append(node)
            for island_id in sorted(adj - exterior_outline_ids):
                island_to_outer[int(island_id)] = node

    for rec in fill_recs:
        adj = _adjacent_labels(rec["pts"], outline_lab)
        if (adj & exterior_outline_ids) or len(adj) != 1:
            continue
        island_id = int(next(iter(adj)))
        parent = island_to_outer.get(int(island_id))
        if parent is None:
            continue
        col = int(Counter(map(int, sub[fill_lab == int(rec["id"])].flatten())).most_common(1)[0][0])
        parent["child"] = {
            "pts": rec["pts"],
            "cent": rec["cent"],
            "size": rec["size"],
            "bbox": rec["bbox"],
            "color": int(col),
        }

    outer_nodes.sort(key=lambda n: (n["cent"][0], n["cent"][1]))
    return outer_nodes


def _tight_crop_nonbg(sub: np.ndarray, bg: int) -> np.ndarray:
    b = bbox(sub != int(bg))
    if b is None:
        return sub.copy()
    return crop_bbox(sub, b)


def _count_children(nodes: list[dict]) -> int:
    return sum(1 for node in nodes if node.get("child") is not None)


def _measure_variant_features(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    groups = _logical_groups(g, bg)
    if len(groups) < 2:
        return {"logical_objects": int(len(groups)), "nested_holes": 0, "dst_raw_components": 0}
    objs = [_extract_group_subgrid(g, grp, bg) for grp in groups[:2]]
    hole_info = [_enclosed_bg_records(obj, bg) for obj in objs]
    dest_i = int(np.argmax([len(hh) for hh in hole_info]))
    src_i = int(1 - dest_i)
    outline = _detect_outline_color(objs[src_i], objs[dest_i], bg)
    dest_tree = _build_dest_tree(objs[dest_i], bg, outline)
    dst_raw_components = int(label_mask(objs[dest_i] != int(bg), connectivity=8)[1])
    return {
        "logical_objects": 2,
        "nested_holes": int(_count_children(dest_tree)),
        "dst_raw_components": int(dst_raw_components),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    groups = _logical_groups(g, bg)
    assert len(groups) >= 2
    groups = sorted(groups, key=lambda grp: grp["size"], reverse=True)[:2]
    objs = [_extract_group_subgrid(g, grp, bg) for grp in groups]

    holes_list = [_enclosed_bg_records(obj, bg) for obj in objs]
    dest_i = int(np.argmax([sum(int(h["size"]) for h in holes) for holes in holes_list]))
    src_i = int(1 - dest_i)

    src = objs[src_i]
    dest = objs[dest_i].copy()
    outline = _detect_outline_color(src, dest, bg)

    src_tree = _build_src_tree(src, bg, outline)
    dest_tree = _build_dest_tree(dest, bg, outline)
    if not src_tree or not dest_tree:
        return _tight_crop_nonbg(dest, bg)

    dest_pts = [node["cent"] for node in dest_tree]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    best: tuple[int, bool] | None = None
    for k in range(4):
        for flip in (False, True):
            tpts = [
                _transform_point(node["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
                for node in src_tree
            ]
            tkeys, _ranks_r, _ranks_c = _rank_keys(tpts)
            if sorted(tkeys) == sorted(dest_keys):
                best = (int(k), bool(flip))
                break
        if best is not None:
            break
    if best is None:
        best = (0, False)
    k, flip = best

    dest_list = []
    for node in dest_tree:
        key = (int(d_rr[int(node["cent"][0])]), int(d_cc[int(node["cent"][1])]))
        dest_list.append((key, node["cent"], node))
    dest_list.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))

    tpts_all = [
        _transform_point(node["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        for node in src_tree
    ]
    _src_keys, s_rr, s_cc = _rank_keys(tpts_all)
    src_list = []
    for node in src_tree:
        tp = _transform_point(node["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, node))
    src_list.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))

    for (_key_s, _tp, src_node), (_key_d, _cent_d, dest_node) in zip(src_list, dest_list):
        # Level 0: fill the outer cavity ring / plain cavity first.
        for rr, cc in dest_node["pts"]:
            dest[int(rr), int(cc)] = int(src_node["color"])
        # Level 1: then fill the inner hole inside the matched parent island.
        if src_node.get("child") is not None and dest_node.get("child") is not None:
            for rr, cc in dest_node["child"]["pts"]:
                dest[int(rr), int(cc)] = int(src_node["child"]["color"])

    return _tight_crop_nonbg(dest, bg)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (8, 17)
OBJECT_W_RANGE = (7, 13)
OUTER_COUNT_RANGE = (3, 4)
NESTED_COUNT_RANGE = (1, 2)
FILL_COLOR_COUNT_RANGE = (3, 5)
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 4
MAX_GEN_ATTEMPTS = 500
NONRECT_PROBABILITY = 0.65
MAX_MUTATE_ATTEMPTS = 40


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int) -> np.ndarray:
    m = mask.copy()
    H, W = m.shape
    n_remove = random.randint(1, max(1, int(n_remove_max)))
    removed = 0
    tries = 0
    while removed < n_remove and tries < MAX_MUTATE_ATTEMPTS:
        tries += 1
        cand = []
        for r in range(H):
            for c in range(W):
                if not m[r, c]:
                    continue
                nb = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < H and 0 <= nc < W and m[nr, nc]:
                        nb += 1
                if nb <= 2:
                    cand.append((r, c))
        if not cand:
            break
        r, c = random.choice(cand)
        m2 = m.copy()
        m2[int(r), int(c)] = False
        if not m2.any():
            continue
        _, n = label_mask(m2, connectivity=4)
        if int(n) == 1:
            m = m2
            removed += 1
    return m


def _gen_shape(h: int, w: int, *, nonrect: bool) -> np.ndarray:
    m = _random_rect_shape(int(h), int(w))
    if nonrect and h * w >= 3:
        m = _mutate_shape(m, n_remove_max=max(1, (h * w) // 3))
    return m


def _make_plain_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    attempts = 0
    while True:
        src = _gen_shape(h, w, nonrect=(random.random() < NONRECT_PROBABILITY))
        dst = _gen_shape(h, w, nonrect=(random.random() < NONRECT_PROBABILITY))
        attempts += 1
        if attempts >= 4 or not np.array_equal(src, dst):
            return src, dst


def _make_nested_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inner_h = 1 if h == 5 else random.choice([1, 2])
    inner_w = 1 if w == 5 else random.choice([1, 2])
    hole_h = inner_h + 2
    hole_w = inner_w + 2
    hole_r0 = (h - hole_h) // 2
    hole_c0 = (w - hole_w) // 2

    base_outer = np.ones((h, w), dtype=bool)
    base_outer[hole_r0 : hole_r0 + hole_h, hole_c0 : hole_c0 + hole_w] = False
    outer_src = base_outer.copy()
    outer_dst = base_outer.copy()

    inner_src = np.zeros((h, w), dtype=bool)
    inner_dst = np.zeros((h, w), dtype=bool)
    src_small = _gen_shape(inner_h, inner_w, nonrect=(random.random() < 0.5))
    dst_small = _gen_shape(inner_h, inner_w, nonrect=(random.random() < 0.5))
    inner_src[hole_r0 + 1 : hole_r0 + 1 + inner_h, hole_c0 + 1 : hole_c0 + 1 + inner_w] = src_small
    inner_dst[hole_r0 + 1 : hole_r0 + 1 + inner_h, hole_c0 + 1 : hole_c0 + 1 + inner_w] = dst_small
    return outer_src, outer_dst, inner_src, inner_dst


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _choose_dims(nested: bool) -> tuple[int, int]:
    if nested:
        return random.choice([5, 6]), 5
    return random.choice([2, 3, 4]), random.choice([2, 3, 4])


def _design_slots() -> list[dict] | None:
    outer_count = random.randint(*OUTER_COUNT_RANGE)
    nested_count = random.randint(NESTED_COUNT_RANGE[0], min(NESTED_COUNT_RANGE[1], outer_count - 1))
    flags = [True] * int(nested_count) + [False] * int(outer_count - nested_count)
    random.shuffle(flags)

    if outer_count == 3:
        row_counts = random.choice([[1, 2], [2, 1]])
    else:
        row_counts = [2, 2]

    temp_rows = []
    max_row_w = 0
    flag_i = 0
    for row_n in row_counts:
        row_specs = []
        row_h = 0
        row_w = 0
        for _ in range(int(row_n)):
            nested = bool(flags[flag_i])
            flag_i += 1
            h, w = _choose_dims(nested)
            row_specs.append({"nested": nested, "h": int(h), "w": int(w)})
            row_h = max(int(row_h), int(h))
            row_w += int(w)
        row_w += max(0, int(row_n) - 1)
        max_row_w = max(int(max_row_w), int(row_w))
        temp_rows.append((row_specs, int(row_h), int(row_w)))

    if max_row_w > OBJECT_W_RANGE[1] - 2:
        return None

    slots = []
    cur_r = 0
    slot_id = 0
    for row_specs, row_h, row_w in temp_rows:
        cur_c = (int(max_row_w) - int(row_w)) // 2
        for spec in row_specs:
            slots.append(
                {
                    "slot_id": int(slot_id),
                    "nested": bool(spec["nested"]),
                    "h": int(spec["h"]),
                    "w": int(spec["w"]),
                    "r0": int(cur_r + (row_h - spec["h"]) // 2),
                    "c0": int(cur_c),
                }
            )
            slot_id += 1
            cur_c += int(spec["w"]) + 1
        cur_r += int(row_h) + random.choice([1, 1, 2])

    max_r = max(int(slot["r0"] + slot["h"]) for slot in slots)
    max_c = max(int(slot["c0"] + slot["w"]) for slot in slots)
    if not (OBJECT_H_RANGE[0] - 2 <= max_r <= OBJECT_H_RANGE[1] - 2):
        return None
    if not (OBJECT_W_RANGE[0] - 2 <= max_c <= OBJECT_W_RANGE[1] - 2):
        return None
    return slots


def _assign_colors(slots: list[dict], fill_pool: list[int]) -> None:
    refs = []
    for slot in slots:
        refs.append((slot, "outer_color"))
        if slot["nested"]:
            refs.append((slot, "inner_color"))

    cols = list(fill_pool)
    while len(cols) < len(refs):
        cols.append(random.choice(fill_pool))
    random.shuffle(cols)

    for (slot, key), col in zip(refs, cols):
        slot[key] = int(col)

    for slot in slots:
        if slot["nested"] and slot["inner_color"] == slot["outer_color"] and len(fill_pool) > 1:
            choices = [c for c in fill_pool if int(c) != int(slot["outer_color"])]
            slot["inner_color"] = int(random.choice(choices))


def _materialize_slot_masks(slots: list[dict]) -> None:
    for slot in slots:
        h = int(slot["h"])
        w = int(slot["w"])
        if slot["nested"]:
            outer_src, outer_dst, inner_src, inner_dst = _make_nested_masks(h, w)
            slot["outer_src"] = outer_src
            slot["outer_dst"] = outer_dst
            slot["inner_src"] = inner_src
            slot["inner_dst"] = inner_dst
        else:
            outer_src, outer_dst = _make_plain_masks(h, w)
            slot["outer_src"] = outer_src
            slot["outer_dst"] = outer_dst
            slot["inner_src"] = None
            slot["inner_dst"] = None
            slot["inner_color"] = None


def _collect_components(slots: list[dict], *, variant: str) -> list[dict]:
    comps = []
    for slot in slots:
        comps.append(
            {
                "r0": int(slot["r0"]),
                "c0": int(slot["c0"]),
                "mask": slot[f"outer_{variant}"],
                "color": int(slot["outer_color"]),
            }
        )
        if slot["nested"]:
            comps.append(
                {
                    "r0": int(slot["r0"]),
                    "c0": int(slot["c0"]),
                    "mask": slot[f"inner_{variant}"],
                    "color": int(slot["inner_color"]),
                }
            )
    return comps


def _render_object(
    components: list[dict], bg: int, outline: int, *, filled: bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    placed = []
    for comp in components:
        rr, cc = np.where(comp["mask"])
        if len(rr) == 0:
            continue
        placed.append((comp, rr + int(comp["r0"]), cc + int(comp["c0"])))
    if not placed:
        return None, None

    all_rows = np.concatenate([item[1] for item in placed])
    all_cols = np.concatenate([item[2] for item in placed])
    r_min = int(all_rows.min())
    r_max = int(all_rows.max())
    c_min = int(all_cols.min())
    c_max = int(all_cols.max())

    pad = 2
    H = int(r_max - r_min + 1 + 2 * pad)
    W = int(c_max - c_min + 1 + 2 * pad)
    canvas = np.full((H, W), int(bg), dtype=int)
    fill_mask = np.zeros((H, W), dtype=bool)

    for comp, rows, cols in placed:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        canvas[rr, cc] = int(comp["color"])
        fill_mask[rr, cc] = True

    outline_mask = np.zeros_like(fill_mask)
    outline_mask[1:, :] |= fill_mask[:-1, :]
    outline_mask[:-1, :] |= fill_mask[1:, :]
    outline_mask[:, 1:] |= fill_mask[:, :-1]
    outline_mask[:, :-1] |= fill_mask[:, 1:]
    outline_mask &= ~fill_mask
    canvas[outline_mask] = int(outline)

    filled_grid = canvas.copy()
    if not filled:
        canvas[fill_mask] = int(bg)

    b = bbox(canvas != int(bg))
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    r0, c0, r1, c1 = b
    filled_cropped = filled_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    return cropped, filled_cropped


def _is_single_logical_object(grid: np.ndarray, bg: int) -> bool:
    groups = _logical_groups(grid, int(bg))
    return len(groups) == 1


def _build_object_pair(
    slots: list[dict], bg: int, outline: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    src_components = _collect_components(slots, variant="src")
    dst_components = _collect_components(slots, variant="dst")

    src, _src_filled = _render_object(src_components, bg, outline, filled=True)
    dst, expected = _render_object(dst_components, bg, outline, filled=False)
    if src is None or dst is None or expected is None:
        return None

    if not _is_single_logical_object(src, bg):
        return None
    if not _is_single_logical_object(dst, bg):
        return None
    if not (OBJECT_H_RANGE[0] <= src.shape[0] <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= src.shape[1] <= OBJECT_W_RANGE[1]):
        return None
    if not (OBJECT_H_RANGE[0] <= dst.shape[0] <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= dst.shape[1] <= OBJECT_W_RANGE[1]):
        return None

    outline_color = _detect_outline_color(src, dst, bg)
    src_tree = _build_src_tree(src, bg, outline_color)
    dst_tree = _build_dest_tree(dst, bg, outline_color)
    expected_outer = len(slots)
    expected_inner = sum(1 for slot in slots if slot["nested"])
    if len(src_tree) != expected_outer or len(dst_tree) != expected_outer:
        return None
    if _count_children(src_tree) != expected_inner or _count_children(dst_tree) != expected_inner:
        return None
    raw_dst_components = int(label_mask(dst != int(bg), connectivity=8)[1])
    if expected_inner > 0 and raw_dst_components <= 1:
        return None
    return src, dst, expected


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, bg: int) -> np.ndarray | None:
    H = random.randint(*CANVAS_SIZE_RANGE)
    W = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((H, W), int(bg), dtype=int)
    try:
        r0, c0 = random_free_location(
            canvas,
            src,
            transparent=int(bg),
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
    except ValueError:
        return None
    paste(canvas, src, r0, c0, transparent=int(bg))
    try:
        r1, c1 = random_free_location(
            canvas,
            dst,
            transparent=int(bg),
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
    except ValueError:
        return None
    paste(canvas, dst, r1, c1, transparent=int(bg))
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline = int(palette[0])

    slots = _design_slots()
    if slots is None:
        return None
    _materialize_slot_masks(slots)

    total_components = len(slots) + sum(1 for slot in slots if slot["nested"])
    max_fill_cols = min(FILL_COLOR_COUNT_RANGE[1], total_components, len(palette) - 1)
    if max_fill_cols < FILL_COLOR_COUNT_RANGE[0]:
        return None
    n_fill_cols = random.randint(FILL_COLOR_COUNT_RANGE[0], max_fill_cols)
    fill_pool = palette[1 : 1 + n_fill_cols]
    _assign_colors(slots, fill_pool)

    built = _build_object_pair(slots, bg, outline)
    if built is None:
        return None
    src, dst, expected = built

    canvas = _place_on_canvas(src, dst, bg)
    if canvas is None:
        return None

    metrics = _measure_variant_features(canvas)
    if metrics["nested_holes"] < 1 or metrics["dst_raw_components"] <= 1:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
