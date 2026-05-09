# Variant of 8b9c3697
# Dimension: D13 — 匹配歧义消解 (match_cond)
# Change: 当多个红色色块在同一方向的射线上时，选择命中次数最多（hit_count）、最近距离（min_dist）的色块；结构间贪... → 面积匹配优先
#
# Design analysis
# spec:
#   variant_goal:
#     keep: 梳齿结构、沿开口方向滑入、轨迹涂黑、未被选中的红块消失、结构间仍按分数贪心占用红块
#     change: 同一结构同一开口方向上故意放入两个可见红矩形；solver 不再先看 hit_count/min_dist，而先看面积和插槽容量的接近程度
#   a_concretization:
#     objects:
#       structure: 仍是原题 4 层 comb/bracket；为了让面积规则稳定可计算，真正参与匹配的结构只用 teeth=2 或 3
#       intended_red: 与原题一致，面积 = teeth * min(teeth, 2)，即 2x2 或 2x3
#       decoy_red: 放在更近位置，但只覆盖 opening 的一部分列；面积更小，为 2x1 或 2x2
#     behavior_trace:
#       original_logic: teeth=3 时，近处 2x2 decoy 覆盖 2 条射线，远处 2x3 target 只剩 1 条射线可见；旧规则会因为 hit_count=2>1 选 decoy
#       variant_logic: 同一场景下，slot_area=3*2=6；decoy area=4，target area=6，因此新规则优先选 target，即便它更远、命中更少
#       difference: 输入里同向可见的多个红块都保留，但输出只让面积最匹配的那一个进入 cavity
#   b_complexity_calibration:
#     matched_structures: 每个匹配结构都带 1 个 decoy，确保样例里重复出现“近处/多命中但面积不合适”的冲突
#     values_controlled: 只有两种 slot_area（4, 6），decoy 面积对应为（2, 4）；自由度低，便于人眼归纳
#     clutter_control: 适当降低无关 distractor 数量，避免面积线索被大量随机红块淹没
#   c_hidden_assumptions_audit:
#     old_dependency_1: 原 main 用 rid_best=max(...hit_count,-min_dist...)，默认“更近/命中更多”就是正确匹配；变体中此假设失效
#     old_dependency_2: 原 generator 用 _unique_matched_visibility 强制每个结构至多看到 1 个红块；这会直接抹掉 D13 需要的歧义，必须删除
#     old_dependency_3: 原 _place_structures 只按 output 占用来摆放 patch；原题 input/output 占用近似一致，因此无害。现在 input 里多了 decoy，若仍按 output 放置会导致输入重叠，必须改为用 footprint（input/output/structure 的并集）摆放
#     old_dependency_4: 新 solver 用 slot_area = socket_count * min(socket_count, 2)，这依赖原 comb 几何仍成立；generator 必须继续只生成这类结构
#   d_placement_legality:
#     matched_patch:
#       target: 放在 opening 方向更远的位置
#       decoy: 放在同方向更近的位置，并只占 target 宽度的一侧子集，保证二者都在同一组射线上可见但互不接触
#     global_layout:
#       structure patch 用 footprint 放置，避免输入新增 decoy 后与其他 patch 重叠
#       distractor 仍禁止落在任一结构的清晰射线上，避免产生第三个候选
#   e_visual_observability:
#     human_signal: 解题者能直接看到“近的小块没进去，远但尺寸更贴 cavity 的块进去了”
#     anti_degenerate: 验证器要求每个 matched structure 恰好看到 2 个红块，且 area-rule 的选择与 legacy-rule 不同
#   f_visual_cleanliness:
#     decoy_count: 每个结构只加 1 个 decoy，且只沿 opening 同轴放置，避免多层覆盖
#     distractors: 只在非射线区域散布，减少干扰
#   g_color_reasoning:
#     no_color_logic: 排序与选择完全不使用颜色 id；黑色只表示滑行轨迹，不参与任何比较

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


def _extract_red_data(g: np.ndarray) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]], dict[tuple[int, int], int]]:
    red_comps = components_mask(g == int(Color.RED), connectivity=4)
    assert red_comps

    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_comps):
        pts = [(int(r), int(c)) for r, c in coords]
        red_coords.append(pts)
        red_bbox.append(tuple(map(int, bbox_coords(coords))))
        for rc in pts:
            red_at[rc] = int(rid)
    return red_coords, red_bbox, red_at


def _extract_structures(g: np.ndarray, bg: int) -> tuple[list[set[tuple[int, int]]], list[tuple[int, int, int, int]]]:
    struct_mask = (g != int(bg)) & (g != int(Color.RED))
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg non-red structure to dock into"

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))
    return struct_coords, struct_bbox


def _slot_candidates(
    g: np.ndarray,
    bg: int,
    red_at: dict[tuple[int, int], int],
) -> tuple[list[dict], list[list[int]], list[tuple[int, int, int, int]], list[list[tuple[int, int]]]]:
    H, W = g.shape
    red_coords, red_bbox, _ = _extract_red_data(g)
    red_areas = [len(coords) for coords in red_coords]
    struct_coords, struct_bbox = _extract_structures(g, bg)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    struct_infos: list[dict] = []
    visible_per_struct: list[list[int]] = []

    for sid, S in enumerate(struct_coords):
        sr0, sc0, sr1, sc1 = map(int, struct_bbox[sid])
        best_area = None
        best_legacy = None
        visible_rids: set[int] = set()

        for dr, dc in dirs:
            hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
            socket_cells: set[tuple[int, int]] = set()

            for r, c in S:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(g[nr, nc]) != int(bg):
                    continue
                if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                    continue

                socket_cells.add((int(nr), int(nc)))
                rr, cc = int(nr), int(nc)
                while 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(bg):
                    rr += int(dr)
                    cc += int(dc)

                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(Color.RED):
                    rid = int(red_at[(int(rr), int(cc))])
                    visible_rids.add(int(rid))
                    rec = hits[rid]
                    rec[0] += 1
                    dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
                    if int(dist) < int(rec[1]):
                        rec[1] = int(dist)
                        rec[2] = (int(r), int(c))

            if not hits or not socket_cells:
                continue

            max_proj = max(int(r * dr + c * dc) for r, c in socket_cells)
            mouth_width = sum(1 for r, c in socket_cells if int(r * dr + c * dc) == int(max_proj))
            slot_area = int(mouth_width * min(mouth_width, 2))

            def area_key(rid: int) -> tuple[int, int, int]:
                return (
                    -abs(int(red_areas[rid]) - int(slot_area)),
                    int(hits[rid][0]),
                    -int(hits[rid][1]),
                )

            def legacy_key(rid: int) -> tuple[int, int, int]:
                return (
                    int(hits[rid][0]),
                    -int(hits[rid][1]),
                    -abs(int(red_areas[rid]) - int(slot_area)),
                )

            rid_area = max(hits.keys(), key=area_key)
            rid_legacy = max(hits.keys(), key=legacy_key)

            cnt_a, mind_a, anchor_a = hits[rid_area]
            cand_area = {
                "score": (
                    -abs(int(red_areas[rid_area]) - int(slot_area)),
                    int(cnt_a),
                    -int(mind_a),
                ),
                "rid": int(rid_area),
                "dir": (int(dr), int(dc)),
                "anchor": anchor_a,
                "sid": int(sid),
                "slot_area": int(slot_area),
                "hit_count": int(cnt_a),
                "min_dist": int(mind_a),
            }

            cnt_l, mind_l, anchor_l = hits[rid_legacy]
            cand_legacy = {
                "score": (
                    int(cnt_l),
                    -int(mind_l),
                    -abs(int(red_areas[rid_legacy]) - int(slot_area)),
                ),
                "rid": int(rid_legacy),
                "dir": (int(dr), int(dc)),
                "anchor": anchor_l,
                "sid": int(sid),
                "slot_area": int(slot_area),
                "hit_count": int(cnt_l),
                "min_dist": int(mind_l),
            }

            if best_area is None or cand_area["score"] > best_area["score"]:
                best_area = cand_area
            if best_legacy is None or cand_legacy["score"] > best_legacy["score"]:
                best_legacy = cand_legacy

        struct_infos.append(
            {
                "sid": int(sid),
                "best_area": best_area,
                "best_legacy": best_legacy,
            }
        )
        visible_per_struct.append(sorted(int(rid) for rid in visible_rids))

    return struct_infos, visible_per_struct, red_bbox, red_coords


def _assign_structures(struct_infos: list[dict], mode: str) -> list[dict]:
    key_name = "best_area" if mode == "area" else "best_legacy"
    ranked = [info[key_name] for info in struct_infos if info[key_name] is not None]
    ranked.sort(reverse=True, key=lambda item: item["score"])

    used_red: set[int] = set()
    assignments: list[dict] = []
    for item in ranked:
        if int(item["rid"]) in used_red or item["anchor"] is None:
            continue
        used_red.add(int(item["rid"]))
        assignments.append(item)
    return assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide selected red rectangles into comb cavities, preferring the size that fits the cavity best, and leave a black path.

    Input structure:
    - The dominant color is background; large non-red comb/bracket shapes create interior sockets.
    - Several separate red rectangles can sit on the same opening side of one structure, including closer but smaller decoys.

    Rule:
    - Ray-cast outward from each structure's interior socket cells to gather visible red components by direction.
    - Estimate that direction's cavity capacity from the socket width, then choose the visible red whose area is closest to that capacity; ties fall back to hit coverage and distance.
    - Greedily reserve chosen reds across structures, slide each reserved component into the cavity, paint its traversed cells black, and erase every unchosen red.

    Key insight: The deciding signal is area fit to the socket implied by the comb teeth, not simply the nearest red or the one hit by the most rays.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    RED = int(Color.RED)
    BLACK = int(Color.BLACK)

    assert int(RED) in set(map(int, np.unique(g)))

    red_coords, red_bbox, red_at = _extract_red_data(g)
    struct_infos, _visible_per_struct, _red_bbox2, _red_coords2 = _slot_candidates(g, bg, red_at)
    assignments = _assign_structures(struct_infos, mode="area")

    out = g.copy()
    out[out == int(RED)] = int(bg)

    for item in assignments:
        rid = int(item["rid"])
        dr, dc = map(int, item["dir"])
        ar, ac = map(int, item["anchor"])

        coords = red_coords[rid]
        r0, c0, r1, c1 = map(int, red_bbox[rid])
        tr, tc = int(ar + dr), int(ac + dc)

        if dr == 0 and dc == 1:
            shift_r, shift_c = 0, int(tc - c0)
        elif dr == 0 and dc == -1:
            shift_r, shift_c = 0, int(tc - c1)
        elif dr == 1 and dc == 0:
            shift_r, shift_c = int(tr - r0), 0
        else:
            shift_r, shift_c = int(tr - r1), 0

        steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)

        for step in range(0, steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = int(BLACK)

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(RED)

    return out


GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

MATCHED_WIDTH_TEETH = ((8, 2), (9, 3))
UNMATCHED_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.3
SLIDE_RANGE = (4, 8)
DECOY_SLIDE_RANGE = (2, 3)
STRUCT_PADDING = 4

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 6
TOTAL_RED_MAX = 10
DISTRACTOR_MIN = 1
DISTRACTOR_MAX = 4

MAX_RETRIES = 400

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = sc
    g[1, 1] = g[1, width - 2] = sc
    g[2:4, 1] = g[2:4, width - 2] = sc
    g[1, start:start + teeth] = sc
    return g


def _matched_patches(width: int, teeth: int, slide: int, decoy_slide: int, rot_k: int, bg: int, sc: int) -> dict:
    red = int(Color.RED)
    black = int(Color.BLACK)
    red_h = min(teeth, 2)
    decoy_w = max(1, teeth - 1)
    decoy_side = random.choice((0, 1))
    start = (width - teeth) // 2
    decoy_c0 = int(start if decoy_side == 0 else start + teeth - decoy_w)

    patch_h = 2 + slide + red_h
    structure = np.full((patch_h, width), bg, dtype=int)
    base = _structure_array(width, teeth, bg, sc)
    paste(structure, base, 0, 0, transparent=bg)

    input_patch = structure.copy()
    output_patch = structure.copy()
    intended_mask = np.zeros_like(structure, dtype=bool)

    for off in range(1, slide + 1):
        output_patch[2 + off:2 + off + red_h, start:start + teeth] = black
    output_patch[2:2 + red_h, start:start + teeth] = red

    target_r0 = 2 + slide
    input_patch[target_r0:target_r0 + red_h, start:start + teeth] = red
    intended_mask[target_r0:target_r0 + red_h, start:start + teeth] = True

    decoy_r0 = 2 + decoy_slide
    input_patch[decoy_r0:decoy_r0 + red_h, decoy_c0:decoy_c0 + decoy_w] = red

    footprint = np.where(
        (input_patch != bg) | (output_patch != bg) | (structure != bg),
        sc,
        bg,
    )

    return {
        "matched": True,
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "footprint": np.rot90(footprint, k=rot_k),
        "intended_mask": np.rot90(intended_mask, k=rot_k),
    }


def _unmatched_patches(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    structure = np.rot90(_structure_array(width, teeth, bg, sc), k=rot_k)
    return {
        "matched": False,
        "structure": structure,
        "input": structure.copy(),
        "output": structure.copy(),
        "footprint": structure.copy(),
        "intended_mask": np.zeros_like(structure, dtype=bool),
    }


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        for dr, dc in _DIRS:
            for r, c in item["coords"]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if struct_grid[nr, nc] != bg:
                    continue
                if not (r0 < nr < r1 and c0 < nc < c1):
                    continue
                rr, cc = nr, nc
                while 0 <= rr < h and 0 <= cc < w and struct_grid[rr, cc] == bg:
                    forbidden[rr, cc] = True
                    rr += dr
                    cc += dc
    return forbidden


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _place_structures(
    specs: list[dict],
    h: int,
    w: int,
    bg: int,
    sc: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    occupancy = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(occupancy, spec["footprint"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(occupancy, spec["footprint"], r0, c0, transparent=bg)
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == sc)
        coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]

        intended_coords = [
            (r0 + int(rr), c0 + int(cc))
            for rr, cc in np.argwhere(spec["intended_mask"])
        ]

        placed.append(
            {
                "matched": bool(spec["matched"]),
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "intended_coords": intended_coords,
            }
        )

    return inp, out, struct_grid, placed


def _validate_variant_grid(grid: np.ndarray, placed: list[dict], bg: int) -> bool:
    red_coords, _red_bbox, red_at = _extract_red_data(grid)
    struct_infos, visible_per_struct, _red_bbox2, _red_coords2 = _slot_candidates(grid, bg, red_at)
    area_best = {info["sid"]: info["best_area"] for info in struct_infos}
    legacy_best = {info["sid"]: info["best_legacy"] for info in struct_infos}

    differing = 0
    for sid, item in enumerate(placed):
        visible = visible_per_struct[sid]
        best_area = area_best[sid]
        best_legacy = legacy_best[sid]

        if item["matched"]:
            if len(visible) != 2 or best_area is None or best_legacy is None:
                return False
            intended_rids = {int(red_at[(r, c)]) for r, c in item["intended_coords"]}
            if len(intended_rids) != 1:
                return False
            intended_rid = next(iter(intended_rids))
            if int(best_area["rid"]) != int(intended_rid):
                return False
            if int(best_legacy["rid"]) != int(best_area["rid"]):
                differing += 1
        else:
            if visible:
                return False

    return differing >= max(1, sum(1 for item in placed if item["matched"]) // 2)


def _place_distractors(
    inp: np.ndarray,
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
    sc: int,
    count: int,
) -> bool:
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0:r1 + 1, c0:c1 + 1] = True
    red_mask = inp == int(Color.RED)

    shapes = list(DISTRACTOR_SHAPES)
    for _ in range(count):
        blocked = forbidden | bbox_blocked | (inp != bg) | _dilate4(red_mask)
        canvas = np.full((h, w), bg, dtype=int)
        canvas[blocked] = sc
        random.shuffle(shapes)
        ok = False
        for rh, rw in shapes:
            patch = np.full((rh, rw), int(Color.RED), dtype=int)
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is not None:
                paste(inp, patch, loc[0], loc[1], transparent=bg)
                red_mask[loc[0]:loc[0] + rh, loc[1]:loc[1] + rw] = True
                ok = True
                break
        if not ok:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1 if n_structs == 3 and random.random() < UNMATCHED_PROB else 0
    matched = n_structs - unmatched
    if matched <= 0:
        return None

    base_reds = 2 * matched
    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - base_reds)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - base_reds)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)

    specs: list[dict] = []
    for idx in range(n_structs):
        rot_k = random.randrange(4)
        if idx < matched:
            width, teeth = random.choice(MATCHED_WIDTH_TEETH)
            slide = random.randint(*SLIDE_RANGE)
            decoy_slide = random.randint(*DECOY_SLIDE_RANGE)
            if decoy_slide >= slide:
                return None
            specs.append(_matched_patches(width, teeth, slide, decoy_slide, rot_k, bg, sc))
        else:
            width, teeth = random.choice(UNMATCHED_WIDTH_TEETH)
            specs.append(_unmatched_patches(width, teeth, rot_k, bg, sc))

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["footprint"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    if not _validate_variant_grid(inp, placed, bg):
        return None

    if not _place_distractors(inp, struct_grid, placed, bg, sc, distractor_count):
        return None

    if not _validate_variant_grid(inp, placed, bg):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D13 variant")
