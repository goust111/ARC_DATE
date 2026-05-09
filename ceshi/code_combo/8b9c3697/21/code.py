# Combo variant of 8b9c3697 (index 21)
# combo_spec: D4/0+D5/1
# Name: 静默滑入保留干扰

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for combo D4/0 + D5/1
# ======================================
# a) 具体化与行为追踪
# - 原题里，每个梳齿结构会从开口方向吸入一个矩形红块：源位置被清空，滑行经过的每一步都涂成黑色，
#   最终只在结构内部 socket 里留下红块；其他未匹配红块会被擦掉。
# - 本融合沿用原题的“内部 socket + 射线匹配”几何，只改输出表现：
#   1) matched 红块仍然真正滑入插槽，所以源位置恢复为背景；
#   2) 整条滑行路径不再涂黑，保持背景；
#   3) unmatched 红色干扰块完整保留在原位，不再删除。
# - 具体小例子：若一个 2x3 红块在结构下方 5 格处且正对开口，原题输出会是“内部 2x3 红块 + 一条 2x3
#   黑拖尾”；本融合输出则只剩“内部 2x3 红块”，外部原位置变回背景。与此同时，别处那些不在任何 socket
#   射线上的 1x2 / 2x2 红色干扰块会继续留在原地。
#
# b) 复杂度校准
# - 保留原题的结构数量、旋转方向、红块尺寸、4-8 格滑移距离和 2-8 个干扰块，避免退化成简单复制题。
# - 输出少了黑拖尾，因此视觉线索减少；为保持难度，继续保留多结构和多个 distractor，让解题者必须区分
#   “哪些红块被吸入了”与“哪些红块只是保留下来”。
# - 规则自由度仍然有限：移动方向只有四个正交方向，插槽宽度只由 teeth 数决定，干扰块尺寸也只取六个重复档位。
#
# c) 隐式假设审查
# - 原版的 socket 搜索、射线命中统计和一红块只分配给一个结构的贪心逻辑都仍然成立，因为几何匹配规则没变。
# - 被破坏的原假设有两条：
#   1) `out[out == RED] = bg` 隐含“所有红块默认消失”，融合后不成立，因为 unmatched 红块必须保留；
#   2) 逐步把经过位置写成黑色，隐含“路径需要显式编码”，融合后不成立，因为路径必须保持背景。
# - 新版 `main()` 的额外假设是：matched 红块的终点仍位于结构内部，且不会与其他保留的红块重叠。
#   生成器继续利用原题的结构间距和 distractor 射线屏蔽来保证这一点。
#
# d) 放置合法性
# - 生成器继续采用 output-first：先构造“结构 + 插槽内终点红块”的目标，再反推出输入中的源红块位置。
# - matched patch 的 output 不再包含黑色轨迹，也不包含外部源红块；distractor 会同时写入 input 和 output，
#   保证它们在输出中保留。
# - 结构之间仍保持 padding；distractor 仍禁止落在任何 socket 向外的清晰视线通道上，因此不会误变成 matched。
#
# e) 视觉可观察性
# - 每个样例至少有 1 个 matched 结构和至少 2 个 distractor，所以观察者会同时看到：
#   1) 某些正对结构的红块消失于原地并出现在结构内部；
#   2) 其他红块完全不动；
#   3) 整个输出里没有黑色拖尾。
# - 这三点在 2-3 组样例里就足以归纳出“静默滑入 + 保留干扰”的融合规则。
#
# f) 可视化整洁度
# - 去掉黑轨迹后，输出比原题更干净。为避免“内部红块和外部干扰挤作一团”，生成器仍对结构 bbox、结构射线、
#   已放置红块的 4 邻域做避让，保证对象之间有明确分隔。
# - unmatched 红块只出现在结构外部背景区，matched 红块只出现在结构内部 socket，因此两类红块的角色清晰可辨。
#
# g) 变化的可推断性
# - 颜色体系仍然只有背景色、结构色和红色三类可见角色；黑色在本融合输出中应完全消失。
# - 规则完全由可见几何关系驱动：是否处在某个结构开口的视线通道上，决定它是“滑入”还是“原地保留”。
#   没有用颜色 id、组件编号或隐藏排序作为判断依据。
#
# h) 新 solver 假设与 generator 保证
# - `main()` 假设每个 matched 红块与其目标结构之间存在一条纯背景直线走廊，generator 沿用原题 slide patch
#   构造来保证这一点。
# - `main()` 假设 distractor 不会被任何结构的 socket 射线命中；generator 的 `_structure_rays()` 和放置屏蔽区
#   专门保证这一点。
# - `main()` 假设至少有一个红块和至少一个结构，generator 保持原题的 2-3 个结构与 5-11 个总红块配置。


GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 4

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

MAX_RETRIES = 300

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _dilate4(mask: np.ndarray) -> np.ndarray:
    """4-connected dilation of a boolean mask."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _find_assignments(
    g: np.ndarray, bg: int
) -> tuple[
    list[list[tuple[int, int]]],
    list[tuple[int, int, int, int]],
    set[int],
    list[tuple[int, tuple[int, int], tuple[int, int]]],
]:
    red_val = int(Color.RED)
    h, w = g.shape

    red_comps = components_mask(g == red_val, connectivity=4)
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

    struct_mask = (g != int(bg)) & (g != int(red_val))
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg non-red structure to dock into"

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))

    struct_best: list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int], int]] = []

    for sid, pts in enumerate(struct_coords):
        sr0, sc0, sr1, sc1 = map(int, struct_bbox[sid])
        best = None

        for dr, dc in _DIRS:
            hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
            for r, c in pts:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) != int(bg):
                    continue
                if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                    continue

                rr, cc = int(nr), int(nc)
                while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(bg):
                    rr += int(dr)
                    cc += int(dc)

                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(red_val):
                    rid = int(red_at[(int(rr), int(cc))])
                    rec = hits[rid]
                    rec[0] += 1
                    dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
                    if int(dist) < int(rec[1]):
                        rec[1] = int(dist)
                        rec[2] = (int(r), int(c))

            if not hits:
                continue

            rid_best = max(hits.keys(), key=lambda k: (int(hits[k][0]), -int(hits[k][1])))
            cnt, mind, anchor = hits[rid_best]
            cand = (int(cnt), -int(mind), int(rid_best), (int(dr), int(dc)), anchor)
            if best is None or cand > best:
                best = cand

        if best is not None:
            cnt, neg_mind, rid, dirv, anchor = best
            struct_best.append(
                ((int(cnt), int(neg_mind)), int(rid), (int(dirv[0]), int(dirv[1])), anchor, int(sid))
            )

    struct_best.sort(reverse=True, key=lambda t: t[0])
    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for _score, rid, dirv, anchor, _sid in struct_best:
        if int(rid) in used_red or anchor is None:
            continue
        used_red.add(int(rid))
        assignments.append((int(rid), (int(dirv[0]), int(dirv[1])), (int(anchor[0]), int(anchor[1]))))

    return red_coords, red_bbox, used_red, assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide only the structure-aligned red rectangles into interior sockets, without leaving any trail behind.

    Input structure:
    - The most common color is background. Large non-background, non-red comb/bracket structures contain interior cavities.
    - Multiple red rectangles lie outside the structures. Some align with a structure opening; others are distractors that do not line up with any socket ray.

    Rule:
    - For each structure, inspect interior socket cells and ray-cast outward to identify the red component seen through the opening.
    - Move that matched red component into the socket, clearing its source cells back to background and keeping every traversed path cell unchanged.
    - Leave every unmatched red component exactly where it started.

    Key insight: The geometry for deciding which red belongs to which structure is unchanged from the base task; the fusion only changes the rendering, so matched reds silently relocate while unmatched reds remain visible.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_val = int(Color.RED)

    assert red_val in set(map(int, np.unique(g)))

    red_coords, red_bbox, _used_red, assignments = _find_assignments(g, bg)

    out = g.copy()
    for rid, (dr, dc), (ar, ac) in assignments:
        coords = red_coords[int(rid)]
        r0, c0, r1, c1 = map(int, red_bbox[int(rid)])
        tr, tc = int(ar + dr), int(ac + dc)

        if dr == 0 and dc == 1:
            shift_r, shift_c = 0, int(tc - c0)
        elif dr == 0 and dc == -1:
            shift_r, shift_c = 0, int(tc - c1)
        elif dr == 1 and dc == 0:
            shift_r, shift_c = int(tr - r0), 0
        else:
            shift_r, shift_c = int(tr - r1), 0

        for r, c in coords:
            out[int(r), int(c)] = int(bg)

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = int(red_val)

    return out


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    """Build a 4-row normalized comb structure (opening facing down)."""
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = sc
    g[1, 1] = g[1, width - 2] = sc
    g[2:4, 1] = g[2:4, width - 2] = sc
    g[1, start:start + teeth] = sc
    return g


def _matched_patches(width: int, teeth: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    """Build input/output/structure patches for one matched structure."""
    red_val = int(Color.RED)
    red_h = min(teeth, 2)
    base = _structure_array(width, teeth, bg, sc)
    patch_h = 2 + int(slide) + red_h

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    start = (width - teeth) // 2
    input_patch[2 + slide : 2 + slide + red_h, start : start + teeth] = red_val
    output_patch[2 : 2 + red_h, start : start + teeth] = red_val

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
    }


def _unmatched_patches(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    """Build patches for a structure that has no matched red."""
    structure = np.rot90(_structure_array(width, teeth, bg, sc), k=rot_k)
    return {"structure": structure, "input": structure.copy(), "output": structure.copy()}


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    """Return a mask of cells lying on any socket-to-outside line of sight."""
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        for dr, dc in _DIRS:
            for r, c in item["coords"]:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(struct_grid[nr, nc]) != int(bg):
                    continue
                if not (int(r0) < nr < int(r1) and int(c0) < nc < int(c1)):
                    continue
                rr, cc = int(nr), int(nc)
                while 0 <= rr < h and 0 <= cc < w and int(struct_grid[rr, cc]) == int(bg):
                    forbidden[rr, cc] = True
                    rr += int(dr)
                    cc += int(dc)
    return forbidden


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    """Try to place patch on canvas; return (r, c) or None."""
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _unique_matched_visibility(grid: np.ndarray, bg: int, expected_matched: int) -> bool:
    """Check that exactly expected_matched reds are each visible from exactly one structure."""
    h, w = grid.shape
    red_val = int(Color.RED)
    red_comps = components_mask(grid == red_val, connectivity=4)
    if len(red_comps) != expected_matched:
        return False

    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_comps):
        for r, c in coords:
            red_at[(int(r), int(c))] = int(rid)

    struct_mask = (grid != int(bg)) & (grid != int(red_val))
    struct_comps = components_mask(struct_mask, connectivity=4)
    red_to_structs: dict[int, set[int]] = {}
    struct_hits = 0

    for sid, coords in enumerate(struct_comps):
        pts = {(int(r), int(c)) for r, c in coords}
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        visible_rids: set[int] = set()
        for dr, dc in _DIRS:
            for r, c in pts:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(grid[nr, nc]) != int(bg):
                    continue
                if not (r0 < nr < r1 and c0 < nc < c1):
                    continue
                rr, cc = int(nr), int(nc)
                while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(bg):
                    rr += int(dr)
                    cc += int(dc)
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(red_val):
                    visible_rids.add(int(red_at[(int(rr), int(cc))]))
        if len(visible_rids) > 1:
            return False
        if visible_rids:
            struct_hits += 1
            rid = next(iter(visible_rids))
            red_to_structs.setdefault(int(rid), set()).add(int(sid))

    if struct_hits != expected_matched:
        return False
    return len(red_to_structs) == expected_matched and all(len(sids) == 1 for sids in red_to_structs.values())


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    """Place all structure patches on fresh canvases."""
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(out, spec["output"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == sc)
        coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        placed.append({"coords": coords, "bbox": (min(rows), min(cols), max(rows), max(cols))})

    return inp, out, struct_grid, placed


def _place_distractors(
    inp: np.ndarray,
    out: np.ndarray,
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
    sc: int,
    count: int,
) -> bool:
    """Scatter distractor red rectangles and keep them in both input and output."""
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    red_mask = inp == int(Color.RED)

    shapes = list(DISTRACTOR_SHAPES)
    for _ in range(count):
        blocked = forbidden | bbox_blocked | (inp != int(bg)) | _dilate4(red_mask)
        canvas = np.full((h, w), bg, dtype=int)
        canvas[blocked] = sc
        random.shuffle(shapes)
        ok = False
        for rh, rw in shapes:
            patch = np.full((rh, rw), int(Color.RED), dtype=int)
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is None:
                continue
            paste(inp, patch, loc[0], loc[1], transparent=bg)
            paste(out, patch, loc[0], loc[1], transparent=bg)
            red_mask[loc[0] : loc[0] + rh, loc[1] : loc[1] + rw] = True
            ok = True
            break
        if not ok:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1 if random.random() < UNMATCHED_PROB else 0
    matched = n_structs - unmatched
    if matched <= 0:
        return None

    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)

    specs = []
    rot_pool = list(range(4))
    random.shuffle(rot_pool)
    for idx in range(n_structs):
        width, teeth = random.choice(STRUCT_WIDTH_TEETH)
        # Prefer distinct structure orientations inside one puzzle so the four docking
        # directions receive clearer and more even coverage across small sample sets.
        rot_k = rot_pool[idx % len(rot_pool)]
        if idx < matched:
            slide = random.randint(*SLIDE_RANGE)
            specs.append(_matched_patches(width, teeth, slide, rot_k, bg, sc))
        else:
            specs.append(_unmatched_patches(width, teeth, rot_k, bg, sc))
    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    if not _unique_matched_visibility(inp, bg, matched):
        return None

    if not _place_distractors(inp, out, struct_grid, placed, bg, sc, distractor_count):
        return None

    if np.count_nonzero(out == int(Color.BLACK)) != 0:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for combo 8b9c3697 D4/0+D5/1")
