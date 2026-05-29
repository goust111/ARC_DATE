# Combo variant of 8b9c3697 (index 23)
# combo_spec: D2/1+D4/1
# Name: 多色精灵彩色轨迹

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 保留原题的“大型单色梳齿结构 + 外部对象沿开口直线滑入 + 未匹配对象消失”主线，但把可移动对象
#      从纯红矩形改成双色精灵。规范朝向下，精灵固定为 `2 x teeth`：靠结构的一整行是红色功能面，
#      远离结构的一整行是主色装饰面。
#    - 小例子：原题里一个 `2x3` 红块在结构下方 5 格，输出是黑轨迹 + 最终 `2x3` 红块。融合后同一位置
#      放的是 `[红红红; 橙橙橙]` 精灵。它仍然凭借上边的红色功能面被射线识别并滑入，但输出改为：
#      中间所有经过的位置都涂成橙色，最终停靠时保留 `[红红红; 橙橙橙]`。
#    - 差异集中在两个环节：匹配对象不再是“整块都红”，而是“红色前缘 + 主色尾部”的单个精灵；
#      轨迹也不再使用统一黑色，而是复制该精灵自己的主色。
#
# b) 复杂度校准
#    - 只保留 teeth=2 和 teeth=3 两档。这样精灵至少是 `2x2`，双色结构肉眼稳定可见；teeth=1 会退化成
#      无法做成多色连通精灵的 `1x1` 红点。
#    - 每个样例至少有 2 个匹配精灵，并且它们的主色互不相同。这样输出一定能出现至少两条不同颜色的轨迹，
#      融合效果不会退化成“只有一个彩色拖尾”。
#    - 精灵图案固定成“整条红前缘 + 整条主色后缘”，避免加入过多自由度。人类只需归纳一条视觉规则：
#      红色那一面负责对接，非红主色负责留下路径并在最终图案中保留。
#
# c) 隐式假设审查
#    - 原 solver 假设“所有红色像素组成可移动物体，所有非红非背景像素都是结构”。融合后这不成立，
#      因为精灵包含红色和主色两种像素。新 solver 改为先提取全部非背景连通块，再按“是否含红”分类。
#    - 原 solver 在最终对齐时直接使用整个红块 bbox。融合后必须按红色功能面对齐插槽，否则主色尾部会改变
#      停靠位置。新版本先恢复命中的 socket 面，再把精灵的前缘红面平移到这个 socket 面上。
#    - 原 solver 把所有中间位置刷成黑色。融合后这会抹掉新维度，因此改为把整个中间精灵占位统一刷成该精灵
#      的主色，最后再覆盖最终的双色精灵图案。
#    - 新 solver 额外依赖一个前提：精灵在开口垂直方向上已经与结构对齐，只需沿滑动轴平移。generator 通过
#      output-first 构造和 oracle 校验显式保证这一点。
#
# d) 放置合法性
#    - 匹配结构 patch 仍按 output-first 构造，并把完整滑行走廊包含进 patch 尺寸，所以结构、最终精灵、
#      彩色轨迹三者不会在局部模板里互相越界。
#    - 全局放置继续使用 `random_free_location(..., padding=4)` 保证不同结构彼此分离，避免一条轨迹挤到另一
#      个结构附近造成视觉干扰。
#    - 干扰项也统一使用同型双色精灵，但必须避开所有 socket 向外的可视射线、结构 bbox 以及已有精灵的
#      4 邻域，防止组件粘连、误匹配或遮挡。
#
# e) 视觉可观察性
#    - 输入里所有可移动对象都共享一眼可见的模式：一边整条红色，一边整条非红色。这让解题者能够把它们看成
#      一类“精灵”，而不是把主色条误认为独立结构。
#    - 输出里每个成功滑入的精灵都会留下与自己尾部同色的长轨迹，并且最终停靠后仍保留红+主色两层图案。
#      因此从 2-4 个样例就能看出“黑轨迹”规则已经失效。
#
# f) 可视化整洁度
#    - 结构仍是单色大块，精灵始终只有两行/两列，轨迹也只使用精灵主色中的一种，不引入第三层符号。
#    - 彩色轨迹被限制在各自的直线走廊里，且不同结构之间有固定 padding，因此输出不会出现多条彩色路径
#      在同一区域纠缠的情况。
#
# g) 变化的可推断性
#    - 颜色逻辑只增加一条可见规则：轨迹颜色来自该精灵自己的主色，而不是来自结构颜色或固定黑色。
#      这个对应关系可以直接从“输入精灵尾部颜色 -> 输出轨迹颜色”观察得到。
#    - 为了避免颜色种类失控，匹配精灵的主色从固定的 4 色池里选取，并保证每个样例中已匹配的主色两两不同。
#      这样人类能在少量样例里看到重复支撑，而不是面对一次性随机配色。


RED = int(Color.RED)
BLACK = int(Color.BLACK)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox_of(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not points:
        return None
    return tuple(map(int, bbox_coords(np.array(points, dtype=int))))


def _component_colors(grid: np.ndarray, coords: np.ndarray) -> set[int]:
    return {int(grid[int(r), int(c)]) for r, c in coords}


def _sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _front_red_face(red_coords: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = map(int, direction)
    if (dr, dc) == (1, 0):
        edge = min(int(r) for r, _ in red_coords)
        return [(int(r), int(c)) for r, c in red_coords if int(r) == int(edge)]
    if (dr, dc) == (-1, 0):
        edge = max(int(r) for r, _ in red_coords)
        return [(int(r), int(c)) for r, c in red_coords if int(r) == int(edge)]
    if (dr, dc) == (0, 1):
        edge = min(int(c) for _, c in red_coords)
        return [(int(r), int(c)) for r, c in red_coords if int(c) == int(edge)]
    edge = max(int(c) for _, c in red_coords)
    return [(int(r), int(c)) for r, c in red_coords if int(c) == int(edge)]


def _sprite_info(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    colors = _component_colors(grid, coords) - {int(bg)}
    other_colors = colors - {int(RED)}
    if int(RED) not in colors or len(other_colors) != 1:
        return None

    pts = [(int(r), int(c)) for r, c in coords]
    red_pts = [(int(r), int(c)) for r, c in pts if int(grid[int(r), int(c)]) == int(RED)]
    if not red_pts:
        return None

    return {
        "coords": pts,
        "bbox": tuple(map(int, bbox_coords(coords))),
        "red_coords": red_pts,
        "main_color": int(next(iter(other_colors))),
    }


def _structure_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    pts = {(int(r), int(c)) for r, c in coords}
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    sample_r, sample_c = next(iter(pts))
    return {
        "coords": pts,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "color": int(grid[int(sample_r), int(sample_c)]),
    }


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    sprites: list[dict] = []
    structures: list[dict] = []
    for coords in components_mask(g != int(bg), connectivity=4):
        colors = _component_colors(g, coords)
        if int(RED) in colors:
            info = _sprite_info(g, coords, bg)
            if info is not None:
                sprites.append(info)
        else:
            structures.append(_structure_info(g, coords))

    red_at: dict[tuple[int, int], int] = {}
    for rid, sprite in enumerate(sprites):
        for rc in sprite["red_coords"]:
            red_at[(int(rc[0]), int(rc[1]))] = int(rid)

    struct_best: list[dict] = []
    for sid, structure in enumerate(structures):
        sr0, sc0, sr1, sc1 = map(int, structure["bbox"])
        best: dict | None = None

        for dr, dc in _DIRS:
            hits: dict[int, dict] = defaultdict(lambda: {"count": 0, "mind": 10**9, "sockets": []})
            for r, c in structure["coords"]:
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

                rid = red_at.get((int(rr), int(cc)))
                if rid is None:
                    continue

                rec = hits[int(rid)]
                rec["count"] += 1
                rec["sockets"].append((int(nr), int(nc)))
                dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
                if int(dist) < int(rec["mind"]):
                    rec["mind"] = int(dist)

            if not hits:
                continue

            rid_best = max(hits.keys(), key=lambda k: (int(hits[k]["count"]), -int(hits[k]["mind"])))
            rec = hits[int(rid_best)]
            socket_bbox = _bbox_of(rec["sockets"])
            if socket_bbox is None:
                continue

            cand = {
                "score": (int(rec["count"]), -int(rec["mind"])),
                "rid": int(rid_best),
                "direction": (int(dr), int(dc)),
                "socket_bbox": socket_bbox,
                "sid": int(sid),
            }
            if best is None or cand["score"] > best["score"]:
                best = cand

        if best is not None:
            struct_best.append(best)

    struct_best.sort(reverse=True, key=lambda rec: rec["score"])
    used_rids: set[int] = set()
    assignments: list[dict] = []
    for rec in struct_best:
        rid = int(rec["rid"])
        if rid in used_rids:
            continue
        used_rids.add(rid)
        assignments.append(rec)

    return {
        "bg": int(bg),
        "sprites": sprites,
        "structures": structures,
        "assignments": assignments,
    }


def _assignment_shift(sprite: dict, direction: tuple[int, int], socket_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    face = _front_red_face(sprite["red_coords"], direction)
    face_bbox = _bbox_of(face)
    assert face_bbox is not None
    return int(socket_bbox[0] - face_bbox[0]), int(socket_bbox[1] - face_bbox[1])


def _trail_and_final_masks(sprite: dict, shift_r: int, shift_c: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    final_cells = {(int(r + shift_r), int(c + shift_c)) for r, c in sprite["coords"]}
    steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
    step_dr = _sign(int(shift_r))
    step_dc = _sign(int(shift_c))
    trail_cells: set[tuple[int, int]] = set()
    for step in range(steps):
        off_r = int(step * step_dr)
        off_c = int(step * step_dc)
        for r, c in sprite["coords"]:
            trail_cells.add((int(r + off_r), int(c + off_c)))
    trail_cells -= final_cells
    return trail_cells, final_cells


def _fusion_stats(inp: np.ndarray, out: np.ndarray) -> dict:
    analysis = _analyze_scene(inp)
    visible_trail_cells = 0
    multicolor_docks = 0
    trail_colors: set[int] = set()

    for plan in analysis["assignments"]:
        sprite = analysis["sprites"][int(plan["rid"])]
        shift_r, shift_c = _assignment_shift(sprite, plan["direction"], plan["socket_bbox"])
        trail_cells, final_cells = _trail_and_final_masks(sprite, shift_r, shift_c)
        main_color = int(sprite["main_color"])

        visible_trail_cells += sum(
            1 for r, c in trail_cells if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and int(out[r, c]) == int(main_color)
        )
        if any(0 <= r < out.shape[0] and 0 <= c < out.shape[1] and int(out[r, c]) == int(main_color) for r, c in trail_cells):
            trail_colors.add(int(main_color))

        final_colors = {
            int(out[r, c])
            for r, c in final_cells
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
        }
        if int(RED) in final_colors and int(main_color) in final_colors:
            multicolor_docks += 1

    return {
        "assigned": int(len(analysis["assignments"])),
        "distinct_trail_colors": int(len(trail_colors)),
        "visible_trail_cells": int(visible_trail_cells),
        "multicolor_docks": int(multicolor_docks),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide two-color sprites into comb sockets and leave each swept path in the sprite's own main color.

    Input structure:
    - Background is the dominant color.
    - Large monochrome comb/bracket structures contain interior socket cells just behind their openings.
    - Movable objects are disconnected two-color sprites: a red leading face plus a non-red main-color trailing face.

    Rule:
    - For each structure, ray-cast outward from interior socket cells and identify the sprite whose red leading face is visible.
    - Align that sprite by its red face, slide it straight into the socket, and recolor every intermediate occupied position to the sprite's main color.
    - Redraw the final docked sprite with its original two-color pattern and erase every unmatched sprite to background.

    Key insight:
    - The red subshape still determines docking geometry, but the full component must be preserved. The solver therefore separates
      "which pixels are used to detect/align the sprite" from "which pixels are repainted for the trail and final imprint".
    """
    g = np.array(grid, dtype=int)
    analysis = _analyze_scene(g)
    bg = int(analysis["bg"])
    h, w = g.shape

    assert analysis["sprites"], "expected at least one multicolor sprite"
    assert analysis["structures"], "expected at least one structure"

    out = g.copy()
    for sprite in analysis["sprites"]:
        for r, c in sprite["coords"]:
            out[int(r), int(c)] = int(bg)

    for plan in analysis["assignments"]:
        sprite = analysis["sprites"][int(plan["rid"])]
        shift_r, shift_c = _assignment_shift(sprite, plan["direction"], plan["socket_bbox"])
        steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
        step_dr = _sign(int(shift_r))
        step_dc = _sign(int(shift_c))
        trail_color = int(sprite["main_color"])

        for step in range(steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in sprite["coords"]:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(trail_color)

        for r, c in sprite["coords"]:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(g[int(r), int(c)])

    return out


# --------------- generator (combo variant) ---------------

GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 8)
SPRITE_MAIN_COLORS = (5, 6, 7, 9)

STRUCT_WIDTH_TEETH = ((8, 2), (9, 3))
SCENARIOS = (
    (2, 0),  # two matched structures
    (3, 0),  # three matched structures
    (3, 1),  # two matched structures + one empty distractor structure
)
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 4

TOTAL_SPRITES_MIN = 5
TOTAL_SPRITES_MAX = 9
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 6

MAX_RETRIES = 300


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _sprite_patch(teeth: int, bg: int, main_color: int) -> np.ndarray:
    patch = np.full((2, int(teeth)), int(main_color), dtype=int)
    patch[0, :] = int(RED)
    return patch


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    g = np.full((4, int(width)), int(bg), dtype=int)
    start = (int(width) - int(teeth)) // 2
    g[0, :] = int(sc)
    g[1, 1] = g[1, int(width) - 2] = int(sc)
    g[2:4, 1] = g[2:4, int(width) - 2] = int(sc)
    g[1, start : start + int(teeth)] = int(sc)
    return g


def _matched_patches(width: int, teeth: int, slide: int, rot_k: int, bg: int, sc: int, main_color: int) -> dict:
    base = _structure_array(width, teeth, bg, sc)
    sprite = _sprite_patch(teeth, bg, main_color)
    patch_h = 2 + int(slide) + sprite.shape[0]
    start = (int(width) - int(teeth)) // 2

    structure = np.full((patch_h, int(width)), int(bg), dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    for off in range(1, int(slide) + 1):
        output_patch[2 + off : 2 + off + sprite.shape[0], start : start + int(teeth)] = int(main_color)
    paste(output_patch, sprite, 2, start, transparent=bg)
    paste(input_patch, sprite, 2 + int(slide), start, transparent=bg)

    return {
        "structure": np.rot90(structure, k=int(rot_k)),
        "input": np.rot90(input_patch, k=int(rot_k)),
        "output": np.rot90(output_patch, k=int(rot_k)),
        "matched": True,
        "main_color": int(main_color),
    }


def _unmatched_patches(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    structure = np.rot90(_structure_array(width, teeth, bg, sc), k=int(rot_k))
    return {
        "structure": structure,
        "input": structure.copy(),
        "output": structure.copy(),
        "matched": False,
        "main_color": None,
    }


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
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
    try:
        return random_free_location(canvas, patch, transparent=int(bg), padding=int(padding))
    except ValueError:
        return None


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    inp = np.full((int(h), int(w)), int(bg), dtype=int)
    out = np.full((int(h), int(w)), int(bg), dtype=int)
    struct_grid = np.full((int(h), int(w)), int(bg), dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(out, spec["output"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(out, spec["output"], int(r0), int(c0), transparent=bg)
        paste(inp, spec["input"], int(r0), int(c0), transparent=bg)
        paste(struct_grid, spec["structure"], int(r0), int(c0), transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == int(sc))
        coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in coords_arr]
        rows = [int(r) for r, _ in coords]
        cols = [int(c) for _, c in coords]
        placed.append({"coords": coords, "bbox": (min(rows), min(cols), max(rows), max(cols))})

    return inp, out, struct_grid, placed


def _place_distractors(inp: np.ndarray, struct_grid: np.ndarray, placed: list[dict], bg: int, sc: int, count: int) -> bool:
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    teeth_options = [2, 3]
    for _ in range(int(count)):
        sprite_mask = (inp == int(RED)) | np.isin(inp, np.array(SPRITE_MAIN_COLORS, dtype=int))
        blocked = forbidden | bbox_blocked | (inp != int(bg)) | _dilate4(sprite_mask)
        canvas = np.full((h, w), int(bg), dtype=int)
        canvas[blocked] = int(sc)

        random.shuffle(teeth_options)
        ok = False
        for teeth in teeth_options:
            patch = _sprite_patch(int(teeth), bg, random.choice(SPRITE_MAIN_COLORS))
            patch = np.rot90(patch, k=random.randrange(4))
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is None:
                continue
            paste(inp, patch, int(loc[0]), int(loc[1]), transparent=bg)
            ok = True
            break
        if not ok:
            return False
    return True


def _gen_attempt(scenario: tuple[int, int] | None = None) -> np.ndarray | None:
    bg = int(random.choice(BG_COLORS))
    sc = int(random.choice(STRUCT_COLORS))
    h = int(random.randint(*GRID_H_RANGE))
    w = int(random.randint(*GRID_W_RANGE))
    n_structs, unmatched = random.choice(SCENARIOS) if scenario is None else scenario
    matched = int(n_structs - unmatched)
    if matched < 2:
        return None

    min_dist = max(DISTRACTOR_MIN, TOTAL_SPRITES_MIN - matched)
    max_dist = min(DISTRACTOR_MAX, TOTAL_SPRITES_MAX - matched)
    if min_dist > max_dist:
        return None
    distractor_count = int(random.randint(min_dist, max_dist))

    matched_colors = random.sample(list(SPRITE_MAIN_COLORS), k=matched)
    specs: list[dict] = []
    for idx in range(n_structs):
        width, teeth = random.choice(STRUCT_WIDTH_TEETH)
        rot_k = random.randrange(4)
        if idx < matched:
            slide = random.randint(*SLIDE_RANGE)
            specs.append(_matched_patches(width, teeth, slide, rot_k, bg, sc, matched_colors[idx]))
        else:
            specs.append(_unmatched_patches(width, teeth, rot_k, bg, sc))
    random.shuffle(specs)
    specs.sort(key=lambda spec: np.count_nonzero(spec["output"] != int(bg)), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    analysis = _analyze_scene(inp)
    if len(analysis["structures"]) != n_structs or len(analysis["assignments"]) != matched:
        return None

    if not _place_distractors(inp, struct_grid, placed, bg, sc, distractor_count):
        return None

    try:
        actual = main(inp)
        stats = _fusion_stats(inp, out)
    except Exception:
        return None

    if not np.array_equal(actual, out):
        return None

    if stats["assigned"] != matched:
        return None
    if stats["multicolor_docks"] != matched:
        return None
    if stats["distinct_trail_colors"] < 2:
        return None
    if stats["visible_trail_cells"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    scenario = random.choice(SCENARIOS)
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(scenario=scenario)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo variant 23")
