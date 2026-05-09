# Combo variant of 8b9c3697 (index 5)
# combo_spec: D3/2+D4/0+D5/1
# Name: 纯物理平移模型

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for combo D3/2 + D4/0 + D5/1
# ===============================================
# a) 具体化与行为追踪
# - 以开口朝下的一个 matched 结构为例：原题里，结构内部有预设插槽，位于外部下方的红色矩形会被吸入
#   内部，路径全部涂黑，所有未匹配红块被擦掉。
# - 本融合后，结构改成“前脸实心、无插槽”的碰撞面。红块仍从外部沿单一轴向靠近结构，但终点不再是
#   bbox 内的某个 socket，而是“再前进一步就会撞到结构”的最后安全位置，也就是紧贴结构前脸的外侧。
# - 同时去掉轨迹涂黑；融合描述明确写了“源位置恢复为背景色”，因此这里采用的是“只取消黑拖尾”而
#   不是 D4/0 单变体里那种“保留 matched 源红块副本”。于是 matched 红块真正发生平移：源位置清空，
#   路径保持背景色，只在终点出现红块。未匹配红块按 D5/1 保留原位不擦除。
# - 具体小例子：若一个 2x4 红块在结构下方 6 格处，原题输出会出现内部红块 + 6 步黑拖尾；本融合输出
#   则是该 2x4 红块移动到结构外侧贴边位置，原位置变回背景，6 格通道保持纯背景，其他干扰红块不动。
#
# b) 复杂度校准
# - 保留 2-3 个结构、四向旋转、4-8 格滑移、2-8 个干扰红块，避免题目退化为“把所有红块贴到最近结构”。
# - 碰撞面的接触宽度只使用 3/4/5 三档，重复出现在同一批样例里，保证人类能从少量样例中学到规则。
# - 输出里没有黑拖尾，信息量下降，因此必须保留多结构和大量未匹配红块，让“哪些红块移动了、哪些保留”
#   仍需要整体空间推理，而不是靠局部邻接直接看出。
#
# c) 隐式假设审查
# - 原 solver 依赖“插槽在结构 bbox 内且相邻 bg 单元可作为 anchor”；D3/2 破坏了这个假设，因此整个
#   socket 搜索必须替换成“从结构边缘向外看见红块，再反向平移直到碰撞”的逻辑。
# - 原 build-output 阶段默认 `out[out == RED] = bg`，等价于“所有源红都要消失”。融合后这个假设只对
#   matched 红块成立；unmatched 红块必须保留，所以输出构造改成“仅擦除被分配到结构的源红块”。
# - D3 单变体默认会把路径涂黑；D4/0 破坏了“路径需要显式编码”的假设，因此所有逐步 `BLACK` 写入都删掉。
# - 新 solver 的额外假设是：matched 红块与结构之间存在清晰的纯背景直线路径，且最终停靠位置不会与其他
#   红块重叠。生成器通过 front-ray 保留区、footprint 预留和 assignment 复验来保证这些条件成立。
#
# d) 放置合法性
# - 生成器仍然 output-first，但为了解决“输出里没有拖尾，无法自然占位”的问题，额外维护一个 footprint：
#   它包含结构本体、最终停靠位、源红块位置以及两者之间整条直线走廊。结构放置时按 footprint 防重叠。
# - distractor 放置时同时写入 input 和 output，因为 D5/1 规定 unmatched 红块在输出中也要保留。
# - distractor 仍需避开所有结构前方射线，否则会被 solver 误识别为 matched；放置后再次用 solver 计数
#   校验 `matched` 数量，拒绝任何歧义样本。
#
# e) 视觉可观察性
# - 与原题相比，最明显的差别有三点：终点位于结构外侧而非内部；没有任何黑拖尾；未匹配红块不会消失。
# - 每个有效样例至少有 1 个 matched 结构和至少 2 个 distractor，因此观察者能同时看到“有些红块移动了”
#   和“另一些红块完全不动”，足以从 2-3 个样例归纳出融合规则。
#
# f) 可视化整洁度
# - 结构之间保留 padding，且每个 matched 结构的整条走廊都被 footprint 预留，避免多个对象挤到同一条
#   推理通道里。
# - 没有黑拖尾后，输出更干净；为了防止“终点和原地红块混成一团”，front-ray 约束禁止在终点通道上放置
#   干扰红块，使移动结果始终可分辨。
#
# g) 变化的可推断性
# - 颜色体系仍只有背景色、结构色、红色三类可见角色，不引入新的颜色映射。
# - 规则完全由几何关系决定：是否在结构前方同轴可见、是否会撞到结构、是否被任何结构匹配。没有使用颜色
#   数值大小、组件编号顺序等人类不可见因素参与行为判断。


GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_SPAN = ((7, 3), (8, 4), (9, 3), (9, 5))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 2

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

MAX_RETRIES = 300

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _rotate_dir(dirv: tuple[int, int], k: int) -> tuple[int, int]:
    dr, dc = map(int, dirv)
    for _ in range(int(k) % 4):
        dr, dc = -dc, dr
    return int(dr), int(dc)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _collect_red_data(
    g: np.ndarray,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]], dict[tuple[int, int], int]]:
    red_val = int(Color.RED)
    red_comps = components_mask(g == red_val, connectivity=4)
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


def _collect_structures(
    g: np.ndarray, bg: int
) -> tuple[list[set[tuple[int, int]]], list[tuple[int, int, int, int]], np.ndarray]:
    red_val = int(Color.RED)
    struct_mask = (g != int(bg)) & (g != red_val)
    struct_comps = components_mask(struct_mask, connectivity=4)
    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))
    return struct_coords, struct_bbox, struct_mask


def _visibility_info(
    g: np.ndarray,
    bg: int,
) -> tuple[
    list[list[tuple[int, int]]],
    list[tuple[int, int, int, int]],
    list[set[tuple[int, int]]],
    list[tuple[int, int, int, int]],
    list[tuple[tuple[int, int], int, tuple[int, int], int]] | None,
]:
    red_coords, red_bbox, red_at = _collect_red_data(g)
    struct_coords, struct_bbox, _struct_mask = _collect_structures(g, bg)

    if not red_coords or not struct_coords:
        return red_coords, red_bbox, struct_coords, struct_bbox, None

    h, w = g.shape
    red_val = int(Color.RED)
    struct_best: list[tuple[tuple[int, int], int, tuple[int, int], int]] = []

    for sid, pts in enumerate(struct_coords):
        best = None
        for dr, dc in _DIRS:
            hits: dict[int, list[int]] = defaultdict(lambda: [0, 10**9])
            for r, c in pts:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) != int(bg):
                    continue

                rr, cc = int(nr), int(nc)
                while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(bg):
                    rr += int(dr)
                    cc += int(dc)
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == red_val:
                    rid = int(red_at[(int(rr), int(cc))])
                    rec = hits[rid]
                    rec[0] += 1
                    dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
                    rec[1] = min(int(rec[1]), int(dist))

            if not hits:
                continue
            rid_best = max(hits, key=lambda rid: (int(hits[rid][0]), -int(hits[rid][1])))
            cnt, mind = hits[rid_best]
            cand = ((int(cnt), -int(mind)), int(rid_best), (int(dr), int(dc)), int(sid))
            if best is None or cand > best:
                best = cand
        struct_best.append(best)

    return red_coords, red_bbox, struct_coords, struct_bbox, struct_best


def _move_until_collision(
    coords: list[tuple[int, int]],
    move_vec: tuple[int, int],
    struct_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], int]:
    h, w = struct_mask.shape
    mv_r, mv_c = map(int, move_vec)
    cur = [(int(r), int(c)) for r, c in coords]
    steps = 0
    while True:
        nxt = [(int(r + mv_r), int(c + mv_c)) for r, c in cur]
        blocked = False
        for rr, cc in nxt:
            if not (0 <= rr < h and 0 <= cc < w):
                blocked = True
                break
            if bool(struct_mask[rr, cc]):
                blocked = True
                break
        if blocked:
            return cur, int(steps)
        cur = nxt
        steps += 1


def _solver_assignments(
    g: np.ndarray, bg: int
) -> tuple[list[list[tuple[int, int]]], np.ndarray, list[tuple[int, tuple[int, int]]]]:
    red_coords, _red_bbox, _struct_coords, _struct_bbox, struct_best = _visibility_info(g, bg)
    assert red_coords, "expected at least one red component"
    assert struct_best is not None, "expected at least one non-bg non-red structure"

    _struct_coords2, _struct_bbox2, struct_mask = _collect_structures(g, bg)
    scored = [item for item in struct_best if item is not None]
    scored.sort(reverse=True, key=lambda t: t[0])

    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int]]] = []
    for _score, rid, dirv, _sid in scored:
        if int(rid) in used_red:
            continue
        used_red.add(int(rid))
        assignments.append((int(rid), (int(dirv[0]), int(dirv[1]))))
    return red_coords, struct_mask, assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide only the structure-aligned red rectangles until they physically stop against a solid structure face.

    Input structure:
    - The most common color is background. Large non-background, non-red components are solid comb/bracket-like structures with an exposed front face rather than an interior socket.
    - Red rectangles lie outside the structures. Some sit on a clear row/column in front of one structure; the rest are unmatched distractors.

    Rule:
    - For each structure, ray-cast outward from its exposed boundary cells to locate the visible red component aligned with that face.
    - Move that matched red component back along the same axis until the next step would hit the structure. Erase its source cells to background and redraw it only at the collision-stop position.
    - Leave the traversed path unchanged, and keep every unmatched red component exactly where it started.

    Key insight: This is a pure translation task with no preset socket and no trail encoding; the final red position is determined solely by collision geometry, while unmatched reds remain untouched.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_val = int(Color.RED)

    assert red_val in set(map(int, np.unique(g)))

    red_coords, struct_mask, assignments = _solver_assignments(g, bg)

    out = g.copy()
    for rid, dirv in assignments:
        coords = red_coords[int(rid)]
        move_vec = (-int(dirv[0]), -int(dirv[1]))
        final_coords, _steps = _move_until_collision(coords, move_vec, struct_mask)

        for r, c in coords:
            out[int(r), int(c)] = int(bg)
        for rr, cc in final_coords:
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = red_val

    return out


def _structure_array(width: int, span: int, bg: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.full((4, width), bg, dtype=int)
    front = np.zeros((4, width), dtype=bool)
    start = (int(width) - int(span)) // 2

    g[0, :] = sc
    g[1, 1] = g[2, 1] = sc
    g[1, width - 2] = g[2, width - 2] = sc
    g[1:4, start:start + span] = sc
    front[3, start:start + span] = True
    return g, front


def _matched_patches(width: int, span: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    red_val = int(Color.RED)
    red_h = 2

    base, front = _structure_array(width, span, bg, sc)
    patch_h = 4 + int(slide) + red_h

    structure = np.full((patch_h, width), bg, dtype=int)
    front_mask = np.zeros((patch_h, width), dtype=bool)
    footprint = np.full((patch_h, width), bg, dtype=int)

    paste(structure, base, 0, 0, transparent=bg)
    paste(front_mask, front.astype(int), 0, 0, transparent=0)
    front_mask = front_mask.astype(bool)

    input_patch = structure.copy()
    output_patch = structure.copy()
    start = (int(width) - int(span)) // 2

    final_r0 = 4
    source_r0 = 4 + int(slide)

    input_patch[source_r0:source_r0 + red_h, start:start + span] = red_val
    output_patch[final_r0:final_r0 + red_h, start:start + span] = red_val

    paste(footprint, base, 0, 0, transparent=bg)
    footprint[final_r0:source_r0 + red_h, start:start + span] = sc

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "footprint": np.rot90(footprint, k=rot_k),
        "front_mask": np.rot90(front_mask, k=rot_k),
        "front_dir": _rotate_dir((1, 0), rot_k),
    }


def _unmatched_patches(width: int, span: int, rot_k: int, bg: int, sc: int) -> dict:
    structure, front = _structure_array(width, span, bg, sc)
    rotated_structure = np.rot90(structure, k=rot_k)
    return {
        "structure": rotated_structure,
        "input": rotated_structure.copy(),
        "output": rotated_structure.copy(),
        "footprint": rotated_structure.copy(),
        "front_mask": np.rot90(front, k=rot_k),
        "front_dir": _rotate_dir((1, 0), rot_k),
    }


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        dr, dc = item["front_dir"]
        for r, c in item["front_coords"]:
            rr, cc = int(r + dr), int(c + dc)
            while 0 <= rr < h and 0 <= cc < w and int(struct_grid[rr, cc]) == int(bg):
                forbidden[rr, cc] = True
                rr += int(dr)
                cc += int(dc)
    return forbidden


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _assignment_count_matches(grid: np.ndarray, bg: int, expected_matched: int) -> bool:
    try:
        _red_coords, _struct_mask, assignments = _solver_assignments(grid, bg)
    except AssertionError:
        return False
    return len(assignments) == int(expected_matched)


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    reserve_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(reserve_grid, spec["footprint"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(reserve_grid, spec["footprint"], r0, c0, transparent=bg)
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == sc)
        coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]

        front_arr = np.argwhere(spec["front_mask"])
        front_coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in front_arr]

        placed.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "front_coords": front_coords,
                "front_dir": spec["front_dir"],
            }
        )

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
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0:r1 + 1, c0:c1 + 1] = True

    input_red_mask = inp == int(Color.RED)
    output_red_mask = out == int(Color.RED)
    shapes = list(DISTRACTOR_SHAPES)

    for _ in range(int(count)):
        # Keep unmatched reds isolated from both source reds in the input and
        # the future collision-stop reds in the output, so moved/static objects
        # remain visually separable after translation.
        blocked = (
            forbidden
            | bbox_blocked
            | (inp != bg)
            | _dilate4(input_red_mask)
            | _dilate4(output_red_mask)
        )
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
            input_red_mask[loc[0]:loc[0] + rh, loc[1]:loc[1] + rw] = True
            output_red_mask[loc[0]:loc[0] + rh, loc[1]:loc[1] + rw] = True
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
    unmatched = 1 if random.random() < UNMATCHED_PROB else 0
    matched = int(n_structs - unmatched)
    if matched <= 0:
        return None

    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)

    specs = []
    for idx in range(n_structs):
        width, span = random.choice(STRUCT_WIDTH_SPAN)
        rot_k = random.randrange(4)
        if idx < matched:
            slide = random.randint(*SLIDE_RANGE)
            specs.append(_matched_patches(width, span, slide, rot_k, bg, sc))
        else:
            specs.append(_unmatched_patches(width, span, rot_k, bg, sc))

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["footprint"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    if not _assignment_count_matches(inp, bg, matched):
        return None

    if not _place_distractors(inp, out, struct_grid, placed, bg, sc, distractor_count):
        return None

    if not _assignment_count_matches(inp, bg, matched):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo 5")
