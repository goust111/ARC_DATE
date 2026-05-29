# Combo variant of 8b9c3697 (index 8)
# combo_spec: D2/2+D3/1
# Name: 方向标记停靠杆

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


"""
design_analysis:
  fused_rule:
    summary: |
      原题是“结构内部插槽 + 远距离看见红块就把它滑回去”。
      这里同时替换两件事：
      1) 红块升级成“红核 + 单像素方向标记”的精灵，方向从局部读出；
      2) 目标不再是封闭空腔，而是开放式停靠杆，精灵沿自己的标记方向前进，碰到杆就停。

  concrete_pixel_design:
    structure_shape: |
      每个结构仍然保留“大型支架”观感：一条 7-9 格的背板 + 一根细连接臂 + 一段 1-3 格的停靠杆。
      规范朝向下（opening-down）时，结构是 4 行高：
      row0 整行背板，row1-row2 一列连接臂，row3 中央短横杆。
      旋转后可得到上下左右四个开口方向。
    matched_sprite: |
      红核仍是实心矩形，尺寸继续从原题 teeth 数继承：
      垂直停靠时尺寸 = min(rod_len,2) x rod_len；
      水平停靠时是它的旋转。
      单像素粉色标记贴在红核将要前进的一侧中央，表示移动方向。
    output_effect: |
      匹配精灵只移动红核；标记不保留。
      红核从源位置到停靠位置之间的每一个中间红核位置都涂成黑色。
      所有不匹配精灵整组擦除。

  behavior_trace:
    local_example: |
      设一个 opening-down 结构：顶部是长背板，下面吊着一段 3 格横杆。
      输入里在更下方放一个 2x3 红核，它上边外侧中央再贴 1 个粉色像素。
      solver 先从粉色像素读出“向上移动”，再检查这块红核的列范围是否已经和横杆对齐；
      若对齐，就沿竖直方向上移，直到红核顶边紧贴横杆底边为止。
      原题在这里会尝试寻找“结构内部 socket”；融合后不存在封闭 cavity，只剩开放杆件碰撞停靠。
    visible_difference: |
      人眼会看到：目标结构从“梳齿槽”变成“背板吊杆”，红块多了彩色方向触角，
      输出不再是塞进腔体，而是挂靠在开放杆旁边。

  complexity_calibration:
    why_this_is_not_too_simple: |
      若只做“一根孤立小杆”，画面会过小且容易和噪声混淆；保留宽背板能维持原题的大结构视觉锚点。
      若让方向靠颜色 id 编码，人类难以归纳；这里方向完全由标记落在哪一侧决定。
    ambiguity_control: |
      匹配不仅要求“杆射线先看见某个标记”，还要求：
      - 标记方向必须朝向该杆；
      - 红核尺寸必须与杆长兼容；
      - 红核在与运动垂直的轴上必须已经和杆对齐。
      这样干扰精灵即使带标记，也很难误配。
    repeated_support: |
      方向只取四个正交档位，杆长只取 1/2/3 三档，样例中这些档位会重复出现，便于人类从少量例子总结规则。

  implicit_assumption_audit_of_original_main:
    structure_mask: |
      原代码把所有 non-bg 且 non-red 像素都当结构；现在方向标记会被误并入结构。
      新版先抽取“红核 + 单像素标记”的精灵，再把标记排除出结构连通域。
    socket_search: |
      原代码依赖“严格位于 bbox 内部的 socket cell”，这是封闭插槽假设。
      开放杆没有内部空腔，因此必须改为识别“满宽/满高背板 + 最深处停靠杆”的结构几何。
    direction_inference: |
      原代码从结构往外看见红色，就同时完成“找到对象 + 推断方向”。
      现在方向由标记局部编码，结构射线只能命中标记像素，不能再靠红色位置反推。
    alignment_logic: |
      原代码默认最终位置由插槽 face 决定。
      新版改成“红核与杆在垂直运动轴上已经对齐，最终只沿标记方向平移到紧贴杆的一格外侧”。
    cleanup: |
      原代码只需清除红色；现在输出还必须额外清除粉色标记。

  placement_legality:
    matched_generation: |
      继续使用 output-first：先画好“杆 + 已停靠红核 + 黑轨迹”，再反推出输入中的源红核和方向标记。
      标记被放在未来轨迹线上，因此输出阶段自然被黑轨迹覆盖或直接擦除，不会留下多余像素。
    distractors: |
      干扰对象也必须是完整精灵（红核+标记），放置时禁止落在任意停靠杆的可视射线上，
      并与已有精灵保持至少一层 4 邻域间距，避免视觉缠绕。

  visual_observability_and_cleanliness:
    observability: |
      2-4 组样例足以看到两个新规律同时生效：
      1) 真正移动的红精灵都带着同色单像素方向触角；
      2) 目标结构不是封闭槽，而是开放杆，红块最终停在杆的外侧。
    cleanliness: |
      结构仍然稀疏分布，杆长只有 1-3，标记只有 1 像素，背景颜色仍单一；
      因此即便有 5-11 个红精灵，总体画面仍然可读。

  new_solver_assumptions_guaranteed_by_generator:
    - 每个红色连通块恰好对应一个四邻域单像素方向标记。
    - 每个结构恰好只有一个满宽/满高背板边界，且最深层只有一段单行/单列停靠杆。
    - 真正可匹配的红核在与运动垂直的轴上已和目标杆完全对齐，因此只需单轴平移。
    - 干扰精灵不会把自己的标记放到任何停靠杆的可视射线上。
"""


MARKER_COLOR = int(Color.PINK)
RED = int(Color.RED)
BLACK = int(Color.BLACK)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _rot_dir(dirv: tuple[int, int], k: int) -> tuple[int, int]:
    dr, dc = map(int, dirv)
    for _ in range(int(k) % 4):
        dr, dc = -int(dc), int(dr)
    return int(dr), int(dc)


def _side_center(r0: int, c0: int, r1: int, c1: int, dirv: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, dirv)
    if dr == -1:
        return int(r0 - 1), int(c0 + (c1 - c0) // 2)
    if dr == 1:
        return int(r1 + 1), int(c0 + (c1 - c0) // 2)
    if dc == -1:
        return int(r0 + (r1 - r0) // 2), int(c0 - 1)
    return int(r0 + (r1 - r0) // 2), int(c1 + 1)


def _marker_dir(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int] | None:
    mr, mc = map(int, marker)
    r0, c0, r1, c1 = map(int, bbox)
    if mr == r0 - 1 and c0 <= mc <= c1:
        return (-1, 0)
    if mr == r1 + 1 and c0 <= mc <= c1:
        return (1, 0)
    if mc == c0 - 1 and r0 <= mr <= r1:
        return (0, -1)
    if mc == c1 + 1 and r0 <= mr <= r1:
        return (0, 1)
    return None


def _sprite_patch(height: int, width: int, move_dir: tuple[int, int]) -> np.ndarray:
    dr, dc = map(int, move_dir)
    if dr != 0:
        patch = np.full((height + 1, width), -1, dtype=int)
        if dr == -1:
            patch[0, (width - 1) // 2] = MARKER_COLOR
            patch[1:, :] = RED
        else:
            patch[:height, :] = RED
            patch[height, (width - 1) // 2] = MARKER_COLOR
    else:
        patch = np.full((height, width + 1), -1, dtype=int)
        if dc == -1:
            patch[(height - 1) // 2, 0] = MARKER_COLOR
            patch[:, 1:] = RED
        else:
            patch[:, :width] = RED
            patch[(height - 1) // 2, width] = MARKER_COLOR
    return patch


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _red_dims(rod_len: int) -> tuple[int, int]:
    return min(int(rod_len), 2), int(rod_len)


def _is_full_rectangle(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return len(coords) == int((r1 - r0 + 1) * (c1 - c0 + 1))


def _extract_sprites(
    g: np.ndarray,
) -> tuple[
    list[list[tuple[int, int]]],
    list[tuple[int, int, int, int]],
    dict[int, tuple[int, int]],
    dict[int, tuple[int, int]],
    np.ndarray,
]:
    red_comps = components_mask(g == RED, connectivity=4)
    assert red_comps, "expected at least one red sprite"

    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_comps):
        pts = [(int(r), int(c)) for r, c in coords]
        red_coords.append(pts)
        red_bbox.append(tuple(map(int, bbox_coords(coords))))
        for rc in pts:
            red_at[rc] = int(rid)

    marker_of_red: dict[int, tuple[int, int]] = {}
    move_dir_of_red: dict[int, tuple[int, int]] = {}
    marker_mask = np.zeros_like(g, dtype=bool)

    for coords in components_mask(g == MARKER_COLOR, connectivity=4):
        if len(coords) != 1:
            continue
        mr, mc = map(int, coords[0])
        adj_rids = {
            int(red_at[(rr, cc)])
            for rr, cc in ((mr - 1, mc), (mr + 1, mc), (mr, mc - 1), (mr, mc + 1))
            if (rr, cc) in red_at
        }
        if len(adj_rids) != 1:
            continue
        rid = int(next(iter(adj_rids)))
        dirv = _marker_dir((mr, mc), red_bbox[rid])
        if dirv is None:
            continue
        if rid in marker_of_red:
            raise AssertionError("each red component must have exactly one direction marker")
        marker_of_red[rid] = (int(mr), int(mc))
        move_dir_of_red[rid] = (int(dirv[0]), int(dirv[1]))
        marker_mask[int(mr), int(mc)] = True

    assert len(marker_of_red) == len(red_coords), "every red component must carry one side marker"
    return red_coords, red_bbox, marker_of_red, move_dir_of_red, marker_mask


def _infer_structure(coords: np.ndarray) -> dict:
    bbox = tuple(map(int, bbox_coords(coords)))
    r0, c0, r1, c1 = bbox
    pts = {(int(r), int(c)) for r, c in coords}
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)

    boundary_counts = {
        (-1, 0): sum(1 for r, _c in pts if int(r) == int(r0)),
        (1, 0): sum(1 for r, _c in pts if int(r) == int(r1)),
        (0, -1): sum(1 for _r, c in pts if int(c) == int(c0)),
        (0, 1): sum(1 for _r, c in pts if int(c) == int(c1)),
    }
    expected = {(-1, 0): w, (1, 0): w, (0, -1): h, (0, 1): h}
    full_sides = [side for side in boundary_counts if int(boundary_counts[side]) == int(expected[side])]
    assert len(full_sides) == 1, "each docking structure must have exactly one full back wall"
    back_side = full_sides[0]
    open_dir = (-int(back_side[0]), -int(back_side[1]))

    def depth(rc: tuple[int, int]) -> int:
        r, c = rc
        if back_side == (-1, 0):
            return int(r - r0)
        if back_side == (1, 0):
            return int(r1 - r)
        if back_side == (0, -1):
            return int(c - c0)
        return int(c1 - c)

    max_depth = max(depth(rc) for rc in pts)
    rod_coords = [rc for rc in pts if int(depth(rc)) == int(max_depth)]
    if open_dir[0] != 0:
        assert len({int(r) for r, _c in rod_coords}) == 1, "vertical approach expects a horizontal rod"
        rod_coords.sort(key=lambda x: (int(x[1]), int(x[0])))
    else:
        assert len({int(c) for _r, c in rod_coords}) == 1, "horizontal approach expects a vertical rod"
        rod_coords.sort(key=lambda x: (int(x[0]), int(x[1])))

    rod_bbox = tuple(map(int, bbox_coords(np.array(rod_coords, dtype=int))))
    return {
        "bbox": bbox,
        "open_dir": (int(open_dir[0]), int(open_dir[1])),
        "rod_coords": rod_coords,
        "rod_bbox": rod_bbox,
        "rod_len": int(len(rod_coords)),
    }


def _sprite_matches_rod(
    coords: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    open_dir: tuple[int, int],
    rod_bbox: tuple[int, int, int, int],
    rod_len: int,
) -> bool:
    if not _is_full_rectangle(coords, bbox):
        return False

    r0, c0, r1, c1 = map(int, bbox)
    rh = int(r1 - r0 + 1)
    rw = int(c1 - c0 + 1)

    if open_dir[0] != 0:
        expect = (min(int(rod_len), 2), int(rod_len))
        return (rh, rw) == expect and int(c0) == int(rod_bbox[1]) and int(c1) == int(rod_bbox[3])

    expect = (int(rod_len), min(int(rod_len), 2))
    return (rh, rw) == expect and int(r0) == int(rod_bbox[0]) and int(r1) == int(rod_bbox[2])


def _desired_top_left(
    red_bbox: tuple[int, int, int, int],
    rod_bbox: tuple[int, int, int, int],
    open_dir: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, red_bbox)
    rh = int(r1 - r0 + 1)
    rw = int(c1 - c0 + 1)
    rr0, rc0, rr1, rc1 = map(int, rod_bbox)
    dr, dc = map(int, open_dir)

    if dr == 1:
        return int(rr1 + 1), int(rc0)
    if dr == -1:
        return int(rr0 - rh), int(rc0)
    if dc == 1:
        return int(rr0), int(rc1 + 1)
    return int(rr0), int(rc0 - rw)


def _rod_structure_array(width: int, rod_len: int, bg: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.full((4, width), bg, dtype=int)
    rod_mask = np.zeros((4, width), dtype=bool)
    start = (int(width) - int(rod_len)) // 2
    stem_c = int(start + (int(rod_len) - 1) // 2)
    g[0, :] = int(sc)
    g[1:3, int(stem_c)] = int(sc)
    g[3, int(start) : int(start) + int(rod_len)] = int(sc)
    rod_mask[3, int(start) : int(start) + int(rod_len)] = True
    return g, rod_mask


def _matched_patches(width: int, rod_len: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    red_h, red_w = _red_dims(rod_len)
    base, rod_base = _rod_structure_array(width, rod_len, bg, sc)
    patch_h = 4 + int(slide) + int(red_h)

    structure = np.full((patch_h, width), bg, dtype=int)
    rod_mask = np.zeros((patch_h, width), dtype=bool)
    paste(structure, base, 0, 0, transparent=bg)
    rod_mask[: rod_base.shape[0], : rod_base.shape[1]] = rod_base

    input_patch = structure.copy()
    output_patch = structure.copy()

    start = (int(width) - int(rod_len)) // 2
    src_r0 = 4 + int(slide)
    input_patch[int(src_r0) : int(src_r0) + int(red_h), int(start) : int(start) + int(red_w)] = RED
    mr, mc = _side_center(
        int(src_r0),
        int(start),
        int(src_r0 + red_h - 1),
        int(start + red_w - 1),
        (-1, 0),
    )
    input_patch[int(mr), int(mc)] = MARKER_COLOR

    for off in range(1, int(slide) + 1):
        output_patch[4 + int(off) : 4 + int(off) + int(red_h), int(start) : int(start) + int(red_w)] = BLACK
    output_patch[4 : 4 + int(red_h), int(start) : int(start) + int(red_w)] = RED

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "rod_mask": np.rot90(rod_mask.astype(int), k=rot_k).astype(bool),
        "open_dir": _rot_dir((1, 0), rot_k),
    }


def _unmatched_patches(width: int, rod_len: int, rot_k: int, bg: int, sc: int) -> dict:
    base, rod_base = _rod_structure_array(width, rod_len, bg, sc)
    return {
        "structure": np.rot90(base, k=rot_k),
        "input": np.rot90(base, k=rot_k),
        "output": np.rot90(base, k=rot_k),
        "rod_mask": np.rot90(rod_base.astype(int), k=rot_k).astype(bool),
        "open_dir": _rot_dir((1, 0), rot_k),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move direction-marked red sprites onto open docking rods, leaving a black trail.

    Input structure:
    - bg is the most common color.
    - Large non-bg structures are no longer closed sockets: each one is a wide back wall with a short open docking rod.
    - Every red rectangle carries one adjacent single-pixel side marker that tells which orthogonal direction it must move.

    Rule:
    - Read the movement direction from each sprite's side marker.
    - For each structure, identify its open side from the full back wall and cast outward from the exposed rod cells.
    - A sprite can dock only if the ray first hits its marker, the marker direction points toward the rod, and the red core is already aligned with the rod on the perpendicular axis.
    - Slide the red core until it becomes edge-adjacent to the rod, paint all intermediate red positions black, erase every marker, and delete unmatched sprites.

    Key insight:
    - The target is now an open collision stop rather than an interior cavity, so the solver must combine local direction decoding from the sprite with geometric rod detection on the structure.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    assert RED in set(map(int, np.unique(g)))

    red_coords, red_bbox, marker_of_red, move_dir_of_red, marker_mask = _extract_sprites(g)

    struct_mask = (g != int(bg)) & (g != RED) & (g != MARKER_COLOR)
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one docking structure"

    structures = [_infer_structure(coords) for coords in struct_comps]
    marker_to_red = {marker_of_red[rid]: int(rid) for rid in marker_of_red}

    struct_best: list[tuple[tuple[int, int], int, int]] = []
    for sid, info in enumerate(structures):
        open_dr, open_dc = map(int, info["open_dir"])
        wanted_move = (-int(open_dr), -int(open_dc))
        hits: dict[int, list[int]] = defaultdict(lambda: [0, 10**9])

        for rr, cc in info["rod_coords"]:
            r = int(rr + open_dr)
            c = int(cc + open_dc)
            while 0 <= r < H and 0 <= c < W and int(g[r, c]) == int(bg):
                r += int(open_dr)
                c += int(open_dc)

            rid = marker_to_red.get((int(r), int(c)))
            if rid is None:
                continue
            if move_dir_of_red[int(rid)] != wanted_move:
                continue
            if not _sprite_matches_rod(
                red_coords[int(rid)],
                red_bbox[int(rid)],
                info["open_dir"],
                info["rod_bbox"],
                info["rod_len"],
            ):
                continue

            rec = hits[int(rid)]
            rec[0] += 1
            dist = abs(int(r) - int(rr)) + abs(int(c) - int(cc))
            if int(dist) < int(rec[1]):
                rec[1] = int(dist)

        if not hits:
            continue

        rid_best = max(hits.keys(), key=lambda k: (int(hits[k][0]), -int(hits[k][1])))
        cnt, mind = hits[int(rid_best)]
        struct_best.append(((int(cnt), -int(mind)), int(rid_best), int(sid)))

    struct_best.sort(reverse=True, key=lambda t: t[0])
    used_red: set[int] = set()
    assignments: list[tuple[int, int]] = []
    for _score, rid, sid in struct_best:
        if int(rid) in used_red:
            continue
        used_red.add(int(rid))
        assignments.append((int(rid), int(sid)))

    out = g.copy()
    out[out == RED] = int(bg)
    out[marker_mask] = int(bg)

    for rid, sid in assignments:
        coords = red_coords[int(rid)]
        bbox = red_bbox[int(rid)]
        dst_r0, dst_c0 = _desired_top_left(bbox, structures[int(sid)]["rod_bbox"], structures[int(sid)]["open_dir"])
        shift_r = int(dst_r0 - bbox[0])
        shift_c = int(dst_c0 - bbox[1])

        assert shift_r == 0 or shift_c == 0, "docking move must stay on a single axis"
        steps = abs(int(shift_r)) + abs(int(shift_c))
        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)

        for step in range(0, int(steps)):
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


# --------------- generator ---------------

GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_ROD = ((7, 1), (8, 2), (9, 1), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 4

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

MAX_RETRIES = 500

_ROD_WIDTHS_BY_LEN = {
    int(rod_len): tuple(sorted(width for width, rl in STRUCT_WIDTH_ROD if int(rl) == int(rod_len)))
    for rod_len in {int(rl) for _w, rl in STRUCT_WIDTH_ROD}
}
_ROD_LENS = tuple(sorted(_ROD_WIDTHS_BY_LEN))
_VERTICAL_ROTS = (0, 2)
_HORIZONTAL_ROTS = (1, 3)


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
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(out, spec["output"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = map(int, loc)
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == sc)
        coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in coords_arr]
        rows = [int(r) for r, _c in coords]
        cols = [int(c) for _r, c in coords]

        rod_arr = np.argwhere(spec["rod_mask"])
        rod_coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in rod_arr]
        rod_bbox = tuple(map(int, bbox_coords(np.array(rod_coords, dtype=int))))

        placed.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "rod_coords": rod_coords,
                "rod_bbox": rod_bbox,
                "open_dir": tuple(map(int, spec["open_dir"])),
            }
        )

    return inp, out, struct_grid, placed


def _rod_ray_mask(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        dr, dc = map(int, item["open_dir"])
        for r, c in item["rod_coords"]:
            rr = int(r + dr)
            cc = int(c + dc)
            while 0 <= rr < h and 0 <= cc < w and int(struct_grid[rr, cc]) == int(bg):
                forbidden[int(rr), int(cc)] = True
                rr += int(dr)
                cc += int(dc)
    return forbidden


def _place_distractors(
    inp: np.ndarray,
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
    sc: int,
    count: int,
) -> bool:
    h, w = inp.shape
    forbidden = _rod_ray_mask(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = map(int, item["bbox"])
        bbox_blocked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    shapes = list(DISTRACTOR_SHAPES)
    dirs = list(_DIRS)
    for _ in range(int(count)):
        sprite_mask = (inp == RED) | (inp == MARKER_COLOR)
        blocked = forbidden | bbox_blocked | (inp != bg) | _dilate4(sprite_mask)
        canvas = np.full((h, w), bg, dtype=int)
        canvas[blocked] = sc

        random.shuffle(shapes)
        random.shuffle(dirs)
        ok = False
        for rh, rw in shapes:
            for move_dir in dirs:
                patch = _sprite_patch(rh, rw, move_dir)
                loc = _place_patch(canvas, patch, bg, padding=0)
                if loc is None:
                    continue
                paste(inp, patch, int(loc[0]), int(loc[1]), transparent=-1)
                ok = True
                break
            if ok:
                break
        if not ok:
            return False
    return True


def _pick_rotations(count: int, *, force_axis_mix: bool) -> list[int]:
    if count <= 0:
        return []

    picks: list[int] = []
    if force_axis_mix and count >= 2:
        picks.append(random.choice(_VERTICAL_ROTS))
        picks.append(random.choice(_HORIZONTAL_ROTS))

    remaining = [rot for rot in range(4) if rot not in picks]
    random.shuffle(remaining)
    while len(picks) < int(count):
        if remaining:
            picks.append(remaining.pop())
        else:
            picks.append(random.randrange(4))

    random.shuffle(picks)
    return picks


def _pick_rod_lens(count: int) -> list[int]:
    if count <= 0:
        return []

    pool = list(_ROD_LENS)
    random.shuffle(pool)
    picks: list[int] = []
    while len(picks) < int(count):
        if pool:
            picks.append(int(pool.pop()))
        else:
            picks.append(int(random.choice(_ROD_LENS)))
    return picks


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, RED, BLACK, MARKER_COLOR)])
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
    matched_rots = _pick_rotations(matched, force_axis_mix=True)
    unmatched_rots = _pick_rotations(n_structs - matched, force_axis_mix=False)
    matched_lens = _pick_rod_lens(matched)
    unmatched_lens = _pick_rod_lens(n_structs - matched)

    for rod_len, rot_k in zip(matched_lens, matched_rots):
        width = int(random.choice(_ROD_WIDTHS_BY_LEN[int(rod_len)]))
        slide = random.randint(*SLIDE_RANGE)
        specs.append(_matched_patches(width, int(rod_len), slide, int(rot_k), bg, sc))

    for rod_len, rot_k in zip(unmatched_lens, unmatched_rots):
        width = int(random.choice(_ROD_WIDTHS_BY_LEN[int(rod_len)]))
        specs.append(_unmatched_patches(width, int(rod_len), int(rot_k), bg, sc))

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    if not np.array_equal(main(inp), out):
        return None

    if not _place_distractors(inp, struct_grid, placed, bg, sc, distractor_count):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo variant")
