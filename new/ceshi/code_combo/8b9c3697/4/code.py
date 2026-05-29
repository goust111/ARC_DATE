# Combo variant of 8b9c3697 (index 4)
# combo_spec: D1/2+D2/2
# Name: 连接器方向标记对接

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


"""
design_analysis:
  concretization_and_behavior_trace:
    scene_model: |
      保留原题的大型梳齿结构和“红块直线滑入、轨迹涂黑”的结果，但把 source 对象升级成
      三段式精灵：红色矩形核 + 一枚单像素方向标记 + 一个脱离式小连接器。
      同时每个大结构在开口唇边带一个附着式连接器接口。
    local_example: |
      以 opening-down 的标准朝向举例：结构下方真正可配对的红核上边贴着粉色方向点，
      表示它要向上滑；其左侧隔 1 格放着一个橙色/栗色的小 2-3 像素连接器。
      结构开口右下角也附着一个同色同形连接器。若另一个更靠近结构的红精灵也朝上，
      但它的连接器形状或颜色不同，则它不会入槽。若另一个精灵连接器相同但方向点不朝向
      结构，它也不会入槽。
    old_vs_new: |
      原题在“找谁滑入”与“朝哪边滑入”两件事上都依赖全局射线。融合后这两件事被拆开：
      连接器决定身份匹配，方向点决定运动方向。solver 必须先读局部标记，再做结构-精灵对接。

  complexity_calibration:
    connector_vocab: |
      连接器词表限制为 4 类：2 种颜色 x 2 种小形状（domino / ell）。每题只用 2-3 个结构，
      所以可以让结构签名互异，同时在样例中重复这些签名，保证人眼可追踪。
    why_two_decoy_types: |
      仅有“错连接器但同方向”的干扰，可能退化成纯 connector 题；仅有“同连接器但错方向”的
      干扰，又可能退化成纯 marker 题。因此生成器强制同时出现两类干扰：
      1) 同方向但错连接器，证明不能只看箭头；
      2) 同连接器但错方向，证明不能只看接口图案。
    controlled_freedom: |
      方向只取四个正交方向，连接器只取四类签名，红核仍保留原题矩形尺寸约束。
      这样变化明显，但不会引入无法从 2-4 个样例归纳的高自由度映射。

  implicit_assumptions_audit:
    original_structure_mask_breaks: |
      原题默认所有 non-bg/non-red 都是结构。融合后不成立，因为方向标记和红侧连接器也满足
      non-bg/non-red。新版先用固定颜色识别单像素方向标记，再用面积阈值把大结构与小连接器分开。
    original_ray_identity_breaks: |
      原题把“射线首先碰到红色”同时当作身份识别和方向推断。融合后射线不再负责身份识别；
      身份来自连接器签名，方向来自 marker 所在边。
    new_solver_assumptions: |
      新 solver 隐含要求：
      - 每个红核恰有一个正交相邻的单像素方向标记；
      - 每个红核恰有一个距离最近且唯一的脱离式连接器；
      - 每个大结构恰有一个附着式合法连接器；
      - 任一结构的签名对应至多一个“签名正确且方向正确”的红精灵。
      generator 需要逐项保证这些条件。
    output_cleanup: |
      输出中只有结构、黑轨迹和最终红核；所有方向标记与红侧连接器都必须清除。

  placement_legality:
    structure_patch: |
      结构连接器始终附着在开口外侧，且不进入滑行走廊；标准朝向中固定贴在结构右下唇边，
      再整体旋转到四个方向。
    sprite_patch: |
      红精灵的方向标记固定贴在前进侧中央，连接器固定在侧后方并与红核留 1 格空隙。
      这样 solver 可以稳定切出红核、marker 和 free connector 三类局部元素。
    overlap_control: |
      结构放置使用 input/output 联合 footprint，避免输入专有的精灵或干扰和其它结构重叠。
      外部干扰精灵额外禁止落在任意结构 cavity 向外的清晰走廊上，避免意外生成多个合法候选。

  visual_observability_and_cleanliness:
    observability: |
      人类在少量样例中能看到三种稳定重复的局部信号：
      - 结构边缘总带一个小接口；
      - 每个红精灵总带一个方向点；
      - 只有“接口图案相同且方向点朝向结构”的精灵会滑入。
    cleanliness: |
      标记只有 1 像素，连接器只有 2-3 像素；同一结构附近虽然会有对比干扰，但数量被限制，
      且结构间距保留原题量级，不会把网格压得过于拥挤。
"""


RED = int(Color.RED)
BLACK = int(Color.BLACK)
MARKER_COLOR = int(Color.PINK)

GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
SLIDE_RANGE = (5, 8)
STRUCT_PADDING = 4

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 9
SPECIAL_SAME_SIG_WRONG_DIR = 1
SPECIAL_SAME_DIR_WRONG_SIG = 1
MAX_RETRIES = 500

STRUCT_MIN_AREA = 8
FREE_CONNECTOR_MAX_AREA = 3
CONNECTOR_NEAR_DIST = 6

CONNECTOR_SHAPES = {
    "domino": ((0, 0), (0, 1)),
    "ell": ((0, 0), (1, 0), (1, 1)),
}
CONNECTOR_TYPES = (
    (int(Color.ORANGE), "domino"),
    (int(Color.ORANGE), "ell"),
    (int(Color.MAROON), "domino"),
    (int(Color.MAROON), "ell"),
)

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_TO_ROT = {
    (-1, 0): 0,
    (0, -1): 1,
    (1, 0): 2,
    (0, 1): 3,
}


def _normalize_shape(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0, c0 = min(rows), min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _rot_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape([(-int(c), int(r)) for r, c in shape])


ALLOWED_CONNECTOR_SHAPES: set[tuple[tuple[int, int], ...]] = set()
for _shape in CONNECTOR_SHAPES.values():
    cur = _normalize_shape(list(_shape))
    for _ in range(4):
        ALLOWED_CONNECTOR_SHAPES.add(cur)
        cur = _rot_shape(cur)


def _rotate_dir(dirv: tuple[int, int], k: int) -> tuple[int, int]:
    dr, dc = map(int, dirv)
    for _ in range(int(k) % 4):
        dr, dc = -int(dc), int(dr)
    return int(dr), int(dc)


def _opposite(dirv: tuple[int, int]) -> tuple[int, int]:
    return -int(dirv[0]), -int(dirv[1])


def _component_signature(
    grid: np.ndarray, coords: list[tuple[int, int]] | np.ndarray
) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    pts = [(int(r), int(c)) for r, c in coords]
    colors = {int(grid[r, c]) for r, c in pts}
    if len(colors) != 1:
        return None
    sig = int(next(iter(colors))), _normalize_shape(pts)
    if sig[0] not in {int(t[0]) for t in CONNECTOR_TYPES}:
        return None
    if sig[1] not in ALLOWED_CONNECTOR_SHAPES:
        return None
    return sig


def _min_manhattan(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    best = 10**9
    for ar, ac in a:
        for br, bc in b:
            best = min(best, abs(int(ar) - int(br)) + abs(int(ac) - int(bc)))
    return int(best)


def _nearest_connector_sig(
    coords: list[tuple[int, int]], free_connectors: list[dict]
) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    nearby: list[tuple[int, int]] = []
    for cid, info in enumerate(free_connectors):
        dist = _min_manhattan(coords, info["coords"])
        if int(dist) <= int(CONNECTOR_NEAR_DIST):
            nearby.append((int(dist), int(cid)))
    if not nearby:
        return None
    nearby.sort()
    best_dist = int(nearby[0][0])
    best_ids = [int(cid) for dist, cid in nearby if int(dist) == int(best_dist)]
    if len(best_ids) != 1:
        return None
    return free_connectors[int(best_ids[0])]["sig"]


def _split_points4(points: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = {tuple(map(int, p)) for p in points}
    groups: list[list[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        group = [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    group.append(nxt)
        groups.append(group)
    return groups


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


def _structure_direction(
    connector_pts: list[tuple[int, int]], body_bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, body_bbox)
    rows = [int(r) for r, _ in connector_pts]
    cols = [int(c) for _, c in connector_pts]
    if max(rows) > r1:
        return (1, 0)
    if min(rows) < r0:
        return (-1, 0)
    if max(cols) > c1:
        return (0, 1)
    if min(cols) < c0:
        return (0, -1)
    raise AssertionError("connector must sit outside the structure body bbox")


def _socket_cells(
    body_pts: list[tuple[int, int]],
    body_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    grid: np.ndarray,
    bg: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, body_bbox)
    dr, dc = map(int, direction)
    out: set[tuple[int, int]] = set()
    for r, c in body_pts:
        nr, nc = int(r + dr), int(c + dc)
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            continue
        if int(grid[nr, nc]) != int(bg):
            continue
        if not (r0 < nr < r1 and c0 < nc < c1):
            continue
        out.add((int(nr), int(nc)))
    return sorted(out)


def _red_on_open_side(
    red_bbox: tuple[int, int, int, int],
    body_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> bool:
    rr0, rc0, rr1, rc1 = map(int, red_bbox)
    sr0, sc0, sr1, sc1 = map(int, body_bbox)
    dr, dc = map(int, direction)
    if (dr, dc) == (1, 0):
        return rr0 > sr1
    if (dr, dc) == (-1, 0):
        return rr1 < sr0
    if (dr, dc) == (0, 1):
        return rc0 > sc1
    return rc1 < sc0


def _connector_dims(conn_type: tuple[int, str]) -> tuple[int, int]:
    offs = CONNECTOR_SHAPES[str(conn_type[1])]
    return max(dr for dr, _ in offs) + 1, max(dc for _, dc in offs) + 1


def _stamp_connector(
    patch: np.ndarray, anchor_r: int, anchor_c: int, conn_type: tuple[int, str], bg: int
) -> bool:
    color, shape_name = conn_type
    for dr, dc in CONNECTOR_SHAPES[str(shape_name)]:
        rr = int(anchor_r + dr)
        cc = int(anchor_c + dc)
        if not (0 <= rr < patch.shape[0] and 0 <= cc < patch.shape[1]):
            return False
        if int(patch[rr, cc]) != int(bg):
            return False
    for dr, dc in CONNECTOR_SHAPES[str(shape_name)]:
        rr = int(anchor_r + dr)
        cc = int(anchor_c + dc)
        patch[rr, cc] = int(color)
    return True


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
    g[0, :] = int(sc)
    g[1, 1] = int(sc)
    g[1, width - 2] = int(sc)
    g[2:4, 1] = int(sc)
    g[2:4, width - 2] = int(sc)
    g[1, start:start + teeth] = int(sc)
    return g


def _sprite_up_patch(rect_h: int, rect_w: int, bg: int, conn_type: tuple[int, str]) -> np.ndarray | None:
    conn_h, conn_w = _connector_dims(conn_type)
    gap = 1
    patch_h = max(int(rect_h + 1), int(1 + rect_h + conn_h))
    patch_w = int(conn_w + gap + rect_w)
    patch = np.full((patch_h, patch_w), bg, dtype=int)
    red_r0 = 1
    red_c0 = int(conn_w + gap)
    patch[red_r0:red_r0 + rect_h, red_c0:red_c0 + rect_w] = int(RED)
    patch[0, red_c0 + (rect_w - 1) // 2] = int(MARKER_COLOR)
    conn_r0 = int(red_r0 + rect_h)
    if not _stamp_connector(patch, conn_r0, 0, conn_type, bg):
        return None
    return patch


def _sprite_patch(
    rect_h: int, rect_w: int, bg: int, conn_type: tuple[int, str], move_dir: tuple[int, int]
) -> np.ndarray | None:
    base = _sprite_up_patch(rect_h, rect_w, bg, conn_type)
    if base is None:
        return None
    return np.rot90(base, k=DIR_TO_ROT[tuple(map(int, move_dir))])


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _extract_red_sprites(
    g: np.ndarray,
) -> tuple[
    list[list[tuple[int, int]]],
    list[tuple[int, int, int, int]],
    dict[int, tuple[int, int]],
    dict[int, tuple[int, int]],
    dict[tuple[int, int], int],
    np.ndarray,
]:
    red_comps = components_mask(g == int(RED), connectivity=4)
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

    for coords in components_mask(g == int(MARKER_COLOR), connectivity=4):
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
        assert rid not in marker_of_red, "each red component must have exactly one direction marker"
        marker_of_red[rid] = (int(mr), int(mc))
        move_dir_of_red[rid] = tuple(map(int, dirv))
        marker_mask[mr, mc] = True

    assert len(marker_of_red) == len(red_coords), "every red component must carry one side marker"
    return red_coords, red_bbox, marker_of_red, move_dir_of_red, red_at, marker_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert connector-matched, direction-marked red sprites into interior sockets of large structures, leaving a black trail.

    Input structure:
    - The background is the most common color. Large comb/bracket structures are non-bg and non-red.
    - Each structure has one attached connector interface on its opening side.
    - Each red sprite consists of a red rectangle, one adjacent marker pixel showing its travel direction, and one detached nearby connector piece.

    Rule:
    - Read each structure's connector signature from the attached interface.
    - Read each red sprite's movement direction from its single-pixel marker and its connector signature from the unique nearby detached connector.
    - A sprite docks only when its connector signature matches the structure's interface and its marker points toward that structure's opening.
    - Move only the red rectangle into the socket, paint all intermediate red positions black, and erase every unmatched sprite marker/connector.

    Key insight: Matching is now the conjunction of two local cues: the connector says which structure a sprite can dock with, and the side marker says whether it is facing the correct opening.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    assert int(RED) in set(map(int, np.unique(g)))

    red_coords, red_bbox, _marker_of_red, move_dir_of_red, _red_at, marker_mask = _extract_red_sprites(g)

    aux_mask = (g != int(bg)) & (g != int(RED)) & (g != int(MARKER_COLOR)) & (g != int(BLACK))
    aux_comps = components_mask(aux_mask, connectivity=4)
    assert aux_comps, "expected structures and connectors"

    struct_comps = [comp for comp in aux_comps if int(comp.shape[0]) >= int(STRUCT_MIN_AREA)]
    free_conn_comps = [comp for comp in aux_comps if int(comp.shape[0]) <= int(FREE_CONNECTOR_MAX_AREA)]
    assert struct_comps, "expected at least one structure component"

    free_connectors = []
    for comp in free_conn_comps:
        pts = [(int(r), int(c)) for r, c in comp]
        sig = _component_signature(g, pts)
        if sig is None:
            continue
        free_connectors.append({"coords": pts, "sig": sig})

    red_infos = []
    for rid, coords in enumerate(red_coords):
        red_infos.append(
            {
                "rid": int(rid),
                "coords": coords,
                "bbox": tuple(map(int, red_bbox[rid])),
                "sig": _nearest_connector_sig(coords, free_connectors),
                "move_dir": tuple(map(int, move_dir_of_red[int(rid)])),
            }
        )

    sig_to_reds: dict[tuple[int, tuple[tuple[int, int], ...]], list[int]] = {}
    for info in red_infos:
        if info["sig"] is not None:
            sig_to_reds.setdefault(info["sig"], []).append(int(info["rid"]))

    struct_infos = []
    for sid, comp in enumerate(struct_comps):
        pts = [(int(r), int(c)) for r, c in comp]
        color_counts = Counter(int(g[r, c]) for r, c in pts)
        body_color = int(color_counts.most_common(1)[0][0])
        body_pts = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == int(body_color)]
        raw_conn_pts = [(int(r), int(c)) for r, c in pts if int(g[r, c]) != int(body_color)]
        assert raw_conn_pts, "structure must carry an attached connector"
        body_bbox = tuple(map(int, bbox_coords(np.array(body_pts, dtype=int))))
        body_set = set(body_pts)

        conn_groups = []
        for color in sorted({int(g[r, c]) for r, c in raw_conn_pts}):
            color_pts = [(int(r), int(c)) for r, c in raw_conn_pts if int(g[r, c]) == int(color)]
            conn_groups.extend(_split_points4(color_pts))
        touching = [
            grp
            for grp in conn_groups
            if any((int(r + dr), int(c + dc)) in body_set for r, c in grp for dr, dc in DIRS)
        ]
        touching = [
            grp
            for grp in touching
            if len(grp) in (2, 3) and _normalize_shape(grp) in ALLOWED_CONNECTOR_SHAPES
        ]
        assert len(touching) == 1, "expected exactly one legal attached connector"
        conn_pts = touching[0]
        conn_sig = _component_signature(g, conn_pts)
        assert conn_sig is not None
        direction = _structure_direction(conn_pts, body_bbox)
        sockets = _socket_cells(body_pts, body_bbox, direction, g, bg)
        assert sockets, "expected interior sockets"
        struct_infos.append(
            {
                "sid": int(sid),
                "bbox": body_bbox,
                "sig": conn_sig,
                "out_dir": tuple(map(int, direction)),
                "anchor": tuple(map(int, sockets[0])),
            }
        )

    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    used_red: set[int] = set()
    for sinfo in struct_infos:
        wanted_move = _opposite(sinfo["out_dir"])
        candidate_rids = []
        for rid in sig_to_reds.get(sinfo["sig"], []):
            rinfo = red_infos[int(rid)]
            if tuple(map(int, rinfo["move_dir"])) != tuple(map(int, wanted_move)):
                continue
            if not _red_on_open_side(rinfo["bbox"], sinfo["bbox"], sinfo["out_dir"]):
                continue
            candidate_rids.append(int(rid))
        if len(candidate_rids) != 1:
            continue
        rid = int(candidate_rids[0])
        if rid in used_red:
            continue
        used_red.add(rid)
        assignments.append((int(rid), tuple(map(int, sinfo["out_dir"])), tuple(map(int, sinfo["anchor"]))))

    out = g.copy()
    out[out == int(RED)] = int(bg)
    out[marker_mask] = int(bg)
    for info in free_connectors:
        for r, c in info["coords"]:
            out[int(r), int(c)] = int(bg)

    for rid, (out_dr, out_dc), (tr, tc) in assignments:
        coords = red_infos[int(rid)]["coords"]
        r0, c0, r1, c1 = map(int, red_infos[int(rid)]["bbox"])

        if (out_dr, out_dc) == (0, 1):
            shift_r, shift_c = 0, int(tc - c0)
        elif (out_dr, out_dc) == (0, -1):
            shift_r, shift_c = 0, int(tc - c1)
        elif (out_dr, out_dc) == (1, 0):
            shift_r, shift_c = int(tr - r0), 0
        else:
            shift_r, shift_c = int(tr - r1), 0

        steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
        step_dr = 0 if int(shift_r) == 0 else (1 if int(shift_r) > 0 else -1)
        step_dc = 0 if int(shift_c) == 0 else (1 if int(shift_c) > 0 else -1)

        for step in range(0, steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(BLACK)

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(RED)

    return out


def _matched_patches(
    width: int,
    teeth: int,
    slide: int,
    rot_k: int,
    bg: int,
    sc: int,
    conn_type: tuple[int, str],
) -> dict | None:
    red_h = min(int(teeth), 2)
    base = _structure_array(width, teeth, bg, sc)
    start = (width - teeth) // 2
    true_top = int(2 + slide)

    true_sprite = _sprite_up_patch(red_h, teeth, bg, conn_type)
    if true_sprite is None:
        return None
    sprite_h, sprite_w = true_sprite.shape
    sprite_left = int(start - (sprite_w - teeth))
    if sprite_left < 0:
        return None

    patch_h = int(max(6, true_top - 1 + sprite_h))
    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    if not _stamp_connector(structure, 4, width - 2, conn_type, bg):
        return None

    input_patch = structure.copy()
    paste(input_patch, true_sprite, true_top - 1, sprite_left, transparent=bg)
    output_patch = main(input_patch)

    footprint = np.full_like(input_patch, bg)
    footprint[(input_patch != int(bg)) | (output_patch != int(bg))] = int(sc)

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "footprint": np.rot90(footprint, k=rot_k),
        "conn_type": conn_type,
        "move_dir": _rotate_dir((-1, 0), rot_k),
    }


def _unmatched_patches(
    width: int, teeth: int, rot_k: int, bg: int, sc: int, conn_type: tuple[int, str]
) -> dict | None:
    structure = np.full((6, width), bg, dtype=int)
    paste(structure, _structure_array(width, teeth, bg, sc), 0, 0, transparent=bg)
    if not _stamp_connector(structure, 4, width - 2, conn_type, bg):
        return None
    rotated = np.rot90(structure, k=rot_k)
    return {
        "structure": rotated,
        "input": rotated.copy(),
        "output": rotated.copy(),
        "footprint": rotated.copy(),
        "conn_type": conn_type,
        "move_dir": None,
    }


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(inp, spec["footprint"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == sc)
        coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        placed.append({"coords": coords, "bbox": (min(rows), min(cols), max(rows), max(cols))})

    return inp, out, struct_grid, placed


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        for dr, dc in DIRS:
            for r, c in item["coords"]:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(struct_grid[nr, nc]) != int(bg):
                    continue
                if not (r0 < nr < r1 and c0 < nc < c1):
                    continue
                rr, cc = int(nr), int(nc)
                while 0 <= rr < h and 0 <= cc < w and int(struct_grid[rr, cc]) == int(bg):
                    forbidden[rr, cc] = True
                    rr += int(dr)
                    cc += int(dc)
    return forbidden


def _blocked_canvas(inp: np.ndarray, struct_grid: np.ndarray, placed: list[dict], bg: int, sc: int) -> np.ndarray:
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0:r1 + 1, c0:c1 + 1] = True
    mobile_mask = (inp != int(bg)) & (struct_grid == int(bg))
    blocked = forbidden | bbox_blocked | (inp != int(bg)) | _dilate4(mobile_mask)
    canvas = np.full((h, w), bg, dtype=int)
    canvas[blocked] = int(sc)
    return canvas


def _place_special_same_sig_wrong_dir(
    inp: np.ndarray,
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
    sc: int,
    sig_to_move: dict[tuple[int, str], tuple[int, int]],
    count: int,
) -> bool:
    shape_choices = list(DISTRACTOR_SHAPES)
    sig_items = list(sig_to_move.items())
    for _ in range(count):
        random.shuffle(sig_items)
        placed_one = False
        for conn_type, correct_move in sig_items:
            wrong_dirs = [d for d in DIRS if tuple(map(int, d)) != tuple(map(int, correct_move))]
            random.shuffle(shape_choices)
            random.shuffle(wrong_dirs)
            for rh, rw in shape_choices:
                for move_dir in wrong_dirs:
                    patch = _sprite_patch(rh, rw, bg, conn_type, move_dir)
                    if patch is None:
                        continue
                    canvas = _blocked_canvas(inp, struct_grid, placed, bg, sc)
                    loc = _place_patch(canvas, patch, bg, padding=1)
                    if loc is None:
                        continue
                    paste(inp, patch, loc[0], loc[1], transparent=bg)
                    placed_one = True
                    break
                if placed_one:
                    break
            if placed_one:
                break
        if not placed_one:
            return False
    return True


def _place_special_same_dir_wrong_sig(
    inp: np.ndarray,
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
    sc: int,
    sig_to_move: dict[tuple[int, str], tuple[int, int]],
    wrong_pool: list[tuple[int, str]],
    count: int,
) -> bool:
    shape_choices = list(DISTRACTOR_SHAPES)
    move_dirs = list(sig_to_move.values())
    for _ in range(count):
        random.shuffle(move_dirs)
        random.shuffle(shape_choices)
        placed_one = False
        for move_dir in move_dirs:
            pool = list(wrong_pool)
            random.shuffle(pool)
            for rh, rw in shape_choices:
                for conn_type in pool:
                    patch = _sprite_patch(rh, rw, bg, conn_type, move_dir)
                    if patch is None:
                        continue
                    canvas = _blocked_canvas(inp, struct_grid, placed, bg, sc)
                    loc = _place_patch(canvas, patch, bg, padding=1)
                    if loc is None:
                        continue
                    paste(inp, patch, loc[0], loc[1], transparent=bg)
                    placed_one = True
                    break
                if placed_one:
                    break
            if placed_one:
                break
        if not placed_one:
            return False
    return True


def _place_generic_distractors(
    inp: np.ndarray,
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
    sc: int,
    wrong_pool: list[tuple[int, str]],
    count: int,
) -> bool:
    shape_choices = list(DISTRACTOR_SHAPES)
    for _ in range(count):
        random.shuffle(shape_choices)
        random.shuffle(wrong_pool)
        placed_one = False
        for rh, rw in shape_choices:
            dirs = list(DIRS)
            random.shuffle(dirs)
            for conn_type in wrong_pool:
                for move_dir in dirs:
                    patch = _sprite_patch(rh, rw, bg, conn_type, move_dir)
                    if patch is None:
                        continue
                    canvas = _blocked_canvas(inp, struct_grid, placed, bg, sc)
                    loc = _place_patch(canvas, patch, bg, padding=1)
                    if loc is None:
                        continue
                    paste(inp, patch, loc[0], loc[1], transparent=bg)
                    placed_one = True
                    break
                if placed_one:
                    break
            if placed_one:
                break
        if not placed_one:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    forbidden = {bg, int(RED), int(BLACK), int(MARKER_COLOR), int(Color.ORANGE), int(Color.MAROON)}
    sc = random.choice([c for c in STRUCT_COLORS if c not in forbidden])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1 if random.random() < UNMATCHED_PROB else 0
    matched = n_structs - unmatched
    if matched <= 0:
        return None

    if n_structs > len(CONNECTOR_TYPES):
        return None
    struct_conn_types = random.sample(list(CONNECTOR_TYPES), k=n_structs)
    wrong_pool = [t for t in CONNECTOR_TYPES if t not in struct_conn_types]
    if not wrong_pool:
        return None

    specs = []
    sig_to_move: dict[tuple[int, str], tuple[int, int]] = {}
    for idx in range(n_structs):
        width, teeth = random.choice(STRUCT_WIDTH_TEETH)
        rot_k = random.randrange(4)
        conn_type = struct_conn_types[idx]
        if idx < matched:
            slide = random.randint(*SLIDE_RANGE)
            spec = _matched_patches(
                width,
                teeth,
                slide,
                rot_k,
                bg,
                sc,
                conn_type,
            )
            if spec is not None:
                sig_to_move[conn_type] = tuple(map(int, spec["move_dir"]))
        else:
            spec = _unmatched_patches(width, teeth, rot_k, bg, sc, conn_type)
        if spec is None:
            return None
        specs.append(spec)

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["footprint"] != bg), reverse=True)

    placed = _place_structures(specs, h, w, bg, sc)
    if placed is None:
        return None
    inp, out, struct_grid, placed_structs = placed

    forced_decoys = SPECIAL_SAME_SIG_WRONG_DIR + SPECIAL_SAME_DIR_WRONG_SIG
    current_reds = len(components_mask(inp == int(RED), connectivity=4))
    min_total = max(TOTAL_RED_MIN, current_reds + forced_decoys)
    if min_total > TOTAL_RED_MAX:
        return None
    target_total = random.randint(min_total, TOTAL_RED_MAX)
    extra = target_total - current_reds
    if extra < forced_decoys:
        return None

    if not _place_special_same_sig_wrong_dir(
        inp, struct_grid, placed_structs, bg, sc, sig_to_move, SPECIAL_SAME_SIG_WRONG_DIR
    ):
        return None

    if not _place_special_same_dir_wrong_sig(
        inp,
        struct_grid,
        placed_structs,
        bg,
        sc,
        sig_to_move,
        wrong_pool,
        SPECIAL_SAME_DIR_WRONG_SIG,
    ):
        return None

    if not _place_generic_distractors(
        inp,
        struct_grid,
        placed_structs,
        bg,
        sc,
        wrong_pool,
        extra - forced_decoys,
    ):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo variant D1/2+D2/2")
