# Combo variant of cbebaa4b (index 31)
# combo_spec: D8/2+D16/0
# Name: 双像素方向编码连接器 + 模糊对齐

# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
# design:
#   connector_form:
#     - 每个连接点都是一个 2 格红色 domino；横向 domino 表示 horizontal port，纵向 domino 表示
#       vertical port。
#     - 与 D8/2 一样，端口方向是可见语义；solver 不能再把单个红像素当成独立 peg。
#   fuzzy_alignment:
#     - 在原 D8/2 的“整块红 domino 必须完全重合”基础上，再加入 D16/0 的 1 格模糊对齐：
#       同方向的两个 domino 可以沿自身轴向滑移 -1 / 0 / +1 格后视为一对。
#     - 例：assembly 里已有水平端口 {(5,6),(5,7)}。
#       * 旧 D8/2：moving piece 只能把自己的水平 domino 放到 {(5,6),(5,7)}。
#       * 新组合：还允许放到 {(5,5),(5,6)} 或 {(5,7),(5,8)}。
#       * 这三种情况都保持“水平对水平”，但只有第 2 种是精确重合；其余两种会在输出里形成
#         3 格红线段，而不是 2 格完全重合。
#   visible_output_effect:
#     - 精确对接边在输出里留下 2 格红 domino。
#     - 模糊对接边在输出里留下 3 格红 triomino（两个 domino 交错 1 格）。
#     - 因此人类能从样例里同时看到“端口有方向”与“允许 1 格滑移”这两个变化。
#
# b) 复杂度校准
# design:
#   kept_small:
#     - 方向类别仍只有 horizontal / vertical 两档。
#     - 模糊对齐也只开放三个离散滑移值 {-1,0,+1}，不做更大范围搜索。
#   enough_signal:
#     - 生成器要求每题同时出现横向与纵向端口。
#     - 至少有一个节点同时拥有两种方向的端口，避免退化成纯横链或纯竖链。
#     - 至少有一条连接使用非零滑移，因此输出中必有 3 格红连接段，肉眼可见区别于 D8/2。
#
# c) 隐式假设审查
# audit:
#   old_red_pixels_are_ports:
#     - 原 main() 把每个 arr==2 的坐标当成独立端口；在组合后失效。
#     - 新 solver 必须先把红色 4-连通块拆成 2 格 port，再在 port 层面做匹配。
#   old_exact_keying:
#     - D8/2 版本用“全局 port 坐标完全相同”作为配对 key。
#     - 组合后同一条边可出现 2 格完全重合或 3 格交错重合，不能再只用 exact global key。
#   old_all_red_counts_finish_at_2:
#     - 精确重合时每个红格最终计数都为 2；模糊滑移时连接两端会留下计数 1 的端点。
#     - 所以终态正确性不能再依赖 `all(red_count == 2)`，而应改为“所有 port 都恰好配成对，
#       不存在未闭合 open port”。
#   old_move_generation:
#     - 旧代码枚举“piece 的某个红像素 / 某个红 domino 与目标精确对齐”。
#     - 现在要枚举“同方向 port + 允许滑移 -1/0/+1”的候选平移。
#   old_overlap_feasibility:
#     - 原题只检查红坐标累计次数是否超过 2。
#     - 组合后还要检查：
#       1. 一个新 port 若声明为未配对，则它不能压到任何已有红格；
#       2. 一个新 port 若去配对某个 open port，则它只能压到这一个目标 port 的重叠区；
#       3. 一个红格不能被 3 个原始 port 共用。
#
# d) 放置合法性
# design:
#   solver_state:
#     - DFS 显式维护 open_ports（当前仍暴露、尚未配对的全局端口列表）。
#     - 另维护 red_counts，用来阻止未来 piece 压到已经封闭的红格，或把同一红格叠到 3 次。
#   per_piece_matching:
#     - 对一个待放 piece 的所有 global ports，枚举它们与当前 open_ports 的局部二分匹配。
#     - 每个 piece port 只能匹配一个 open port；每个 open port 也只能被一个 piece port 消耗。
#     - 未匹配的 piece ports 会成为新的 open_ports。
#   generator_side:
#     - 仍采用 output-first：先在局部坐标里长出树，再为每条边随机指定滑移值，再渲染 assembled
#       output，最后把 piece 打散得到输入。
#     - 组装校验与 solver 共用同一套 fuzzy port 规则，避免生成器产出“视觉上像滑移、逻辑上却不合法”
#       的样本。
#
# e) 视觉可观察性
# design:
#   observable_cues:
#     - 输入里可直接看到红端口始终是 2 格 domino，且有横放/竖放两类。
#     - 输出里可直接看到有些连接是 2 格红块，有些连接是 3 格红线段；这正是 1 格模糊滑移的可视结果。
#     - 推理不依赖颜色 id 顺序、对象生成顺序等人类不可见因素。
#
# f) 可视化整洁度
# design:
#   cleanliness:
#     - 仍保持 piece 单连通、散布时至少 2 格 Chebyshev 间距、最终结构为一棵树。
#     - 模糊滑移只在端口局部产生 2 或 3 格红段，不增加额外标记层，也不引入大面积重叠。
#
# g) 变化的可推断性
# design:
#   colors:
#     - 没有新增颜色映射；红色仍只表示连接器。
#     - 变化完全由红色几何结构承载：2 格 domino 编码方向，2/3 格终态红段编码精确或模糊对齐。
#
# h) 融合行为量化指标
# metrics:
#   - horizontal_ports / vertical_ports: 输入对象中两类方向端口的数量。
#   - orthogonal_nodes: 同时拥有两种方向端口的节点数量。
#   - slipped_edges: 使用 ±1 滑移的边数。
#   - triomino_components: 输出里长度为 3 的红色连通段数量；该值 > 0 直接证明模糊滑移被真正使用。
#   - strict_solver_same_output: 把同一输入交给“只允许精确重合”的 D8/2 solver 后，是否仍得到相同输出。
#     生成器要求这个值为假，防止样本退化回纯精确对接。
#
# i) 新 main() 的隐式假设与 generator 保证
# assumptions:
#   - 输入中的每个红色 4-连通块都恰好是一个 2 格 horizontal/vertical domino。
#   - 每个非 anchor piece 的度数为 1 或 2，因此每个 piece 的 port 数很小，局部匹配枚举可控。
#   - 合法样本里，不会出现一个新 port 同时合理匹配两个不同 open_ports 的必要歧义；若发生，DFS 会分支，
#     而 generator 通过唯一解检查筛掉这类样本。
# generator_guarantees:
#   - `_make_piece` / `_make_anchor` 只生成合法 2 格端口。
#   - `_grow_tree` 强制同时出现横竖两种方向端口、至少一个正交节点、至少一条非零滑移边。
#   - `_assemble` 与 `main()` 共用 fuzzy port 匹配逻辑。
#   - `_gen_attempt` 额外要求输出中存在 3 格红连接段，并要求严格精确 solver 无法得到同一输出。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


PortCells = tuple[tuple[int, int], tuple[int, int]]


def _normalize_cells(cells) -> PortCells:
    pts = tuple(sorted((int(r), int(c)) for r, c in cells))
    assert len(pts) == 2, "each connector must contain exactly two red cells"
    return pts


def _port_orientation(cells: PortCells) -> str:
    (r0, c0), (r1, c1) = cells
    if r0 == r1 and abs(c0 - c1) == 1:
        return "h"
    if c0 == c1 and abs(r0 - r1) == 1:
        return "v"
    raise AssertionError("connector cells must form a horizontal or vertical domino")


def _shift_port(cells: PortCells, top: int, left: int) -> PortCells:
    return tuple((r + top, c + left) for r, c in cells)


def _port_axis_shift(base: PortCells, other: PortCells) -> int | None:
    orient = _port_orientation(base)
    if _port_orientation(other) != orient:
        return None
    if orient == "h":
        if base[0][0] != other[0][0]:
            return None
        d0 = other[0][1] - base[0][1]
        d1 = other[1][1] - base[1][1]
    else:
        if base[0][1] != other[0][1]:
            return None
        d0 = other[0][0] - base[0][0]
        d1 = other[1][0] - base[1][0]
    if d0 != d1:
        return None
    return int(d0)


def _target_alignment_top_left(target: PortCells, local: PortCells, slip: int) -> tuple[int, int]:
    orient = _port_orientation(target)
    (tr, tc), _ = target
    (lr, lc), _ = local
    if orient == "h":
        return tr - lr, tc + slip - lc
    return tr + slip - lr, tc - lc


def _extract_ports(arr: np.ndarray) -> list[dict]:
    ports = []
    for pts in components_mask(arr == 2, connectivity=4):
        cells = _normalize_cells(pts)
        ports.append({"cells": cells, "orientation": _port_orientation(cells)})
    ports.sort(key=lambda p: p["cells"])
    return ports


def _globalize_ports(local_ports: list[dict], top: int, left: int) -> list[dict]:
    out = []
    for port in local_ports:
        out.append(
            {
                "cells": _shift_port(port["cells"], top, left),
                "orientation": port["orientation"],
            }
        )
    return out


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        ports = _extract_ports(arr)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "ports": ports,
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
            }
        )
    return pieces


def _try_place(
    canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int
) -> np.ndarray | None:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None

    sub = canvas[top : top + ph, left : left + pw]
    mask = arr != bg
    if not np.any(mask):
        return None

    conflict = mask & (sub != bg) & ~(((arr == 2) & (sub == 2)) | (arr == sub))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _init_red_counts(global_ports: list[dict]) -> dict[tuple[int, int], int]:
    red_counts: dict[tuple[int, int], int] = {}
    for port in global_ports:
        for cell in port["cells"]:
            red_counts[cell] = red_counts.get(cell, 0) + 1
    return red_counts


def _sorted_open_ports(open_ports: list[dict]) -> list[dict]:
    return sorted(open_ports, key=lambda p: (p["cells"], p["orientation"]))


def _placement_matchings_for_ports(
    piece_ports: list[dict],
    open_ports: list[dict],
    red_counts: dict[tuple[int, int], int],
    allowed_shifts: tuple[int, ...],
) -> list[dict]:
    allowed = set(int(s) for s in allowed_shifts)
    open_cell_sets = [set(port["cells"]) for port in open_ports]
    piece_cell_sets = [set(port["cells"]) for port in piece_ports]
    options = []

    for i, port in enumerate(piece_ports):
        if any(red_counts.get(cell, 0) >= 2 for cell in port["cells"]):
            return []
        hits = {cell for cell in port["cells"] if red_counts.get(cell, 0) > 0}
        candidates = []
        for j, target in enumerate(open_ports):
            if port["orientation"] != target["orientation"]:
                continue
            shift = _port_axis_shift(target["cells"], port["cells"])
            if shift is None or shift not in allowed:
                continue
            overlap = piece_cell_sets[i] & open_cell_sets[j]
            if hits == overlap and overlap:
                candidates.append(
                    {
                        "target_idx": j,
                        "shift": int(shift),
                        "overlap_cells": tuple(sorted(overlap)),
                    }
                )
        if hits and not candidates:
            return []
        options.append({"hits": hits, "candidates": candidates})

    order = sorted(
        range(len(piece_ports)),
        key=lambda i: (
            0 if options[i]["hits"] else 1,
            len(options[i]["candidates"]) if options[i]["hits"] else 0,
            i,
        ),
    )
    out = []

    def rec(k: int, used_targets: set[int], matches: list[dict]) -> None:
        if k == len(order):
            matched_piece_indices = {m["piece_port_index"] for m in matches}
            matched_target_indices = tuple(sorted(m["target_idx"] for m in matches))
            unmatched_piece_indices = tuple(
                i for i in range(len(piece_ports)) if i not in matched_piece_indices
            )
            out.append(
                {
                    "matches": tuple(
                        sorted(
                            matches,
                            key=lambda m: (m["piece_port_index"], m["target_idx"]),
                        )
                    ),
                    "matched_target_indices": matched_target_indices,
                    "unmatched_piece_indices": unmatched_piece_indices,
                    "matched_port_count": len(matches),
                    "score": int(sum(len(m["overlap_cells"]) for m in matches)),
                }
            )
            return

        i = order[k]
        info = options[i]
        if not info["hits"]:
            rec(k + 1, used_targets, matches)
            return

        for cand in info["candidates"]:
            tgt = cand["target_idx"]
            if tgt in used_targets:
                continue
            used_targets.add(tgt)
            matches.append(
                {
                    "piece_port_index": i,
                    "target_idx": tgt,
                    "shift": cand["shift"],
                    "overlap_cells": cand["overlap_cells"],
                }
            )
            rec(k + 1, used_targets, matches)
            matches.pop()
            used_targets.remove(tgt)

    rec(0, set(), [])
    return out


def _advance_state(
    piece_ports: list[dict],
    open_ports: list[dict],
    red_counts: dict[tuple[int, int], int],
    matching: dict,
) -> tuple[list[dict], dict[tuple[int, int], int]]:
    consumed = set(matching["matched_target_indices"])
    next_open = [port for i, port in enumerate(open_ports) if i not in consumed]
    for i in matching["unmatched_piece_indices"]:
        next_open.append(piece_ports[i])
    next_open = _sorted_open_ports(next_open)

    next_red = dict(red_counts)
    for port in piece_ports:
        for cell in port["cells"]:
            next_red[cell] = next_red.get(cell, 0) + 1
    return next_open, next_red


def _solve_with_allowed_shifts(grid: np.ndarray, allowed_shifts: tuple[int, ...]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"
    assert all(
        len(port["cells"]) == 2 for piece in pieces for port in piece["ports"]
    ), "all connectors must be 2-cell dominoes"

    anchors = [piece for piece in pieces if np.any(piece["arr"] == 4)]
    assert len(anchors) == 1, "input must have exactly one yellow (4) anchor piece"
    anchor = anchors[0]
    rest = [piece for piece in pieces if piece is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    open_ports = _globalize_ports(anchor["ports"], ar0, ac0)
    red_counts = _init_red_counts(open_ports)
    solutions: list[np.ndarray] = []
    seen_solution_keys: set[bytes] = set()

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        cur_open_ports: list[dict],
        cur_red_counts: dict[tuple[int, int], int],
    ) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            if not cur_open_ports:
                key = cur.tobytes()
                if key not in seen_solution_keys:
                    seen_solution_keys.add(key)
                    solutions.append(cur)
            return
        if not cur_open_ports:
            return

        best_moves = None
        for target_idx, target in enumerate(cur_open_ports):
            moves = []
            for idx, piece in enumerate(remaining):
                ph, pw = piece["shape"]
                for port in piece["ports"]:
                    if port["orientation"] != target["orientation"]:
                        continue
                    for slip in allowed_shifts:
                        top, left = _target_alignment_top_left(
                            target["cells"], port["cells"], int(slip)
                        )
                        if top < 0 or left < 0 or top + ph > h or left + pw > w:
                            continue
                        piece_ports = _globalize_ports(piece["ports"], top, left)
                        matchings = _placement_matchings_for_ports(
                            piece_ports, cur_open_ports, cur_red_counts, allowed_shifts
                        )
                        for matching in matchings:
                            if target_idx not in matching["matched_target_indices"]:
                                continue
                            if matching["matched_port_count"] <= 0:
                                continue
                            moves.append((idx, top, left, piece_ports, matching))

            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible fuzzy-port move"

        dedup: dict[tuple, tuple] = {}
        for idx, top, left, piece_ports, matching in best_moves:
            key = (
                idx,
                top,
                left,
                tuple(
                    (m["piece_port_index"], m["target_idx"], m["shift"])
                    for m in matching["matches"]
                ),
                matching["unmatched_piece_indices"],
            )
            prev = dedup.get(key)
            cur_score = (
                matching["score"],
                matching["matched_port_count"],
                remaining[idx]["area"],
            )
            if prev is None:
                dedup[key] = (idx, top, left, piece_ports, matching)
                continue
            prev_matching = prev[4]
            prev_score = (
                prev_matching["score"],
                prev_matching["matched_port_count"],
                remaining[prev[0]]["area"],
            )
            if cur_score > prev_score:
                dedup[key] = (idx, top, left, piece_ports, matching)

        moves2 = list(dedup.values())
        best_quality = max(
            (m[4]["score"], m[4]["matched_port_count"]) for m in moves2
        )
        moves2 = [
            m
            for m in moves2
            if (m[4]["score"], m[4]["matched_port_count"]) == best_quality
        ]
        moves2.sort(
            key=lambda x: (
                -x[4]["score"],
                -x[4]["matched_port_count"],
                -remaining[x[0]]["area"],
                x[1],
                x[2],
                x[0],
            )
        )

        for idx, top, left, piece_ports, matching in moves2:
            piece = remaining[idx]
            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue
            next_open_ports, next_red_counts = _advance_state(
                piece_ports, cur_open_ports, cur_red_counts, matching
            )
            dfs(
                nxt,
                remaining[:idx] + remaining[idx + 1 :],
                next_open_ports,
                next_red_counts,
            )
            if len(solutions) >= 2:
                return

    dfs(placed, rest, _sorted_open_ports(open_ports), red_counts)
    assert (
        len(solutions) == 1
    ), f"fuzzy domino-port assembly must have exactly one valid solution, got {len(solutions)}"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites by docking directional red domino ports with small axial slip.

    Input structure:
    - The background is black (0). One sprite is the yellow anchor containing a solid 3x3 or 4x5 block of color 4.
    - Every connector is a red 2-cell domino on a sprite boundary.
    - Horizontal red dominoes may only dock to horizontal ones; vertical dominoes may only dock to vertical ones.

    Rule:
    - Keep the anchor fixed and translate the other sprites onto the same canvas.
    - A legal attachment matches one or more whole red domino ports against currently exposed ports of the same orientation.
    - The match may be exact, or it may slide by one cell along the port axis; this means a connected pair can leave either a 2-cell red domino or a 3-cell staggered red segment in the output.
    - When all sprites are placed, every original port must be paired exactly once, so no open connector remains.

    Key insight:
    - The red marks are typed ports, not independent pixels.
    - First use the domino orientation to choose which ports are compatible, then use the 1-cell fuzzy slide to locate the only near-alignment that lets all ports close consistently.
    """
    return _solve_with_allowed_shifts(grid, (-1, 0, 1))


def _strict_domino_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_allowed_shifts(grid, (0,))


def _output_fusion_metrics(out: np.ndarray) -> dict[str, int]:
    red_components = components_mask(out == 2, connectivity=4)
    sizes = [len(comp) for comp in red_components]
    return {
        "red_components": len(sizes),
        "domino_components": int(sum(sz == 2 for sz in sizes)),
        "triomino_components": int(sum(sz == 3 for sz in sizes)),
        "other_red_components": int(sum(sz not in (2, 3) for sz in sizes)),
    }


# --------------- generator (combo) ---------------

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]

SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 2500

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}
SIDE_ORIENTATION = {"top": "h", "bottom": "h", "left": "v", "right": "v"}
FUZZY_SLIPS = (-1, 0, 1)


def _new_node(kind, h, w, color, top=0, left=0, cap=PIECE_CAP):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "sides": set(),
        "ports": {},
        "edge_slips": {},
        "parent_slip": None,
    }


def _ensure_port(node, side):
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _port_cells(node, side):
    h, w = node["shape"]
    k = node["ports"][side]
    if side == "top":
        return [(0, k), (0, k + 1)]
    if side == "bottom":
        return [(h - 1, k), (h - 1, k + 1)]
    if side == "left":
        return [(k, 0), (k + 1, 0)]
    return [(k, w - 1), (k + 1, w - 1)]


def _global_port(node, side):
    tr, tc = node["top"], node["left"]
    return [(tr + r, tc + c) for r, c in _port_cells(node, side)]


def _bbox_of(node):
    h, w = node["shape"]
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _rects_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union_bbox(nodes):
    boxes = [_bbox_of(node) for node in nodes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _draw_path(arr, start, end, color, order):
    r, c = start
    er, ec = end
    axes = (
        [(r, er, c, True), (c, ec, er, False)]
        if order == 0
        else [(c, ec, r, False), (r, er, ec, True)]
    )
    for a_start, a_end, fixed, is_row in axes:
        step = 1 if a_end >= a_start else -1
        for v in range(a_start, a_end + step, step):
            rr, cc = (v, fixed) if is_row else (fixed, v)
            if arr[rr, cc] != 2:
                arr[rr, cc] = color


def _paint_bar(arr, side, color, used):
    h, w = arr.shape
    is_horizontal = side in ("top", "bottom")
    fixed = 0 if side in ("top", "left") else (h - 1 if side == "bottom" else w - 1)
    span = w if is_horizontal else h
    lo_conflict = "left" if is_horizontal else "top"
    hi_conflict = "right" if is_horizontal else "bottom"
    for i in range(span):
        if (i == 0 and lo_conflict in used) or (i == span - 1 and hi_conflict in used):
            continue
        if is_horizontal:
            arr[fixed, i] = color
        else:
            arr[i, fixed] = color


def _bar_endpoint(shape, side, hub):
    h, w = shape
    hr, hc = hub
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        return (row, min(max(hc, 1), w - 2))
    col = 0 if side == "left" else w - 1
    return (min(max(hr, 1), h - 2), col)


def _make_piece(node):
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))

    bar_sides = [side for side in SIDES if side not in used]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, used)

    terminals = []
    for side in used:
        dr, dc = INWARD[side]
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
            rr, cc = r + dr, c + dc
            arr[rr, cc] = color
            terminals.append((rr, cc))

    arr[hub] = color
    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if len(_extract_ports(arr)) != len(used):
        return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    arr = np.zeros(node["shape"], dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
    if len(_extract_ports(arr)) != len(node["sides"]):
        return None
    return arr


def _candidate_fits(nodes, parent_i, child):
    cb = _bbox_of(child)
    if not _rects_overlap(cb, _bbox_of(nodes[parent_i])):
        return False
    for i, other in enumerate(nodes):
        if i == parent_i:
            continue
        if _rects_overlap(cb, _bbox_of(other)):
            return False
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    return (r1 - r0 + 1) <= MAX_ASSEMBLY_DIM and (c1 - c0 + 1) <= MAX_ASSEMBLY_DIM


def _variant_metrics(nodes) -> dict[str, int]:
    horizontal_ports = 0
    vertical_ports = 0
    orthogonal_nodes = 0
    slipped_edges = 0
    slipped_horizontal = 0
    slipped_vertical = 0
    for node in nodes:
        orients = {SIDE_ORIENTATION[side] for side in node["sides"]}
        horizontal_ports += sum(SIDE_ORIENTATION[side] == "h" for side in node["sides"])
        vertical_ports += sum(SIDE_ORIENTATION[side] == "v" for side in node["sides"])
        if len(orients) == 2:
            orthogonal_nodes += 1
        if node["parent_slip"] not in (None, 0):
            slipped_edges += 1
            if node.get("parent_side") in ("top", "bottom"):
                slipped_horizontal += 1
            elif node.get("parent_side") in ("left", "right"):
                slipped_vertical += 1
    return {
        "horizontal_ports": horizontal_ports,
        "vertical_ports": vertical_ports,
        "orthogonal_nodes": orthogonal_nodes,
        "slipped_edges": slipped_edges,
        "slipped_horizontal": slipped_horizontal,
        "slipped_vertical": slipped_vertical,
    }


def _variant_structure_ok(nodes) -> bool:
    m = _variant_metrics(nodes)
    return (
        m["horizontal_ports"] > 0
        and m["vertical_ports"] > 0
        and m["orthogonal_nodes"] > 0
        and m["slipped_edges"] > 0
    )


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)

    for i in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            parents = [j for j, node in enumerate(nodes) if len(node["sides"]) < node["cap"]]
            random.shuffle(parents)
            for parent_i in parents:
                parent = nodes[parent_i]
                sides = [side for side in SIDES if side not in parent["sides"]]
                random.shuffle(sides)
                for side in sides:
                    incoming = OPPOSITE[side]
                    _ensure_port(parent, side)
                    slip_order = list(FUZZY_SLIPS)
                    random.shuffle(slip_order)
                    for slip in slip_order:
                        child = _new_node("piece", ph, pw, colors[i])
                        _ensure_port(child, incoming)
                        pr, pc = _global_port(parent, side)[0]
                        cr, cc = _port_cells(child, incoming)[0]
                        if SIDE_ORIENTATION[side] == "h":
                            child["top"] = pr - cr
                            child["left"] = pc + slip - cc
                        else:
                            child["top"] = pr + slip - cr
                            child["left"] = pc - cc
                        child["sides"].add(incoming)
                        child["edge_slips"][incoming] = int(slip)
                        child["parent_slip"] = int(slip)
                        child["parent_side"] = incoming
                        if not _candidate_fits(nodes, parent_i, child):
                            continue
                        parent["sides"].add(side)
                        parent["edge_slips"][side] = int(slip)
                        nodes.append(child)
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    if not _variant_structure_ok(nodes):
        return None
    return nodes


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None
        node["local_ports"] = _extract_ports(node["arr"])

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    for node in nodes:
        node["pos"] = (node["top"] + sr, node["left"] + sc)

    out = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    ar, ac = anchor["pos"]
    placed = _try_place(out, anchor["arr"], ar, ac, 0)
    if placed is None:
        return None
    open_ports = _sorted_open_ports(_globalize_ports(anchor["local_ports"], ar, ac))
    red_counts = _init_red_counts(open_ports)

    def rec(
        idx: int,
        cur: np.ndarray,
        cur_open_ports: list[dict],
        cur_red_counts: dict[tuple[int, int], int],
    ) -> np.ndarray | None:
        if idx == len(nodes):
            if cur_open_ports:
                return None
            if len(components_mask(cur != 0, connectivity=4)) != 1:
                return None
            return cur

        node = nodes[idx]
        top, left = node["pos"]
        nxt = _try_place(cur, node["arr"], top, left, 0)
        if nxt is None:
            return None

        piece_ports = _globalize_ports(node["local_ports"], top, left)
        matchings = _placement_matchings_for_ports(
            piece_ports, cur_open_ports, cur_red_counts, FUZZY_SLIPS
        )
        matchings = [m for m in matchings if m["matched_port_count"] > 0]
        if not matchings:
            return None
        best_quality = max((m["score"], m["matched_port_count"]) for m in matchings)
        matchings = [
            m for m in matchings if (m["score"], m["matched_port_count"]) == best_quality
        ]
        matchings.sort(
            key=lambda m: (
                -m["score"],
                -m["matched_port_count"],
                top,
                left,
            )
        )
        for matching in matchings:
            next_open_ports, next_red_counts = _advance_state(
                piece_ports, cur_open_ports, cur_red_counts, matching
            )
            solved = rec(idx + 1, nxt, next_open_ports, next_red_counts)
            if solved is not None:
                return solved
        return None

    solved = rec(1, placed, open_ports, red_counts)
    if solved is None:
        return None

    out_metrics = _output_fusion_metrics(solved)
    if out_metrics["other_red_components"] != 0:
        return None
    if out_metrics["triomino_components"] <= 0:
        return None
    return solved


def _scatter(nodes, h, w):
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    rest = nodes[1:]
    random.shuffle(rest)
    for node in rest:
        try:
            r, c = random_free_location(
                inp,
                node["arr"],
                transparent=0,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, node["arr"], r, c, transparent=0)
    return inp


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_tree()
    if nodes is None:
        return None
    out = _assemble(nodes, h, w)
    if out is None:
        return None
    inp = _scatter(nodes, h, w)
    if inp is None:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None

    strict_same = False
    try:
        strict_out = _strict_domino_main(inp)
        strict_same = np.array_equal(strict_out, solved)
    except Exception:
        strict_same = False
    if strict_same:
        return None

    out_metrics = _output_fusion_metrics(solved)
    if out_metrics["triomino_components"] <= 0 or out_metrics["other_red_components"] != 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
