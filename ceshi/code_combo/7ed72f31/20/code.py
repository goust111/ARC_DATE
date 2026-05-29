# Combo variant of 7ed72f31 (index 20)
# combo_spec: D8/1+D9/0
# Name: 步进移动链

from __future__ import annotations

# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 原题: 每个红标记组都独立读取静态输入中的本地 payload, 做一次镜像, 并保留原像素。
#    - 融合后: 所有红标记按空间顺序串成一条链; 只有第一个标记在输入里带真实 payload。
#      第一步把 seed payload 反射成 P1 并删除 seed; 第二步把 P1 反射成 P2 并删除 P1; 依此类推。
#      所以输出中红标记全部保留, 但彩色 sprite 只出现在最终位置。
#    - 小例子:
#      * 输入: 左上有蓝色 L 形贴着一条竖红线, 更右下还有一条横红线和一个孤立红点, 后两者附近为空。
#      * 原题 solver: 只会把蓝色 L 在第一条竖线处复制到另一侧, 后两步几乎不起作用, 且原 L 保留。
#      * 本变体: 第一步得到 P1, 第二步把 P1 再翻到横线另一侧得 P2, 第三步绕红点点反射得 P3。
#        输出里只剩 P3, P0/P1/P2 都被视作中间态而消失, 看上去像棋子沿红标记链逐步跳动。
# b) 复杂度校准
#    - 保留三种可见标记几何: 横线 / 竖线 / 单点。
#    - 强制至少 3 个标记, 且后续标记输入中不带本地 payload, 否则会退化成普通单步移动。
#    - seed 形状强制非对称, 避免多次反射后方向变化肉眼不可见。
#    - 不引入无关散点和共享颜色对象, 否则删除语义会让“哪些像素属于正在移动的棋子”变得模糊。
# c) 隐式假设审查
#    - 原 solver 依赖“每组独立、扫描整张静态输入就能找到源 payload”; 链式传递后失效。
#      修正: 先抽取全部红标记并排序, 显式维护 current_payload。
#    - 原 solver 依赖“反射只叠加, 不需要擦除源像素”; 删除语义后失效。
#      修正: 输出不再从整张输入复制彩色像素, 而是只保留红标记, 最后仅绘制链末端 payload。
#    - 原 isolated-red 分支依赖“红点只处理输入中相邻组件”; 链式后后续点标记在输入里可能是空的。
#      修正: 点标记和线标记统一都只消费 current_payload, 不再从点周围临时抓组件。
# d) 放置合法性
#    - 第一个标记必须是横线或竖线, 并与 seed 4 连通, 保证 solver 能从同一组件抽到初始 payload。
#    - 后续标记可为横线 / 竖线 / 单点, 但都放在上一步 payload 的右侧、下方或右下方, 使空间顺序与执行顺序一致。
#    - 所有中间 payload 状态、最终 payload 和全部红标记两两不重叠且不越界; 生成器以整条链的所有中间态做占位检查。
# e) 视觉可观察性
#    - 解题者会看到: 输入里后续红标记旁边为空, 输出却只有最末端出现 sprite, 这迫使其推断“中间结果继续移动”。
#    - 删除语义也稳定可见: 起点 sprite 在输出中完全消失, 与原题“源和镜像同时保留”形成鲜明差异。
# f) 可视化整洁度
#    - 只保留一条链, 各阶段沿右/下/右下展开, 避免多个机制在同一区域叠层。
#    - 标记与 payload 之间维持 1 格式的几何关系, 便于肉眼识别“经过该标记后发生反射”。
# g) 变化的可推断性
#    - 不做颜色映射; 颜色只表示同一个 sprite 身份, 变化只发生在位置和朝向上。
#    - 关键控制变量是链长度和标记方向序列, 而不是颜色值; 这些都能从 2-3 个样例中直接观察出来。
# new_main_assumptions:
#    - 整张图只有一条需要执行的标记链, 排序方式可用左上到右下的行优先顺序表示。
#    - 输入中只有第一个标记带 seed payload; 后续标记周围为空。
#    - 对于线标记, current_payload 在执行该步时完整落在红线跨度内且位于线的一侧。
#    - 生成器通过构造保证每一步反射后的 payload 不出界、不碰撞, 因而 solver 可把“无法反射”视为非法输入并提前停止。

import random as _rng
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


MARK = 2
Payload = list[tuple[int, int, int]]


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _marker_info(red_pts: list[tuple[int, int]]) -> dict | None:
    pts = sorted((int(r), int(c)) for r, c in red_pts)
    if not pts:
        return None

    rows = {int(r) for r, _c in pts}
    cols = {int(c) for _r, c in pts}
    if len(pts) == 1:
        r, c = pts[0]
        return {
            "kind": "point",
            "pts": pts,
            "anchor": (int(r), int(c)),
            "sort_key": (int(r), int(c)),
        }
    if len(rows) == 1:
        row = pts[0][0]
        c0 = min(c for _r, c in pts)
        c1 = max(c for _r, c in pts)
        return {
            "kind": "row",
            "pts": pts,
            "row": int(row),
            "c0": int(c0),
            "c1": int(c1),
            "anchor": (int(row), int(c0)),
            "sort_key": (int(row), int(c0)),
        }
    if len(cols) == 1:
        col = pts[0][1]
        r0 = min(r for r, _c in pts)
        r1 = max(r for r, _c in pts)
        return {
            "kind": "col",
            "pts": pts,
            "col": int(col),
            "r0": int(r0),
            "r1": int(r1),
            "anchor": (int(r0), int(col)),
            "sort_key": (int(r0), int(col)),
        }
    return None


def _extract_seed_payload(
    g: np.ndarray,
    bg: int,
    marker: dict,
    nonbg_lab: np.ndarray,
) -> Payload:
    labels = {int(nonbg_lab[r, c]) for r, c in marker["pts"] if int(nonbg_lab[r, c]) != 0}
    if len(labels) == 1:
        lbl = next(iter(labels))
        payload = [
            (int(r), int(c), int(g[r, c]))
            for r, c in np.argwhere(nonbg_lab == int(lbl))
            if int(g[r, c]) != int(MARK)
        ]
        if payload:
            return sorted(payload)

    if marker["kind"] != "point":
        return []

    h, w = g.shape
    pr, pc = marker["pts"][0]
    seed: Payload = []
    seen: set[tuple[int, int]] = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = int(pr + dr), int(pc + dc)
            if not _inb(h, w, rr, cc):
                continue
            v = int(g[rr, cc])
            if v == int(bg) or v == int(MARK):
                continue
            same_lab, _ = label_mask(g == int(v), connectivity=8)
            lbl = int(same_lab[rr, cc])
            for r, c in np.argwhere(same_lab == int(lbl)):
                pt = (int(r), int(c))
                if pt in seen:
                    continue
                seen.add(pt)
                seed.append((int(r), int(c), int(v)))
    return sorted(seed)


def _reflect_payload(payload: Payload, marker: dict, h: int, w: int) -> Payload:
    if not payload:
        return []

    out: Payload = []
    if marker["kind"] == "row":
        row = int(marker["row"])
        c0 = int(marker["c0"])
        c1 = int(marker["c1"])
        cols = [c for _r, c, _v in payload]
        rows = [r for r, _c, _v in payload]
        if min(cols) < c0 or max(cols) > c1:
            return []
        if not (max(rows) < row or min(rows) > row):
            return []
        for r, c, v in payload:
            rr, cc = int(2 * row - r), int(c)
            if not _inb(h, w, rr, cc):
                return []
            out.append((int(rr), int(cc), int(v)))
    elif marker["kind"] == "col":
        col = int(marker["col"])
        r0 = int(marker["r0"])
        r1 = int(marker["r1"])
        cols = [c for _r, c, _v in payload]
        rows = [r for r, _c, _v in payload]
        if min(rows) < r0 or max(rows) > r1:
            return []
        if not (max(cols) < col or min(cols) > col):
            return []
        for r, c, v in payload:
            rr, cc = int(r), int(2 * col - c)
            if not _inb(h, w, rr, cc):
                return []
            out.append((int(rr), int(cc), int(v)))
    else:
        pr, pc = marker["pts"][0]
        for r, c, v in payload:
            rr, cc = int(2 * pr - r), int(2 * pc - c)
            if not _inb(h, w, rr, cc):
                return []
            out.append((int(rr), int(cc), int(v)))

    if len({(int(r), int(c)) for r, c, _v in out}) != len(out):
        return []
    return sorted(out)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    red_lab, red_n = label_mask(g == int(MARK), connectivity=4)
    markers: list[dict] = []
    for k in range(1, int(red_n) + 1):
        pts = [(int(r), int(c)) for r, c in np.argwhere(red_lab == int(k))]
        info = _marker_info(pts)
        if info is not None:
            markers.append(info)
    if not markers:
        return g.copy()

    markers.sort(key=lambda m: tuple(map(int, m["sort_key"])))
    nonbg_lab, _ = label_mask(g != int(bg), connectivity=4)
    current_payload = _extract_seed_payload(g, int(bg), markers[0], nonbg_lab)
    if not current_payload:
        return g.copy()

    for marker in markers:
        current_payload = _reflect_payload(current_payload, marker, int(h), int(w))
        if not current_payload:
            return g.copy()

    out = np.full((h, w), int(bg), dtype=int)
    out[g == int(MARK)] = int(MARK)
    for r, c, v in current_payload:
        out[int(r), int(c)] = int(v)
    return out


def _norm_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _random_seed_shape() -> set[tuple[int, int]]:
    for _ in range(300):
        hh = _rng.randint(2, 4)
        ww = _rng.randint(2, 4)
        target = _rng.randint(4, min(7, hh * ww))
        start = (_rng.randint(0, hh - 1), _rng.randint(0, ww - 1))
        cells = {start}
        while len(cells) < target:
            r, c = _rng.choice(tuple(cells))
            nbrs = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < hh and 0 <= cc < ww:
                    nbrs.append((rr, cc))
            _rng.shuffle(nbrs)
            for nb in nbrs:
                if nb not in cells:
                    cells.add(nb)
                    break
            else:
                break

        cells = _norm_shape(cells)
        max_r = max(r for r, _c in cells)
        max_c = max(c for _r, c in cells)
        if max_r < 1 or max_c < 1:
            continue
        row_m = _norm_shape({(max_r - r, c) for r, c in cells})
        col_m = _norm_shape({(r, max_c - c) for r, c in cells})
        pt_m = _norm_shape({(max_r - r, max_c - c) for r, c in cells})
        if cells == row_m or cells == col_m or cells == pt_m:
            continue
        return cells
    raise RuntimeError("failed to generate asymmetric seed")


def _bbox_payload(payload: Payload) -> tuple[int, int, int, int]:
    rows = [r for r, _c, _v in payload]
    cols = [c for _r, c, _v in payload]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _coords_only(payload: Payload) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c, _v in payload}


def _try_place_step(
    current: Payload,
    step_kind: str,
    occupied: set[tuple[int, int]],
) -> tuple[dict, Payload] | None:
    r0, c0, r1, c1 = _bbox_payload(current)
    width = int(c1 - c0 + 1)
    height = int(r1 - r0 + 1)

    if step_kind == "row":
        extra_cap = max(0, min(2, 6 - width))
        extra = _rng.randint(0, extra_cap)
        left_pad = _rng.randint(0, extra)
        right_pad = extra - left_pad
        row = int(r1 + 1)
        mk_c0 = int(c0 - left_pad)
        mk_c1 = int(c1 + right_pad)
        marker_pts = [(int(row), int(c)) for c in range(int(mk_c0), int(mk_c1) + 1)]
        marker = {
            "kind": "row",
            "pts": marker_pts,
            "row": int(row),
            "c0": int(mk_c0),
            "c1": int(mk_c1),
            "anchor": (int(row), int(mk_c0)),
            "sort_key": (int(row), int(mk_c0)),
        }
    elif step_kind == "col":
        extra_cap = max(0, min(2, 6 - height))
        extra = _rng.randint(0, extra_cap)
        top_pad = _rng.randint(0, extra)
        bot_pad = extra - top_pad
        col = int(c1 + 1)
        mk_r0 = int(r0 - top_pad)
        mk_r1 = int(r1 + bot_pad)
        marker_pts = [(int(r), int(col)) for r in range(int(mk_r0), int(mk_r1) + 1)]
        marker = {
            "kind": "col",
            "pts": marker_pts,
            "col": int(col),
            "r0": int(mk_r0),
            "r1": int(mk_r1),
            "anchor": (int(mk_r0), int(col)),
            "sort_key": (int(mk_r0), int(col)),
        }
    elif step_kind == "point":
        bottom_right = (int(r1), int(c1))
        if bottom_right not in _coords_only(current):
            return None
        pr, pc = int(r1 + 1), int(c1 + 1)
        marker_pts = [(int(pr), int(pc))]
        marker = {
            "kind": "point",
            "pts": marker_pts,
            "anchor": (int(pr), int(pc)),
            "sort_key": (int(pr), int(pc)),
        }
    else:
        return None

    marker_cells = {(int(r), int(c)) for r, c in marker["pts"]}
    if marker_cells & occupied:
        return None

    nxt = _reflect_payload(current, marker, 10_000, 10_000)
    if not nxt:
        return None
    nxt_cells = _coords_only(nxt)
    if nxt_cells & occupied:
        return None
    if nxt_cells & marker_cells:
        return None
    return marker, nxt


def _build_chain(bg: int) -> tuple[list[dict], list[Payload], int]:
    palette = [v for v in range(1, 10) if v not in {int(MARK), int(bg)}]
    color = int(_rng.choice(palette))
    seed_shape = _random_seed_shape()
    states: list[Payload] = [sorted((int(r), int(c), int(color)) for r, c in seed_shape)]
    markers: list[dict] = []
    occupied = _coords_only(states[0])
    chain_len = _rng.randint(3, 5)

    for step_idx in range(chain_len):
        current = states[-1]
        if step_idx == 0:
            kinds = ["row", "col"]
        else:
            kinds = ["row", "col"]
            if (max(r for r, _c, _v in current), max(c for _r, c, _v in current)) in _coords_only(current):
                kinds.append("point")
        _rng.shuffle(kinds)

        placed = None
        for kind in kinds:
            placed = _try_place_step(current, kind, occupied)
            if placed is not None:
                break
        if placed is None:
            raise RuntimeError("failed to place chain step")

        marker, nxt = placed
        markers.append(marker)
        states.append(nxt)
        occupied |= {(int(r), int(c)) for r, c in marker["pts"]}
        occupied |= _coords_only(nxt)

    return markers, states, color


def _local_to_grid(markers: list[dict], states: list[Payload], bg: int) -> tuple[np.ndarray, np.ndarray]:
    all_cells: list[tuple[int, int]] = []
    all_cells.extend((int(r), int(c)) for r, c, _v in states[0])
    all_cells.extend((int(r), int(c)) for r, c, _v in states[-1])
    for marker in markers:
        all_cells.extend((int(r), int(c)) for r, c in marker["pts"])

    min_r = min(r for r, _c in all_cells)
    max_r = max(r for r, _c in all_cells)
    min_c = min(c for _r, c in all_cells)
    max_c = max(c for _r, c in all_cells)
    need_h = int(max_r - min_r + 1)
    need_w = int(max_c - min_c + 1)

    h_lo = max(18, need_h + 2)
    h_hi = min(28, need_h + 8)
    w_lo = max(18, need_w + 2)
    w_hi = min(28, need_w + 8)
    if h_lo > h_hi or w_lo > w_hi:
        raise RuntimeError("chain too large for target grid")

    h = _rng.randint(h_lo, h_hi)
    w = _rng.randint(w_lo, w_hi)
    off_r = _rng.randint(1 - min_r, h - 2 - max_r)
    off_c = _rng.randint(1 - min_c, w - 2 - max_c)

    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)

    for r, c, v in states[0]:
        rr, cc = int(r + off_r), int(c + off_c)
        inp[rr, cc] = int(v)

    for marker in markers:
        for r, c in marker["pts"]:
            rr, cc = int(r + off_r), int(c + off_c)
            inp[rr, cc] = int(MARK)
            out[rr, cc] = int(MARK)

    for r, c, v in states[-1]:
        rr, cc = int(r + off_r), int(c + off_c)
        out[rr, cc] = int(v)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    bg = int(_rng.choice([v for v in range(10) if v != int(MARK)]))
    try:
        markers, states, _ = _build_chain(int(bg))
        inp, expected = _local_to_grid(markers, states, int(bg))
    except RuntimeError:
        return None

    if np.array_equal(inp, expected):
        return None
    if np.count_nonzero(inp == int(MARK)) < 3:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid stepped-move chain input")
