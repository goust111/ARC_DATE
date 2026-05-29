# Variant of 7ed72f31
# Dimension: D8 — 多标记组的处理顺序 (multi_step)
# Change: 各标记组独立处理，无特定顺序要求，结果直接叠加... → 链式反射：一次反射的结果参与下一次反射

from __future__ import annotations

# Design analysis for the D8 variant
# ----------------------------------
# spec:
#   core_change:
#     old: 每个红标记组都只读取静态输入里的本地 payload，然后把镜像结果直接叠加到输出。
#     new: 红标记组按照空间顺序（这里具体化为左上到右下的行优先顺序）串成一条链；
#          只有第一个标记从输入读取初始 payload，之后每一步都对“上一步刚生成的镜像”
#          再做一次反射，并把中间结果继续传给下一步。
#   concrete_objects:
#     markers:
#       - 横向红线：表示沿该行反射
#       - 纵向红线：表示沿该列反射
#       - 单个红点：表示绕该点中心做点反射
#     payload:
#       - 单色、不对称、4 连通的小 sprite
#       - 输入中只附着在第一个标记旁边
#       - 输出中会出现原 sprite + 每一步链式反射产生的所有中间副本
#   small_trace:
#     input:
#       - 第 1 组是右侧竖线标记，左边贴着蓝色 payload P0
#       - 第 2 组是更靠下的一条横线，输入里它附近是空的
#       - 第 3 组是更靠右下的一个红点，输入里它附近仍然是空的
#     old_solver_behavior:
#       - 只会处理第 1 组；第 2/3 组因为输入里没有本地 payload，基本不起作用
#     new_behavior:
#       - 第 1 组把 P0 镜像成 P1
#       - 第 2 组把 P1 再镜像成 P2
#       - 第 3 组把 P2 再点反射成 P3
#       - 输出累积显示 P0、P1、P2、P3
#   complexity_calibration:
#     - 关键提升不是新增很多对象类别，而是让“处理顺序”和“中间状态传递”成为必要推理步骤。
#     - 为避免视觉上退化成原题，生成器强制至少 3 个标记，并且后续标记在输入中不带本地 payload。
#     - 为避免链式规则肉眼不可见，整个链只朝右/下/右下推进，使空间顺序和执行顺序一致。
#   implicit_assumption_audit:
#     original_main_dependencies:
#       - 用整个静态输入 g 扫描 eligible 像素；这依赖“每组独立、源 payload 永远在输入里”。
#       - isolated red 的逻辑从 g 的邻域抓取组件；这依赖“单点标记只处理静态相邻组件”。
#       - out 只是叠加层，不会反过来影响后续步骤。
#     variant_fix:
#       - 改成先识别全部红标记，再显式维护 current_payload。
#       - 第一步从输入抽 seed；后续步骤只读取 current_payload，不再回头扫描整张 g。
#       - out 继续累积，但真正传递给下一步的是本步 newly reflected payload，而不是整张 out。
#   placement_legality:
#     - 第一个标记必须是横线或竖线，并与 seed payload 4 连通，保证初始 payload 可提取。
#     - 后续标记单独放置，不与任何已有非背景像素重叠；它们只在输出语义上接收前一步 payload。
#     - 所有 payload 状态、所有红标记、所有镜像副本两两不重叠且不越界。
#   visual_observability:
#     - 由于输入里只有第一个标记有真实 payload，解题者会看到“后面的红标记输入为空、输出却出现形状”，
#       从而被迫推断它们消耗的是前一步结果而不是本地对象。
#     - payload 强制不对称，避免反射后看不出方向变化。
#   cleanliness:
#     - 不放自由散点和无关彩色对象，避免链的中间副本与杂物混淆。
#     - 标记按对角线方向依次展开，避免多个阶段在同一区域堆叠。

import random as _rng

import numpy as np

from arc_utils import get_bg, label_mask


MARK = 2


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _marker_info(red_pts: list[tuple[int, int]]) -> dict | None:
    pts = sorted((int(r), int(c)) for r, c in red_pts)
    if not pts:
        return None
    rs = {r for r, _c in pts}
    cs = {c for _r, c in pts}
    if len(pts) == 1:
        r, c = pts[0]
        return {
            "kind": "point",
            "pts": pts,
            "anchor": (int(r), int(c)),
            "sort_key": (int(r), int(c)),
        }
    if len(rs) == 1:
        r = pts[0][0]
        c0 = min(c for _r, c in pts)
        c1 = max(c for _r, c in pts)
        return {
            "kind": "row",
            "pts": pts,
            "row": int(r),
            "c0": int(c0),
            "c1": int(c1),
            "anchor": (int(r), int(c0)),
            "sort_key": (int(r), int(c0)),
        }
    if len(cs) == 1:
        c = pts[0][1]
        r0 = min(r for r, _c in pts)
        r1 = max(r for r, _c in pts)
        return {
            "kind": "col",
            "pts": pts,
            "col": int(c),
            "r0": int(r0),
            "r1": int(r1),
            "anchor": (int(r0), int(c)),
            "sort_key": (int(r0), int(c)),
        }
    return None


def _extract_seed_payload(
    g: np.ndarray,
    bg: int,
    marker: dict,
    nonbg_lab: np.ndarray,
) -> list[tuple[int, int, int]]:
    labels = {int(nonbg_lab[r, c]) for r, c in marker["pts"] if int(nonbg_lab[r, c]) != 0}
    if len(labels) == 1:
        lbl = next(iter(labels))
        seed = [
            (int(r), int(c), int(g[r, c]))
            for r, c in np.argwhere(nonbg_lab == int(lbl))
            if int(g[r, c]) != int(MARK)
        ]
        if seed:
            return sorted(seed)

    if marker["kind"] != "point":
        return []

    h, w = g.shape
    pr, pc = marker["pts"][0]
    seed: list[tuple[int, int, int]] = []
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
            same_lab, _n = label_mask(g == int(v), connectivity=8)
            lbl = int(same_lab[rr, cc])
            for r, c in np.argwhere(same_lab == int(lbl)):
                pt = (int(r), int(c))
                if pt in seen:
                    continue
                seen.add(pt)
                seed.append((int(r), int(c), int(v)))
    return sorted(seed)


def _reflect_payload(
    payload: list[tuple[int, int, int]],
    marker: dict,
    h: int,
    w: int,
) -> list[tuple[int, int, int]]:
    if not payload:
        return []

    out: list[tuple[int, int, int]] = []
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
            if _inb(h, w, rr, cc):
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
            if _inb(h, w, rr, cc):
                out.append((int(rr), int(cc), int(v)))
    else:
        pr, pc = marker["pts"][0]
        for r, c, v in payload:
            rr, cc = int(2 * pr - r), int(2 * pc - c)
            if _inb(h, w, rr, cc):
                out.append((int(rr), int(cc), int(v)))
    return sorted(out)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    red_lab, red_n = label_mask(g == int(MARK), connectivity=4)
    markers: list[dict] = []
    for k in range(1, int(red_n) + 1):
        pts = [(int(r), int(c)) for r, c in np.argwhere(red_lab == int(k))]
        info = _marker_info(pts)
        if info is not None:
            markers.append(info)
    if not markers:
        return out

    markers.sort(key=lambda m: tuple(map(int, m["sort_key"])))

    nonbg_lab, _n = label_mask(g != int(bg), connectivity=4)
    current_payload = _extract_seed_payload(g, int(bg), markers[0], nonbg_lab)
    if not current_payload:
        return out

    for marker in markers:
        current_payload = _reflect_payload(current_payload, marker, int(h), int(w))
        if not current_payload:
            break
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


def _bbox_payload(payload: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c, _v in payload]
    cs = [c for _r, c, _v in payload]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _coords_only(payload: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c, _v in payload}


def _try_place_step(
    current: list[tuple[int, int, int]],
    step_kind: str,
    occupied: set[tuple[int, int]],
) -> tuple[dict, list[tuple[int, int, int]]] | None:
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
        if (int(r1), int(c1)) not in _coords_only(current):
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
    if len(nxt_cells) != len(nxt):
        return None
    return marker, nxt


def _build_chain(bg: int) -> tuple[
    list[dict],
    list[list[tuple[int, int, int]]],
    int,
]:
    color_choices = [v for v in range(1, 10) if v not in {int(MARK), int(bg)}]
    color = int(_rng.choice(color_choices))
    seed_shape = _random_seed_shape()
    states: list[list[tuple[int, int, int]]] = [
        sorted((int(r), int(c), int(color)) for r, c in seed_shape)
    ]
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


def _local_to_grid(
    markers: list[dict],
    states: list[list[tuple[int, int, int]]],
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_cells: list[tuple[int, int]] = []
    for state in states:
        all_cells.extend((int(r), int(c)) for r, c, _v in state)
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
        inp[rr, cc] = out[rr, cc] = int(v)

    for marker in markers:
        for r, c in marker["pts"]:
            rr, cc = int(r + off_r), int(c + off_c)
            inp[rr, cc] = out[rr, cc] = int(MARK)

    for state in states[1:]:
        for r, c, v in state:
            rr, cc = int(r + off_r), int(c + off_c)
            out[rr, cc] = int(v)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    bg = int(_rng.choice([v for v in range(10) if v != int(MARK)]))
    try:
        markers, states, _color = _build_chain(int(bg))
        inp, out = _local_to_grid(markers, states, int(bg))
    except RuntimeError:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid chained-reflection input")
