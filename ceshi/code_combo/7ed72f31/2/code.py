# Combo variant of 7ed72f31 (index 2)
# combo_spec: D1/0+D8/1
# Name: 旋转链式传递

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask


"""
design_analysis:
  fused_rule:
    summary:
      - 红标记不再做镜像，而是都表示 90 度顺时针旋转中心。
      - 多个标记不是独立处理，而是按空间顺序串成一条链；第 i 步的旋转结果作为第 i+1 步的输入。
      - 输出保留初始 payload，并叠加每一步链式旋转得到的中间副本。
    spatial_order_choice:
      - "空间顺序"具体化为 marker 中心从左到右排序，列坐标相同时再按行坐标排序。
      - 这样 generator 可以把红标记明确排成一条水平推进的链，人类从 2-3 个样例里能直接看出顺序。
      - 不使用颜色值、面积等不可视属性做排序。
  concrete_visualization:
    small_trace:
      - 初始有一个不对称蓝色 payload P0，贴着最左侧红线。
      - 第 1 个 marker 是水平红线，P0 绕其中点顺时针 90 度，生成 P1，P1 出现在右侧。
      - 第 2 个 marker 是更靠右的竖直红线。输入里它附近是空的，但 solver 不再回看静态输入，而是把 P1 再顺时针旋转成 P2。
      - 第 3 个 marker 是更靠右下的单红点，再把 P2 顺时针旋转成 P3。
      - 输出同时显示 P0、P1、P2、P3，所以链中累计出现 90/180/270 度三种朝向。
    difference_from_original:
      - 原题后续 marker 彼此独立，且线 marker 做轴镜像，不会出现 "空 marker 消耗上一步结果" 的现象。
      - 融合版里后两个 marker 在输入中局部没有 payload，但输出仍产生新副本；这是链式传递的关键可见线索。
      - 同时线 marker 产生的是正交侧的旋转副本，不是轴对侧的镜像副本。
  complexity_calibration:
    - 每个样本固定至少 3 个步骤，并同时包含 row、col、point 三类 marker，避免退化成单类型旋转。
    - 所有线 marker 只允许奇数长度 3 或 5，既保证离散格点旋转中心是整数，又把可推断的档位控制在少数几种。
    - payload 强制为 4 连通且对旋转、水平镜像、竖直镜像都不对称，保证旋转链与原始镜像任务肉眼可区分。
  implicit_assumption_audit:
    original_main_assumptions_that_break:
      - 原 solver 假设每个 marker 都能直接从静态输入读取本地 payload；链式版本下只对第一个 marker 成立。
      - 原 solver 把 row/col marker 简化成一条反射轴；旋转版本必须真正求中点，因此偶数长度线不再合法。
      - 原 isolated-point 逻辑从当前网格邻域抓 payload；链式版本里后续单点 marker 周围是空的，不能再依赖邻域。
    fused_fix:
      - 先解析全部红 marker，再按可见空间顺序排序。
      - 只从第一个 marker 抽取 seed payload；之后每一步只消费上一轮 newly rotated payload。
      - row/col marker 只接受连续奇数长度红线；否则直接忽略，避免半格中心。
      - 旋转方向统一为顺时针 90 度，包括单红点，避免 "线是 90 度、点是 180 度" 的混搭歧义。
    new_solver_assumptions:
      - 第一个 marker 必须确实能在输入中抓到 seed payload。
      - 后续 marker 虽然在输入中局部为空，但它们的几何位置必须能容纳上一状态的 bbox 约束，否则链会提前终止。
      - marker 的左右顺序必须唯一清晰；generator 因此强制中心列严格递增。
  placement_legality:
    - generator 在局部坐标系里先构造整条链的所有 marker 和所有中间状态，再整体平移进 18-28 的网格。
    - 每一步都检查: 新 marker 不与已有 marker/状态重叠；新旋转状态也不与任何旧对象重叠。
    - 第一个 marker 与 seed payload 保持题意要求的接触关系；后续 marker 则单独放置在空背景中。
  observability:
    - 输入里只有最左 marker 带 payload，右侧 marker 是空的；输出却在这些空 marker 之后继续出现同色副本，能直接逼出链式推理。
    - 链中固定出现 row -> col -> point 三步结构：先由水平线把 payload 送到右侧，再由竖线把它转到新的朝向，最后由单点完成第 3 次 90 度旋转。
    - 这种固定模板减少自由度，同时仍然保留三类 marker 的共同参与，足以支撑解题者从少量样例看出累计旋转链。
    - 由于最终输出里同时保留三个中间朝向，人类能看到累计 90/180/270 度效果，而不必靠心算终态。
  cleanliness:
    - 不生成额外干扰散点，也不让不同副本重叠。
    - 整条链按从左到右展开，个体分区清晰；虽然部分步骤会上下蛇形，但 marker 顺序仍由可见的横向推进明确给出。
  post_edit_checklist:
    - main() 的每个分支都只依赖 generator 明确保证的条件。
    - generate_input() 通过 oracle main(input) == output 验证正确性。
    - 额外退化检测会比较本融合 solver 与原题 solver / 单变体 solver 的输出差异，确认样本没有退化回原行为。
"""


MARK = 2


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _rot90_cw_center2(center2: tuple[int, int], r: int, c: int) -> tuple[int, int] | None:
    cr2, cc2 = map(int, center2)
    rr2 = cr2 + 2 * int(c) - cc2
    cc2_new = cc2 - 2 * int(r) + cr2
    if rr2 % 2 != 0 or cc2_new % 2 != 0:
        return None
    return rr2 // 2, cc2_new // 2


def _marker_from_row(row: int, c0: int, c1: int) -> dict:
    pts = [(int(row), int(c)) for c in range(int(c0), int(c1) + 1)]
    center2 = (2 * int(row), int(c0 + c1))
    return {
        "kind": "row",
        "pts": pts,
        "row": int(row),
        "c0": int(c0),
        "c1": int(c1),
        "center2": center2,
        "sort_key": (int(center2[1]), int(center2[0])),
    }


def _marker_from_col(col: int, r0: int, r1: int) -> dict:
    pts = [(int(r), int(col)) for r in range(int(r0), int(r1) + 1)]
    center2 = (int(r0 + r1), 2 * int(col))
    return {
        "kind": "col",
        "pts": pts,
        "col": int(col),
        "r0": int(r0),
        "r1": int(r1),
        "center2": center2,
        "sort_key": (int(center2[1]), int(center2[0])),
    }


def _marker_from_point(r: int, c: int) -> dict:
    center2 = (2 * int(r), 2 * int(c))
    return {
        "kind": "point",
        "pts": [(int(r), int(c))],
        "center2": center2,
        "sort_key": (int(center2[1]), int(center2[0])),
    }


def _marker_info(red_pts: list[tuple[int, int]]) -> dict | None:
    pts = sorted((int(r), int(c)) for r, c in red_pts)
    if not pts:
        return None

    if len(pts) == 1:
        return _marker_from_point(pts[0][0], pts[0][1])

    rs = {r for r, _c in pts}
    cs = {c for _r, c in pts}
    if len(rs) == 1:
        row = pts[0][0]
        cols = sorted(c for _r, c in pts)
        if cols == list(range(cols[0], cols[-1] + 1)) and len(cols) % 2 == 1:
            return _marker_from_row(int(row), int(cols[0]), int(cols[-1]))
        return None
    if len(cs) == 1:
        col = pts[0][1]
        rows = sorted(r for r, _c in pts)
        if rows == list(range(rows[0], rows[-1] + 1)) and len(rows) % 2 == 1:
            return _marker_from_col(int(col), int(rows[0]), int(rows[-1]))
        return None
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
            if not _inb(int(h), int(w), rr, cc):
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


def _rotate_payload(
    payload: list[tuple[int, int, int]],
    marker: dict,
    h: int,
    w: int,
) -> list[tuple[int, int, int]]:
    if not payload:
        return []

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

    out: list[tuple[int, int, int]] = []
    for r, c, v in payload:
        dst = _rot90_cw_center2(marker["center2"], int(r), int(c))
        if dst is None:
            return []
        rr, cc = dst
        if not _inb(int(h), int(w), int(rr), int(cc)):
            return []
        out.append((int(rr), int(cc), int(v)))

    if len({(int(r), int(c)) for r, c, _v in out}) != len(out):
        return []
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
        current_payload = _rotate_payload(current_payload, marker, int(h), int(w))
        if not current_payload:
            break
        for r, c, v in current_payload:
            out[int(r), int(c)] = int(v)

    return out


def _neighbors4(p: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = p
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _norm(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _rot90_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    if not cells:
        return set()
    max_r = max(r for r, _c in cells)
    return {(int(c), int(max_r - r)) for r, c in cells}


def _flip_h(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_r = max(r for r, _c in cells)
    return {(int(max_r - r), int(c)) for r, c in cells}


def _flip_v(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_c = max(c for _r, c in cells)
    return {(int(r), int(max_c - c)) for r, c in cells}


def _shape_is_informative(cells: set[tuple[int, int]]) -> bool:
    if len(cells) < 4:
        return False
    norm = _norm(cells)
    max_r = max(r for r, _c in norm)
    max_c = max(c for _r, c in norm)
    if max_r < 1 or max_c < 1:
        return False

    seen = norm
    rot1 = _norm(_rot90_cells(norm))
    rot2 = _norm(_rot90_cells(rot1))
    rot3 = _norm(_rot90_cells(rot2))
    if seen == rot1 or seen == rot2 or seen == rot3:
        return False
    if seen == _norm(_flip_h(norm)) or seen == _norm(_flip_v(norm)):
        return False
    return True


def _random_seed_shape() -> set[tuple[int, int]]:
    for _ in range(400):
        hh = random.randint(2, 4)
        ww = random.randint(2, 4)
        target = random.randint(4, min(6, hh * ww))
        start = (random.randint(0, hh - 1), random.randint(0, ww - 1))
        cells = {start}
        while len(cells) < target:
            frontier = set()
            for cell in cells:
                for nxt in _neighbors4(cell):
                    rr, cc = nxt
                    if 0 <= rr < hh and 0 <= cc < ww and nxt not in cells:
                        frontier.add(nxt)
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        cells = _norm(cells)
        if _shape_is_informative(cells):
            return cells
    raise RuntimeError("failed to build informative seed")


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
    min_center_col2: int,
    require_touching_seed: bool,
) -> tuple[dict, list[tuple[int, int, int]]] | None:
    r0, c0, r1, c1 = _bbox_payload(current)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    candidates: list[tuple[dict, list[tuple[int, int, int]]]] = []

    def accept(marker: dict) -> None:
        marker_cells = {(int(r), int(c)) for r, c in marker["pts"]}
        if marker_cells & occupied:
            return
        if int(marker["center2"][1]) <= int(min_center_col2):
            return
        nxt = _rotate_payload(current, marker, 10_000, 10_000)
        if not nxt:
            return
        nxt_cells = _coords_only(nxt)
        if nxt_cells & occupied:
            return
        if nxt_cells & marker_cells:
            return
        if len(nxt_cells) != len(nxt):
            return
        candidates.append((marker, nxt))

    if step_kind == "row":
        for line_len in (3, 5):
            if width > line_len:
                continue
            for gap in ((1,) if require_touching_seed else (1, 2, 3)):
                row = int(r1 + gap)
                start_lo = int(c1 - line_len + 1)
                start_hi = int(c0)
                for start in range(start_lo, start_hi + 1):
                    end = int(start + line_len - 1)
                    accept(_marker_from_row(int(row), int(start), int(end)))
    elif step_kind == "col":
        for line_len in (3, 5):
            if height > line_len:
                continue
            for gap in ((1,) if require_touching_seed else (1, 2, 3)):
                col = int(c1 + gap)
                start_lo = int(r1 - line_len + 1)
                start_hi = int(r0)
                for start in range(start_lo, start_hi + 1):
                    end = int(start + line_len - 1)
                    accept(_marker_from_col(int(col), int(start), int(end)))
    elif step_kind == "point":
        if require_touching_seed:
            return None
        for dr in (1, 2, 3):
            for dc in (1, 2, 3):
                accept(_marker_from_point(int(r1 + dr), int(c1 + dc)))
    else:
        return None

    if not candidates:
        return None
    return random.choice(candidates)


def _build_chain(bg: int) -> tuple[list[dict], list[list[tuple[int, int, int]]]]:
    color_choices = [v for v in range(10) if v not in {int(bg), int(MARK)}]
    color = int(random.choice(color_choices))
    seed_shape = _random_seed_shape()
    states: list[list[tuple[int, int, int]]] = [
        sorted((int(r), int(c), int(color)) for r, c in seed_shape)
    ]
    occupied = _coords_only(states[0])
    markers: list[dict] = []
    last_center_col2 = -10_000

    kinds = ["row", "col", "point"]

    for idx, kind in enumerate(kinds):
        placed = _try_place_step(
            states[-1],
            kind,
            occupied,
            int(last_center_col2),
            require_touching_seed=(idx == 0),
        )
        if placed is None:
            raise RuntimeError(f"failed to place {kind} step")
        marker, nxt = placed
        markers.append(marker)
        states.append(nxt)
        occupied |= {(int(r), int(c)) for r, c in marker["pts"]}
        occupied |= _coords_only(nxt)
        last_center_col2 = int(marker["center2"][1])

    return markers, states


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

    h = random.randint(h_lo, h_hi)
    w = random.randint(w_lo, w_hi)
    off_r = random.randint(1 - min_r, h - 2 - max_r)
    off_c = random.randint(1 - min_c, w - 2 - max_c)

    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)

    for r, c, v in states[0]:
        rr, cc = int(r + off_r), int(c + off_c)
        inp[rr, cc] = int(v)
        out[rr, cc] = int(v)

    for marker in markers:
        for r, c in marker["pts"]:
            rr, cc = int(r + off_r), int(c + off_c)
            inp[rr, cc] = int(MARK)
            out[rr, cc] = int(MARK)

    for state in states[1:]:
        for r, c, v in state:
            rr, cc = int(r + off_r), int(c + off_c)
            out[rr, cc] = int(v)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice([v for v in range(10) if v != int(MARK)]))
    try:
        markers, states = _build_chain(int(bg))
        inp, out = _local_to_grid(markers, states, int(bg))
    except RuntimeError:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(2000):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid chained-rotation input")
