# Variant of 3e6067c3
# Dimension: D8 — 多步处理：控制序列处理顺序 (multi_step)
# Change: 从左到右顺序处理控制像素对，每对连续像素定义一段桥梁... → 从两端向中间交替处理

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask

"""
Design analysis for the D8 variant

a) 具体化与行为追踪
- 原题输入仍然是若干同尺寸带框方块，底部一行控制像素给出颜色序列；输出是在对应方块之间补上彩色桥梁。
- 这次改变的不是“桥梁长什么样”，而是“哪一段先画、哪一段后画”。为了让这个顺序在最终图像上可见，
  generator 不再只生成互不相交的相邻桥，而是生成至少一处横桥和竖桥在空白区域中相交的布局。
- 小例子：若控制色依次是 [A,B,C,D,E,F]，原题处理 pair 索引 [0,1,2,3,4]；本变体处理 [0,4,1,3,2]。
  令 pair1(B->C) 的横桥和 pair4(E->F) 的竖桥在中心交叉：
  原题最终交叉点颜色来自 E，因为 pair4 最后画；变体最终交叉点颜色来自 B，因为 pair1 在变体里晚于 pair4。
  因此“从两端向中间交替处理”会直接改变输出中交叉点的可见颜色。

b) 复杂度校准
- 若仍使用原 generator 的不相交路径，换顺序不会改变最终输出，D8 会退化成不可观察的隐藏变化。
- 因此保留“框 + 控制序列 + 按连续色对画桥”的整体任务，只把几何布局改成稀疏对齐布局，让部分桥相交。
- 颜色种类控制在 6 个方块、6 种内部色，避免多重重复颜色带来的额外歧义；复杂度集中在“顺序覆写”而不是颜色匹配。

c) 隐式假设审查（main 中与 D8 相关的旧依赖）
- 旧版 `cur = start_frame` 假设存在一条单向遍历路径，后一段桥总从前一段终点继续出发；这在“先左端 pair、再右端 pair”下失效。
- 旧版 `assert cur["inner"] == a` 依赖当前状态与控制序列同步；变体中 pair 是独立处理的，不能再保留这个断言。
- 旧版按 `for i in range(len(ctrl)-1)` 线性处理；变体必须显式构造新的 pair 索引顺序。
- 旧版 generator 用 `if cells & bridge_cells_all: fail` 禁止所有桥重叠；这会抹掉 D8 的可见性，因此必须改成允许重叠并利用覆写次序。

d) 放置合法性
- generator 使用一个固定的六框稀疏骨架，并对其做旋转/镜像。连续控制色对仍然只连接同行或同列的方块。
- 每条桥的通道都要求非空、在网格内、且不覆盖任何框或控制行；只允许桥与桥之间相交。
- 输入通过“先构造输出，再擦除全部桥单元，再跑 main 验证”保证合法。

e) 视觉可观察性
- 关键视觉线索是交叉点颜色取决于哪条桥后画。多组样例中，控制序列两端对应的桥会先被画出，较内侧的一条桥再覆盖中心交叉点。
- 这个规律完全由位置与相交关系体现，不依赖任何颜色编号大小或不可见排序。

f) 可视化整洁度
- 框仍然大小统一、控制行仍在底部固定奇数列，画面结构与原题一致。
- 相交只发生在空白区域中心，不会穿过方块内部，也不会与控制行叠加，避免杂乱。

g) 颜色变化可推断性
- 每条桥仍使用该 pair 左端颜色，和原题一致；新增的只有“后画者覆盖先画者”。
- 方块内部色全局唯一，解题者不需要额外推断重复颜色的对应关系。

9) main() 的新隐式假设与 generator 对应保证
- 新 main 假设每个控制色对都能在现有框中找到一个最合理的同行/同列匹配；generator 通过固定骨架保证这一点。
- 新 main 允许桥相交，并默认后处理的 pair 可以覆写先处理的 pair；generator 专门构造至少一处这样的交叉。
- 新 main 不再依赖单一遍历状态，而是把每个 pair 独立解析；generator 因此把每个内部色设为全局唯一，使匹配稳定且可观察。
"""


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(frame_a: dict, frame_b: dict) -> tuple[bool, bool]:
    a = frame_a["inner_bbox"]
    b = frame_b["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _pair_order_from_ends(n_ctrl: int) -> list[int]:
    order: list[int] = []
    left = 0
    right = int(n_ctrl) - 2
    while left <= right:
        order.append(int(left))
        left += 1
        if left <= right:
            order.append(int(right))
            right -= 1
    return order


def _most_common_nonbg(grid: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, grid.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(grid: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    grid[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    grid[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _bridge_cells(frame_a: dict, frame_b: dict) -> set[tuple[int, int]]:
    r0a, c0a, r1a, c1a = map(int, frame_a["bbox"])
    r0b, c0b, r1b, c1b = map(int, frame_b["bbox"])
    ia = frame_a["inner_bbox"]
    ib = frame_b["inner_bbox"]
    if int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2]):
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    return set()


def _draw_bridge(grid: np.ndarray, frame_a: dict, frame_b: dict, color: int) -> None:
    for r, c in _bridge_cells(frame_a, frame_b):
        if 0 <= int(r) < grid.shape[0] and 0 <= int(c) < grid.shape[1]:
            grid[int(r), int(c)] = int(color)


def _detect_frames(grid: np.ndarray, border: int) -> list[dict]:
    lab, n = label_mask(grid == int(border), connectivity=4)
    frames: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if int(pts.shape[0]) < 8:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        thick = None
        inner_col = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = grid[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            vals = set(map(int, inner.flatten()))
            if len(vals) != 1:
                continue
            ok = True
            if not bool((grid[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((grid[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((grid[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                ok = False
            if not bool((grid[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                ok = False
            if not ok:
                continue
            thick = int(tt)
            inner_col = int(next(iter(vals)))
            break
        if thick is None or inner_col is None:
            continue
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(thick),
                "inner": int(inner_col),
                "inner_bbox": (int(r0 + thick), int(c0 + thick), int(r1 - thick), int(c1 - thick)),
            }
        )
    return frames


def _best_pair(by_color: dict[int, list[dict]], color_a: int, color_b: int) -> tuple[dict, dict] | None:
    cands_a = by_color.get(int(color_a), [])
    cands_b = by_color.get(int(color_b), [])
    best = None
    best_pair = None
    for frame_a in cands_a:
        for frame_b in cands_b:
            if frame_a is frame_b:
                continue
            same_row, same_col = _aligned(frame_a, frame_b)
            aligned = 1 if (same_row or same_col) else 0
            cells = _bridge_cells(frame_a, frame_b)
            usable = 1 if cells else 0
            ra, ca = _center(frame_a)
            rb, cb = _center(frame_b)
            dist = abs(int(ra) - int(rb)) + abs(int(ca) - int(cb))
            cand = (int(usable), int(aligned), -int(dist))
            if best is None or cand > best:
                best = cand
                best_pair = (frame_a, frame_b)
    return best_pair


def _control_colors(grid: np.ndarray, bg: int, frames: list[dict]) -> list[int]:
    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_r = None
    for r in range(int(grid.shape[0] - 1), int(max_frame_r), -1):
        if int((grid[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None
    ctrl = [int(grid[int(ctrl_r), int(c)]) for c in range(int(grid.shape[1])) if int(grid[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2
    return ctrl


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert border is not None

    frames = _detect_frames(g, int(border))
    assert frames

    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["inner"])].append(frame)

    ctrl = _control_colors(g, int(bg), frames)

    out = g.copy()
    for idx in _pair_order_from_ends(len(ctrl)):
        color_a = int(ctrl[int(idx)])
        color_b = int(ctrl[int(idx) + 1])
        pair = _best_pair(by_color, int(color_a), int(color_b))
        if pair is None:
            continue
        frame_a, frame_b = pair
        _draw_bridge(out, frame_a, frame_b, int(color_a))
    return out


_DIGITS = [int(i) for i in range(10)]
_BASE_SLOTS = [(0, 0), (2, 0), (2, 4), (0, 4), (0, 2), (4, 2)]


def _transform_slot(slot: tuple[int, int], mode: int) -> tuple[int, int]:
    r, c = map(int, slot)
    if int(mode) == 0:
        return int(r), int(c)
    if int(mode) == 1:
        return int(c), int(4 - r)
    if int(mode) == 2:
        return int(4 - r), int(4 - c)
    if int(mode) == 3:
        return int(4 - c), int(r)
    if int(mode) == 4:
        return int(r), int(4 - c)
    if int(mode) == 5:
        return int(4 - r), int(c)
    if int(mode) == 6:
        return int(c), int(r)
    return int(4 - c), int(4 - r)


def _render_with_order(
    *,
    shape: tuple[int, int],
    bg: int,
    border: int,
    frames: list[dict],
    colors: list[int],
    ctrl_r: int,
    ctrl_cols: list[int],
    order: list[int],
) -> tuple[np.ndarray, list[set[tuple[int, int]]]]:
    out = np.full(shape, int(bg), dtype=int)
    for frame, color in zip(frames, colors):
        r0, c0, r1, _c1 = map(int, frame["bbox"])
        outer = int(r1 - r0 + 1)
        _draw_frame(out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(frame["t"]), border=int(border), inner_col=int(color))
    pair_cells: list[set[tuple[int, int]]] = []
    for idx in range(len(colors) - 1):
        cells = _bridge_cells(frames[int(idx)], frames[int(idx) + 1])
        pair_cells.append(set(cells))
    for idx in order:
        _draw_bridge(out, frames[int(idx)], frames[int(idx) + 1], int(colors[int(idx)]))
    for color, col in zip(colors, ctrl_cols):
        out[int(ctrl_r), int(col)] = int(color)
    return out, pair_cells


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 6:
        return None

    t = 1
    inner_side = int(np.random.choice(np.array([1, 2], dtype=int)))
    outer = int(inner_side + 2 * t)

    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))
    slot_row_gap = [int(np.random.choice(np.array([1, 1, 2], dtype=int))) for _ in range(4)]
    slot_col_gap = [int(np.random.choice(np.array([1, 1, 2], dtype=int))) for _ in range(4)]

    r0s = [int(top_margin)]
    for gap in slot_row_gap:
        r0s.append(int(r0s[-1] + outer + int(gap)))
    c0s = [int(left_margin)]
    for gap in slot_col_gap:
        c0s.append(int(c0s[-1] + outer + int(gap)))

    mode = int(np.random.randint(0, 8))
    slots = [_transform_slot(slot, int(mode)) for slot in _BASE_SLOTS]

    frames: list[dict] = []
    occupied = set()
    for rr, cc in slots:
        if (int(rr), int(cc)) in occupied:
            return None
        occupied.add((int(rr), int(cc)))
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
                "t": int(t),
            }
        )

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)
    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 4))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=len(frames), replace=False)]
    ctrl_cols = [int(1 + 2 * i) for i in range(len(colors))]
    if int(ctrl_cols[-1]) >= int(W):
        return None

    base_grid = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame, color in zip(frames, colors):
        r0, c0, r1, _c1 = map(int, frame["bbox"])
        _draw_frame(base_grid, r0=int(r0), c0=int(c0), outer=int(r1 - r0 + 1), t=int(frame["t"]), border=int(border), inner_col=int(color))

    pair_cells: list[set[tuple[int, int]]] = []
    for idx in range(len(frames) - 1):
        cells = _bridge_cells(frames[int(idx)], frames[int(idx) + 1])
        if not cells:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(base_grid[int(r), int(c)]) != int(bg):
                return None
        pair_cells.append(set(cells))

    alt_order = _pair_order_from_ends(len(colors))
    ltr_order = list(range(len(colors) - 1))
    alt_out, _ = _render_with_order(
        shape=(int(H), int(W)),
        bg=int(bg),
        border=int(border),
        frames=frames,
        colors=colors,
        ctrl_r=int(ctrl_r),
        ctrl_cols=ctrl_cols,
        order=alt_order,
    )
    ltr_out, _ = _render_with_order(
        shape=(int(H), int(W)),
        bg=int(bg),
        border=int(border),
        frames=frames,
        colors=colors,
        ctrl_r=int(ctrl_r),
        ctrl_cols=ctrl_cols,
        order=ltr_order,
    )
    if bool(np.array_equal(alt_out, ltr_out)):
        return None

    overlap_cells = 0
    for i in range(len(pair_cells)):
        for j in range(i + 1, len(pair_cells)):
            overlap_cells += int(len(pair_cells[int(i)] & pair_cells[int(j)]))
    if int(overlap_cells) <= 0:
        return None

    bridge_union: set[tuple[int, int]] = set()
    for cells in pair_cells:
        bridge_union |= set(cells)

    inp = alt_out.copy()
    for r, c in bridge_union:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if pred.shape != alt_out.shape:
        return None
    if not bool(np.array_equal(pred, alt_out)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
