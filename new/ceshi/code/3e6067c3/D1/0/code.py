# Variant of 3e6067c3
# Dimension: D1 — 匹配条件：颜色匹配方式 (match_cond)
# Change: 控制像素颜色与方块内部填充颜色相同 (s.colors == d.inner_squares.colors)... → 边框颜色匹配

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


# Design analysis
# a) 具体化与行为追踪
# - 原任务输入里，所有方块共享同一种边框色，控制行的颜色序列对应“内部填充色”。
#   solver 先用“最常见的非背景色”锁定边框，再从该颜色的连通块里提取所有 framed square。
# - 本变体把“识别标签”从内部颜色挪到边框颜色。每个方块边框色都不同，控制行按边框色编码。
#   为了让变化肉眼可见，内部填充不再承担识别职责，而是刻意做成重复的公共颜色。
# - 小例子：
#   原逻辑：控制序列 [2, 5, 7] -> 找 inner=2 的方块，再连到 inner=5，再连到 inner=7，桥颜色分别画 2、5。
#   新逻辑：三个方块内部都可能是同一个 6，但边框分别是 2、5、7；控制序列仍是 [2, 5, 7]，
#   此时必须看边框，桥颜色也画当前源方块的边框色 2、5。
# - 差异发生在两个环节：
#   1. 方块检测：不能再假设“存在单一全局边框色”，而要按颜色枚举候选环形组件。
#   2. 控制匹配：`by[color]` 必须按 `frame["border"]` 建，而不是按 `frame["inner"]` 建。
#
# b) 复杂度校准
# - 不把规则做成“任意花纹边框”或“边框和内部都参与匹配”，否则自由度过高。
# - 仅改变一个核心条件：控制像素只匹配边框色。几何布局、桥接方式、路径连续性都保持原任务风格。
# - 为了避免视觉退化回原任务，生成器让多个方块复用同一种内部颜色；因此仅看内部无法解释控制序列。
#
# c) 隐式假设审查
# - 原 `main()` 里“most common non-bg == border”在本变体失效，因为每个方块边框色不同。
# - 原 `label_mask(g == border)` 只在单一边框色前提下成立；现在必须对每种非背景色分别找组件。
# - 原 `by[f["inner"]]`、`assert cur["inner"] == a` 绑定了旧匹配条件；现在改为 `border`。
# - 其余如 `aligned()`、桥宽等逻辑只依赖几何，不依赖匹配颜色来源，保留即可。
#
# d) 放置合法性
# - 仍沿用 output-first：先放方块、再按路径画桥、最后擦掉桥得到输入。
# - 自避免路径 + 行列间至少 1 个背景像素，保证方块不重叠且桥有空间落下。
# - 控制行放在所有方块下方的单独一行，避免和方块或桥连接。
#
# e) 视觉可观察性
# - 人类从样例中会看到：控制行颜色出现在方块边框上，而不是出现在中心填充里。
# - 内部大面积重复颜色提供了反证：若仍按内部颜色理解，控制行会失去对应物。
#
# f) 可视化整洁度
# - 不增加额外标记，不在同一区域叠多层提示。仍只有三类元素：方块、桥、控制行。
# - 统一内部色让画面更整齐，同时突出轮廓色。
#
# g) 颜色变化可推断性
# - 颜色角色只有三类：背景色、若干唯一边框色、一个重复内部色。
# - 控制行只复用边框色，不引入额外映射表；桥颜色直接等于当前源方块的边框色。
#
# 9) 新的 solver 假设与 generator 保证
# - 假设每个有效方块的边框是单色矩形环，内部是单色实心矩形。generator 用 `_draw_frame()` 保证。
# - 假设控制行颜色都能在某个方块边框上找到。generator 直接用路径上的边框色生成控制行。
# - 假设同一路径相邻方块共享同一行带或同一列带。generator 的路径仍是格点 4 邻路径，保证这一点。


def _detect_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    colors = sorted(set(map(int, g.flatten())) - {int(bg)})
    for border_col in colors:
        lab, n = label_mask(g == int(border_col), connectivity=4)
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

            found = None
            for tt in range(1, int(min(h, w) // 2) + 1):
                inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
                if inner.size == 0:
                    continue
                if bool((inner == int(border_col)).any()):
                    continue
                inner_vals = set(map(int, inner.flatten()))
                if len(inner_vals) != 1:
                    continue
                inner_col = int(next(iter(inner_vals)))
                if int(inner_col) == int(bg):
                    continue

                ring_area = int(h * w - (h - 2 * tt) * (w - 2 * tt))
                if int(pts.shape[0]) != int(ring_area):
                    continue

                ok = True
                if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border_col)).all()):
                    ok = False
                if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border_col)).all()):
                    ok = False
                if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border_col)).all()):
                    ok = False
                if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border_col)).all()):
                    ok = False
                if not ok:
                    continue

                found = {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "t": int(tt),
                    "border": int(border_col),
                    "inner": int(inner_col),
                    "inner_bbox": (int(r0 + tt), int(c0 + tt), int(r1 - tt), int(c1 - tt)),
                }
                break

            if found is not None:
                frames.append(found)

    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a control-pixel sequence.

    Variant rule:
    - Every framed square is identified by its border color, not its inner fill color.
    - Control pixels below the lattice list border colors left-to-right.
    - For each consecutive pair, connect the matching squares with a bridge colored
      by the current square's border color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    frames = _detect_frames(g, int(bg))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["border"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)

    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert not (len(ctrl) < 2)

    def center(f: dict) -> tuple[int, int]:
        r0, c0, r1, c1 = map(int, f["bbox"])
        return int((r0 + r1) // 2), int((c0 + c1) // 2)

    def aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
        a = f1["inner_bbox"]
        b = f2["inner_bbox"]
        same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
        same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
        return same_row, same_col

    start_color = int(ctrl[0])
    assert not (start_color not in by)
    start_cands = by[start_color]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        nxt_cands = by.get(nxt_color, [])
        best = None
        bestf = None
        for f in start_cands:
            for g2 in (nxt_cands if nxt_cands else [f]):
                sr, sc = aligned(f, g2)
                al = 1 if (sr or sc) else 0
                (r1_, c1_) = center(f)
                (r2_, c2_) = center(g2)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    bestf = f
        cur = bestf

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
        (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
        (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
        ia = fa["inner_bbox"]
        sr, sc = aligned(fa, fb)
        if sr:
            rr0, rr1 = int(ia[0]), int(ia[2])
            if int(c1a) < int(c0b):
                cc0, cc1 = int(c1a + 1), int(c0b - 1)
            else:
                cc0, cc1 = int(c1b + 1), int(c0a - 1)
            for r in range(int(rr0), int(rr1) + 1):
                for c in range(int(cc0), int(cc1) + 1):
                    if 0 <= r < H and 0 <= c < W:
                        out[int(r), int(c)] = int(color)
            return
        if sc:
            cc0, cc1 = int(ia[1]), int(ia[3])
            if int(r1a) < int(r0b):
                rr0, rr1 = int(r1a + 1), int(r0b - 1)
            else:
                rr0, rr1 = int(r1b + 1), int(r0a - 1)
            for r in range(int(rr0), int(rr1) + 1):
                for c in range(int(cc0), int(cc1) + 1):
                    if 0 <= r < H and 0 <= c < W:
                        out[int(r), int(c)] = int(color)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["border"]) == int(a)
        assert int(b) in by
        cands = by[b]
        best = None
        bestf = None
        for f in cands:
            sr, sc = aligned(cur, f)
            al = 1 if (sr or sc) else 0
            (r1_, c1_) = center(cur)
            (r2_, c2_) = center(f)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
        nxt = bestf
        draw_bridge(cur, nxt, int(a))
        cur = nxt

    return out


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]
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


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    avail = [int(d) for d in _DIGITS if int(d) != int(bg)]

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(2, 4))  # 2..3: larger repeated interior is visually obvious
    outer = int(inner + 2 * t)
    if not (4 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(8, max_frames)) + 1))
    if len(avail) < int(n_frames + 1):
        return None

    path: list[tuple[int, int]] | None = None
    for _path_try in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _ in range(int(n_frames) - 1):
            r, c = map(int, p[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in vis
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            p.append((int(nxt[0]), int(nxt[1])))
            vis.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in p}) < 2 or len({c for _r, c in p}) < 2:
            continue
        path = p
        break
    if path is None:
        return None

    row_gap = [int(np.random.randint(1, 6)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 6)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    r0s = [int(top_margin)]
    for k in range(int(R) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(C) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    border_colors = [int(x) for x in np.random.choice(np.array(avail, dtype=int), size=int(n_frames), replace=False)]
    border_set = set(map(int, border_colors))
    inner_choices = [int(d) for d in avail if int(d) not in border_set]
    if not inner_choices:
        return None
    inner_col = int(np.random.choice(np.array(inner_choices, dtype=int)))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    border_by_slot: dict[tuple[int, int], int] = {}
    for idx, (rr, cc) in enumerate(path):
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        border_col = int(border_colors[int(idx)])
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
            "border": int(border_col),
            "inner": int(inner_col),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f
        border_by_slot[(int(rr), int(cc))] = int(border_col)

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)

    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(f["border"]),
            inner_col=int(inner_col),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells or cells & bridge_cells_all:
            ok = False
            break
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                ok = False
                break
            if int(out[int(r), int(c)]) != int(bg):
                ok = False
                break
        if not ok:
            break
        paint = int(border_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    ctrl_colors = [int(border_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cc in zip(ctrl_colors, ctrl_cols):
        out[int(ctrl_r), int(cc)] = int(col)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None

    frames_found = _detect_frames(inp, int(bg))
    if int(len(frames_found)) != int(len(frames)):
        return None
    if len({int(f["border"]) for f in frames_found}) != int(len(frames)):
        return None
    if {int(f["inner"]) for f in frames_found} != {int(inner_col)}:
        return None
    if set(ctrl_colors) != {int(f["border"]) for f in frames_found}:
        return None
    if int(inner_col) in set(ctrl_colors):
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
