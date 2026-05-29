# Combo variant of 3e6067c3 (index 2)
# combo_spec: D1/0+D3/1
# Name: 边框匹配+左侧列控制

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4


# Design analysis
# a) 具体化与行为追踪
# - 原任务里，所有方块共享同一种边框色，控制信息位于底部一行，颜色序列按左到右读取，
#   并且这些颜色对应的是方块内部填充色。
# - 本融合变体同时改两点：
#   1. 控制像素搬到左侧单列，按上到下编码访问顺序。
#   2. 控制像素匹配的是方块边框色，不再匹配内部填充色。
# - 小例子：假设有 4 个 framed square，内部都填同一个颜色 6，但边框分别是 2/3/4/8。
#   左侧控制列从上到下是 [3, 8, 2]。那么解题者必须先找 border=3 的方块，
#   再连到 border=8 的方块，再连到 border=2 的方块，且第一段桥颜色画 3、第二段桥颜色画 8。
# - 与原始逻辑的差异有两个协同点：
#   1. 读取顺序不再来自底部水平条，而来自左侧垂直条。
#   2. 读取到的颜色不再去找 inner，而要去找 border。由于内部色被刻意做成共享，
#      若仍按原任务思路看内部颜色，就无法解释控制序列。
#
# b) 复杂度校准
# - 只改“控制位置”和“颜色匹配对象”两个核心维度，桥接几何、格点路径和相邻方块的连接方式保持原任务风格。
# - generator 强制所有方块共享同一个内部色、每个方块使用唯一边框色，避免视觉上退化回原题。
# - 控制列至少包含 4 个像素，且使用完整路径顺序，不做压缩编码，使“左侧列 + 边框匹配”在样例中反复出现。
#
# c) 隐式假设审查
# - 原 `main()` 假设“最常见非背景色就是全局边框色”，这里失效，因为每个方块边框色不同，
#   而共享内部色反而更可能成为最常见非背景色。
# - 原 `label_mask(g == border)` 只适用于单一边框色；现在必须对每种非背景色分别检测矩形环组件。
# - 原控制读取逻辑依赖“方块下方存在一行控制像素”，这里失效，改为在所有方块左侧寻找控制列。
# - 原 `by[f["inner"]]`、`assert cur["inner"] == a` 都绑定了旧匹配条件，必须切换到 `border`。
# - 桥绘制逻辑本身只依赖两方块的几何对齐关系，不依赖颜色语义，所以可以保留。
#
# d) 放置合法性
# - generator 仍采用 output-first：先放方块，再按路径画桥，再擦除桥得到输入。
# - 方块放在带间隙的格点上，路径是 4 邻自避免路径，所以相邻方块必然共享同一行带或同一列带。
# - 左侧控制列与主区域之间至少保留两列背景，避免控制像素与边框连通。
# - 控制像素在同一列按隔行放置，互不接触，也不会与桥或方块重叠。
#
# e) 视觉可观察性
# - 解题者从少量样例就能直接看到控制信息不在底部，而是在左侧竖列。
# - 控制列颜色都出现在方块边框上，却不出现在内部填充上；共享内部色提供了明显反证。
# - 桥颜色等于当前源方块的边框色，因此 input/output 对照会强化“看边框而非看内部”的规律。
#
# f) 可视化整洁度
# - 画面只保留三类元素：主区域 framed square、左侧控制列、输出中的桥。
# - 控制列与主区域分区明确，不和边框或桥叠在一起；共享内部色也让视觉焦点集中在外框。
#
# g) 变化的可推断性
# - 非背景颜色角色受控：一个共享内部色、若干唯一边框色、以及桥复用边框色。
# - 不引入任意映射表。控制色 = 边框色，桥色 = 当前源方块边框色，规律可由结构直接观察得到。
#
# h) 融合反退化约束
# - generator 永远只放左侧控制列，不放可被解释为底部控制行的有效序列。
# - generator 永远让内部色与任何控制色不同，且所有方块共享这个内部色。
# - 因此若不用“左侧列读取 + 边框匹配”这两个变化，规则无法正确解释。
#
# i) 新 solver 假设与 generator 保证
# - solver 假设有效方块是单色矩形边框环 + 单色实心内部；generator 用 `_draw_frame()` 保证。
# - solver 假设控制列严格位于全部方块左侧且至少含 2 个非背景像素；generator 保证这一点。
# - solver 假设控制列颜色都能在某个方块边框上唯一找到；generator 为每个路径方块分配唯一边框色并直接复用。


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


def _read_left_control_sequence(g: np.ndarray, bg: int, frames: list[dict]) -> list[int]:
    min_frame_c = min(int(f["bbox"][1]) for f in frames)
    ctrl_c = None
    for c in range(int(min_frame_c) - 1, -1, -1):
        if int((g[:, int(c)] != int(bg)).sum()) >= 2:
            ctrl_c = int(c)
            break
    assert not (ctrl_c is None)
    ctrl = [int(g[int(r), int(ctrl_c)]) for r in range(int(g.shape[0])) if int(g[int(r), int(ctrl_c)]) != int(bg)]
    assert not (len(ctrl) < 2)
    return ctrl


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _choose_best(cur: dict, cands: list[dict]) -> dict:
    best = None
    bestf = None
    for f in cands:
        sr, sc = _aligned(cur, f)
        al = 1 if (sr or sc) else 0
        r1_, c1_ = _center(cur)
        r2_, c2_ = _center(f)
        d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
        cand = (int(al), -int(d))
        if best is None or cand > best:
            best = cand
            bestf = f
    assert bestf is not None
    return bestf


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    h, w = map(int, out.shape)
    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    sr, sc = _aligned(fa, fb)
    if sr:
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        for r in range(int(rr0), int(rr1) + 1):
            for c in range(int(cc0), int(cc1) + 1):
                if 0 <= r < h and 0 <= c < w:
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
                if 0 <= r < h and 0 <= c < w:
                    out[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a left-side control sequence.

    Fused rule:
    - Control pixels live in a dedicated column left of the square field and are read top-to-bottom.
    - Each control color identifies a square by its border color, not by its shared inner fill color.
    - For each consecutive pair, draw a bridge colored by the current square's border color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frames = _detect_frames(g, int(bg))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["border"])].append(f)

    ctrl = _read_left_control_sequence(g, int(bg), frames)

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
                sr, sc = _aligned(f, g2)
                al = 1 if (sr or sc) else 0
                r1_, c1_ = _center(f)
                r2_, c2_ = _center(g2)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    bestf = f
        assert bestf is not None
        cur = bestf

    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["border"]) == int(a)
        assert int(b) in by
        nxt = _choose_best(cur, by[b])
        _draw_bridge(out, cur, nxt, int(a))
        cur = nxt

    return out


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])
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
    inner = int(np.random.randint(2, 4))
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
    for _ in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _step in range(int(n_frames) - 1):
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
    ctrl_c = int(np.random.randint(0, 3))
    ctrl_gap = int(np.random.randint(2, 6))
    left_margin = int(ctrl_c + ctrl_gap + 1)

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

    ctrl_top = int(np.random.randint(0, 3))
    ctrl_rows = [int(ctrl_top + 2 * i) for i in range(int(n_frames))]
    bottom_margin = int(np.random.randint(1, 4))
    right_margin = int(np.random.randint(1, 5))
    H = int(max(int(max_r1 + 1), int(ctrl_rows[-1] + 1)) + bottom_margin)
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
    for row, col in zip(ctrl_rows, ctrl_colors):
        out[int(row), int(ctrl_c)] = int(col)

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
    if int(inner_col) in set(ctrl_colors):
        return None
    if set(ctrl_colors) != {int(f["border"]) for f in frames_found}:
        return None

    ctrl = _read_left_control_sequence(inp, int(bg), frames_found)
    if ctrl != ctrl_colors:
        return None

    min_frame_c = min(int(f["bbox"][1]) for f in frames_found)
    left_ctrl_cols = [int(c) for c in range(int(min_frame_c)) if int((inp[:, int(c)] != int(bg)).sum()) >= 2]
    if left_ctrl_cols != [int(ctrl_c)]:
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
