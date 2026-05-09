# Combo variant of 3e6067c3 (index 1)
# combo_spec: D1/0+D5/0
# Name: 边框匹配+终点色桥梁

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


# Design analysis for combo D1/0 + D5/0
# ======================================
# a) 具体化与行为追踪
# - 原任务里，所有方块共用一种边框色，控制行颜色匹配“内部填充色”，桥段也用起点内部色上色。
# - 本融合版同时翻转两件事：
#   1. 控制行不再指向内部色，而是指向每个方块各自唯一的边框色。
#   2. 从 A 方块走到 B 方块时，桥段不再涂 A 的颜色，而是涂 B 的边框色。
# - 小例子：若三个方块内部都填 6，边框依次为 2、5、7，控制序列也是 [2, 5, 7]。
#   原任务思路会失败，因为内部没有 2/5/7 可匹配；融合后要先找 border=2 的方块，再连到 border=5，
#   而 2->5 这段桥的颜色应是 5，不是 2；5->7 段应是 7，不是 5。
# - 因此控制像素找到的是“当前要访问的边框色对象”，真正落笔的却是“下一对象的边框色”，
#   颜色归属链从“当前方块决定桥色”重构为“下一方块决定桥色”。
#
# b) 复杂度校准
# - 若仍保留多种内部色，人类可能继续按原任务把内部当标签；因此生成器强制所有方块共享同一种内部色。
# - 若相邻桥段可能同色，则 D5 的终点着色不明显；因此生成器强制路径上每个方块边框色全局唯一，
#   从而每一段桥都满足 source_border != dest_border。
# - 几何结构、桥宽、控制行位置仍保留原任务风格，避免引入与本 combo 无关的新自由度。
#
# c) 隐式假设审查
# - 原 `most common non-bg == border` 已失效，因为现在每个方块边框色不同；solver 必须枚举所有非背景色找环。
# - 原 `by[f["inner"]]` 和 `assert cur["inner"] == a` 绑定了“控制匹配内部色”的旧语义，现在必须改为 border。
# - 原 `draw_bridge(cur, nxt, a)` 绑定了“桥色来自起点”的旧语义，现在必须改为 `nxt["border"]` / `b`。
# - `aligned()`、最近候选选择、桥的像素几何只依赖 inner_bbox 的位置关系，不依赖颜色来源，因此可以保留。
# - 多项改动叠加后，新冲突点是：控制颜色与桥颜色都来自边框色，但角色不同。solver 必须明确把
#   `ctrl[i]` 当作当前节点标签，把桥色当作 `ctrl[i+1]`，不能混成同一个方向。
#
# d) 放置合法性
# - 仍采用 output-first：先放框，再沿路径画桥，最后擦桥得到输入。
# - 路径是格点 4 邻自避免路径，相邻方块天然共享同一行带或列带；`_bridge_cells()` 只会在两框之间的空带落笔。
# - 生成器逐段检查桥像素是否越界、是否与已有框或桥重叠，控制行单独放在所有方块下方。
#
# e) 视觉可观察性
# - 输入中，控制色只出现在“边框”而不出现在内部；这迫使解题者把注意力转到外轮廓。
# - 输出中，每段桥都与其终点方块边框同色；连续看 2-3 组样例即可看出桥色总是“朝向下一块”。
# - 统一内部色提供反证：若按原任务读内部色，将无法解释控制行；若按起点着色，也会与输出桥色系统性矛盾。
#
# f) 可视化整洁度
# - 画面仍只有三类元素：方块、桥、控制行，没有额外图标或重叠编码。
# - 统一内部色让大块区域更安静，唯一边框色与桥色承担全部逻辑提示，视觉层次更清晰。
#
# g) 变化的可推断性
# - 颜色种类被限制为：1 个背景色、1 个公共内部色、若干路径边框色。
# - 没有隐藏映射表；控制行直接复用边框色，桥也直接复用“终点边框色”。
# - 量化退化指标可设为 `source_border != dest_border` 的桥段数。生成器保证该值等于总桥段数，
#   因而融合逻辑与“源端着色”旧逻辑在同一输入上的输出必然不同。
#
# 9) main() 新增假设与 generator 保证
# - 假设每个目标方块是单色矩形环包住单色实心矩形；`_draw_frame()` 保证这一点。
# - 假设控制行中的每个非背景色都对应唯一一个方块边框；生成器给路径上的每个方块分配全局唯一边框色。
# - 假设公共内部色不与任何边框色冲突；生成器从剩余颜色中单独抽取一个公共内部色。
# - 假设相邻控制色对应的两个方块在几何上可桥接；路径按 4 邻格生成并用 `_bridge_cells()` 验证。


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


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _pick_best(cands: list[dict], cur: dict) -> dict:
    best = None
    bestf = None
    for f in cands:
        sr, sc = _aligned(cur, f)
        al = 1 if (sr or sc) else 0
        r1, c1 = _center(cur)
        r2, c2 = _center(f)
        d = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        cand = (int(al), -int(d))
        if best is None or cand > best:
            best = cand
            bestf = f
    assert bestf is not None
    return bestf


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    h, w = out.shape
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
    Combo rule:
    - Control pixels match square border colors, not inner colors.
    - Every bridge segment is painted with the destination square's border color.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    frames = _detect_frames(g, int(bg))
    assert frames

    by_border: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_border[int(frame["border"])].append(frame)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_r = None
    for r in range(int(h - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(w)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    start_color = int(ctrl[0])
    assert int(start_color) in by_border
    start_cands = by_border[int(start_color)]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        nxt_cands = by_border.get(int(nxt_color), [])
        if nxt_cands:
            best = None
            bestf = None
            for frame in start_cands:
                for g2 in nxt_cands:
                    sr, sc = _aligned(frame, g2)
                    al = 1 if (sr or sc) else 0
                    r1, c1 = _center(frame)
                    r2, c2 = _center(g2)
                    d = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
                    cand = (int(al), -int(d))
                    if best is None or cand > best:
                        best = cand
                        bestf = frame
            assert bestf is not None
            cur = bestf
        else:
            cur = start_cands[0]

    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["border"]) == int(a)
        assert int(b) in by_border
        nxt = _pick_best(by_border[int(b)], cur)
        _draw_bridge(out, cur, nxt, int(b))
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

    r_count = int(np.random.randint(2, 6))
    c_count = int(np.random.randint(2, 6))
    max_frames = int(r_count * c_count)
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
                if in_bounds((int(r_count), int(c_count)), int(rr), int(cc)) and (int(rr), int(cc)) not in vis
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

    row_gap = [int(np.random.randint(1, 6)) for _ in range(int(r_count) - 1)]
    col_gap = [int(np.random.randint(1, 6)) for _ in range(int(c_count) - 1)]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    r0s = [int(top_margin)]
    for k in range(int(r_count) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(c_count) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    border_colors = [int(x) for x in np.random.choice(np.array(avail, dtype=int), size=int(n_frames), replace=False)]
    border_set = set(map(int, border_colors))
    inner_choices = [int(d) for d in avail if int(d) not in border_set]
    if not inner_choices:
        return None
    inner_col = int(np.random.choice(np.array(inner_choices, dtype=int)))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    border_by_slot: dict[tuple[int, int], int] = {}
    frames: list[dict] = []
    for idx, (rr, cc) in enumerate(path):
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        border_col = int(border_colors[int(idx)])
        frame = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
            "border": int(border_col),
            "inner": int(inner_col),
        }
        frames.append(frame)
        slot_to_frame[(int(rr), int(cc))] = frame
        border_by_slot[(int(rr), int(cc))] = int(border_col)

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)
    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    h = int(ctrl_r + 1 + bottom_margin)
    w = int(max_c1 + 1 + right_margin)
    if not (7 <= int(h) <= 30 and 7 <= int(w) <= 30):
        return None

    out = np.full((int(h), int(w)), int(bg), dtype=int)
    for frame in frames:
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(frame["border"]),
            inner_col=int(inner_col),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    changed_segments = 0
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells or cells & bridge_cells_all:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(h) and 0 <= int(c) < int(w)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        src = int(border_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        dst = int(border_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        if int(src) != int(dst):
            changed_segments += 1
        for r, c in cells:
            out[int(r), int(c)] = int(dst)
        bridge_cells_all |= set(cells)

    if int(changed_segments) != int(len(path) - 1):
        return None

    ctrl_colors = [int(border_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(w):
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
    if len({int(frame["border"]) for frame in frames_found}) != int(len(frames)):
        return None
    if {int(frame["inner"]) for frame in frames_found} != {int(inner_col)}:
        return None
    if set(ctrl_colors) != {int(frame["border"]) for frame in frames_found}:
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
