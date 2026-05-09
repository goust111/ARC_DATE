# Combo variant of 3e6067c3 (index 34)
# combo_spec: D15/0+D5/0
# Name: 间隙桥梁+终点色桥梁

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4


# Design analysis for combo D15/0 + D5/0
# =======================================
# a) 具体化与行为追踪
# - 原任务：控制序列若为 [2, 5, 7]，则先找到 2 色方块，再连到 5 色方块、再连到 7 色方块。
#   原桥梁会把两方块之间的整段空隙全部填满，且 2->5 段涂 2 色，5->7 段涂 5 色。
# - 本融合变体：桥梁几何仍是轴对齐矩形，但两端都各留 1 个背景像素空隙；同时颜色改成终点色。
#   若两同排方块外框右边界/左边界分别在 c=4 和 c=10，则原桥填 5..9，本变体只填 6..8。
#   且 2->5 段现在填 5 色，5->7 段填 7 色。也就是“桥梁悬浮在中间，颜色指向终点”。
# - 因此 solver 中真正变化的只有两处：桥梁矩形的求法，以及绘制时用 `b` 而不是 `a`。
#   generator 也必须同步按这两个规则构造虚拟输出，否则 `main(inp) == out` 的自校验会失配。
#
# b) 复杂度校准
# - D15 的留白规则要求两方块之间至少有 3 格间隙，才能形成“1 格空隙 + 至少 1 格桥梁 + 1 格空隙”。
#   所以生成器把横纵 gap 都限制为 3 或 4，避免桥梁被完全裁掉，确保几何变化稳定可见。
# - D5 的终点着色若恰好与起点色相同，会退化成原任务着色。为避免这一点，生成器继续约束：
#   同一晶格行/列上的路径方块颜色互异。由于相邻路径步总在同一行或列，这保证每段桥的起终点颜色不同。
# - 不引入额外自由度，仍然只保留原任务的“行/列对齐 + 最近候选 + 控制序列”主逻辑。
#
# c) 隐式假设审查
# - 框检测逻辑依赖“边框色构成连通方框，内部是单色矩形”，融合后不变。
# - `by[color]`、`ctrl`、起点选择、最近候选选择只依赖方块颜色与几何，不依赖桥梁颜色或是否留空，仍成立。
# - 原代码的隐藏旧假设有两类：
#   1. `_bridge_cells()` / `draw_bridge()` 认为桥梁覆盖整个 gap；D15 破坏了这一点，所以要统一改成 `_bridge_rect(..., end_gap=1)`。
#   2. `draw_bridge(cur, nxt, int(a))` 和 generator 里的 `paint = colors_by_slot[a_slot]` 认为颜色来自起点；
#      D5 破坏了这一点，所以都要改成终点色。
# - `assert cur["inner"] == a` 仍然安全，因为控制序列表示访问顺序而不是桥梁颜色归属。
# - `draw_bridge()` 现在隐含要求相邻路径方块的 gap 在裁边后仍非空；generator 通过 gap >= 3 保证这一点。
#
# d) 放置合法性
# - generator 仍然采用 output-first：先摆方块，再画融合桥梁，再擦掉桥梁得到输入。
# - 对每段桥梁都检查：桥梁集合非空、不越界、不与已有桥梁重叠、不覆盖非背景像素。
# - 控制行仍在所有方块下方，桥梁只出现在相邻路径方块的共同行/列带中，因此不会碰到控制行。
#
# e) 视觉可观察性
# - 解题者从少量样例可以同时看见两个稳定信号：
#   1. 桥梁与两端方块之间都有一格背景“护城河”；
#   2. 每段桥都与终点方块同色，而不是与起点方块同色。
# - 这两个信号叠加后，桥梁看起来像“漂在空中并指向终点”的矩形条带，足够从 2-4 个样例归纳。
#
# f) 可视化整洁度
# - 仍然只有三层元素：背景、带框方块、桥梁；没有额外标记。
# - 几何裁边只会减少桥梁面积，不会增加重叠风险；颜色复用已有方块内色，不会新增噪声颜色。
#
# g) 颜色变化的可推断性
# - 桥梁颜色不是任意映射，而是直接取终点方块的内部色，结构线索非常强。
# - 颜色种类仍受 ARC 十色盘限制，且路径长度至少 4，通常能在一个样例里看到多段“桥段颜色跟着终点走”。
#
# h) 防退化量化指标
# - 对每个样本统计：
#   1. `detached_segments`: 留空后仍存在的桥段数，应该等于总桥段数；
#   2. `dest_colored_segments`: 终点色 != 起点色且实际桥梁使用终点色的段数，也应该等于总桥段数。
# - 若任一指标经常小于总桥段数，说明样本可能退化回原任务或单一变体，需要修正生成器。
#
# i) main() 新增假设与 generator 保证
# - `_bridge_rect(..., end_gap=1)` 假设相邻路径方块沿同一行/列且裁边后仍有剩余桥体；generator 用 1-touch 路径和 gap >= 3 保证。
# - 终点着色假设 `ctrl[i + 1]` 与 `nxt["inner"]` 一致；generator 直接用路径方块内色写控制行，保证一致。
# - 候选选择假设同一行/列里颜色足以区分目标，至少不会让真实路径被更近的同色干扰；
#   generator 对每行/列施加颜色唯一性约束，因此不会出现相邻路径段的颜色歧义。


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _bridge_rect(fa: dict, fb: dict, *, end_gap: int = 1) -> tuple[int, int, int, int] | None:
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]
    same_row = bool(int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2]))
    same_col = bool(int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]))
    gap = int(end_gap)

    if same_row:
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1 + gap), int(c0b - 1 - gap)
        else:
            cc0, cc1 = int(c1b + 1 + gap), int(c0a - 1 - gap)
        if int(cc0) > int(cc1):
            return None
        return int(rr0), int(cc0), int(rr1), int(cc1)

    if same_col:
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1 + gap), int(r0b - 1 - gap)
        else:
            rr0, rr1 = int(r1b + 1 + gap), int(r0a - 1 - gap)
        if int(rr0) > int(rr1):
            return None
        return int(rr0), int(cc0), int(rr1), int(cc1)

    return None


def _bridge_cells(fa: dict, fb: dict, *, end_gap: int = 1) -> set[tuple[int, int]]:
    rect = _bridge_rect(fa, fb, end_gap=int(end_gap))
    if rect is None:
        return set()
    rr0, cc0, rr1, cc1 = map(int, rect)
    return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via a control-color sequence.

    Combo rule:
    - Each bridge is a trimmed axis-aligned rectangle with one background cell left
      empty next to both endpoint squares.
    - Each bridge segment is painted with the destination square's inner color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert not (border is None)

    lab, n = label_mask(g == int(border), connectivity=4)
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

        t = None
        inner_col = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            u = set(map(int, inner.flatten()))
            if len(u) != 1:
                continue

            ok = True
            if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                ok = False
            if not ok:
                continue

            t = int(tt)
            inner_col = int(next(iter(u)))
            break

        if t is None or inner_col is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(t),
                "inner": int(inner_col),
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            }
        )

    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

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
                r1_, c1_ = center(f)
                r2_, c2_ = center(g2)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    bestf = f
        cur = bestf

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
        rect = _bridge_rect(fa, fb, end_gap=1)
        if rect is None:
            return
        rr0, cc0, rr1, cc1 = map(int, rect)
        for r in range(int(rr0), int(rr1) + 1):
            for c in range(int(cc0), int(cc1) + 1):
                if 0 <= r < H and 0 <= c < W:
                    out[int(r), int(c)] = int(color)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        cands = by[b]
        best = None
        bestf = None
        for f in cands:
            sr, sc = _aligned(cur, f)
            al = 1 if (sr or sc) else 0
            r1_, c1_ = center(cur)
            r2_, c2_ = center(f)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
        nxt = bestf
        draw_bridge(cur, nxt, int(b))
        cur = nxt

    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 3:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    max_dim = int(max(2, min(4, 24 // (int(outer) + 3))))
    R = int(np.random.randint(2, int(max_dim) + 1))
    C = int(np.random.randint(2, int(max_dim) + 1))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(12, max_frames)) + 1))

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

    # Keep the D15 moat visible: 1 bg cell + >=1 bridge cell + 1 bg cell.
    row_gap = [int(np.random.randint(3, 5)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(3, 5)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))

    r0s = [int(top_margin)]
    for k in range(int(R) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(C) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in path:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)

    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)

    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    # Same-row / same-col uniqueness keeps every path edge color-distinct, making D5 visible.
    used_in_row: dict[int, set[int]] = defaultdict(set)
    used_in_col: dict[int, set[int]] = defaultdict(set)

    start_color = int(np.random.choice(np.array(inner_palette, dtype=int)))
    colors_by_slot: dict[tuple[int, int], int] = {(0, 0): int(start_color)}
    used_in_row[0].add(int(start_color))
    used_in_col[0].add(int(start_color))

    for rr, cc in path[1:]:
        rr = int(rr)
        cc = int(cc)
        forbid = set(used_in_row[int(rr)]) | set(used_in_col[int(cc)]) | {int(start_color)}
        choices = [int(x) for x in inner_palette if int(x) not in forbid]
        if not choices:
            return None
        col = int(np.random.choice(np.array(choices, dtype=int)))
        colors_by_slot[(int(rr), int(cc))] = int(col)
        used_in_row[int(rr)].add(int(col))
        used_in_col[int(cc)].add(int(col))

    if len(colors_by_slot) != len(path):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        rr, cc = map(int, f["slot"])
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(rr, cc)]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb, end_gap=1)
        if not cells:
            ok = False
            break
        if cells & bridge_cells_all:
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
        paint = int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    if not ok:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None
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
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
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
