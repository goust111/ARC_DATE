# Variant of 3e6067c3
# Dimension: D7 — 网格形状：输出裁剪策略 (grid_shape)
# Change: 输出网格与输入相同大小，不做裁剪或扩展... → 仅输出桥梁区域
#
# Design analysis
# a) 具体化与行为追踪
#    原题 output 保留整张场景：方块、控制行、背景都还在，只是在相邻方块之间补上彩色桥梁。
#    这个变体把“识别与连接”保持不变，但把“呈现结果”的载体改掉：
#    1. 先按原逻辑找到控制序列对应的访问路径。
#    2. 对每一对相邻颜色，仍然在两个方块之间的对齐通道里画出同色桥梁。
#    3. 只保留这些桥梁像素，其余位置全部视为背景。
#    4. 最后把整张桥梁画布裁成所有桥梁像素的最小外接矩形。
#    小例子：若原图里有三个同厚度方块 A-B-C 横向排列，控制序列是 [A,B,C]。
#    原逻辑会输出“完整原图 + 两段桥”。新逻辑会输出一个只含两段横桥的窄条，
#    中间保留被删掉的方块位置形成的背景空洞；看不到方块本体和控制行。
#    差异落在 output 构造阶段，而不是匹配阶段。
# b) 复杂度校准
#    变体没有简化连接规则；解题者仍然必须先从输入里恢复访问顺序和桥梁位置，
#    只是最终只观察到桥梁的裁剪结果。为保证视觉差异明显，生成器要求至少 4 个方块，
#    因而至少 3 段桥梁，并过滤掉“裁剪后几乎不变”的退化样本。
# c) 隐式假设审查
#    原 main() 的关键旧假设是 `out = g.copy()`：这等价于默认方块、控制行、背景都应保留。
#    在本变体里该假设失效，若继续 copy 输入就会把不该出现在输出里的方块和控制像素带过去。
#    因此必须改成从纯背景画布开始，只写入桥梁像素。其余关于：
#    - 边框色是最常见非背景色
#    - 方块由边框连通域识别
#    - 控制行在所有方块之下
#    - 目标方块通过同 row-band / col-band + 最近距离选取
#    这些前提在变体中仍然成立，可以保持不变。
# d) 放置合法性
#    生成器继续使用 output-first：先构造完整场景中的桥梁关系，再擦掉桥梁得到输入。
#    它保证：
#    - 方块外框互不重叠，且行列间至少隔 1 个背景像素；
#    - 路径是格点上的 self-avoiding path，因此连续两步必同一行或同一列；
#    - 每段桥梁所在通道不与已有方块、桥梁冲突；
#    - 裁剪后的输出至少含一个非背景像素。
# e) 视觉可观察性
#    从 2-4 组样例里，人会看到：输入中仍是“带框方块 + 底部控制色列”，
#    输出却突然变成一个比输入小很多的桥梁拼图，且桥梁颜色顺序仍由控制行决定。
#    这能直接暴露“只输出桥梁并裁剪”的新规则。
# f) 可视化整洁度
#    输出里去掉了方块与控制行，剩下的元素层次更少。桥梁之间最多通过背景孔洞分隔，
#    不会和方块叠压，因此比原题更整洁。
# g) 颜色变化可推断性
#    颜色规则完全沿用原题：每段桥梁用起点方块的内色着色。没有新增隐藏颜色映射。
#    裁剪只改变画幅，不改变颜色语义。
#
# Reference-task note
# 题面要求先读 source_task `a251c730` 的 solver；在给定资源里只提供了该任务的 JSON 样例，
# 没有对应 solver 文件。因此这里以其样例中“提取目标区域并紧凑裁剪”的表现作为实现基线。

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


def _crop_to_nonbg(g: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(g != int(bg))
    if int(pts.shape[0]) == 0:
        return np.full((1, 1), int(bg), dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


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
        (r1_, c1_) = _center(cur)
        (r2_, c2_) = _center(f)
        d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
        cand = (int(al), -int(d))
        if best is None or cand > best:
            best = cand
            bestf = f
    assert bestf is not None
    return bestf


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
        return {
            (int(r), int(c))
            for r in range(int(rr0), int(rr1) + 1)
            for c in range(int(cc0), int(cc1) + 1)
        }
    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1):
            return set()
        return {
            (int(r), int(c))
            for r in range(int(rr0), int(rr1) + 1)
            for c in range(int(cc0), int(cc1) + 1)
        }
    return set()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a control-pixel sequence.

    Variant D7:
    Keep the original bridge-finding logic, but output only the bridge pixels and
    crop the result to the bridges' tight bounding box.
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
    assert border is not None

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

    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    start_color = int(ctrl[0])
    assert start_color in by
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
                (r1_, c1_) = _center(f)
                (r2_, c2_) = _center(g2)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    bestf = f
        assert bestf is not None
        cur = bestf

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        nxt = _choose_best(cur, by[b])
        for r, c in _bridge_cells(cur, nxt):
            if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
                out[int(r), int(c)] = int(a)
        cur = nxt

    return _crop_to_nonbg(out, int(bg))


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(
    g: np.ndarray,
    *,
    r0: int,
    c0: int,
    outer: int,
    t: int,
    border: int,
    inner_col: int,
) -> None:
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

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
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

    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

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

    full_scene = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        rr, cc = map(int, f["slot"])
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            full_scene,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(rr, cc)]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    bridge_canvas = np.full((int(H), int(W)), int(bg), dtype=int)
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
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
            if int(full_scene[int(r), int(c)]) != int(bg):
                ok = False
                break
        if not ok:
            break
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            bridge_canvas[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok or not bridge_cells_all:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cc in zip(ctrl_colors, ctrl_cols):
        full_scene[int(ctrl_r), int(cc)] = int(col)

    inp = full_scene.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    expected = _crop_to_nonbg(bridge_canvas, int(bg))

    # Keep the D7 change visually obvious: output must shrink and must exclude square area.
    if expected.shape == inp.shape:
        return None
    if int(expected.shape[0]) >= int(inp.shape[0]) - 1 and int(expected.shape[1]) >= int(inp.shape[1]) - 1:
        return None
    if int(np.count_nonzero(expected != int(bg))) >= int(np.count_nonzero(inp != int(bg))):
        return None

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if pred.shape != expected.shape:
        return None
    if not bool(np.array_equal(pred, expected)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
