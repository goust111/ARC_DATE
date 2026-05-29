# Combo variant of 3e6067c3 (index 12)
# combo_spec: D5/0+D7/1
# Name: 终点色桥梁+仅输出桥梁

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask

# Design analysis for combo D5/0 + D7/1
# ======================================
# a) 具体化与行为追踪
#    原题先在完整场景里保留方块、控制行与背景，再把每一段桥涂成“起点方块色”。
#    本融合版同时改两个环节：
#    1. 桥仍画在同一条对齐通道里，但 A->B 这段改用 B 的内色。
#    2. 输出不再保留方块或控制行，只保留所有桥梁像素，并裁成桥梁外接矩形。
#    例子：若控制序列是 [2,5,7]，路径是 2 方块 -> 5 方块 -> 7 方块，
#    原题会输出完整场景，且两段桥颜色为 [2,5]；
#    本题会输出一个只含桥梁的裁剪拼图，两段桥颜色变为 [5,7]。
#    解题者必须先恢复原路径，才能“想象”出只剩桥的终态。
#
# b) 复杂度校准
#    不能因为只输出桥而把匹配逻辑简化掉；仍保留：
#    - 带框方块检测
#    - control row 读取顺序
#    - 同 row-band / col-band 的对齐约束
#    - 同色候选里优先对齐、再取最近
#    为了让 D5 的颜色归属变化在裁剪后仍明显可见，生成器继续要求路径长度 >= 4，
#    并对同一行/列施加颜色唯一性约束，使相邻路径节点颜色必异。
#
# c) 隐式假设审查
#    原 main() 里与本融合最直接冲突的旧假设有两条：
#    - `out = g.copy()` 假设方块和控制行应该留在输出中；D7/1 使其失效。
#    - `draw_bridge(..., a)` 假设桥段颜色来自起点；D5/0 使其失效。
#    其他关键前提仍成立：
#    - 最常见非背景色仍可作为边框色；
#    - 方块仍可由边框连通域恢复；
#    - 控制行仍位于所有方块下方；
#    - 候选匹配仍仅依赖几何对齐与距离，而不依赖桥色。
#    新 solver 的附加前提是：终点颜色能由 `nxt["inner"]` 唯一读取；
#    generator 通过把控制序列直接编码为路径方块内色并保留这些方块来保证此点。
#
# d) 放置合法性
#    generator 仍使用 output-first：
#    - 先布置完整方块场景；
#    - 再在相邻路径方块之间计算桥梁像素；
#    - 最后擦掉桥得到输入。
#    self-avoiding lattice path 保证每一对相邻节点共享一行或一列；
#    `_bridge_cells()` 保证桥只落在两框之间的空白通道里，不越界、不压框、不互相重叠。
#
# e) 视觉可观察性
#    人在 2-4 个样例中会看到：
#    - 输入仍是“方块 + 底部控制色列”；
#    - 输出突然变成更小的桥梁裁剪图；
#    - 每段桥的颜色总与它所通向的方块一致，而不是来自离开的方块。
#    这两层变化叠加后，和原题的视觉差异足够直接。
#
# f) 可视化整洁度
#    输出只含矩形桥带和必要的背景孔洞，不再叠加方块轮廓、控制行或整张背景。
#    因而虽然推理更难，成图反而更干净，不会出现多层元素压在同一区域的问题。
#
# g) 变化的可推断性
#    融合版没有引入新的隐藏颜色映射，只复用“终点方块内色”。
#    颜色变化与桥梁方向强绑定；同时裁剪策略又把注意力集中在桥本身，便于归纳。
#
# h) 防退化指标
#    对同一输入，比对“终点着色桥梁输出”和“原题起点着色桥梁输出”。
#    若两者完全相同，则 D5 变化退化了；生成器显式拒绝这种样本。
#    同时要求裁剪输出比输入明显更小，避免 D7 变化在视觉上不成立。
#
# i) main() 新假设逐项审查
#    - frame detection 循环假设每个有效方块是单一边框厚度包住单一内色：generator 保证。
#    - start frame 选择在首色重复时依赖“与第二色候选的对齐+最近”打破平局：沿用原题并由构造支持。
#    - `_choose_best(cur, by[b])` 假设 path 下一站在候选中几何上最优：generator 让真实路径满足这一原题机制。
#    - `_crop_to_nonbg` 假设至少存在一段桥：generator 让路径长度始终 >= 4，因此桥段数 >= 3。


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
        r1, c1 = _center(cur)
        r2, c2 = _center(f)
        d = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        cand = (int(al), -int(d))
        if best is None or cand > best:
            best = cand
            bestf = f
    assert bestf is not None
    return bestf


def _detect_frames(g: np.ndarray, border: int) -> list[dict]:
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
            vals = set(map(int, inner.flatten()))
            if len(vals) != 1:
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
            inner_col = int(next(iter(vals)))
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
    return frames


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
    Recover the control-path bridges, color each segment by its destination square,
    then return only the bridge pixels cropped to their tight bounding box.
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

    frames = _detect_frames(g, int(border))
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
                r1, c1 = _center(f)
                r2, c2 = _center(g2)
                d = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
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
                out[int(r), int(c)] = int(b)
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
    source_bridge_canvas = np.full((int(H), int(W)), int(bg), dtype=int)
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
        dst_paint = int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        src_paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            bridge_canvas[int(r), int(c)] = int(dst_paint)
            source_bridge_canvas[int(r), int(c)] = int(src_paint)
        bridge_cells_all |= set(cells)
    if not ok or not bridge_cells_all:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cc in zip(ctrl_colors, ctrl_cols):
        full_scene[int(ctrl_r), int(cc)] = int(col)

    inp = full_scene.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    expected = _crop_to_nonbg(bridge_canvas, int(bg))
    source_expected = _crop_to_nonbg(source_bridge_canvas, int(bg))

    if expected.shape == inp.shape:
        return None
    if int(expected.shape[0]) >= int(inp.shape[0]) - 1 and int(expected.shape[1]) >= int(inp.shape[1]) - 1:
        return None
    if int(np.count_nonzero(expected != int(bg))) >= int(np.count_nonzero(inp != int(bg))):
        return None
    if bool(np.array_equal(expected, source_expected)):
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
