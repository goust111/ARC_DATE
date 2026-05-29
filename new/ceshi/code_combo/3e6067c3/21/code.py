# Combo variant of 3e6067c3 (index 21)
# combo_spec: D10/1+D1/0
# Name: 边框色定桥梁+边框色匹配

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import colors_present, get_bg, in_bounds, label_mask, neighbors4


# Design analysis
# a) 具体化与行为追踪
# - 原任务里，所有方块共享同一种边框色；控制行列出的是内部色；桥梁也用当前内部色着色。
# - 本融合变体把整条推理链改成“边框色驱动”：
#   1. 先看控制行里的颜色；
#   2. 用该颜色去匹配某个方块的边框，而不是内部；
#   3. 找到下一个控制边框色对应的方块；
#   4. 在两者间补桥，桥色也取起点方块的边框色。
# - 具体小例子：假设 3 个方块的边框分别是 [2,5,7]，内部都填 6，控制行是 [2,5,7]。
#   原任务会先失败，因为它会去找 inner=2 的方块；融合后会依次找到 border=2、border=5、border=7
#   的方块，并把两段桥涂成 2 和 5。内部色 6 只是一块视觉填充，不参与匹配，也不决定桥色。
#
# b) 复杂度校准
# - 若边框不唯一，按边框匹配会重新出现歧义；因此生成器强制每个方块边框色唯一。
# - 若内部色也各不相同，人类可能误以为内部仍有语义；因此生成器让所有方块共享一个装饰性内部色。
# - 若边框色和内部色重叠，桥色与内部色可能偶然相同，削弱 D10 的可见性；因此两组调色板严格分离。
# - 方块数量控制在 4..7，既能让控制序列和桥梁行为重复出现，又不至于耗尽 10 色调色板。
#
# c) 隐式假设审查
# - 原 `main()` 假设“最常见非背景色 == 全局边框色”；融合后不成立，必须枚举每种非背景色检测环形框。
# - 原 `by[f["inner"]]` 把控制颜色映射到内部色；融合后必须改成 `by[f["border"]]`。
# - 原 `assert cur["inner"] == a` 和 `draw_bridge(..., color=a)` 都隐含“控制色 == 内部色 == 桥色”；
#   融合后应改为“控制色 == 边框色 == 桥色”，内部色彻底脱钩。
# - 几何部分如 `aligned()`、桥宽取 inner bbox 厚度、按最近且同行/同列优先消歧，这些仅依赖空间结构，
#   在融合后仍然成立，可以保留。
#
# d) 放置合法性
# - 仍采用 output-first：先放方块和目标桥梁，再擦掉桥梁得到输入。
# - 方块沿规则晶格放置，行列间至少隔 1 个背景像素；路径是 4 邻自避免路径，所以相邻方块必定同行或同列。
# - `_bridge_cells()` 只返回两框之间的纯背景矩形，生成器逐段检查不越界、不重叠、不覆盖已有对象。
#
# e) 视觉可观察性
# - 输入里，控制行的颜色只会出现在各方块边框上，不会出现在内部。
# - 输出里，新出现的桥梁颜色也只会复用边框色，因此观察者能从 2-3 组样例归纳出“边框色主导全部动作”。
# - 因为内部全部是同一装饰色，任何“按内部匹配”假说都会立即被样例反证。
#
# f) 可视化整洁度
# - 画面仍只有三类元素：带框方块、桥梁、控制行。
# - 内部统一纯色，边框唯一，桥梁是纯矩形填充，避免在同一区域叠加多套编码。
# - 控制行固定在底部单独一行，和主体区域分离。
#
# g) 变化的可推断性
# - 颜色角色被压缩为三类：背景、唯一边框色、统一装饰内部色。
# - 没有额外的颜色映射表，也不依赖颜色数值大小；人类只需观察“控制色能在哪一层找到”即可推断规则。
# - 融合的关键变量是“边框色是否参与匹配与着色”；生成器保证该变量一旦去掉，样例就无法按原题解释。
#
# 9) 新 solver 的隐式假设与 generator 保证
# - 假设每个目标对象都是单色矩形环包围单色实心矩形；`_draw_frame()` 保证这一点。
# - 假设控制行至少含 2 个非背景像素且位于所有方块下方；generator 按固定奇数列生成控制行并保留底边空行。
# - 假设每个控制色都能对应到唯一边框色方块；generator 为每个方块分配唯一边框并直接用路径边框色写控制行。
# - 假设内部色不需要参与任何判别；generator 统一内部色，并验证该色不出现在控制/边框集合中。


def _detect_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for border in colors_present(g, exclude={bg}):
        lab, n = label_mask(g == int(border), connectivity=4)
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

            comp_area = int(pts.shape[0])
            found = None
            for tt in range(1, int(min(h, w) // 2) + 1):
                inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
                if inner.size == 0:
                    continue
                if bool((inner == int(border)).any()):
                    continue

                inner_vals = set(map(int, inner.flatten()))
                if len(inner_vals) != 1:
                    continue
                inner_col = int(next(iter(inner_vals)))
                if int(inner_col) == int(bg):
                    continue

                ring_ok = True
                if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                    ring_ok = False
                if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                    ring_ok = False
                if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                    ring_ok = False
                if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                    ring_ok = False
                if not ring_ok:
                    continue

                inner_h = int(inner.shape[0])
                inner_w = int(inner.shape[1])
                ring_area = int(h * w - inner_h * inner_w)
                if int(comp_area) != int(ring_area):
                    continue

                found = {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "t": int(tt),
                    "border": int(border),
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


def _choose_start(start_cands: list[dict], nxt_cands: list[dict]) -> dict:
    if len(start_cands) == 1:
        return start_cands[0]

    best = None
    bestf = None
    for frame in start_cands:
        for g2 in (nxt_cands if nxt_cands else [frame]):
            sr, sc = _aligned(frame, g2)
            aligned_score = 1 if (sr or sc) else 0
            r1, c1 = _center(frame)
            r2, c2 = _center(g2)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned_score), -int(dist))
            if best is None or cand > best:
                best = cand
                bestf = frame
    assert bestf is not None
    return bestf


def _pick_next(cur: dict, cands: list[dict]) -> dict:
    best = None
    bestf = None
    for frame in cands:
        sr, sc = _aligned(cur, frame)
        aligned_score = 1 if (sr or sc) else 0
        r1, c1 = _center(cur)
        r2, c2 = _center(frame)
        dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        cand = (int(aligned_score), -int(dist))
        if best is None or cand > best:
            best = cand
            bestf = frame
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


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares using a control sequence of border colors.

    Fusion rule:
    - Detect framed squares by their border rings, allowing each square to have its own border color.
    - Read control pixels below the lattice as border colors.
    - For each consecutive pair, connect the current square to the next matched-border square.
    - Paint each bridge with the current square's border color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    frames = _detect_frames(g, int(bg))
    assert frames

    by_border: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_border[int(frame["border"])].append(frame)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)

    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    start_color = int(ctrl[0])
    assert start_color in by_border
    cur = _choose_start(by_border[start_color], by_border.get(int(ctrl[1]), []))

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict) -> None:
        for r, c in _bridge_cells(fa, fb):
            if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
                out[int(r), int(c)] = int(fa["border"])

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["border"]) == int(a)
        assert int(b) in by_border
        nxt = _pick_next(cur, by_border[b])
        draw_bridge(cur, nxt)
        cur = nxt

    return out


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
    max_by_palette = int(len(avail) - 1)
    upper = int(min(7, max_frames, max_by_palette))
    if upper < 4:
        return None
    n_frames = int(np.random.randint(4, int(upper) + 1))

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
        if len({int(r) for r, _c in p}) < 2 or len({int(c) for _r, c in p}) < 2:
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
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
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
        for r, c in cells:
            out[int(r), int(c)] = int(fa["border"])
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    ctrl_colors = [int(border_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if len(ctrl_colors) < 2 or int(ctrl_cols[-1]) >= int(W):
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
