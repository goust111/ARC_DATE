# Combo variant of 3e6067c3 (index 22)
# combo_spec: D1/0+D5/0+D7/0
# Name: 边框匹配+终点色桥梁+紧凑裁剪

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

"""
Design analysis for combo D1/0 + D5/0 + D7/0.

a) 具体化与行为追踪
   - 原任务: 所有方块共享同一边框色, 控制行颜色匹配内部填充色, 桥梁也用起点内部色,
     最后返回整张画布.
   - 融合后: 每个方块改由唯一边框色标识, 控制行按边框色给出访问顺序; 从 A 连到 B 时,
     桥梁不再继承 A, 而是染成终点 B 的边框色; 最终只保留方块与桥梁所在的最小矩形.
   - 小例子: 若三个方块共享同一个内部色 6, 但边框依次是 2,5,7, 控制行为 [2,5,7].
     原逻辑会去匹配 inner=2/5/7, 这里却根本找不到; 融合逻辑应当匹配 border=2/5/7,
     并把 2->5 的桥段画成 5, 5->7 的桥段画成 7, 最后裁掉控制行和外边距.

b) 复杂度校准
   - 为了让 D1 肉眼可见, generator 让所有方块共享同一种内部色, 避免解题者仍靠内部色工作.
   - 为了让 D5 肉眼可见, generator 给路径上每个方块分配不同边框色, 所以任一桥段的起点色
     与终点色都不同, 终点着色与原始起点着色必然产生不同输出.
   - 为了让 D7 稳定可见, generator 强制保留正的上/左/右边距和控制行下方空白.

c) 隐式假设审查
   - 原 solver 假设 “最常见非背景色就是全局边框色”, D1 使其失效; 现在必须枚举所有非背景色
     并逐色找矩形环组件.
   - 原 `by[f["inner"]]` 与 `assert cur["inner"] == a` 绑定内部色匹配; 融合后都改为 `border`.
   - D5 把 `draw_bridge(cur, nxt, a)` 的颜色归属改成终点色. 在本融合里, 终点色必须取
     `nxt["border"]` / 控制序列中的 `b`, 而不是旧任务的内部色.
   - D7 改写最终输出形状, 但桥梁几何仍需在原坐标系绘制, 所以裁剪只能放在全部桥梁画完之后.
   - 新增假设: 控制行颜色必须一一对应某些方块边框, 且所有有意义输出像素都在控制行上方.
     generator 显式保证这两点.

d) 放置合法性
   - 仍使用 output-first: 先放方块, 再验证并记录每一段桥梁像素, 最后擦桥得到输入.
   - 路径是晶格上的 4 邻自避免路径, 所以相邻方块必共享一条行带或列带.
   - 行列间距至少 1, 桥段集合彼此不重叠, 也不能覆盖任何方块或控制行.

e) 视觉可观察性
   - 观察样例时, 控制行颜色只会出现在边框上, 不会出现在内部; 这会把注意力从中心填充移到外框.
   - 桥梁颜色总与“桥段终点”的边框一致, 而不是起点一致.
   - 输出中控制行完全消失, 只剩一个裁紧后的连接图案.

f) 可视化整洁度
   - 网格仍只包含背景、方块、桥梁、控制行四类元素; 输出阶段又移除了控制行和多余留白.
   - 单一内部色 + 唯一边框色让对象角色更清晰, 不会叠出难辨的多层标记.

g) 变化的可推断性
   - 颜色角色很少: 一个背景色, 一个重复内部色, 若干唯一边框色.
   - 颜色变化不是任意映射: 控制行直接复用边框色, 桥梁直接复用终点边框色.
   - generator 额外拒绝“终点着色裁剪图 == 起点着色裁剪图”的样本, 防止 D5 退化.

h) 融合行为量化指标
   - D1 指标: `len({frame.inner}) == 1` 且 `len({frame.border}) == n_frames`, 说明控制匹配只能看边框.
   - D5 指标: `source_crop != destination_crop`, 说明终点着色在裁剪后的最终输出中仍然可见.
   - D7 指标: `output.shape` 在高和宽上都严格小于 `input.shape`.

i) main() 的新假设与 generator 保证
   - `_detect_frames()` 假设每个有效对象都是单色矩形环包围单色实心矩形; `_draw_frame()` 保证.
   - `_aligned()` 假设要连接的相邻目标共享同一 inner band 的行带或列带; 路径采样保证.
   - `_tight_crop_scene()` 假设控制行上方至少存在一个非背景像素; 既然至少有一个方块, 该条件恒成立.
   - 起点/终点选择依赖边框色可区分; generator 为路径上的每个方块分配唯一边框色.
"""


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


def _tight_crop_scene(g: np.ndarray, bg: int, ctrl_r: int) -> np.ndarray:
    fg = np.argwhere(g[: int(ctrl_r), :] != int(bg))
    assert fg.size > 0
    r0, c0 = fg.min(axis=0)
    r1, c1 = fg.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _pick_frame(cur: dict, cands: list[dict]) -> dict:
    best = None
    bestf = None
    r1_, c1_ = _center(cur)
    for f in cands:
        sr, sc = _aligned(cur, f)
        al = 1 if (sr or sc) else 0
        r2_, c2_ = _center(f)
        d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
        cand = (int(al), -int(d))
        if best is None or cand > best:
            best = cand
            bestf = f
    assert bestf is not None
    return bestf


def _choose_start(by: dict[int, list[dict]], ctrl: list[int]) -> dict:
    start_color = int(ctrl[0])
    assert int(start_color) in by
    start_cands = by[int(start_color)]
    if len(start_cands) == 1:
        return start_cands[0]

    nxt_color = int(ctrl[1])
    nxt_cands = by.get(int(nxt_color), [])
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
                if 0 <= int(r) < int(h) and 0 <= int(c) < int(w):
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
                if 0 <= int(r) < int(h) and 0 <= int(c) < int(w):
                    out[int(r), int(c)] = int(color)


def _solve_full(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    g = np.array(grid, dtype=int)
    h, w = map(int, g.shape)
    bg = int(get_bg(g))

    frames = _detect_frames(g, int(bg))
    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by[int(frame["border"])].append(frame)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_r = None
    for r in range(int(h - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(w)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    cur = _choose_start(by, ctrl)
    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["border"]) == int(a)
        assert int(b) in by
        nxt = _pick_frame(cur, by[int(b)])
        _draw_bridge(out, cur, nxt, int(b))
        cur = nxt

    return out, int(bg), int(ctrl_r)


def main(grid: np.ndarray) -> np.ndarray:
    full, bg, ctrl_r = _solve_full(grid)
    return _tight_crop_scene(full, int(bg), int(ctrl_r))


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
    border_by_slot: dict[tuple[int, int], int] = {}
    border_set = set(map(int, border_colors))
    inner_choices = [int(d) for d in avail if int(d) not in border_set]
    if not inner_choices:
        return None
    inner_col = int(np.random.choice(np.array(inner_choices, dtype=int)))

    slot_to_frame: dict[tuple[int, int], dict] = {}
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
    bottom_margin = int(np.random.randint(1, 3))
    right_margin = int(np.random.randint(1, 5))
    height = int(ctrl_r + 1 + bottom_margin)
    width = int(max_c1 + 1 + right_margin)
    if not (7 <= int(height) <= 30 and 7 <= int(width) <= 30):
        return None

    base_scene = np.full((int(height), int(width)), int(bg), dtype=int)
    for frame in frames:
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            base_scene,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(frame["border"]),
            inner_col=int(inner_col),
        )

    segments: list[tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]]] = []
    bridge_cells_all: set[tuple[int, int]] = set()
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells or cells & bridge_cells_all:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(height) and 0 <= int(c) < int(width)):
                return None
            if int(base_scene[int(r), int(c)]) != int(bg):
                return None
        bridge_cells_all |= set(cells)
        segments.append((cells, (int(a_slot[0]), int(a_slot[1])), (int(b_slot[0]), int(b_slot[1]))))

    ctrl_colors = [int(border_by_slot[(int(r), int(c))]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(width):
        return None

    full_out = base_scene.copy()
    source_out = base_scene.copy()
    for cells, a_slot, b_slot in segments:
        src_color = int(border_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        dst_color = int(border_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        if int(src_color) == int(dst_color):
            return None
        for r, c in cells:
            full_out[int(r), int(c)] = int(dst_color)
            source_out[int(r), int(c)] = int(src_color)

    for col, cc in zip(ctrl_colors, ctrl_cols):
        full_out[int(ctrl_r), int(cc)] = int(col)
        source_out[int(ctrl_r), int(cc)] = int(col)

    expected = _tight_crop_scene(full_out, int(bg), int(ctrl_r))
    source_expected = _tight_crop_scene(source_out, int(bg), int(ctrl_r))
    if not (expected.shape[0] < full_out.shape[0] and expected.shape[1] < full_out.shape[1]):
        return None
    if bool(np.array_equal(expected, source_expected)):
        return None

    inp = full_out.copy()
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
