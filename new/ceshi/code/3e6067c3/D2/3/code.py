# Variant of 3e6067c3
# Dimension: D2 — 匹配条件：对齐与消歧规则 (match_cond)
# Change: 当多个方块具有相同内部颜色时，优先选择行/列对齐且距离最近的方块... → 按方块在网格中的位置顺序消歧

"""
Design analysis for the D2 variant

a) 具体化与行为追踪
- 画面仍然由等尺寸带框方块、底部控制色序列、以及被补画出来的彩色桥梁组成。
- 改动点只发生在“控制序列中的某个颜色在网格里对应多个方块”时。原任务会从当前方块出发，
  在同色候选里偏向“与当前方块同一行带/同一列带且距离最近”的那个；本变体改为：
  先把全部方块按位置排序（先上后下、同行先左后右），然后选该颜色里第一个尚未访问的方块。
- 一个具体小例子：起点在左上方块 A，控制序列第二个颜色是红色。网格里有两个红框：
  上方同一行较远的红框 R1，和下方同一列较近的红框 R2。原逻辑会因为 R2 更近而连向 R2；
  新逻辑会因为 R1 在全局排序里更靠前而连向 R1。视觉上第一条桥梁会明显不同。

b) 复杂度校准
- 不能把所有颜色都做成唯一，否则 D2 变化完全不可见；也不能让大量颜色重复，否则控制序列会很乱。
- 因此生成器固定制造少量“同色干扰块”：主路径颜色基本唯一，只对 1 个关键目标颜色复制一个干扰块。
- 这样既能稳定触发新规则，又不会把整个任务改造成别的类型。

c) 隐式假设审查
- 原 main() 中 `start_cands` 和后续 `cands` 的挑选都依赖“当前方块到目标方块的局部几何关系”。
  D2 改动后，这个前提不再成立，选择必须与当前位置解耦。
- 但 `draw_bridge()` 仍然要求真正被选中的连续两个方块在同一行带或同一列带，否则不会画桥。
  所以新 solver 不再靠对齐来“筛选候选”，而是把“被全局顺序选中的那个方块仍与当前方块对齐”
  作为生成器必须保证的场景约束。
- 原代码没有 visited 概念；新规则显式要求“第一个未访问过的方块”，所以 solver 必须记录访问集。

d) 放置合法性
- 生成器继续使用规则网格上的等尺寸方块，方块之间至少隔 1 个背景像素，避免边框连成一个组件。
- 主路径仍是相邻格点的自避路径，因此桥梁只占用格点之间的唯一走廊，不会交叉或重叠。
- 同色干扰块放在主路径起点的正下方；真正目标块放在起点同行。这样二者都与起点对齐，
  但只会有主路径目标块被新规则选中，桥梁也合法。

e) 视觉可观察性
- 每个样例都强制包含至少一处“同色二选一”，且更近的那个故意放在更靠后的全局位置。
- 解题者从 2-3 个样例中可以直接看出：输出不是连向最近同色块，而是连向更靠上的/更靠左的同色块。
- 排序依据只使用可见位置，不依赖颜色编号大小等不可见信息。

f) 可视化整洁度
- 只引入少量干扰块，不叠加额外标记，桥梁仍沿着规则行/列走廊绘制，画面保持原任务的整洁晶格感。
- 控制序列继续放在单独的底行，且只编码主路径，不让未使用干扰块污染控制行。

g) 颜色变化可推断性
- 桥梁颜色仍使用“当前出发方块的内部颜色”，与原任务一致。
- 变体不引入新的颜色映射，只引入“同色候选的选择规则变化”，因此颜色种类保持可读。

h) 量化退化检查思路
- 不能只验证 `main(input) == output`，因为没有真正分叉的样例也会通过。
- 生成器额外要求：同一个输入送入旧策略 solver 时，输出必须与新策略输出不同。
  这确保生成样本确实触发了 D2 的新行为，而不是退化回原任务。

i) 新 solver 的输入假设与生成器保障
- 假设 1：控制行首色对应的首个全局同色方块就是正确起点。生成器通过让起点颜色全局唯一来保证。
- 假设 2：每次按全局顺序选中的下一个同色方块都与当前方块同排或同列。生成器通过主路径构造保证。
- 假设 3：同色干扰块虽然更近，但在全局顺序里更靠后。生成器通过“目标在起点同行，干扰块在起点下方”
  和“水平间距 > 垂直间距”的间距设置同时保证。
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask


_DIGITS = [int(i) for i in range(10)]

_TEMPLATES = [
    {
        "shape": (3, 3),
        "path": [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        "decoys": [((1, 0), 1)],
    },
    {
        "shape": (3, 4),
        "path": [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],
        "decoys": [((1, 0), 1)],
    },
    {
        "shape": (3, 4),
        "path": [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
        "decoys": [((1, 0), 1)],
    },
    {
        "shape": (3, 4),
        "path": [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)],
        "decoys": [((1, 0), 1)],
    },
]


def _frame_order_key(f: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int(r0), int(c0), int(r1), int(c1)


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


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


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


def _paint_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    for r, c in _bridge_cells(fa, fb):
        out[int(r), int(c)] = int(color)


def _extract_frames(g: np.ndarray, border: int) -> list[dict]:
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

    frames.sort(key=_frame_order_key)
    for idx, frame in enumerate(frames):
        frame["order"] = int(idx)
    return frames


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, int, list[dict], dict[int, list[dict]], list[int]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    border = _most_common_nonbg(g, int(bg))
    assert not (border is None)

    frames = _extract_frames(g, int(border))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by[int(frame["inner"])].append(frame)
    for col in by:
        by[int(col)].sort(key=_frame_order_key)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert not (len(ctrl) < 2)

    return g, int(H), int(W), int(bg), int(border), frames, by, ctrl


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def main(grid: np.ndarray) -> np.ndarray:
    g, _H, _W, _bg, _border, _frames, by, ctrl = _parse_scene(grid)
    out = g.copy()

    start_color = int(ctrl[0])
    assert not (start_color not in by)

    visited: set[int] = set()
    cur = None
    for f in by[int(start_color)]:
        order = int(f["order"])
        if order not in visited:
            cur = f
            visited.add(order)
            break
    assert not (cur is None)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)

        nxt = None
        for f in by.get(int(b), []):
            order = int(f["order"])
            if order not in visited:
                nxt = f
                break
        if nxt is None:
            continue

        _paint_bridge(out, cur, nxt, int(a))
        visited.add(int(nxt["order"]))
        cur = nxt

    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g, _H, _W, _bg, _border, _frames, by, ctrl = _parse_scene(grid)
    out = g.copy()

    start_color = int(ctrl[0])
    assert not (start_color not in by)
    start_cands = by[int(start_color)]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        nxt_cands = by.get(int(nxt_color), [])
        best = None
        bestf = None
        for f in start_cands:
            pool = nxt_cands if nxt_cands else [f]
            for g2 in pool:
                sr, sc = _aligned(f, g2)
                al = 1 if (sr or sc) else 0
                r1_, c1_ = _center(f)
                r2_, c2_ = _center(g2)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    bestf = f
        cur = bestf

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if int(b) not in by:
            continue

        best = None
        bestf = None
        for f in by[int(b)]:
            sr, sc = _aligned(cur, f)
            al = 1 if (sr or sc) else 0
            r1_, c1_ = _center(cur)
            r2_, c2_ = _center(f)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f

        nxt = bestf
        _paint_bridge(out, cur, nxt, int(a))
        cur = nxt

    return out


def _transform_slot(slot: tuple[int, int], shape: tuple[int, int], mirror_h: bool) -> tuple[int, int]:
    r, c = map(int, slot)
    rows, cols = map(int, shape)
    if mirror_h:
        c = int(cols - 1 - c)
    assert 0 <= int(r) < int(rows) and 0 <= int(c) < int(cols)
    return int(r), int(c)


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]

    template = _TEMPLATES[int(np.random.randint(0, len(_TEMPLATES)))]
    shape = tuple(map(int, template["shape"]))
    mirror_h = bool(int(np.random.randint(0, 2)))
    path = [_transform_slot(slot, shape, mirror_h) for slot in template["path"]]
    decoys = [(_transform_slot(slot, shape, mirror_h), int(src_idx)) for slot, src_idx in template["decoys"]]

    if len(path) > len(inner_palette):
        return None

    t = 1
    inner = int(np.random.randint(1, 3))
    outer = int(inner + 2 * t)

    rows, cols = map(int, shape)
    if rows < 2 or cols < 2:
        return None

    row_gap = [int(np.random.randint(1, 3)) for _ in range(int(rows) - 1)]
    col_gap = [int(np.random.randint(1, 4)) for _ in range(int(cols) - 1)]

    cur0 = path[0]
    tgt0 = path[1]
    decoy0, _src_idx0 = decoys[0]
    if int(cur0[0]) != int(tgt0[0]):
        return None
    if int(cur0[1]) == int(tgt0[1]):
        return None
    if int(decoy0[1]) != int(cur0[1]):
        return None
    if int(decoy0[0]) <= int(cur0[0]):
        return None

    special_row = int(cur0[0])
    special_col = int(min(cur0[1], tgt0[1]))
    row_gap[int(special_row)] = 1
    if int(cols) - 1 <= int(special_col):
        return None
    col_gap[int(special_col)] = int(np.random.randint(3, 5))

    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))
    right_margin = int(np.random.randint(1, 3))

    r0s = [int(top_margin)]
    for k in range(int(rows) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(cols) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    all_slots = list(path)
    for slot, _src_idx in decoys:
        if slot not in all_slots:
            all_slots.append(slot)

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in all_slots:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        frame = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
        }
        frames.append(frame)
        slot_to_frame[(int(rr), int(cc))] = frame

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)

    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    min_ctrl_w = int(2 * len(path) + 1)
    W = int(max(max_c1 + 1 + right_margin, min_ctrl_w + 1))
    H = int(ctrl_r + 1 + bottom_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    path_colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=len(path), replace=False)]
    colors_by_slot: dict[tuple[int, int], int] = {}
    for slot, col in zip(path, path_colors):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(col)
    for slot, src_idx in decoys:
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(path_colors[int(src_idx)])

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(int(rr), int(cc))]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells:
            return None
        if cells & bridge_cells_all:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(path)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for slot, cc in zip(path, ctrl_cols):
        out[int(ctrl_r), int(cc)] = int(colors_by_slot[(int(slot[0]), int(slot[1]))])

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
        legacy = _legacy_main(inp)
    except Exception:
        return None

    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None
    if bool(np.array_equal(legacy, out)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
