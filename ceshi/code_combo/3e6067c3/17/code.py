# Combo variant of 3e6067c3 (index 17)
# combo_spec: D8/3+D5/4
# Name: 颜色排序处理+虚线桥梁

"""
Design analysis for the combo variant.

a) 具体化与行为追踪
- 画面仍是原题那类场景：若干同边框色的带框方块排在主区域，最底下另有一行控制像素。
- 原题做法是按控制像素从左到右读取颜色 `c0,c1,c2,...`，再依次连接
  `c0->c1, c1->c2, ...`，桥是实心矩形。
- 本融合改成两步同时生效：
  1. 先把控制行里出现的颜色按颜色值从小到大排序，再按这个顺序处理。
  2. 每条桥不再填满整个矩形走廊，而是沿桥方向“画一列/行，空一列/行”。
- 一个具体小例子：
  若控制行左到右是 `[6,2,4,3]`，原题会尝试连 `6->2->4->3`。
  本变体必须先排序成 `[2,3,4,6]`，因此实际输出画的是 `2->3->4->6`。
  假设 `2->3` 是横桥且走廊宽 3，那么原题会把 3 列都涂满；这里则只画第 1、3 列，
  第 2 列保持背景，形成明显虚线。

b) 复杂度校准
- 如果每个样例都用完全不同的一大堆颜色，那么“按颜色排序”会退化成隐藏数字规则，
  人类很难从 2-4 个样例稳定归纳。
- 为了避免这一点，生成器固定使用一个小而稳定的颜色集 `[2,3,4,6]` 作为真实路径色，
  让不同样例反复展示同一组颜色的相对先后关系；解题者可以从多例中看出真实顺序始终是
  红 -> 绿 -> 黄 -> 粉，而不是看控制像素在底部的左右位置。
- 如果桥的 gap 只有 1，虚线和实线看起来一样，所以生成器把行列间隔限制在 2..4，
  并显式过滤“虚线输出和实线输出一样”的退化样本。
- 颜色全部唯一，避免再叠加原题里“同色候选消歧”的额外难点，把注意力集中在
  “顺序不按位置 + 桥是虚线”这两个融合点上。

c) 隐式假设审查
- 原 `main()` 的控制序列假设是“底部非背景像素的空间顺序就是执行顺序”。这一前提失效，
  现在只能把它们当成一个颜色多集，再排序后处理。
- 原 `draw_bridge()` 假设桥走廊中的每个像素都要填色。D5/4 下这不成立，必须改成
  基于桥方向的交替掩码。
- 原求解器对“起点是谁”的假设是第一枚控制像素对应的颜色；这里起点应改成排序后的最小色。
- 原候选匹配里的“按对齐和距离从同色候选中选一个”对生成器样本仍然成立，但本融合
  不再依赖它制造歧义；生成器直接保证内部色全局唯一，从而不把第三条隐藏规则带进来。
- `combo_spec` 的来源说明里，D8/3 原始单变体是“后续桥避开旧桥”。这与本题显式融合名称、
  融合描述都冲突。这里以显式融合描述为准，只实现“颜色排序处理”，不额外叠加绕路规则，
  否则会平白引入第三条未在组合描述中要求的人类可见规则。

d) 放置合法性
- 生成器采用 output-first：先按真实规则画完整输出，再擦掉桥得到输入。
- 方块来自一条 4 点正交路径，任意相邻两点在晶格上相邻，因此桥矩形必然非空。
- 行列间隔、边距、控制行位置都在像素层面检查，确保对象不重叠、不越界，桥不会撞到方块
  或控制行。
- 还会检查所有实心桥走廊两两不重叠，避免“因为只擦掉虚线像素而让两条桥共享同一走廊”
  造成视觉脏乱。

e) 视觉可观察性
- 输入里控制色顺序被故意打乱；输出里桥却稳定地按同一颜色顺序出现。因为所有样例重复使用
  同一小色集，解题者能从多例直接看出“不是按位置，而是按颜色顺序处理”。
- 虚线桥在 gap>=2 时会清楚地留下背景断点，和原题实心桥的差异肉眼可见。
- 这里确实用到了 ARC 颜色编号的固定次序，但不是把黑盒数字规则藏在单例里；生成器用
  固定小颜色表反复提供跨样例证据，让该顺序变成可见、可归纳的任务线索。

f) 可视化整洁度
- 只保留 4 个真实方块，没有额外诱饵对象，主区域结构清爽。
- 桥仍然是轴对齐矩形，只是内部做规则性留白，不会出现锯齿、折线或多层覆盖。
- 控制行继续放在最底部独立一行，与主区域有 2-4 行空白，阅读层次清楚。

g) 变化的可推断性
- 颜色映射没有改：桥颜色仍等于当前起点方块的内部颜色。
- 真正变化的是“控制行只提供一组颜色，而不是提供最终先后顺序”。由于颜色集小且跨样例稳定，
  人类可以从少量例子中归纳出固定优先级。

h) 量化退化检查
- 仅验证 `main(input) == output` 不够，因为若控制行碰巧已经升序、或桥 gap 恰好全为 1，
  样本就会退化回原行为。
- 因此生成器额外要求：
  1. `order_inversions >= 1`：控制行顺序与升序至少有一对逆序。
  2. `order_diff_pixels > 0`：同一输入按“左到右 + 虚线”求出的输出必须不同。
  3. `dash_omission_total > 0`：同一排序若改画实心桥，输出也必须不同。

i) 新 solver 的输入假设与生成器保障
- 假设 1：控制行里每种颜色对应恰好一个方块。生成器用全局唯一内部色保证。
- 假设 2：排序后的相邻颜色在空间上确实能用原题那种同行带/同列带走廊连接。
  生成器按一条真实路径构造，显式保证这一点。
- 假设 3：控制行一定不是已经升序，否则顺序变体不可见。生成器显式排除。
- 假设 4：至少有一个桥走廊长度 >= 2，否则虚线变体不可见。生成器通过 gap 采样和
  `dash_omission_total` 检查保证。
"""

from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import get_bg, label_mask


TRUE_COLORS = [2, 3, 4, 6]
BASE_TEMPLATE = [(0, 0), (0, 1), (1, 1), (1, 2)]


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


def _extract_frames(g: np.ndarray) -> tuple[int, int, list[dict]]:
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

    assert frames
    return int(bg), int(border), frames


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _choose_start_frame(by: dict[int, list[dict]], seq: list[int]) -> dict:
    start_color = int(seq[0])
    assert start_color in by
    start_cands = by[start_color]
    if len(start_cands) == 1:
        return start_cands[0]

    nxt_color = int(seq[1])
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
    return bestf


def _choose_next_frame(cur: dict, target_color: int, by: dict[int, list[dict]]) -> dict | None:
    if int(target_color) not in by:
        return None
    cands = by[int(target_color)]
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
    return bestf


def _bridge_rect(fa: dict, fb: dict) -> tuple[str, int, int, int, int] | None:
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
            return None
        return ("h", int(rr0), int(rr1), int(cc0), int(cc1))

    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1):
            return None
        return ("v", int(rr0), int(rr1), int(cc0), int(cc1))

    return None


def _solid_bridge_cells(rect: tuple[str, int, int, int, int]) -> set[tuple[int, int]]:
    _orient, rr0, rr1, cc0, cc1 = rect
    return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}


def _dashed_bridge_cells(rect: tuple[str, int, int, int, int]) -> set[tuple[int, int]]:
    orient, rr0, rr1, cc0, cc1 = rect
    cells: set[tuple[int, int]] = set()
    if orient == "h":
        for c in range(int(cc0), int(cc1) + 1):
            if (int(c) - int(cc0)) % 2 != 0:
                continue
            for r in range(int(rr0), int(rr1) + 1):
                cells.add((int(r), int(c)))
        return cells

    for r in range(int(rr0), int(rr1) + 1):
        if (int(r) - int(rr0)) % 2 != 0:
            continue
        for c in range(int(cc0), int(cc1) + 1):
            cells.add((int(r), int(c)))
    return cells


def _paint_cells(out: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        out[int(r), int(c)] = int(color)


def _control_row_and_colors(g: np.ndarray, bg: int, max_frame_r: int) -> tuple[int, list[int]]:
    H, W = g.shape
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None
    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2
    return int(ctrl_r), ctrl


def _render_with_sequence(grid: np.ndarray, sequence: list[int], *, dashed: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    _bg, _border, frames = _extract_frames(g)
    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    cur = _choose_start_frame(by, sequence)
    out = g.copy()
    for i in range(int(len(sequence)) - 1):
        a = int(sequence[i])
        b = int(sequence[i + 1])
        assert int(cur["inner"]) == int(a)
        nxt = _choose_next_frame(cur, int(b), by)
        if nxt is None:
            continue
        rect = _bridge_rect(cur, nxt)
        if rect is not None:
            cells = _dashed_bridge_cells(rect) if dashed else _solid_bridge_cells(rect)
            _paint_cells(out, cells, int(a))
        cur = nxt
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, _border, frames = _extract_frames(g)
    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    _ctrl_r, ctrl = _control_row_and_colors(g, int(bg), int(max_frame_r))
    sorted_ctrl = sorted(int(c) for c in ctrl)
    return _render_with_sequence(g, sorted_ctrl, dashed=True)


def _normalize_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(int(r) for r, _c in points)
    min_c = min(int(c) for _r, c in points)
    return [(int(r - min_r), int(c - min_c)) for r, c in points]


def _transform_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = list(points)
    max_r = max(int(r) for r, _c in pts)
    max_c = max(int(c) for _r, c in pts)

    def apply_transform(p: tuple[int, int], code: int) -> tuple[int, int]:
        r, c = map(int, p)
        if code == 0:
            return int(r), int(c)
        if code == 1:
            return int(r), int(max_c - c)
        if code == 2:
            return int(max_r - r), int(c)
        if code == 3:
            return int(max_r - r), int(max_c - c)
        if code == 4:
            return int(c), int(r)
        if code == 5:
            return int(c), int(max_r - r)
        if code == 6:
            return int(max_c - c), int(r)
        return int(max_c - c), int(max_r - r)

    code = int(np.random.randint(0, 8))
    transformed = [apply_transform(p, code) for p in pts]
    return _normalize_points(transformed)


def _permutation_pool() -> list[tuple[int, ...]]:
    sorted_cols = tuple(int(c) for c in TRUE_COLORS)
    return [perm for perm in itertools.permutations(sorted_cols) if tuple(perm) != sorted_cols]


def _inversion_count(seq: list[int]) -> int:
    inv = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if int(seq[i]) > int(seq[j]):
                inv += 1
    return int(inv)


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    bg = 8
    border = 1
    t = 1
    outer = 3
    points = _transform_points(list(BASE_TEMPLATE))

    unique_rows = sorted({int(r) for r, _c in points})
    unique_cols = sorted({int(c) for _r, c in points})
    row_index = {int(r): i for i, r in enumerate(unique_rows)}
    col_index = {int(c): i for i, c in enumerate(unique_cols)}

    gap_choices = np.array([2, 2, 3, 4], dtype=int)
    row_gaps = [int(np.random.choice(gap_choices)) for _ in range(max(0, len(unique_rows) - 1))]
    col_gaps = [int(np.random.choice(gap_choices)) for _ in range(max(0, len(unique_cols) - 1))]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    row_r0s = [int(top_margin)]
    for gap in row_gaps:
        row_r0s.append(int(row_r0s[-1] + outer + int(gap)))
    col_c0s = [int(left_margin)]
    for gap in col_gaps:
        col_c0s.append(int(col_c0s[-1] + outer + int(gap)))

    frames_in_order: list[dict] = []
    max_r1 = 0
    max_c1 = 0
    for idx, (rr, cc) in enumerate(points):
        r0 = int(row_r0s[int(row_index[int(rr)])])
        c0 = int(col_c0s[int(col_index[int(cc)])])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        frame = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "t": int(t),
            "inner": int(TRUE_COLORS[int(idx)]),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
        }
        frames_in_order.append(frame)
        max_r1 = max(int(max_r1), int(r1))
        max_c1 = max(int(max_c1), int(c1))

    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    H = int(ctrl_r + 2)
    W = int(max(max_c1 + 2, 2 * len(TRUE_COLORS) + 2))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames_in_order:
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(frame["inner"]))

    dashed_cells_all: set[tuple[int, int]] = set()
    solid_cells_all: set[tuple[int, int]] = set()
    dash_omission_total = 0
    for i in range(len(frames_in_order) - 1):
        fa = frames_in_order[int(i)]
        fb = frames_in_order[int(i + 1)]
        rect = _bridge_rect(fa, fb)
        if rect is None:
            return None
        solid_cells = _solid_bridge_cells(rect)
        dashed_cells = _dashed_bridge_cells(rect)
        if not solid_cells or not dashed_cells:
            return None
        if solid_cells & solid_cells_all:
            return None
        for r, c in solid_cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        _paint_cells(out, dashed_cells, int(fa["inner"]))
        dashed_cells_all |= set(dashed_cells)
        solid_cells_all |= set(solid_cells)
        dash_omission_total += int(len(solid_cells) - len(dashed_cells))

    if int(dash_omission_total) <= 0:
        return None

    perm = list(_permutation_pool()[int(np.random.randint(0, len(_permutation_pool())))])
    ctrl_cols = [int(1 + 2 * i) for i in range(len(perm))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for color, col in zip(perm, ctrl_cols):
        out[int(ctrl_r), int(col)] = int(color)

    inp = out.copy()
    for r, c in dashed_cells_all:
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

    legacy_lr = _render_with_sequence(inp, list(perm), dashed=True)
    sorted_solid = _render_with_sequence(inp, sorted(TRUE_COLORS), dashed=False)
    order_diff_pixels = int((pred != legacy_lr).sum())
    if order_diff_pixels < 2:
        return None
    if bool(np.array_equal(pred, sorted_solid)):
        return None

    meta = {
        "order_inversions": int(_inversion_count(list(perm))),
        "order_diff_pixels": int(order_diff_pixels),
        "dash_omission_total": int(dash_omission_total),
        "ctrl_perm": tuple(int(x) for x in perm),
    }
    if int(meta["order_inversions"]) <= 0:
        return None
    return inp, meta


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            inp, _meta = result
            return inp
    raise RuntimeError("Failed to generate valid input after 500 attempts")
