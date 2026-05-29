# Variant of 3e6067c3
# Dimension: D8 — 多步处理：控制序列处理顺序 (multi_step)
# Change: 从左到右顺序处理控制像素对，每对连续像素定义一段桥梁... → 迭代处理：后续桥梁避开先前桥梁

from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np

from arc_utils import get_bg, label_mask

# Design analysis
# a) Concrete behavior:
# - 场景仍然是若干同边框色的带框方块，底部控制行仍然按从左到右给出访问顺序。
# - 为了让“绕开旧桥”在 2-3 个样例里肉眼可见，生成器把方块内部固定为 1x1，
#   这样每段桥梁本质上是单像素宽的横线/竖线，后续桥梁遇到旧桥时会变成明显的
#   “狗腿形”折线：先离开原通道，绕到阻挡桥的外侧 1 像素，再回到目标方块。
# - 小例子：若先画一条竖桥，再画一条横桥，而横桥的直线路径会穿过那条旧竖桥，
#   原逻辑会直接覆盖交叉点；变体逻辑会把横桥整体在交叉处上移或下移 1 像素绕过去。
# b) Complexity calibration:
# - 只改 D8：控制序列仍定义同一串方块，匹配规则仍按颜色与共行/共列对齐完成。
# - 为了把难点集中在“顺序依赖”，颜色改为全局唯一，避免颜色重复带来的额外歧义。
# - 生成器强制每个样本至少出现一次直线路径与旧桥冲突，从而保证变体可观察。
# c) Original main() assumptions that break here:
# - 原 draw_bridge 无条件覆盖输出网格，隐含前提是桥梁之间永不相交。
# - 原 generator 还显式拒绝 `cells & bridge_cells_all`，这会把 D8 现象完全过滤掉。
# - 变体里这两个前提都不成立，所以 solver 必须显式检查先前已画桥梁，再决定是否绕行。
# d) Placement legality:
# - 生成器使用少量固定折线路径模板，并在像素层面顺序模拟绘制；如果直连和两侧
#   单像素偏移通道都不可用，就直接丢弃该样本。
# - 只有背景格允许成为桥梁；边框、内部像素和控制行都不会被桥梁覆盖。
# e) Visual observability:
# - 解题者能直接看到：前几段桥梁是直的，后几段在撞上旧桥时会出现偏移 1 像素的折线。
# - 颜色值本身不参与任何隐藏排序；顺序信息仍只来自底部可见控制像素序列。
# f) Visual cleanliness:
# - 只允许单像素桥梁和单像素偏移，避免粗桥叠在一起形成难辨认的大色块。
# - 模板是稀疏的正交折线，框、桥、控制行三类元素分区清楚。
# g) Color inference:
# - 桥梁颜色仍等于当前控制像素/起点方块的内部颜色，沿用原题的结构线索。
# h) Variant metric used during validation:
# - 统计 `reroute_count`：需要绕行的桥梁段数。样本必须满足 `reroute_count >= 1`。
# - 另与“老规则直连覆盖版”比较，只要出现绕行，输出就必须与老规则不同。
# i) New assumptions introduced by this main():
# - 生成输入默认所有框的 inner_bbox 为 1x1，因此“绕到阻挡桥外侧 1 像素”的几何是明确的。
# - 若后续桥梁需要绕行，则其原通道两侧至少有一侧是纯背景；生成器通过顺序模拟保证。


def _extract_frames(g: np.ndarray) -> tuple[int, int, list[dict]]:
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


def _segment_spec(fa: dict, fb: dict) -> dict | None:
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]
    same_row, same_col = _aligned(fa, fb)
    if same_row:
        row = int((int(ia[0]) + int(ia[2])) // 2)
        if int(fa["bbox"][3]) < int(fb["bbox"][1]):
            c0 = int(fa["bbox"][3] + 1)
            c1 = int(fb["bbox"][1] - 1)
        else:
            c0 = int(fb["bbox"][3] + 1)
            c1 = int(fa["bbox"][1] - 1)
        return {"kind": "h", "row": int(row), "c0": int(c0), "c1": int(c1)}
    if same_col:
        col = int((int(ia[1]) + int(ia[3])) // 2)
        if int(fa["bbox"][2]) < int(fb["bbox"][0]):
            r0 = int(fa["bbox"][2] + 1)
            r1 = int(fb["bbox"][0] - 1)
        else:
            r0 = int(fb["bbox"][2] + 1)
            r1 = int(fa["bbox"][0] - 1)
        return {"kind": "v", "col": int(col), "r0": int(r0), "r1": int(r1)}
    return None


def _direct_bridge_cells(spec: dict) -> list[tuple[int, int]]:
    if spec["kind"] == "h":
        if int(spec["c0"]) > int(spec["c1"]):
            return []
        return [(int(spec["row"]), int(c)) for c in range(int(spec["c0"]), int(spec["c1"]) + 1)]
    if int(spec["r0"]) > int(spec["r1"]):
        return []
    return [(int(r), int(spec["col"])) for r in range(int(spec["r0"]), int(spec["r1"]) + 1)]


def _unique_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    return out


def _detour_bridge_cells(spec: dict, detour_coord: int) -> list[tuple[int, int]]:
    if spec["kind"] == "h":
        row = int(spec["row"])
        detour_row = int(detour_coord)
        c0 = int(spec["c0"])
        c1 = int(spec["c1"])
        cells = [(int(detour_row), int(c)) for c in range(int(c0), int(c1) + 1)]
        cells += [(int(r), int(c0)) for r in range(min(int(row), int(detour_row)), max(int(row), int(detour_row)) + 1)]
        cells += [(int(r), int(c1)) for r in range(min(int(row), int(detour_row)), max(int(row), int(detour_row)) + 1)]
        return _unique_cells(cells)
    col = int(spec["col"])
    detour_col = int(detour_coord)
    r0 = int(spec["r0"])
    r1 = int(spec["r1"])
    cells = [(int(r), int(detour_col)) for r in range(int(r0), int(r1) + 1)]
    cells += [(int(r0), int(c)) for c in range(min(int(col), int(detour_col)), max(int(col), int(detour_col)) + 1)]
    cells += [(int(r1), int(c)) for c in range(min(int(col), int(detour_col)), max(int(col), int(detour_col)) + 1)]
    return _unique_cells(cells)


def _cells_are_bg(cells: list[tuple[int, int]], out: np.ndarray, bg: int) -> bool:
    H, W = out.shape
    for r, c in cells:
        if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
            return False
        if int(out[int(r), int(c)]) != int(bg):
            return False
    return True


def _shortest_bg_path(spec: dict, out: np.ndarray, bg: int) -> list[tuple[int, int]] | None:
    direct = _direct_bridge_cells(spec)
    if not direct:
        return None

    start = direct[0]
    end = direct[-1]
    H, W = out.shape
    if spec["kind"] == "h":
        c_min = int(min(spec["c0"], spec["c1"]))
        c_max = int(max(spec["c0"], spec["c1"]))
        r_min = 0
        r_max = int(H - 1)
        dirs = [(0, 1), (-1, 0), (1, 0), (0, -1)]
    else:
        r_min = int(min(spec["r0"], spec["r1"]))
        r_max = int(max(spec["r0"], spec["r1"]))
        c_min = 0
        c_max = int(W - 1)
        dirs = [(1, 0), (0, -1), (0, 1), (-1, 0)]

    q: deque[tuple[int, int]] = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for dr, dc in dirs:
            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            nxt = (int(nr), int(nc))
            if nxt in prev:
                continue
            if not (int(r_min) <= int(nr) <= int(r_max) and int(c_min) <= int(nc) <= int(c_max)):
                continue
            if not (0 <= int(nr) < int(H) and 0 <= int(nc) < int(W)):
                continue
            if nxt != end and int(out[int(nr), int(nc)]) != int(bg):
                continue
            prev[nxt] = cur
            q.append(nxt)

    if end not in prev:
        return None

    path: list[tuple[int, int]] = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _route_bridge(fa: dict, fb: dict, out: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], bool] | None:
    spec = _segment_spec(fa, fb)
    if spec is None:
        return None
    direct = _direct_bridge_cells(spec)
    if not direct:
        return None
    if _cells_are_bg(direct, out, int(bg)):
        return direct, False

    detour = _shortest_bg_path(spec, out, int(bg))
    if detour is None:
        return None
    if not _cells_are_bg(detour, out, int(bg)):
        return None
    return detour, True


def _paint_cells(out: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        out[int(r), int(c)] = int(color)


def _choose_start_frame(by: dict[int, list[dict]], ctrl: list[int]) -> dict:
    start_color = int(ctrl[0])
    assert start_color in by
    start_cands = by[start_color]
    if len(start_cands) == 1:
        return start_cands[0]

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


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg, _border, frames = _extract_frames(g)

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

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(g.shape[1]) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    cur = _choose_start_frame(by, ctrl)
    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        nxt = _choose_next_frame(cur, int(b), by)
        if nxt is None:
            continue
        routed = _route_bridge(cur, nxt, out, int(bg))
        assert routed is not None
        cells, _rerouted = routed
        _paint_cells(out, cells, int(a))
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


_TEMPLATES: list[list[tuple[int, int]]] = [
    [(0, 1), (4, 1), (4, 4), (1, 4), (1, 0)],
]


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    digits = list(range(10))
    bg = int(np.random.choice(np.array(digits, dtype=int)))
    border_choices = np.array([d for d in digits if int(d) != int(bg)], dtype=int)
    border = int(np.random.choice(border_choices))
    inner_palette = [int(d) for d in digits if int(d) not in (int(bg), int(border))]

    template = _TEMPLATES[int(np.random.randint(0, len(_TEMPLATES)))]
    points = _transform_points(template)
    n_frames = len(points)
    if len(inner_palette) < int(n_frames):
        return None

    palette = list(map(int, np.random.permutation(np.array(inner_palette, dtype=int))[: int(n_frames)]))
    outer = 3
    t = 1

    unique_rows = sorted({int(r) for r, _c in points})
    unique_cols = sorted({int(c) for _r, c in points})
    row_index = {int(r): i for i, r in enumerate(unique_rows)}
    col_index = {int(c): i for i, c in enumerate(unique_cols)}

    row_gaps = [int(np.random.randint(2, 5)) for _ in range(max(0, len(unique_rows) - 1))]
    col_gaps = [int(np.random.randint(2, 5)) for _ in range(max(0, len(unique_cols) - 1))]
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
            "inner": int(palette[int(idx)]),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "order_idx": int(idx),
        }
        frames_in_order.append(frame)
        max_r1 = max(int(max_r1), int(r1))
        max_c1 = max(int(max_c1), int(c1))

    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    H = int(ctrl_r + 2)
    W = int(max(max_c1 + 2, 2 * n_frames + 2))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames_in_order:
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(frame["inner"]))

    bridge_cells_all: set[tuple[int, int]] = set()
    reroute_count = 0
    for i in range(int(n_frames) - 1):
        fa = frames_in_order[int(i)]
        fb = frames_in_order[int(i + 1)]
        routed = _route_bridge(fa, fb, out, int(bg))
        if routed is None:
            return None
        cells, rerouted = routed
        if not cells:
            return None
        if not _cells_are_bg(cells, out, int(bg)):
            return None
        _paint_cells(out, cells, int(fa["inner"]))
        bridge_cells_all |= set(cells)
        if rerouted:
            reroute_count += 1

    if int(reroute_count) < 1:
        return None

    ctrl_colors = [int(frame["inner"]) for frame in frames_in_order]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for color, col in zip(ctrl_colors, ctrl_cols):
        out[int(ctrl_r), int(col)] = int(color)

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

    return inp, {"reroute_count": int(reroute_count), "n_frames": int(n_frames)}


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            inp, _meta = result
            return inp
    raise RuntimeError("Failed to generate valid input after 500 attempts")
