# Variant of 3e6067c3
# Dimension: D2 — 匹配条件：对齐与消歧规则 (match_cond)
# Change: 当多个方块具有相同内部颜色时，优先选择行/列对齐且距离最近的方块... → 严格行列交替对齐

from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from arc_utils import get_bg, label_mask

# Design analysis for the D2 variant
# a) Concrete behavior:
#    原题仍然是“框方块 + 底部控制色序列 + 在方块间补桥”。变体只改“下一个同色方块怎么选”：
#    第 1,3,5... 步必须横向连接，同色候选里只能选与当前方块同一行带的目标；
#    第 2,4,6... 步必须纵向连接，同色候选里只能选同一列带的目标。
#    我把它具体化为固定“奇数步横向、偶数步纵向”。generator 会给每个目标色再放一个
#    “更近但方向错误”的同色诱饵，所以旧逻辑会被距离骗走，而新逻辑必须跟踪步号奇偶。
#    例子：当前在红框，下一控制色是绿。输入里既有右侧同排的绿框，也有下方更近的绿框。
#    原逻辑会选“任意行/列对齐且最近”的下方绿框；变体要求第 1 步必须横向，所以必须选右侧绿框。
# b) Complexity calibration:
#    不把控制序列做成全重复色，否则路径搜索会因自由度过高而变脏；改为“路径色唯一、但每个目标色
#    多一个同色诱饵”。这样每步都真有同色消歧，且视觉上能看到“同色二选一”。
#    每个样例至少 3 步，因此会出现横/竖/横 或 横/竖/横/竖 的交替证据。
# c) Hidden assumptions removed from the old solver:
#    - 旧版 `al = 1 if (sr or sc) else 0` 依赖“任意方向都可”，在变体里失效。
#    - 旧版起点选择只看“和第二色是否对齐 + 最近”，没有步号概念；新逻辑必须按第 1 步横向检查。
#    - 旧 generator 通过“同一行同一列颜色唯一”来避免歧义，这会把 D2 变化彻底抹掉，所以必须取消。
# d) Placement legality:
#    generator 先在离散槽位网格上造一条“横/竖交替”的访问路径，再为每一步找一个正交方向诱饵槽位。
#    之后映射到像素网格，先画完整输出、再擦桥得到输入，最后回跑 `main()` 精确比对。
#    任何越界、桥与框重叠、桥之间相交、border/bg 推断失真，都会整样本丢弃。
# e) Visual observability:
#    人看到样例时，会反复遇到“目标色有两个，一个在该步允许方向，一个在禁止方向”，而输出总是选
#    允许方向那一个；这比纯最近距离更容易看出“按步号交替横竖”的新规则。
# f) Visual cleanliness:
#    仍保留原题的统一方框样式和底部控制行，不叠加额外标记。桥只走路径，不连接诱饵；并且显式拒绝
#    交叉桥和压到框上的桥，避免画面脏乱。
# g) Color inferability:
#    不使用颜色值大小参与排序。颜色数控制在 4-5 个路径色，每个目标色只复制成“一真一诱饵”两格，
#    因此人类能从少量样例中归纳：颜色只表示“访问哪个方块类”，方向由步号决定。
# 9) New solver assumptions and generator guarantees:
#    - 假设奇数步必须横向、偶数步必须纵向；generator 始终满足。
#    - 假设首控制色对应的起点是唯一的；generator 不复制首色。
#    - 假设每一步至少有一个满足该步方向的目标色方块；generator 构造时逐步保证。
#    - 假设若同色有多个满足该步方向的候选，则最近者应被选；generator 通常让满足方向者唯一，
#      但 solver 仍保留“同方向内最近优先”作为稳定 tie-break。


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    border = _most_common_nonbg(g, int(bg))
    assert border is not None

    frames = _extract_frames(g, int(border))
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
    assert start_color in by and len(by[start_color]) >= 1

    def trace_from(start: dict) -> list[dict] | None:
        path = [start]
        cur = start
        for i in range(int(len(ctrl)) - 1):
            nxt_color = int(ctrl[int(i) + 1])
            axis = _step_axis(int(i))
            cands = [f for f in by.get(int(nxt_color), []) if _axis_aligned(cur, f, axis)]
            if not cands:
                return None
            cands.sort(key=lambda f: _candidate_key(cur, f))
            cur = cands[0]
            path.append(cur)
        return path

    valid_paths: list[list[dict]] = []
    for start in sorted(by[start_color], key=lambda f: tuple(map(int, f["bbox"]))):
        traced = trace_from(start)
        if traced is not None:
            valid_paths.append(traced)
    assert valid_paths

    path = min(valid_paths, key=_path_key)
    out = g.copy()
    for i in range(int(len(path)) - 1):
        cells = _bridge_cells_axis(path[int(i)], path[int(i) + 1], _step_axis(int(i)))
        assert cells
        for r, c in cells:
            if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
                out[int(r), int(c)] = int(ctrl[int(i)])
    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


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

        thickness = None
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
            thickness = int(tt)
            inner_col = int(next(iter(u)))
            break

        if thickness is None or inner_col is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(thickness),
                "inner": int(inner_col),
                "inner_bbox": (int(r0 + thickness), int(c0 + thickness), int(r1 - thickness), int(c1 - thickness)),
            }
        )
    return frames


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _axis_aligned(f1: dict, f2: dict, axis: str) -> bool:
    sr, sc = _aligned(f1, f2)
    if axis == "row":
        return bool(sr)
    return bool(sc)


def _step_axis(step_idx: int) -> str:
    return "row" if int(step_idx) % 2 == 0 else "col"


def _candidate_key(cur: dict, f: dict) -> tuple[int, int, int, int, int]:
    r1, c1 = _center(cur)
    r2, c2 = _center(f)
    d = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
    return (int(d), int(f["bbox"][0]), int(f["bbox"][1]), int(f["bbox"][2]), int(f["bbox"][3]))


def _path_key(path: list[dict]) -> tuple:
    return tuple(tuple(map(int, f["bbox"])) for f in path)


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _bridge_cells_axis(fa: dict, fb: dict, axis: str) -> set[tuple[int, int]]:
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]
    if axis == "row":
        if not (int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2])):
            return set()
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    if not (int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3])):
        return set()
    cc0, cc1 = int(ia[1]), int(ia[3])
    if int(r1a) < int(r0b):
        rr0, rr1 = int(r1a + 1), int(r0b - 1)
    else:
        rr0, rr1 = int(r1b + 1), int(r0a - 1)
    if int(rr0) > int(rr1):
        return set()
    return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}


_DIGITS = [int(i) for i in range(10)]


def _segment_clear_slots(a: tuple[int, int], b: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    ra, ca = map(int, a)
    rb, cb = map(int, b)
    if int(ra) == int(rb):
        lo, hi = sorted((int(ca), int(cb)))
        for cc in range(int(lo) + 1, int(hi)):
            if (int(ra), int(cc)) in occupied:
                return False
        return True
    if int(ca) == int(cb):
        lo, hi = sorted((int(ra), int(rb)))
        for rr in range(int(lo) + 1, int(hi)):
            if (int(rr), int(ca)) in occupied:
                return False
        return True
    return False


def _slot_candidates(
    cur: tuple[int, int],
    *,
    axis: str,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    min_dist: int,
    max_dist_exclusive: int | None = None,
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r, c = map(int, cur)
    out: list[tuple[int, int]] = []
    if axis == "row":
        for cc in range(int(w)):
            cand = (int(r), int(cc))
            d = abs(int(cc) - int(c))
            if d < int(min_dist):
                continue
            if max_dist_exclusive is not None and d >= int(max_dist_exclusive):
                continue
            if cand == cur or cand in occupied:
                continue
            if _segment_clear_slots(cur, cand, occupied):
                out.append(cand)
    else:
        for rr in range(int(h)):
            cand = (int(rr), int(c))
            d = abs(int(rr) - int(r))
            if d < int(min_dist):
                continue
            if max_dist_exclusive is not None and d >= int(max_dist_exclusive):
                continue
            if cand == cur or cand in occupied:
                continue
            if _segment_clear_slots(cur, cand, occupied):
                out.append(cand)
    return out


def _search_slot_path(
    *,
    shape: tuple[int, int],
    path_len: int,
) -> list[tuple[int, int]] | None:
    starts = [(int(r), int(c)) for r in range(int(shape[0])) for c in range(int(shape[1]))]
    np.random.shuffle(starts)

    def rec(path: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == int(path_len):
            return list(path)

        cur = path[-1]
        axis = _step_axis(len(path) - 1)
        next_cands = _slot_candidates(cur, axis=axis, shape=shape, occupied=occupied, min_dist=2)
        np.random.shuffle(next_cands)
        next_cands.sort(key=lambda x: (abs(int(x[0]) - int(cur[0])) + abs(int(x[1]) - int(cur[1])), int(x[0]), int(x[1])))
        for nxt in next_cands:
            actual_dist = abs(int(nxt[0]) - int(cur[0])) + abs(int(nxt[1]) - int(cur[1]))
            wrong_axis = "col" if axis == "row" else "row"
            probe_occ = set(occupied)
            probe_occ.add(nxt)
            decoy_probe = _slot_candidates(
                cur,
                axis=wrong_axis,
                shape=shape,
                occupied=probe_occ,
                min_dist=1,
                max_dist_exclusive=actual_dist,
            )
            if not decoy_probe:
                continue
            path.append(nxt)
            occupied.add(nxt)
            solved = rec(path, occupied)
            if solved is not None:
                return solved
            occupied.remove(nxt)
            path.pop()
        return None

    for start in starts:
        solved = rec([start], {start})
        if solved is not None:
            return solved
    return None


def _choose_decoys(
    path_slots: list[tuple[int, int]],
    *,
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    occupied = set(path_slots)
    decoys: list[tuple[int, int]] = []
    for i in range(len(path_slots) - 1):
        cur = path_slots[int(i)]
        nxt = path_slots[int(i) + 1]
        axis = _step_axis(int(i))
        wrong_axis = "col" if axis == "row" else "row"
        actual_dist = abs(int(nxt[0]) - int(cur[0])) + abs(int(nxt[1]) - int(cur[1]))
        cands = _slot_candidates(
            cur,
            axis=wrong_axis,
            shape=shape,
            occupied=occupied,
            min_dist=1,
            max_dist_exclusive=actual_dist,
        )
        if not cands:
            return None
        np.random.shuffle(cands)
        cands.sort(key=lambda x: (abs(int(x[0]) - int(cur[0])) + abs(int(x[1]) - int(cur[1])), int(x[0]), int(x[1])))
        dec = cands[0]
        decoys.append(dec)
        occupied.add(dec)
    return decoys


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 5:
        return None

    t = 1
    inner = int(np.random.randint(1, 3))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 4):
        return None

    slot_shape = (4, 5)
    path_len = int(np.random.randint(4, 6))
    path_slots = _search_slot_path(shape=slot_shape, path_len=path_len)
    if path_slots is None:
        return None
    decoy_slots = _choose_decoys(path_slots, shape=slot_shape)
    if decoy_slots is None:
        return None

    row_gap = [int(np.random.randint(1, 3)) for _ in range(int(slot_shape[0]) - 1)]
    col_gap = [int(np.random.randint(1, 3)) for _ in range(int(slot_shape[1]) - 1)]
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))

    r0s = [int(top_margin)]
    for k in range(int(slot_shape[0]) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(slot_shape[1]) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    path_colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=int(path_len), replace=False)]

    frame_slots = list(path_slots) + list(decoy_slots)
    color_by_slot: dict[tuple[int, int], int] = {}
    for idx, slot in enumerate(path_slots):
        color_by_slot[(int(slot[0]), int(slot[1]))] = int(path_colors[int(idx)])
    for idx, slot in enumerate(decoy_slots):
        color_by_slot[(int(slot[0]), int(slot[1]))] = int(path_colors[int(idx) + 1])

    frames: list[dict] = []
    slot_to_frame: dict[tuple[int, int], dict] = {}
    for slot in frame_slots:
        rr, cc = map(int, slot)
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
            "inner": int(color_by_slot[(int(rr), int(cc))]),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)
    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 4))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(f["inner"]))

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for i in range(len(path_slots) - 1):
        fa = slot_to_frame[(int(path_slots[int(i)][0]), int(path_slots[int(i)][1]))]
        fb = slot_to_frame[(int(path_slots[int(i) + 1][0]), int(path_slots[int(i) + 1][1]))]
        cells = _bridge_cells_axis(fa, fb, _step_axis(int(i)))
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
        paint = int(path_colors[int(i)])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    ctrl_colors = [int(x) for x in path_colors]
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
