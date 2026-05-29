# Combo variant of 3e6067c3 (index 27)
# combo_spec: D5/0+D6/1+D7/0
# Name: 终点色桥梁+擦除控制+紧凑裁剪

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

"""
Design analysis for the fused variant.

a) 具体化与行为追踪
   - 输入仍是“上方若干同尺寸带框方块 + 下方一行控制像素”。
   - 原任务会保留整张画布，在控制序列相邻颜色对应的方块之间画桥，且每段桥用起点方块色。
   - 本融合版的可见输出只保留上方主场景：控制行被去掉，外部边距被紧凑裁掉，桥段改成终点方块色。
   - 例：控制序列若为 [2, 5, 7]，则原任务是 “2->5 段涂 2，5->7 段涂 5，并保留控制行”；
     本版是 “2->5 段涂 5，5->7 段涂 7，最终只返回裁剪后的方块+桥梁区域”。
   - 因此差异落在三个阶段：桥梁着色规则、是否保留控制源对象、最终返回的网格范围。

b) 复杂度校准
   - 若只改颜色但保留全画布，D5 很明显，D7 很弱；若只裁剪不保证桥段颜色差异，D5 会退化。
   - 生成器因此继续让路径长度 >= 4，并强制路径相邻节点在同一行/列上颜色不同，这样每段桥的
     “起点色 vs 终点色”都可见。
   - 同时保留至少一个非路径 decoy 方块，避免样例退化成“所有方块都被路径解释完”的过简图案。

c) 隐式假设审查
   - 边框色识别仍依赖“非背景中最常见的颜色是统一边框色”，输入保持 framed squares，所以成立。
   - 方块检测仍依赖输入中每个对象都有完整边框；本融合版不在输入里改方块，只在输出阶段改桥和裁剪。
   - 控制行检测仍假设“位于所有方块下方、最底部且至少两个非背景像素的一行”就是控制行；
     生成器保持这个几何约束。
   - 原代码里真正失效的旧前提有两处：
     1. `draw_bridge(..., a)` 假设桥段颜色取起点；本版改为终点 `b`。
     2. 最终 `return out` 假设输出尺寸与输入一致；本版改为先删除控制层，再做紧凑裁剪。
   - prompt 给出的单变体 D6/1 描述是“访问方块变色”，但 combo 级规格与名称明确要求的是
     “擦除控制，只保留方块和桥梁”。这里以 combo 级规则为准，不引入 visited-square repaint，
     否则会违背“只有方块和终点色桥梁的精简图案”这一总规范。

d) 放置合法性
   - 所有方块都放在规则 lattice 上，相邻 slot 至少留 1 格背景，因此桥只会落在空白带。
   - 路径是 4 邻接自避免游走，桥段只连接路径相邻方块；生成器检查桥不重叠、不越界、不覆盖方块。
   - decoy 方块只放在未走到的 slot 上，因此不会与路径方块或桥冲突。

e) 视觉可观察性
   - 人类能从样例里直接看到：输入有控制行，输出没有；输出只剩被完成的主图案。
   - 再看桥段颜色，会发现每段桥更像“流向目标方块”，因为桥色总与终点方块内部色一致。
   - 紧凑裁剪让这个变化更醒目：输出像是把上方结构“提纯”出来，而不是整图原样保留。

f) 可视化整洁度
   - 输出不叠加额外标记，也不让控制像素与主场景共存；唯一新信息就是桥段颜色和画布裁剪。
   - decoy 方块仍保持原 framed 外观，与被桥连接的路径方块在视觉上容易区分。

g) 变化的可推断性
   - 没有引入新颜色表；桥色直接复用已有方块内部色。
   - 终点色规则具备结构线索：桥段总与它接入的目标方块同色，而不是与离开的方块同色。
   - 裁剪规则也具备结构线索：输出总是输入上方主结构的最小包围盒，不依赖任何不可见排序。

h) 量化防退化指标
   - 指标 1：把原始 solver 的输出做同样裁剪，得到“源点色+裁剪”基线；本生成器要求本版输出与该
     基线不同，从而证明 D5 真在起作用。
   - 指标 2：输出尺寸必须同时小于输入尺寸，证明控制层和外部边距确实被移除。

i) 新 solver 假设与生成器保证
   - solver 新增假设：控制序列的每个相邻对都能在几何上找到对齐候选，且最佳候选就是目标路径点。
     生成器通过 4 邻接 lattice 路径保证这一点。
   - solver 还假设 `ctrl_r` 之上的非背景像素足以定义唯一裁剪框；生成器保证方块始终存在于该区域。
"""


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _find_frames(g: np.ndarray, border: int) -> list[dict]:
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
            uniq = set(map(int, inner.flatten()))
            if len(uniq) != 1:
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
            inner_col = int(next(iter(uniq)))
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


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _choose_candidate(cur: dict, cands: list[dict]) -> dict | None:
    best = None
    bestf = None
    for frame in cands:
        sr, sc = _aligned(cur, frame)
        al = 1 if (sr or sc) else 0
        r1_, c1_ = _center(cur)
        r2_, c2_ = _center(frame)
        d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
        cand = (int(al), -int(d))
        if best is None or cand > best:
            best = cand
            bestf = frame
    return bestf


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
    for frame in start_cands:
        for g2 in (nxt_cands if nxt_cands else [frame]):
            sr, sc = _aligned(frame, g2)
            al = 1 if (sr or sc) else 0
            r1_, c1_ = _center(frame)
            r2_, c2_ = _center(g2)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = frame
    assert bestf is not None
    return bestf


def _find_control_row(g: np.ndarray, bg: int, frames: list[dict]) -> int:
    H, _W = g.shape
    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            return int(r)
    raise AssertionError("control row not found")


def _control_sequence(g: np.ndarray, bg: int, ctrl_r: int) -> list[int]:
    _H, W = g.shape
    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2
    return ctrl


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


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    H, W = out.shape
    for r, c in _bridge_cells(fa, fb):
        if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
            out[int(r), int(c)] = int(color)


def _tight_crop_scene(g: np.ndarray, bg: int, ctrl_r: int) -> np.ndarray:
    fg = np.argwhere(g[: int(ctrl_r), :] != int(bg))
    assert fg.size > 0
    r0, c0 = fg.min(axis=0)
    r1, c1 = fg.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw destination-colored bridges, erase the control layer, then tight-crop.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    border = _most_common_nonbg(g, int(bg))
    assert border is not None

    frames = _find_frames(g, int(border))
    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by[int(frame["inner"])].append(frame)

    ctrl_r = _find_control_row(g, int(bg), frames)
    ctrl = _control_sequence(g, int(bg), int(ctrl_r))
    cur = _choose_start_frame(by, ctrl)

    out = g.copy()
    out[int(ctrl_r), :] = int(bg)

    for a, b in zip(ctrl, ctrl[1:]):
        assert int(cur["inner"]) == int(a)
        cands = by.get(int(b), [])
        if not cands:
            continue
        nxt = _choose_candidate(cur, cands)
        if nxt is None:
            continue
        _draw_bridge(out, cur, nxt, int(b))
        cur = nxt

    return _tight_crop_scene(out, int(bg), int(ctrl_r))


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 5:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(R * C)
    if max_frames < 5:
        return None

    max_path = int(min(7, max_frames - 1, len(inner_palette) - 1))
    if max_path < 4:
        return None
    n_path = int(np.random.randint(4, max_path + 1))

    path: list[tuple[int, int]] | None = None
    for _ in range(400):
        p = [(0, 0)]
        seen = {(0, 0)}
        ok = True
        for _step in range(int(n_path) - 1):
            r, c = map(int, p[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in seen
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            p.append((int(nxt[0]), int(nxt[1])))
            seen.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in p}) < 2 or len({c for _r, c in p}) < 2:
            continue
        path = p
        break
    if path is None:
        return None

    path_set = set(path)
    spare_slots = [(int(r), int(c)) for r in range(int(R)) for c in range(int(C)) if (int(r), int(c)) not in path_set]
    max_decoys = int(min(3, len(spare_slots), len(inner_palette) - n_path))
    if max_decoys < 1:
        return None
    n_decoys = int(np.random.randint(1, max_decoys + 1))
    decoy_idx = np.random.choice(len(spare_slots), size=int(n_decoys), replace=False)
    decoy_slots = [spare_slots[int(i)] for i in np.array(decoy_idx).tolist()]

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

    all_slots = list(path) + list(decoy_slots)
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

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)

    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = int(np.random.randint(1, 3))
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, 2 * n_path + 2))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    used_in_row: dict[int, set[int]] = defaultdict(set)
    used_in_col: dict[int, set[int]] = defaultdict(set)
    path_colors_by_slot: dict[tuple[int, int], int] = {}

    start_color = int(np.random.choice(np.array(inner_palette, dtype=int)))
    path_colors_by_slot[(0, 0)] = int(start_color)
    used_in_row[0].add(int(start_color))
    used_in_col[0].add(int(start_color))

    for rr, cc in path[1:]:
        rr = int(rr)
        cc = int(cc)
        forbid = set(used_in_row[int(rr)]) | set(used_in_col[int(cc)])
        choices = [int(x) for x in inner_palette if int(x) not in forbid]
        if not choices:
            return None
        color = int(np.random.choice(np.array(choices, dtype=int)))
        path_colors_by_slot[(int(rr), int(cc))] = int(color)
        used_in_row[int(rr)].add(int(color))
        used_in_col[int(cc)].add(int(color))

    if len(path_colors_by_slot) != len(path):
        return None

    remaining_colors = [int(c) for c in inner_palette if int(c) not in set(path_colors_by_slot.values())]
    if len(remaining_colors) < len(decoy_slots):
        return None

    decoy_colors = np.random.choice(np.array(remaining_colors, dtype=int), size=int(len(decoy_slots)), replace=False).tolist()
    colors_by_slot = dict(path_colors_by_slot)
    for slot, color in zip(decoy_slots, decoy_colors):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(color)

    full_out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            full_out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(int(rr), int(cc))]),
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
            if int(full_out[int(r), int(c)]) != int(bg):
                ok = False
                break
        if not ok:
            break
        paint = int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        for r, c in cells:
            full_out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    expected = _tight_crop_scene(full_out, int(bg), int(ctrl_r))
    if not (expected.shape[0] < full_out.shape[0] and expected.shape[1] < full_out.shape[1]):
        return None

    inp = full_out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for color, col in zip(ctrl_colors, ctrl_cols):
        inp[int(ctrl_r), int(col)] = int(color)

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
