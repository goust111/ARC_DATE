# Variant of 3e6067c3
# Dimension: D3 — 源对象：控制像素的编码形式 (src)
# Change: 底部区域的单行控制像素序列，每个像素代表一种颜色，从左到右编码访问顺序
#   → 双行控制像素：起点-终点对

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask

# Design analysis for the D3 variant
#
# a) 具体化与行为追踪
# - 主区域仍然是若干等尺寸带框方块，框色统一，内部纯色代表“节点颜色”；输出仍是在同一行带或同一列带的相邻节点之间补出桥梁。
# - 变体只改控制区：原题底部只有一行控制像素，例如 `[2,3,9,4]`，隐含边是 `(2->3),(3->9),(9->4)`。
# - 现在底部改为两行、同列对齐的控制像素：上行 `[2,3,9]`，下行 `[3,9,4]`。每一列直接构成一条起点-终点边，同时两行还满足链式重叠关系 `top[i+1] == bottom[i]`。
# - solver 中受影响的环节只有“控制序列解析”：旧逻辑直接读取单行；新逻辑必须先找出两行、确认列对齐，再把它们重建成访问链 `[top[0]] + bottom`，后续选方块与画桥逻辑继续沿用。
#
# b) 复杂度校准
# - 不把两行做成完全独立的任意配对，否则会引入新的“多源独立绘制”语义，偏离 D3 的最小改动原则。
# - 采用“配对行 = 原序列的相邻边展开”能保留原题的路径复杂度、颜色重复与对齐消歧需求，同时在视觉上明显不同于单行控制。
# - 控制列仍使用固定隔列放置，避免下方区域拥挤，并让“上下对齐成对”一眼可见。
#
# c) 隐式假设审查
# - 原 `main()` 假设控制区只有一行，并用“底部最后一行有 >=2 个非背景像素”定位；该假设在变体中失效，必须改成检测两条连续控制行。
# - 原 `ctrl = ...` 直接得到完整访问链；现在读取任一单行都会丢信息，因此必须组合两行，并检查同列对齐和链式重叠。
# - 原始的起始方块选择、后续候选筛选、桥梁几何都只依赖“恢复出的访问链”，不依赖控制区是一行还是两行，所以这些代码可保持。
#
# d) 放置合法性
# - generator 继续先构造含桥的输出，再擦掉桥得到输入。
# - 只在主区域下方新增一条控制行，因此要保证：控制上行与方块区之间至少隔 1 行背景，控制下行正好在其下方，再留 1 行背景作为底边距。
# - 两控制行在同一组列上放色，其他列保持背景，避免与主区域或彼此重叠。
#
# e) 视觉可观察性
# - 人类可直接看到：输出中的桥还是按一条路径出现，但输入里底部不再是单行顺序，而是“上下配对”的双行编码。
# - 两行之间存在重复错位（下行前 k-1 个颜色等于上行后 k-1 个颜色），这会提示“需要上下合并成链”，而不是单独读取某一行。
#
# f) 可视化整洁度
# - 控制区仍然限制在底部一个窄带，且使用隔列布局；不会覆盖方块、不会和桥梁叠层，整体结构仍是“主区域 + 控制区域”的清晰两段式布局。
#
# g) 颜色变化可推断性
# - 不引入任何新的颜色映射。控制像素直接复用方块内部颜色；桥梁颜色仍取起点方块颜色。
# - 颜色的作用仅是显式匹配已有方块，不依赖数值大小、排序或隐藏编码。
#
# h) 新 `main()` 的隐式假设与对应生成保证
# - `_find_control_rows()` 假设方块区下方存在且只需存在一组“连续两行、非背景列完全相同”的控制行。
#   generator 明确只在底部放这两行控制像素，且两行使用同一组隔列列坐标。
# - `_decode_control_sequence()` 假设上下行不是任意独立列表，而是相邻边展开后的链式重叠编码。
#   generator 始终把路径颜色序列 `[c0,...,ck]` 写成上行 `[c0,...,c{k-1}]`、下行 `[c1,...,ck]`。
# - 后续逐步选点逻辑假设：恢复出的颜色链对应一条真实可画的路径，每一步都能找到同带且最近的目标方块。
#   generator 继续沿用原题的自避免路径和 output-first 校验，只接受 `main(inp) == out` 的样本。


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
            inner_colors = set(map(int, inner.flatten()))
            if len(inner_colors) != 1:
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
            inner_col = int(next(iter(inner_colors)))
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


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(frame_a: dict, frame_b: dict) -> tuple[bool, bool]:
    a = frame_a["inner_bbox"]
    b = frame_b["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _find_control_rows(g: np.ndarray, bg: int, max_frame_r: int) -> tuple[int, int, list[int]]:
    H, W = g.shape
    candidate_rows = [int(r) for r in range(int(max_frame_r) + 1, int(H)) if int((g[int(r)] != int(bg)).sum()) >= 1]
    assert len(candidate_rows) >= 2

    for idx in range(len(candidate_rows) - 2, -1, -1):
        top_r = int(candidate_rows[idx])
        bottom_r = int(candidate_rows[idx + 1])
        if int(bottom_r) != int(top_r + 1):
            continue
        cols_top = [int(c) for c in range(int(W)) if int(g[int(top_r), int(c)]) != int(bg)]
        cols_bottom = [int(c) for c in range(int(W)) if int(g[int(bottom_r), int(c)]) != int(bg)]
        if cols_top and cols_top == cols_bottom:
            return int(top_r), int(bottom_r), cols_top
    raise AssertionError("failed to locate aligned two-row control area")


def _decode_control_sequence(g: np.ndarray, top_r: int, bottom_r: int, cols: list[int]) -> tuple[list[int], list[int], list[int]]:
    starts = [int(g[int(top_r), int(c)]) for c in cols]
    ends = [int(g[int(bottom_r), int(c)]) for c in cols]
    assert len(starts) == len(ends) and len(starts) >= 1
    if len(starts) > 1:
        assert starts[1:] == ends[:-1]
    seq = [int(starts[0]), *map(int, ends)]
    assert len(seq) >= 2
    return starts, ends, seq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges using a two-row control code.

    Input structure:
    - Background color fills empty space.
    - Equal-sized framed squares have a common border color and colored interiors.
    - Bottom control area uses two aligned rows of colored pixels.
      Each column is a (start_color, end_color) pair for one consecutive bridge.

    Rule:
    - Reconstruct the original visitation chain from the paired control rows:
      top = [c0,c1,...,c{k-1}], bottom = [c1,c2,...,ck] -> sequence [c0,c1,...,ck].
    - Starting from the square matching c0, repeat the original logic:
      for each consecutive pair (ci, ci+1), choose the aligned nearest ci+1 square
      and paint the bridge with color ci.
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

    frames = _extract_frames(g, int(border))
    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by[int(frame["inner"])].append(frame)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_top_r, ctrl_bottom_r, ctrl_cols = _find_control_rows(g, int(bg), int(max_frame_r))
    _starts, _ends, ctrl = _decode_control_sequence(g, int(ctrl_top_r), int(ctrl_bottom_r), ctrl_cols)

    start_color = int(ctrl[0])
    assert start_color in by
    start_cands = by[start_color]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        nxt_cands = by.get(nxt_color, [])
        best = None
        best_frame = None
        for frame in start_cands:
            for frame2 in (nxt_cands if nxt_cands else [frame]):
                same_row, same_col = _aligned(frame, frame2)
                aligned_score = 1 if (same_row or same_col) else 0
                r1, c1 = _center(frame)
                r2, c2 = _center(frame2)
                dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
                cand = (int(aligned_score), -int(dist))
                if best is None or cand > best:
                    best = cand
                    best_frame = frame
        cur = best_frame
    assert cur is not None

    out = g.copy()

    def draw_bridge(frame_a: dict, frame_b: dict, color: int) -> None:
        r0a, c0a, r1a, c1a = map(int, frame_a["bbox"])
        r0b, c0b, r1b, c1b = map(int, frame_b["bbox"])
        ia = frame_a["inner_bbox"]
        same_row, same_col = _aligned(frame_a, frame_b)
        if same_row:
            rr0, rr1 = int(ia[0]), int(ia[2])
            if int(c1a) < int(c0b):
                cc0, cc1 = int(c1a + 1), int(c0b - 1)
            else:
                cc0, cc1 = int(c1b + 1), int(c0a - 1)
            for r in range(int(rr0), int(rr1) + 1):
                for c in range(int(cc0), int(cc1) + 1):
                    if 0 <= r < H and 0 <= c < W:
                        out[int(r), int(c)] = int(color)
            return
        if same_col:
            cc0, cc1 = int(ia[1]), int(ia[3])
            if int(r1a) < int(r0b):
                rr0, rr1 = int(r1a + 1), int(r0b - 1)
            else:
                rr0, rr1 = int(r1b + 1), int(r0a - 1)
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
        best = None
        best_frame = None
        for frame in by[b]:
            same_row, same_col = _aligned(cur, frame)
            aligned_score = 1 if (same_row or same_col) else 0
            r1, c1 = _center(cur)
            r2, c2 = _center(frame)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned_score), -int(dist))
            if best is None or cand > best:
                best = cand
                best_frame = frame
        nxt = best_frame
        assert nxt is not None
        draw_bridge(cur, nxt, int(a))
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


def _bridge_cells(frame_a: dict, frame_b: dict) -> set[tuple[int, int]]:
    r0a, c0a, r1a, c1a = map(int, frame_a["bbox"])
    r0b, c0b, r1b, c1b = map(int, frame_b["bbox"])
    ia = frame_a["inner_bbox"]
    ib = frame_b["inner_bbox"]
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
        cand_path = [(0, 0)]
        visited = {(0, 0)}
        ok = True
        for _step in range(int(n_frames) - 1):
            r, c = map(int, cand_path[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in visited
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            cand_path.append((int(nxt[0]), int(nxt[1])))
            visited.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in cand_path}) < 2 or len({c for _r, c in cand_path}) < 2:
            continue
        path = cand_path
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
    ctrl_top_r = int(max_r1 + ctrl_gap)
    ctrl_bottom_r = int(ctrl_top_r + 1)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_bottom_r + 1 + bottom_margin)
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

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(colors_by_slot[(rr, cc)]))

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        frame_a = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        frame_b = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(frame_a, frame_b)
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
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    seq_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    pair_starts = seq_colors[:-1]
    pair_ends = seq_colors[1:]
    if len(pair_starts) < 1:
        return None
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(pair_starts)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cc in zip(pair_starts, ctrl_cols):
        out[int(ctrl_top_r), int(cc)] = int(col)
    for col, cc in zip(pair_ends, ctrl_cols):
        out[int(ctrl_bottom_r), int(cc)] = int(col)

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
