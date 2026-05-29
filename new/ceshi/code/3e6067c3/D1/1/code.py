# Variant of 3e6067c3
# Dimension: D1 — 匹配条件：颜色匹配方式 (match_cond)
# Change: 控制像素颜色与方块内部填充颜色相同 (s.colors == d.inner_squares.colors)... → 形状匹配
#
# Design analysis for this variant
# a) Concrete behavior:
#    原题里每个带框方块的内部是纯色块，底部控制区是同色像素序列；solver 直接按颜色找目标方块。
#    这里把每个方块内部改成“底色 + 单色小图案”的二层结构，桥梁颜色仍取方块底色，但控制区不再给颜色像素，
#    而是给出与方块内部小图案同形的单色图案序列。于是输入中可见的是：
#    - 上半区：同尺寸边框方块；每个方块内部都有 3x3 画布上的 tetromino 风格小图案。
#    - 下半区：同样的小图案按从左到右的顺序排成控制序列。
#    - 输出：仍然在相邻路径方块之间画桥，桥的颜色沿用起点方块的内部底色。
#    小例子追踪：原逻辑看到控制颜色 4，就去找内填色 4 的方块；新逻辑看到控制图案 “L”，就去找内部带 “L” 图案的方块，
#    但桥仍画成该方块的底色，例如蓝底+L 图案的方块会产生蓝桥。
#
# b) Complexity calibration:
#    形状模板使用少量、离散、可肉眼区分的 4 连通小图案，复杂度对齐参考任务 16b78196 的“精确形状匹配、无旋转翻转”。
#    仅使用 6 个模板，避免自由度过多；同时允许全局重复形状，但保持同一行/同一列上的形状唯一，继续复用原题“行列对齐 + 最近”
#    的消歧骨架。
#
# c) Implicit-assumption audit against the original solver:
#    - 原代码 `len(u) == 1`：依赖内部纯色。变体中失效，改为“内部恰有两色：底色 + 图案色”，并抽取图案 mask。
#    - 原代码 `by[inner_color]`：依赖颜色是键。变体中改为 `by[shape_key]`。
#    - 原代码控制区是一行散点像素：变体中控制区是多个连通小图案，因此不再按单行取色，而是对方块下方的非背景连通块做形状提取。
#    - 原代码画桥时直接使用控制颜色 `a`：变体中控制键不是颜色，因此桥色改为当前方块的底色 `cur["fill"]`。
#    这些是与 D1 直接相关、必须替换的依赖；其余路径搜索和桥宽逻辑保持不变。
#
# d) Placement legality:
#    方块仍放在离散 lattice 上，边框尺寸统一；路径仍是 4 邻自避路径；桥仍只发生在同一行带或同一列带之间。
#    generator 继续先构造 output，再抹掉桥得到 input，并用 main 反验，确保不重叠、不越界、路径唯一化成立。
#
# e) Visual observability:
#    控制区所有图案使用同一颜色，故解题者不能靠颜色直接配对，只能看形状。方块底色和桥色仍变化，因此可明显看出：
#    “控制图案决定访问顺序，桥色来自被访问的方块底色”，这和原题的“控制颜色就是桥色/目标色”有可见差异。
#
# f) Visual cleanliness:
#    内部图案固定在 3x3 画布内，且总是单层覆盖在纯底色上；控制图案彼此至少隔 1 列背景；桥只画在方块之间原本的空白带上，
#    不会与控制区或边框重叠。整体仍是整洁的方块 + 底部序列布局。
#
# g) Color inference:
#    本变体不引入新的颜色映射规则。颜色只承担两个可见角色：
#    - 方块底色 / 桥色
#    - 统一的图案色
#    匹配完全依赖形状，因此不会出现“颜色变化很多但无结构线索”的问题。
#
# h) Variant-specific validation target:
#    生成时强制至少一个形状在全局重复，且该重复形状对应不同底色；这样输入不能退化成“每种形状只出现一次”或“形状和颜色一一绑定”。
#
# i) New solver assumptions introduced here:
#    - 每个方块内部恰有两色，且图案色形成单个 4 连通块。
#    - 控制区只包含这些单色图案组件，位于所有方块下方。
#    - 同一行/列上的已访问方块不会共享同一形状，从而 `aligned + nearest` 仍能稳定选中路径上的下一格。
#    generator 明确构造并验证这些条件。

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

ShapeKey = tuple[tuple[int, int], ...]

_DIGITS = [int(i) for i in range(10)]
_SHAPES: list[ShapeKey] = [
    ((0, 0), (1, 0), (2, 0), (2, 1)),  # L
    ((0, 1), (1, 1), (2, 0), (2, 1)),  # mirrored L
    ((0, 0), (0, 1), (0, 2), (1, 1)),  # T
    ((0, 0), (0, 1), (1, 1), (1, 2)),  # Z
    ((0, 1), (0, 2), (1, 0), (1, 1)),  # S
    ((0, 0), (0, 1), (1, 0), (1, 1)),  # 2x2 block
]


def _shape_key_from_coords(coords: np.ndarray) -> ShapeKey | None:
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(rel)


def _shape_key_from_mask(mask: np.ndarray) -> ShapeKey | None:
    return _shape_key_from_coords(np.argwhere(mask))


def _extract_frame_payload(inner: np.ndarray, bg: int, border: int) -> tuple[int, int, ShapeKey] | None:
    cnt = Counter(map(int, inner.flatten()))
    if len(cnt) != 2:
        return None
    fill_color, fill_n = cnt.most_common(1)[0]
    fill_color = int(fill_color)
    fill_n = int(fill_n)
    if fill_color in (int(bg), int(border)):
        return None
    marker_colors = [int(col) for col in cnt if int(col) != int(fill_color)]
    if len(marker_colors) != 1:
        return None
    marker_color = int(marker_colors[0])
    if marker_color in (int(bg), int(border), int(fill_color)):
        return None
    marker_mask = inner == int(marker_color)
    if int(marker_mask.sum()) == 0 or int(marker_mask.sum()) >= int(inner.size):
        return None
    if fill_n <= int(marker_mask.sum()):
        return None
    lab, n = label_mask(marker_mask, connectivity=4)
    if int(n) != 1:
        return None
    shape = _shape_key_from_mask(marker_mask)
    if shape is None:
        return None
    return int(fill_color), int(marker_color), shape


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a control-shape sequence.

    Input structure:
    - bg = background. Border color forms equal-sized framed squares.
    - Each square interior is a solid fill color plus a small monochrome motif.
    - Control area below the squares contains the same motif shapes, read left-to-right.

    Rule:
    - Match control motifs to square motifs by exact shape (no rotation / flip).
    - For each consecutive control pair, find the next matching square aligned in the same row or column band,
      preferring nearest as in the original task.
    - Draw the bridge using the current square's fill color, with the original bridge thickness.
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
    assert not (border is None)

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
        fill_color = None
        shape = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
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

            payload = _extract_frame_payload(inner, int(bg), int(border))
            if payload is None:
                continue
            fill_color, _marker_color, shape = payload
            t = int(tt)
            break

        if t is None or fill_color is None or shape is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(t),
                "fill": int(fill_color),
                "shape": shape,
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            }
        )

    assert not (not frames)

    by: dict[ShapeKey, list[dict]] = defaultdict(list)
    for f in frames:
        by[f["shape"]].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)

    ctrl_mask = np.zeros((int(H), int(W)), dtype=bool)
    ctrl_mask[int(max_frame_r) + 1 :, :] = g[int(max_frame_r) + 1 :, :] != int(bg)
    ctrl_lab, ctrl_n = label_mask(ctrl_mask, connectivity=4)
    ctrl_items: list[tuple[int, int, ShapeKey]] = []
    for cid in range(1, int(ctrl_n) + 1):
        pts = np.argwhere(ctrl_lab == int(cid))
        if pts.size == 0:
            continue
        key = _shape_key_from_coords(pts)
        if key is None:
            continue
        r0, c0 = pts.min(axis=0)
        ctrl_items.append((int(c0), int(r0), key))
    ctrl_items.sort(key=lambda item: (int(item[0]), int(item[1])))
    ctrl = [item[2] for item in ctrl_items]
    assert not (len(ctrl) < 2)

    def center(f: dict) -> tuple[int, int]:
        r0, c0, r1, c1 = map(int, f["bbox"])
        return int((r0 + r1) // 2), int((c0 + c1) // 2)

    def aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
        a = f1["inner_bbox"]
        b = f2["inner_bbox"]
        same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
        same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
        return same_row, same_col

    start_shape = ctrl[0]
    assert not (start_shape not in by)
    start_cands = by[start_shape]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_shape = ctrl[1]
        nxt_cands = by.get(nxt_shape, [])
        best = None
        bestf = None
        for f in start_cands:
            for g2 in (nxt_cands if nxt_cands else [f]):
                sr, sc = aligned(f, g2)
                al = 1 if (sr or sc) else 0
                (r1_, c1_) = center(f)
                (r2_, c2_) = center(g2)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    bestf = f
        cur = bestf

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
        (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
        (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
        ia = fa["inner_bbox"]
        sr, sc = aligned(fa, fb)
        if sr:
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
        if sc:
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
        a = ctrl[i]
        b = ctrl[i + 1]
        assert cur["shape"] == a
        assert b in by
        cands = by[b]
        best = None
        bestf = None
        for f in cands:
            sr, sc = aligned(cur, f)
            al = 1 if (sr or sc) else 0
            (r1_, c1_) = center(cur)
            (r2_, c2_) = center(f)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
        assert not (bestf is None)
        nxt = bestf
        draw_bridge(cur, nxt, int(cur["fill"]))
        cur = nxt

    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _render_shape(g: np.ndarray, *, top: int, left: int, shape: ShapeKey, color: int) -> None:
    for dr, dc in shape:
        g[int(top + dr), int(left + dc)] = int(color)


def _draw_frame(
    g: np.ndarray,
    *,
    r0: int,
    c0: int,
    outer: int,
    t: int,
    border: int,
    fill_color: int,
    marker_color: int,
    shape: ShapeKey,
) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    inner_r0 = int(r0 + t)
    inner_c0 = int(c0 + t)
    g[int(inner_r0) : int(r0 + outer - t), int(inner_c0) : int(c0 + outer - t)] = int(fill_color)
    _render_shape(g, top=int(inner_r0), left=int(inner_c0), shape=shape, color=int(marker_color))


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


def _assign_shapes(path: list[tuple[int, int]]) -> dict[tuple[int, int], ShapeKey] | None:
    used_in_row: dict[int, set[ShapeKey]] = defaultdict(set)
    used_in_col: dict[int, set[ShapeKey]] = defaultdict(set)
    shape_by_slot: dict[tuple[int, int], ShapeKey] = {}

    repeat_pairs = [
        (path[i], path[j])
        for i in range(int(len(path)))
        for j in range(i + 1, int(len(path)))
        if int(path[i][0]) != int(path[j][0]) and int(path[i][1]) != int(path[j][1])
    ]
    if not repeat_pairs:
        return None
    pair_a, pair_b = repeat_pairs[int(np.random.randint(0, len(repeat_pairs)))]
    repeat_shape = _SHAPES[int(np.random.randint(0, len(_SHAPES)))]
    for rr, cc in (pair_a, pair_b):
        rr = int(rr)
        cc = int(cc)
        shape_by_slot[(int(rr), int(cc))] = repeat_shape
        used_in_row[int(rr)].add(repeat_shape)
        used_in_col[int(cc)].add(repeat_shape)

    for rr, cc in path:
        rr = int(rr)
        cc = int(cc)
        if (int(rr), int(cc)) in shape_by_slot:
            continue
        forbid = set(used_in_row[int(rr)]) | set(used_in_col[int(cc)])
        choices = [shape for shape in _SHAPES if shape not in forbid]
        if not choices:
            return None
        shape = choices[int(np.random.randint(0, len(choices)))]
        shape_by_slot[(int(rr), int(cc))] = shape
        used_in_row[int(rr)].add(shape)
        used_in_col[int(cc)].add(shape)

    counts = Counter(shape_by_slot.values())
    if int(len(counts)) < 3:
        return None
    if max(int(v) for v in counts.values()) < 2:
        return None
    return shape_by_slot


def _assign_fill_colors(
    path: list[tuple[int, int]],
    shape_by_slot: dict[tuple[int, int], ShapeKey],
    fill_palette: list[int],
) -> dict[tuple[int, int], int] | None:
    if len(fill_palette) < 3:
        return None
    fill_by_slot: dict[tuple[int, int], int] = {}

    repeated_shape = None
    repeated_slots = None
    counts = Counter(shape_by_slot.values())
    for shape, n in counts.items():
        if int(n) >= 2:
            slots = [slot for slot in path if shape_by_slot[slot] == shape]
            if len(slots) >= 2:
                repeated_shape = shape
                repeated_slots = (slots[0], slots[1])
                break
    if repeated_shape is None or repeated_slots is None:
        return None

    first_fill = int(fill_palette[int(np.random.randint(0, len(fill_palette)))])
    other_choices = [int(col) for col in fill_palette if int(col) != int(first_fill)]
    if not other_choices:
        return None
    second_fill = int(other_choices[int(np.random.randint(0, len(other_choices)))])
    fill_by_slot[repeated_slots[0]] = int(first_fill)
    fill_by_slot[repeated_slots[1]] = int(second_fill)

    for slot in path:
        if slot in fill_by_slot:
            continue
        fill_by_slot[slot] = int(fill_palette[int(np.random.randint(0, len(fill_palette)))])

    collision_exists = any(
        slot_a != slot_b
        and int(fill_by_slot[slot_a]) == int(fill_by_slot[slot_b])
        and shape_by_slot[slot_a] != shape_by_slot[slot_b]
        for slot_a in path
        for slot_b in path
    )
    if not collision_exists:
        diff_shape_pairs = [(a, b) for a in path for b in path if a != b and shape_by_slot[a] != shape_by_slot[b]]
        if not diff_shape_pairs:
            return None
        slot_a, slot_b = diff_shape_pairs[0]
        fill_by_slot[slot_b] = int(fill_by_slot[slot_a])

    if int(fill_by_slot[repeated_slots[0]]) == int(fill_by_slot[repeated_slots[1]]):
        fill_by_slot[repeated_slots[1]] = int(second_fill)

    if len(set(int(fill_by_slot[slot]) for slot in path)) < 3:
        return None
    if int(fill_by_slot[repeated_slots[0]]) == int(fill_by_slot[repeated_slots[1]]):
        return None
    return fill_by_slot


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border_choices = [d for d in _DIGITS if int(d) != int(bg)]
    border = int(np.random.choice(np.array(border_choices, dtype=int)))
    marker_choices = [d for d in _DIGITS if int(d) not in (int(bg), int(border))]
    marker = int(np.random.choice(np.array(marker_choices, dtype=int)))
    fill_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border), int(marker))]
    if len(fill_palette) < 3:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = 3
    outer = int(inner + 2 * t)

    R = int(np.random.randint(2, 4))
    C = int(np.random.randint(2, 4))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(8, max_frames)) + 1))

    path: list[tuple[int, int]] | None = None
    for _path_try in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _ in range(int(n_frames) - 1):
            r, c = p[-1]
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

    row_gap = [int(np.random.randint(1, 4)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 4)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))

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

    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_top = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 4))
    H = int(ctrl_top + 3 + bottom_margin)
    W = int(max(int(max_c1 + 1 + right_margin), int(4 * n_frames)))

    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    shape_by_slot = _assign_shapes(path)
    if shape_by_slot is None:
        return None
    fill_by_slot = _assign_fill_colors(path, shape_by_slot, fill_palette)
    if fill_by_slot is None:
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        rr, cc = map(int, f["slot"])
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            fill_color=int(fill_by_slot[(rr, cc)]),
            marker_color=int(marker),
            shape=shape_by_slot[(rr, cc)],
        )

    bridge_cells_all: set[tuple[int, int]] = set()
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
            if int(out[int(r), int(c)]) != int(bg):
                ok = False
                break
        if not ok:
            break
        paint = int(fill_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    ctrl_cols = [int(1 + 4 * i) for i in range(int(len(path)))]
    if int(ctrl_cols[-1] + 2) >= int(W):
        return None
    for (rr, cc), ctrl_c0 in zip(path, ctrl_cols):
        _render_shape(out, top=int(ctrl_top), left=int(ctrl_c0), shape=shape_by_slot[(int(rr), int(cc))], color=int(marker))

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
