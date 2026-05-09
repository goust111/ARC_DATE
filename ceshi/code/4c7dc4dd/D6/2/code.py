# Variant of 4c7dc4dd
# Dimension: D6 — 框架检测方式 (symbol_instructions)
# Change: 通过搜索单色空心正方形来检测框架，优先选择更大的框架，框架不重叠... → 不完整框架（缺口框架）

# Design analysis
# a) 具体化与行为追踪
#    - 每个框仍然是 1 像素粗、单色、轴对齐的正方形边框，外边长 S 仍在 6..8。
#    - 与原题不同之处在于：四个框都不是闭合边框，而是在四条边中的恰好一条边上挖掉一段
#      连续缺口，宽度为 1 或 2 像素；四个角始终保留，因此人眼仍能把它看成“几乎闭合的框”。
#    - 小例子：S=7、顶部第 3..4 格缺口。旧逻辑在检测候选框时会执行
#      `np.all(g[r0, c0:c1+1] == col)`，顶部出现两个非 `col` 像素后立即否决；
#      新逻辑则允许“恰有一条边存在一段连续 1-2 格缺口，另外三边完整”，因此同一对象会被接受。
#    - 题内训练/测试精灵和输出仍然是框内 `(S-2)x(S-2)` 小图，不因为缺口而改变；缺口只影响
#      “识别哪里是框”这一步。
# b) 复杂度校准
#    - 若允许多条边缺口、角缺失或断裂不连续，会让“这到底还是不是框”变得含糊，且容易与背景噪声冲突。
#      因此限制为：单边、单段、宽度 1-2、角不缺失。
#    - 为了让 D6 的视觉变化明显，每个样例中的四个框都强制带缺口，并让缺口边方向覆盖
#      top/right/bottom/left 四种方位的一个排列；这样不会只偶尔出现“看起来像原题”的完整框。
# c) 隐式假设审查（旧 `main()` 中与 D6 相关的部分）
#    - 旧 `find_square_frames()` 的四个 `np.all(edge == col)` 检查依赖“每条边完全闭合”；
#      在变体中这个前提失效，必须改成逐边判定“完整”或“带一个连续小缺口”。
#    - 旧代码默认左上角颜色就是候选框颜色。这里仍然成立，因为四个角都保留，且生成器保证边框颜色非 0。
#    - 旧代码对候选框按“大优先、互不重叠”筛选，这个前提仍然安全，因此保留。
#    - 其余训练对匹配、模板推断和模板应用逻辑只依赖框内内容，不依赖边框是否闭合，因此不改。
# d) 放置合法性
#    - 生成器继续按象限放置四个同尺寸正方形框，框之间不重叠、不越界。
#    - 绘制时先把整块方框区域清成 0，再画完整边框，最后把选中的 1-2 个边框像素清回 0，
#      于是缺口呈现为“边框向黑色内部开口”，不会把背景条纹引入框内。
#    - 验证时要求新检测器找到恰好 4 个缺口框，并且旧版完整框检测器找不到 4 个完整框，避免退化。
# e) 视觉可观察性
#    - 解题者肉眼最先看到的是四个“缺一小段边”的方框，而不是完整方框，因此 D6 的变化显著可见。
#    - 缺口位置不参与颜色映射或输出规则，只是检测难点；内部规则仍由训练对决定，避免额外隐藏分支。
# f) 可视化整洁度
#    - 边框颜色与背景/精灵颜色集合严格分离，缺口附近只有边框色与黑色内部，视觉上边界清楚。
#    - 缺口不允许吞掉角点，避免框形轮廓坍塌；背景噪声不使用边框颜色，减少伪框。
# g) 颜色变化可推断性
#    - 本变体不新增颜色变换规则；颜色仍只服务于原题的模板推断。
#    - 生成器保持训练对和测试精灵颜色集合的结构约束，确保人类仍能从少量样例中看出 A:B::C:?。
# h) 新 `main()` 的输入假设复查
#    - 假设每个真实框恰有一条边带一个连续 1-2 像素缺口，角完整：`generate_input()` 强制满足。
#    - 假设边框颜色非 0 且不出现在背景/精灵里：`_choose_frame_and_bg()` 与精灵调色盘选择保证。
#    - 假设四个框同尺寸且互不重叠：象限布局与 `validate_scene()` 保证。
#    - 假设三种非空内图中恰好两者共享相同非零颜色集合：`_build_sprites()` 保证。

import numpy as np
import random
from arc_utils import colors_present, draw_line, paste


GAP_SIDES = ("top", "right", "bottom", "left")


def _edge_gap_info(edge: np.ndarray) -> tuple[int, int] | str | None:
    """Return 'full' for a complete edge, (start,width) for a valid gap, else None."""
    if edge[0] == 0 or edge[-1] == 0:
        return None
    missing = np.where(edge == 0)[0]
    if missing.size == 0:
        return "full"
    if missing.size not in (1, 2):
        return None
    if missing[0] == 0 or missing[-1] == len(edge) - 1:
        return None
    if int(missing[-1] - missing[0] + 1) != int(missing.size):
        return None
    return int(missing[0]), int(missing.size)


def find_complete_square_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """Original detector: keep for validation so the generator can reject degenerate samples."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for size in range(4, min(h, w) + 1):
        for r0 in range(0, h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(0, w - size + 1):
                c1 = c0 + size - 1
                col = int(g[r0, c0])
                if col == 0:
                    continue
                if int(g[r0, c1]) != col or int(g[r1, c0]) != col or int(g[r1, c1]) != col:
                    continue
                if not (
                    np.all(g[r0, c0 : c1 + 1] == col)
                    and np.all(g[r1, c0 : c1 + 1] == col)
                    and np.all(g[r0 : r1 + 1, c0] == col)
                    and np.all(g[r0 : r1 + 1, c1] == col)
                ):
                    continue
                cands.append((size, col, r0, c0, r1, c1))
    cands.sort(key=lambda t: (-t[0], t[2], t[3]))
    picked = []
    for cand in cands:
        _, _, r0, c0, r1, c1 = cand
        overlaps = False
        for _, _, pr0, pc0, pr1, pc1 in picked:
            if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                overlaps = True
                break
        if not overlaps:
            picked.append(cand)
    return picked


def find_gap_square_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int, str, int, int]]:
    """
    Detect square frames whose border is complete except for one contiguous 1-2 cell gap on one side.

    Returns tuples:
    (size, color, r0, c0, r1, c1, gap_side, gap_start_along_side, gap_width)
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for size in range(4, min(h, w) + 1):
        for r0 in range(0, h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(0, w - size + 1):
                c1 = c0 + size - 1
                col = int(g[r0, c0])
                if col == 0:
                    continue
                if int(g[r0, c1]) != col or int(g[r1, c0]) != col or int(g[r1, c1]) != col:
                    continue
                edges = {
                    "top": (g[r0, c0 : c1 + 1] == col).astype(int),
                    "right": (g[r0 : r1 + 1, c1] == col).astype(int),
                    "bottom": (g[r1, c0 : c1 + 1] == col).astype(int),
                    "left": (g[r0 : r1 + 1, c0] == col).astype(int),
                }
                gap_info = []
                ok = True
                for side in GAP_SIDES:
                    info = _edge_gap_info(edges[side])
                    if info is None:
                        ok = False
                        break
                    if info != "full":
                        gap_info.append((side, info[0], info[1]))
                if not ok or len(gap_info) != 1:
                    continue
                gap_side, gap_start, gap_width = gap_info[0]
                cands.append((size, col, r0, c0, r1, c1, gap_side, gap_start, gap_width))
    cands.sort(key=lambda t: (-t[0], t[2], t[3]))
    picked = []
    for cand in cands:
        _, _, r0, c0, r1, c1, _, _, _ = cand
        overlaps = False
        for _, _, pr0, pc0, pr1, pc1, _, _, _ in picked:
            if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                overlaps = True
                break
        if not overlaps:
            picked.append(cand)
        if len(picked) >= 6:
            break
    return picked


def stable_colors(src: np.ndarray, dst: np.ndarray) -> set[int]:
    st = set()
    for c in set(int(x) for x in np.unique(src)):
        if c == 0:
            continue
        if np.array_equal(src == c, dst == c):
            st.add(c)
    return st


def pick_var_color(x: np.ndarray, st: set[int]) -> int | None:
    vals = [int(v) for v in np.unique(x) if int(v) != 0 and int(v) not in st]
    if not vals:
        return None
    best = None
    for v in vals:
        cnt = int(np.sum(x == v))
        if best is None or cnt > best[0]:
            best = (cnt, v)
    return best[1]


def symmetries(arr: np.ndarray) -> list[np.ndarray]:
    """All 8 dihedral symmetries of arr (idx 0 = identity, 1 = fliplr, ...)."""
    outs = []
    for k in range(4):
        r = np.rot90(arr, k)
        outs.append(r)
        outs.append(np.fliplr(r))
    return outs


def _bincount_colors(values: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for v in values.tolist():
        if v == 0:
            continue
        out[v] = out.get(v, 0) + 1
    return out


def sym_fill_base(arr: np.ndarray) -> np.ndarray | None:
    y = np.zeros_like(arr)
    for t in (lambda z: z, np.fliplr, np.flipud, lambda z: np.flipud(np.fliplr(z))):
        zz = t(arr)
        rr, cc = np.where(zz != 0)
        for r, c in zip(rr.tolist(), cc.tolist()):
            val = int(zz[r, c])
            if y[r, c] != 0 and int(y[r, c]) != val:
                return None
            y[r, c] = val
    return y


def infer_template(src: np.ndarray, dst: np.ndarray):
    st = stable_colors(src, dst)
    v = pick_var_color(dst, st) or pick_var_color(src, st)

    if np.array_equal(dst == 0, src != 0):
        nz = np.argwhere(dst != 0)
        if nz.size > 0:
            even_cols = set()
            odd_cols = set()
            for r, c in nz:
                col = int(dst[int(r), int(c)])
                if ((int(r) + int(c)) & 1) == 0:
                    even_cols.add(col)
                else:
                    odd_cols.add(col)
            if len(even_cols) == 1 and len(odd_cols) == 1:
                return ("comp_checker", {"stable": st})

    sym_src = sym_fill_base(src)
    if sym_src is not None:
        if np.all(dst[sym_src != 0] == sym_src[sym_src != 0]):
            fill_cols = sorted(set(int(x) for x in dst[sym_src == 0].ravel() if int(x) != 0))
            if len(fill_cols) <= 1:
                fill_col = fill_cols[0] if fill_cols else None
                ok = True
                if fill_col is not None:
                    bad = (sym_src == 0) & (dst != 0) & (dst != fill_col)
                    if np.any(bad):
                        ok = False
                if ok:
                    keep0 = (sym_src == 0) & (dst == 0)
                    return ("sym_fill", {"stable": st, "keep0": keep0})

    if v is not None:
        pred = src.copy()
        m0 = src == 0
        mv = src == v
        pred[m0] = v
        pred[mv] = 0
        for c in st:
            pred[dst == c] = c
        if np.array_equal(pred, dst):
            return ("invert0", {"stable": st})

    for idx, var in enumerate(symmetries(src)):
        if np.array_equal(var, dst):
            return ("sym", {"idx": idx})

    if v is not None:
        coords = np.argwhere(src == v)
        s = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(src, dtype=bool)
        for r, c in s:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in s:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        pred = src.copy()
        pred[src == v] = 0
        pred[keep] = v
        if np.array_equal(pred, dst):
            return ("endpoints", {"stable": st})

        pred = src.copy()
        h, w = pred.shape
        stable_mask = np.zeros_like(pred, dtype=bool)
        for c in st:
            stable_mask |= pred == c
        for r in range(h):
            nz = np.where(pred[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if pred[r, cc] == 0 and not stable_mask[r, cc]:
                        pred[r, cc] = v
        for c in range(w):
            nz = np.where(pred[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if pred[rr, c] == 0 and not stable_mask[rr, c]:
                        pred[rr, c] = v
        if np.array_equal(pred, dst):
            return ("fill", {"stable": st})

    return None


def apply_template(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    if kind == "sym":
        return symmetries(x)[params["idx"]].copy()

    st = set(params.get("stable", set()))
    v = pick_var_color(x, st)

    if kind == "comp_checker":
        h, w = x.shape
        rr, cc = np.indices((h, w))
        parity = (rr + cc) & 1
        even_mask = (x != 0) & (parity == 0)
        odd_mask = (x != 0) & (parity == 1)
        even_counts = _bincount_colors(x[even_mask])
        odd_counts = _bincount_colors(x[odd_mask])
        if even_counts:
            c_even = max(even_counts.items(), key=lambda t: t[1])[0]
        elif odd_counts:
            c_even = max(odd_counts.items(), key=lambda t: t[1])[0]
        else:
            return x.copy()
        c_odd = max(odd_counts.items(), key=lambda t: t[1])[0] if odd_counts else c_even
        y = np.where(parity == 0, c_even, c_odd)
        y[x != 0] = 0
        return y.astype(x.dtype)

    if kind == "sym_fill":
        h, w = x.shape
        nz = np.argwhere(x != 0)
        if nz.size == 0:
            return x.copy()
        best = None
        counts = {}
        for r, c in nz:
            col = int(x[int(r), int(c)])
            d = min(int(r), int(c), h - 1 - int(r), w - 1 - int(c))
            counts[col] = counts.get(col, 0) + 1
            if best is None or d > best[0] or (d == best[0] and counts[col] > best[1]):
                best = (d, counts[col], col)
        fill_col = best[2]
        y = sym_fill_base(x)
        if y is None:
            return x.copy()
        keep0 = params.get("keep0", None)
        if keep0 is None or keep0.shape != y.shape:
            keep0 = np.zeros_like(y, dtype=bool)
        y[(y == 0) & ~keep0] = fill_col
        return y

    if v is None:
        return x.copy()

    if kind == "invert0":
        y = x.copy()
        m0 = x == 0
        mv = x == v
        y[m0] = v
        y[mv] = 0
        return y

    if kind == "endpoints":
        coords = np.argwhere(x == v)
        s = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(x, dtype=bool)
        for r, c in s:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in s:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        y = x.copy()
        y[x == v] = 0
        y[keep] = v
        return y

    if kind == "fill":
        y = x.copy()
        h, w = y.shape
        stable_mask = np.zeros_like(y, dtype=bool)
        for c in st:
            stable_mask |= y == c
        for r in range(h):
            nz = np.where(y[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if y[r, cc] == 0 and not stable_mask[r, cc]:
                        y[r, cc] = v
        for c in range(w):
            nz = np.where(y[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if y[rr, c] == 0 and not stable_mask[rr, c]:
                        y[rr, c] = v
        return y

    return x.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a sprite transformation from a training pair enclosed by broken square frames, then solve the third framed sprite.

    Input structure:
    - The grid contains four non-overlapping square frames of one color; each frame is almost complete, with exactly one 1-2 cell gap on a single side.
    - Every frame surrounds a black interior containing a small sprite, and one interior is entirely blank.
    - Among the three non-blank interiors, two share the same non-zero color set and form the training pair; the remaining one is the test sprite.

    Rule:
    - Detect the gapped frames despite the missing border segment, extract their interiors, and identify the training pair by matching interior palettes.
    - Infer which visual operation turns the source training sprite into the destination training sprite.
    - Apply that same operation to the test sprite and return only the transformed interior.

    Key insight:
    - The missing border cells change frame recognition, not the analogy itself: once the near-square outlines are found, the task is still an A:B::C:? transformation inside the frames.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_gap_square_frames(g)
    if len(frames) < 3:
        return g.copy()

    inners = []
    for size, col, r0, c0, r1, c1, gap_side, gap_start, gap_width in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inners.append(((r0, c0, r1, c1, gap_side, gap_start, gap_width), inner))

    blank_idx = None
    for i, (_, inner) in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for _, inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = []
    for i in nonblank:
        inner = inners[i][1]
        nz = sorted(set(int(x) for x in inner.ravel() if int(x) != 0))
        sigs.append((i, tuple(nz)))

    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        counts = [(i, int(np.count_nonzero(inners[i][1]))) for i in nonblank]
        counts.sort(key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx][1].copy()
    test_idx = remaining[0]

    src_a = inners[pair[0]][1]
    src_b = inners[pair[1]][1]
    test_inner = inners[test_idx][1]

    best = None
    for src, dst in ((src_a, src_b), (src_b, src_a)):
        tpl = infer_template(src, dst)
        if tpl is None:
            continue
        kind, params = tpl
        st = set(params.get("stable", set()))
        v_src = pick_var_color(src, st)
        v_dst = pick_var_color(dst, st)
        score = 1
        if kind == "fill" and v_src is not None and v_dst is not None:
            if int(np.sum(dst == v_dst)) >= int(np.sum(src == v_src)):
                score += 2
        if kind == "endpoints" and v_src is not None and v_dst is not None:
            if int(np.sum(dst == v_dst)) <= int(np.sum(src == v_src)):
                score += 2
        src_n = int(np.sum((src != 0) & ~np.isin(src, list(st))))
        test_n = int(np.sum((test_inner != 0) & ~np.isin(test_inner, list(st))))
        score2 = score * 100 - abs(src_n - test_n)
        if best is None or score2 > best[0]:
            best = (score2, kind, params)

    if best is None:
        return test_inner.copy()

    _, kind, params = best
    return apply_template(kind, params, test_inner)


# --------------- generator ---------------

# ------- CFG: constants governing sampling distribution -------
# Canvas
H = W = 30

# Palette pools
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

# Frame layout
FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8
GAP_WIDTH_CHOICES = (1, 2)

# Rule set
RULES = ["endpoints", "fill", "sym_fill", "comp_checker", "invert0", "sym"]

# Sprite pixel-count bounds
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32

# sym_fill: probability to reserve the (0,0) orbit as forced-zero
SYMFILL_KEEP0_PROB = 0.4

# Rule-specific build bounds
FILL_RECT_AREA_MIN, FILL_RECT_AREA_MAX = 5, 25
ENDPOINTS_LINE_LEN_MIN = 3
COMP_CHECKER_M_MAX = 10
INVERT0_STABLE_MAX = 3
INVERT0_VAR_MAX = 12
SYM_BUILD_RETRIES = 30
SPARSE_COUNT_MAX = 10

# Outer retry loop
GEN_MAX_ATTEMPTS = 300


def orbit_cells(n: int, r: int, c: int) -> list[tuple[int, int]]:
    cells = {
        (r, c),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (n - 1 - r, n - 1 - c),
    }
    return sorted(cells)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def build_fill_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v = colors[0]
    dst = np.zeros((n, n), dtype=int)
    while True:
        h = rng.randint(2, n)
        w = rng.randint(2, n)
        if FILL_RECT_AREA_MIN <= h * w <= FILL_RECT_AREA_MAX:
            break
    r0 = rng.randint(0, n - h)
    c0 = rng.randint(0, n - w)
    dst[r0 : r0 + h, c0 : c0 + w] = v
    src = np.zeros_like(dst)
    src[r0, c0] = v
    src[r0, c0 + w - 1] = v
    src[r0 + h - 1, c0] = v
    src[r0 + h - 1, c0 + w - 1] = v
    return src, dst


def build_endpoints_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v = colors[0]
    src = np.zeros((n, n), dtype=int)
    dst = np.zeros_like(src)
    row_pairs = [(0, n - 1)]
    if n >= 5:
        row_pairs.extend([(0, n - 2), (1, n - 1)])
    r1, r2 = rng.choice(row_pairs)
    len1 = rng.randint(ENDPOINTS_LINE_LEN_MIN, n)
    len2 = rng.randint(ENDPOINTS_LINE_LEN_MIN, n)
    c1 = rng.randint(0, n - len1)
    c2 = rng.randint(0, n - len2)
    draw_line(src, r1, c1, end_r=r1, end_c=c1 + len1 - 1, color=v)
    draw_line(src, r2, c2, end_r=r2, end_c=c2 + len2 - 1, color=v)
    dst[r1, c1] = v
    dst[r1, c1 + len1 - 1] = v
    dst[r2, c2] = v
    dst[r2, c2 + len2 - 1] = v
    return src, dst


def build_comp_checker_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if len(colors) == 1:
        even_color = odd_color = colors[0]
    else:
        even_color, odd_color = colors
    min_src = 4 if n == 6 else 3
    m = rng.randint(min_src, min(COMP_CHECKER_M_MAX, n * n - 3))
    all_cells = [(r, c) for r in range(n) for c in range(n)]
    while True:
        chosen = rng.sample(all_cells, m)
        parities = {(r + c) & 1 for r, c in chosen}
        if len(colors) == 1 or len(parities) == 2:
            break
    rr, cc = np.indices((n, n))
    full = np.where(((rr + cc) & 1) == 0, even_color, odd_color).astype(int)
    hole_mask = np.zeros((n, n), dtype=bool)
    for r, c in chosen:
        hole_mask[r, c] = True
    src = np.where(hole_mask, full, 0)
    dst = np.where(hole_mask, 0, full)
    return src, dst


def build_invert0_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v, s = colors
    all_cells = [(r, c) for r in range(n) for c in range(n)]
    stable_n = rng.randint(1, min(INVERT0_STABLE_MAX, n))
    stable_cells = set(rng.sample(all_cells, stable_n))
    remaining = [cell for cell in all_cells if cell not in stable_cells]
    var_n = rng.randint(2, max(2, min(len(remaining) - 2, INVERT0_VAR_MAX)))
    var_cells = set(rng.sample(remaining, var_n))
    dst = np.zeros((n, n), dtype=int)
    for r, c in stable_cells:
        dst[r, c] = s
    for r, c in var_cells:
        dst[r, c] = v
    src = dst.copy()
    src[dst == 0] = v
    src[dst == v] = 0
    return src, dst


def inverse_symmetry(arr: np.ndarray, idx: int) -> np.ndarray:
    if idx == 0:
        return arr.copy()
    if idx == 1:
        return np.fliplr(arr)
    if idx == 2:
        return np.rot90(arr, -1)
    if idx == 3:
        return np.rot90(np.fliplr(arr), -1)
    if idx == 4:
        return np.rot90(arr, 2)
    if idx == 5:
        return np.flipud(arr)
    if idx == 6:
        return np.rot90(arr, 1)
    if idx == 7:
        return np.rot90(np.fliplr(arr), 1)
    raise ValueError(idx)


def random_sparse_pattern(n: int, colors: list[int], rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    count = rng.randint(3, min(SPARSE_COUNT_MAX, n * n - 1))
    cells = rng.sample([(r, c) for r in range(n) for c in range(n)], count)
    used = [colors[0]] * count
    if len(colors) == 2:
        for i in range(count):
            used[i] = colors[rng.randint(0, 1)]
        used[0] = colors[0]
        used[1] = colors[1]
    for (r, c), col in zip(cells, used):
        arr[r, c] = col
    return arr


def build_sym_pair(n: int, colors: list[int], rng: random.Random, idx: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(SYM_BUILD_RETRIES):
        dst = random_sparse_pattern(n, colors, rng)
        src = inverse_symmetry(dst, idx)
        if not np.array_equal(src, dst):
            return src, dst
    raise ValueError("failed to build sym pair")


def build_sym_fill_pair(
    n: int,
    colors: list[int],
    rng: random.Random,
    keep0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fill_col, accent_col = colors
    src = np.zeros((n, n), dtype=int)
    orbits = []
    seen = set()
    for r in range(n):
        for c in range(n):
            orb = tuple(orbit_cells(n, r, c))
            if orb not in seen:
                seen.add(orb)
                orbits.append(list(orb))
    orbits.sort(
        key=lambda orb: (
            -max(min(r, c, n - 1 - r, n - 1 - c) for r, c in orb),
            -len(orb),
            orb[0],
        )
    )
    fill_orbit = next(orb for orb in orbits if not any(keep0[r, c] for r, c in orb))
    src[fill_orbit[0]] = fill_col
    accent_orbits = [orb for orb in orbits if orb != fill_orbit and not any(keep0[r, c] for r, c in orb)]
    picks = []
    for orb in accent_orbits:
        if len(orb) >= 2:
            picks.append(orb)
        if len(picks) >= 2:
            break
    if len(picks) < 2:
        raise ValueError("not enough accent orbits")
    for orb in picks:
        src[orb[0]] = accent_col
    dst = apply_template("sym_fill", {"keep0": keep0}, src)
    return src, dst


def choose_rule_setup(n: int, rng: random.Random) -> tuple[str, dict, int]:
    kind = rng.choice(RULES)
    params: dict = {}
    if kind == "invert0":
        color_count = 2
    elif kind == "sym_fill":
        color_count = 2
        keep0 = np.zeros((n, n), dtype=bool)
        if n == 6 or rng.random() < SYMFILL_KEEP0_PROB:
            for r, c in orbit_cells(n, 0, 0):
                keep0[r, c] = True
        params["keep0"] = keep0
    elif kind == "sym":
        color_count = rng.choice([1, 2])
        params["idx"] = rng.randint(1, 7)
    elif kind == "comp_checker":
        color_count = rng.choice([1, 2])
    else:
        color_count = 1
    return kind, params, color_count


def build_training_pair(kind: str, params: dict, n: int, colors: list[int], rng: random.Random):
    if kind == "fill":
        return build_fill_pair(n, colors, rng)
    if kind == "endpoints":
        return build_endpoints_pair(n, colors, rng)
    if kind == "comp_checker":
        return build_comp_checker_pair(n, colors, rng)
    if kind == "invert0":
        return build_invert0_pair(n, colors, rng)
    if kind == "sym":
        return build_sym_pair(n, colors, rng, params["idx"])
    if kind == "sym_fill":
        return build_sym_fill_pair(n, colors, rng, params["keep0"])
    raise ValueError(kind)


def quadrant_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    low, high = QUAD_LOW_MIN, QUAD_LOW_MAX
    bottom_low = H - s - QUAD_BOTTOM_MARGIN
    bottom_high = H - s - 1
    return [
        (rng.randint(low, high), rng.randint(low, high)),
        (rng.randint(low, high), rng.randint(bottom_low, bottom_high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(low, high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(bottom_low, bottom_high)),
    ]


def render_background(
    frame_boxes: list[tuple[int, int, int, int]],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    period = len(bg_palette)
    direction = rng.choice([1, -1])
    rr, cc = np.indices((H, W))
    idx = (rr + direction * cc) % period
    grid = np.asarray(bg_palette, dtype=int)[idx]

    reserved = np.zeros((H, W), dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    bg_cells = [(r, c) for r, c in zip(*np.where(~reserved))]
    lo, hi = BG_PERTURB_FRACTION
    perturb_n = int(round(len(bg_cells) * rng.uniform(lo, hi)))
    noise_choices = list(bg_palette) + [c for c in BG_EXTRA_NOISE if c not in bg_palette]
    for r, c in rng.sample(bg_cells, perturb_n):
        grid[r, c] = rng.choice(noise_choices)
    return grid


def choose_gap_start(s: int, width: int, rng: random.Random) -> int:
    starts = list(range(1, s - width))
    centered = [start for start in starts if start >= 2 and start + width <= s - 2]
    if centered:
        starts = centered
    return rng.choice(starts)


def erase_gap(grid: np.ndarray, r0: int, c0: int, s: int, gap_side: str, gap_start: int, gap_width: int) -> None:
    if gap_side == "top":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_width] = 0
    elif gap_side == "bottom":
        grid[r0 + s - 1, c0 + gap_start : c0 + gap_start + gap_width] = 0
    elif gap_side == "left":
        grid[r0 + gap_start : r0 + gap_start + gap_width, c0] = 0
    elif gap_side == "right":
        grid[r0 + gap_start : r0 + gap_start + gap_width, c0 + s - 1] = 0
    else:
        raise ValueError(gap_side)


def draw_gapped_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    s: int,
    frame_color: int,
    sprite: np.ndarray,
    gap_side: str,
    gap_start: int,
    gap_width: int,
) -> None:
    r1, c1 = r0 + s - 1, c0 + s - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    erase_gap(grid, r0, c0, s, gap_side, gap_start, gap_width)
    paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_gap_square_frames(grid)
    if len(frames) != 4:
        return False
    if len(find_complete_square_frames(grid)) != 0:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    gap_sides = {gap_side for _, _, _, _, _, _, gap_side, _, _ in frames}
    if gap_sides != set(GAP_SIDES):
        return False
    widths = [gap_width for _, _, _, _, _, _, _, _, gap_width in frames]
    if any(width not in GAP_WIDTH_CHOICES for width in widths):
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1, _, _, _ in frames]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    nonblank_sigs = [tuple(colors_present(inner, exclude={0})) for inner in inners]
    nonblank_sigs = [s for s in nonblank_sigs if s]
    dup_count = sum(
        1 for i in range(len(nonblank_sigs)) for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False
    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(kind: str, setup_params: dict, n: int, p1: list[int], p2: list[int], rng: random.Random):
    train_src, train_dst = build_training_pair(kind, setup_params, n, p1, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred

    test_src, _unused_test_dst = build_training_pair(kind, setup_params, n, p2, rng)
    test_dst = apply_template(inferred_kind, inferred_params, test_src)

    if np.array_equal(test_dst, test_src):
        return None
    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nz(arr):
            return None
    train_sig = tuple(colors_present(train_src, exclude={0}))
    if train_sig != tuple(colors_present(train_dst, exclude={0})):
        return None
    if tuple(colors_present(test_src, exclude={0})) == train_sig:
        return None
    return train_src, train_dst, test_src, test_dst


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used and c != 0])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def _assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int]:
    order = sorted(range(4), key=lambda i: positions[i])
    train_slots = sorted(rng.sample(order, 2), key=order.index)
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    return train_slots, test_slot


def _assign_gaps(s: int, rng: random.Random) -> list[tuple[str, int, int]]:
    gaps = []
    for side in rng.sample(list(GAP_SIDES), len(GAP_SIDES)):
        width = rng.choice(GAP_WIDTH_CHOICES)
        start = choose_gap_start(s, width, rng)
        gaps.append((side, start, width))
    return gaps


def _render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    bg_palette: list[int],
    gaps: list[tuple[str, int, int]],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite, (gap_side, gap_start, gap_width) in zip(positions, sprites, gaps):
        draw_gapped_frame(grid, r0, c0, s, frame_color, sprite, gap_side, gap_start, gap_width)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    kind, setup_params, color_count = choose_rule_setup(n, rng)
    p1 = choose_palette(rng, color_count, set())
    p2 = choose_palette(rng, color_count, set(p1))

    built = _build_sprites(kind, setup_params, n, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)

    positions = quadrant_positions(s, rng)
    train_slots, test_slot = _assign_slots(positions, rng)
    gaps = _assign_gaps(s, rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    grid = _render_scene(s, positions, sprites, frame_color, bg_palette, gaps, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
