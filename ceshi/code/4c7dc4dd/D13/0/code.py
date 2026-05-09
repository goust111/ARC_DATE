# Variant of 4c7dc4dd
# Dimension: D13 — 精灵颜色数量约束 (meta)
# Change: 每个精灵使用 1-2 种非背景色，训练对和测试精灵的颜色集不相交... → 三色精灵

# Design analysis for the D13 variant
# ----------------------------------
# a) 具体化与行为追踪
#    - 仍然是 30x30 画布、4 个方框、1 个空框、1 对训练精灵、1 个测试精灵。
#    - 变化点不在框架布局，而在每个非空精灵内部：现在每个精灵恰好使用 3 种非背景色。
#      其中最少像素的颜色是稳定色；其余两种都是变量色，主变量色像素更多，次变量色像素更少。
#    - 训练对中的稳定色位置完全不变；两种变量色都按同一条规则变化。测试精灵也遵循同样的
#      “稳定色最少、主变量色最多、次变量色居中”的可见角色分工，但颜色 id 与训练对完全不相交。
#    - 小例子追踪（fill）：
#      原始逻辑：只有 1 个变量色，四个角点被填成整块。
#      变体逻辑：主变量色用“4 角 + 1 个额外点”，次变量色用“4 角”，两者各自被同一 fill 规则扩展；
#      稳定色的 2 个像素原封不动。solver 不再能假设“唯一变量色”，必须把两个变量色分开处理。
#
# b) 复杂度校准
#    - 直接允许任意三色会让角色识别失控，所以这里把角色约束压缩成少量、可见、重复的规律：
#      稳定色总是最小计数；主变量色计数最大；次变量色计数居中。
#    - 为了避免三色版本在视觉上退化回原题，两个变量色都会参与变换，而不是仅仅添加一个静态装饰色。
#    - 只保留 4 类容易承载“双变量色同规则变化”的规则：sym / fill / endpoints / comp_checker。
#      原题中的 invert0 与 sym_fill 在三色下容易出现角色冲突或覆盖歧义，因此不再采样。
#
# c) 隐式假设审查
#    - 原 main() 用 `pick_var_color()` 默认“非稳定色只有一个”。三色变体下这会把两个变量色误并成一个，
#      导致 fill/endpoints/sym 都错判，所以改成按像素数排序提取两个变量角色。
#    - 原 apply_template() 会把训练对里的稳定色 id 直接带到测试精灵；由于训练/测试颜色集不相交，
#      这在三色下完全失效。现在模板只存规则参数，不存训练颜色 id；测试时重新从测试精灵自身推断角色。
#    - 原 fill/endpoints/comp_checker 的实现都默认“唯一可变色”。现在每个分支都要对两个变量色分别执行，
#      且稳定色只作为保留/阻挡区域。
#
# d) 放置合法性
#    - 仍然使用 output-first：先构造训练源/目标和测试源，再渲染到四个框里。
#    - 生成器为每条规则显式保证：两种变量色的工作区域不重叠，稳定色插入在训练源与训练目标都为空的位置，
#      因此不会覆盖规则结果，也不会让 solver 看到“额外变化”。
#    - 4 个框仍按四象限采样并检测不重叠；背景先画、框后盖，内部先清成 0 再贴精灵。
#
# e) 视觉可观察性
#    - 颜色角色全部由像素数量和训练对中的位置变化决定，不依赖颜色 id。
#    - 训练对里能肉眼看到：最小计数的小标记保持不动，较大的两种颜色一起按同类规则变化。
#      测试框里也有同样的“少/中/多”三档颜色分工，因此人可以把角色迁移过去。
#
# f) 可视化整洁度
#    - 两个变量色使用互不重叠的局部结构；稳定色只占 2 个像素，避免把小内框塞满。
#    - 为了让三色关系清楚，去掉最小内框尺寸 n=4，改用 n=5/6。
#
# g) 颜色变化可推断性
#    - 没有任意颜色置换。颜色只承担角色：稳定 / 主变量 / 次变量。
#    - comp_checker 中两种变量色分别对应棋盘两种奇偶位；其余规则中两种变量色各自独立地执行同一空间规则。
#
# h) 量化的变体核心行为
#    - 每个非空精灵恰好 3 色。
#    - 训练源→训练目标中，发生位置变化的非背景颜色数恰好为 2。
#    - 测试源里 3 色计数严格有序，保证 solver 可重新识别角色而不是沿用训练颜色 id。
#
# i) 新 main() 的额外假设与 generator 保证
#    - 假设每个非空精灵恰好 3 色且像素数互异；generator 强制满足。
#    - 假设稳定色是最小计数且训练对里其位置完全一致；generator 强制满足。
#    - 假设两个变量色的规则应用后互不覆盖；各规则 builder 都按不重叠区域构造并用 main() 回验。

import numpy as np
import random
from arc_utils import colors_present, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a framed-sprite analogy rule from one training source/target pair, then solve the framed test sprite.

    Input structure:
    - The canvas contains four same-size hollow square frames over a noisy background.
    - One frame interior is blank; the other three interiors are sprites that each use exactly three non-black colors.
    - The training pair shares the same three-color palette, while the test sprite uses a disjoint three-color palette.

    Rule:
    - Match the training pair by identical interior color sets.
    - Within that pair, treat the smallest-count color as stable and the other two colors as the primary and secondary variables.
    - Infer which shared transformation moves both variable colors while leaving the stable color fixed, then apply it to the test sprite after re-identifying its three roles from its own color counts.

    Key insight:
    - The puzzle is still an A:B::C:? analogy, but now the solver must reason about color roles rather than a single moving color: one color stays fixed, while two distinct variable colors undergo the same spatial rule.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    def find_square_frames():
        cands = []
        for size in range(4, min(H, W) + 1):
            for r0 in range(0, H - size + 1):
                r1 = r0 + size - 1
                for c0 in range(0, W - size + 1):
                    c1 = c0 + size - 1
                    col = int(g[r0, c0])
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
        for size, col, r0, c0, r1, c1 in cands:
            overlaps = False
            for _, _, pr0, pc0, pr1, pc1 in picked:
                if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                    overlaps = True
                    break
            if not overlaps:
                picked.append((size, col, r0, c0, r1, c1))
            if len(picked) >= 6:
                break
        return picked

    def nonzero_counts(arr):
        return {int(c): int(np.sum(arr == c)) for c in np.unique(arr) if int(c) != 0}

    def symmetries(arr):
        outs = []
        for k in range(4):
            r = np.rot90(arr, k)
            outs.append(r)
            outs.append(np.fliplr(r))
        return outs

    def endpoint_mask(mask):
        coords = np.argwhere(mask)
        S = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(mask, dtype=bool)
        for r, c in S:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in S:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        return keep

    def fill_mask(mask, blocked):
        y = mask.copy()
        h, w = y.shape
        for r in range(h):
            nz = np.where(y[r, :])[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if not blocked[r, cc]:
                        y[r, cc] = True
        for c in range(w):
            nz = np.where(y[:, c])[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if not blocked[rr, c]:
                        y[rr, c] = True
        return y

    def place_mask(base, mask, color):
        clash = mask & (base != 0) & (base != color)
        if np.any(clash):
            return None
        out = base.copy()
        out[mask] = color
        return out

    def extract_roles_from_sprite(arr):
        counts = nonzero_counts(arr)
        if len(counts) != 3:
            return None
        items = sorted((cnt, color) for color, cnt in counts.items())
        if len({cnt for cnt, _ in items}) != 3:
            return None
        stable = items[0][1]
        variables = [items[2][1], items[1][1]]
        return stable, variables

    def extract_roles_from_pair(src, dst):
        counts_src = nonzero_counts(src)
        counts_dst = nonzero_counts(dst)
        colors = sorted(set(counts_src) | set(counts_dst))
        if len(colors) != 3:
            return None
        stable = []
        for color in colors:
            if np.array_equal(src == color, dst == color):
                stable.append(color)
        if len(stable) != 1:
            return None
        stable = stable[0]
        if stable not in counts_src or stable not in counts_dst:
            return None
        variables = [c for c in colors if c != stable]
        ranked = sorted(
            ((counts_src.get(color, 0), color) for color in variables),
            reverse=True,
        )
        if ranked[0][0] == ranked[1][0]:
            return None
        if counts_src[stable] >= ranked[1][0]:
            return None
        return stable, [ranked[0][1], ranked[1][1]]

    def infer_template(src, dst):
        roles = extract_roles_from_pair(src, dst)
        if roles is None:
            return None
        stable, variables = roles
        stable_mask = src == stable
        if not np.array_equal(stable_mask, dst == stable):
            return None

        eligible = ~stable_mask
        src_mask = (src != 0) & eligible
        dst_mask = (dst != 0) & eligible
        rr, cc = np.indices(src.shape)
        if np.array_equal(dst_mask, eligible & ~src_mask):
            combined = np.where(src_mask, src, dst)
            even_cols = set(int(v) for v in combined[(eligible) & (((rr + cc) & 1) == 0)] if int(v) != 0)
            odd_cols = set(int(v) for v in combined[(eligible) & (((rr + cc) & 1) == 1)] if int(v) != 0)
            if len(even_cols) == 1 and len(odd_cols) == 1 and even_cols != odd_cols:
                return ("comp_checker", {})

        for idx in range(1, 8):
            pred = np.zeros_like(src)
            pred[stable_mask] = stable
            ok = True
            for color in variables:
                mask = symmetries(src == color)[idx]
                pred2 = place_mask(pred, mask, color)
                if pred2 is None:
                    ok = False
                    break
                pred = pred2
            if ok and np.array_equal(pred, dst):
                return ("sym", {"idx": idx})

        pred = np.zeros_like(src)
        pred[stable_mask] = stable
        ok = True
        for color in variables:
            pred2 = place_mask(pred, endpoint_mask(src == color), color)
            if pred2 is None:
                ok = False
                break
            pred = pred2
        if ok and np.array_equal(pred, dst):
            return ("endpoints", {})

        pred = np.zeros_like(src)
        pred[stable_mask] = stable
        blocked = stable_mask.copy()
        ok = True
        for color in variables:
            pred2 = place_mask(pred, fill_mask(src == color, blocked), color)
            if pred2 is None:
                ok = False
                break
            pred = pred2
        if ok and np.array_equal(pred, dst):
            return ("fill", {})

        return None

    def checker_parity_colors(arr, stable):
        rr, cc = np.indices(arr.shape)
        colors = [int(c) for c in np.unique(arr) if int(c) not in (0, stable)]
        if len(colors) != 2:
            return None
        mapping = {}
        for color in colors:
            parities = set(int((int(r) + int(c)) & 1) for r, c in np.argwhere(arr == color))
            if len(parities) != 1:
                return None
            parity = next(iter(parities))
            if parity in mapping:
                return None
            mapping[parity] = color
        if len(mapping) != 2:
            return None
        return mapping

    def apply_template(kind, params, x):
        roles = extract_roles_from_sprite(x)
        if roles is None:
            return x.copy()
        stable, variables = roles
        stable_mask = x == stable

        if kind == "sym":
            y = np.zeros_like(x)
            y[stable_mask] = stable
            idx = params["idx"]
            for color in variables:
                pred = place_mask(y, symmetries(x == color)[idx], color)
                if pred is None:
                    return x.copy()
                y = pred
            return y

        if kind == "endpoints":
            y = np.zeros_like(x)
            y[stable_mask] = stable
            for color in variables:
                pred = place_mask(y, endpoint_mask(x == color), color)
                if pred is None:
                    return x.copy()
                y = pred
            return y

        if kind == "fill":
            y = np.zeros_like(x)
            y[stable_mask] = stable
            blocked = stable_mask.copy()
            for color in variables:
                pred = place_mask(y, fill_mask(x == color, blocked), color)
                if pred is None:
                    return x.copy()
                y = pred
            return y

        if kind == "comp_checker":
            mapping = checker_parity_colors(x, stable)
            if mapping is None:
                return x.copy()
            rr, cc = np.indices(x.shape)
            parity = (rr + cc) & 1
            y = np.zeros_like(x)
            y[stable_mask] = stable
            eligible = ~stable_mask
            fill_even = eligible & (x == 0) & (parity == 0)
            fill_odd = eligible & (x == 0) & (parity == 1)
            y[fill_even] = mapping[0]
            y[fill_odd] = mapping[1]
            return y

        return x.copy()

    frames = find_square_frames()
    if len(frames) < 3:
        return g.copy()

    inners = []
    for _, _, r0, c0, r1, c1 in frames:
        inners.append(g[r0 + 1 : r1, c0 + 1 : c1])

    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = []
    for i in nonblank:
        sigs.append((i, tuple(sorted(int(x) for x in np.unique(inners[i]) if int(x) != 0))))

    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        counts = [(i, int(np.count_nonzero(inners[i]))) for i in nonblank]
        counts.sort(key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx].copy()
    test_idx = remaining[0]

    src_a = inners[pair[0]]
    src_b = inners[pair[1]]
    test_inner = inners[test_idx]

    best = None
    for src, dst in ((src_a, src_b), (src_b, src_a)):
        tpl = infer_template(src, dst)
        if tpl is None:
            continue
        kind, params = tpl
        roles = extract_roles_from_pair(src, dst)
        test_roles = extract_roles_from_sprite(test_inner)
        if roles is None or test_roles is None:
            continue
        stable, variables = roles
        src_var_n = sum(int(np.sum(src == color)) for color in variables)
        dst_var_n = sum(int(np.sum(dst == color)) for color in variables)
        test_stable, test_variables = test_roles
        test_var_n = sum(int(np.sum(test_inner == color)) for color in test_variables)
        score = 100 - abs(src_var_n - test_var_n)
        if kind == "fill" and dst_var_n > src_var_n:
            score += 20
        if kind == "endpoints" and dst_var_n < src_var_n:
            score += 20
        if kind == "comp_checker":
            score += 5
        if best is None or score > best[0]:
            best = (score, kind, params)

    if best is None:
        return test_inner.copy()

    _, kind, params = best
    return apply_template(kind, params, test_inner)


# --------------- generator ---------------

H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (3,)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

RULES = ["endpoints", "fill", "comp_checker", "sym"]

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 6, 32
STABLE_COUNT = 2
PRIMARY_FILL_AREA_MIN = 6
PRIMARY_FILL_AREA_MAX = 12
SECONDARY_FILL_AREA_MIN = 4
SECONDARY_FILL_AREA_MAX = 8
PRIMARY_LINE_MIN = 4
SECONDARY_LINE_MIN = 3
SYM_BUILD_RETRIES = 40
GEN_MAX_ATTEMPTS = 400


def find_square_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for size in range(4, min(h, w) + 1):
        for r0 in range(0, h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(0, w - size + 1):
                c1 = c0 + size - 1
                col = int(g[r0, c0])
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


def nonzero_counts(arr: np.ndarray) -> dict[int, int]:
    return {int(c): int(np.sum(arr == c)) for c in np.unique(arr) if int(c) != 0}


def stable_colors(src: np.ndarray, dst: np.ndarray) -> set[int]:
    st = set()
    for c in set(int(x) for x in np.unique(src)):
        if c == 0:
            continue
        if np.array_equal(src == c, dst == c):
            st.add(c)
    return st


def extract_roles_from_sprite(arr: np.ndarray):
    counts = nonzero_counts(arr)
    if len(counts) != 3:
        return None
    items = sorted((cnt, color) for color, cnt in counts.items())
    if len({cnt for cnt, _ in items}) != 3:
        return None
    stable = items[0][1]
    variables = [items[2][1], items[1][1]]
    return stable, variables


def extract_roles_from_pair(src: np.ndarray, dst: np.ndarray):
    colors = sorted(set(int(x) for x in np.unique(src)) | set(int(x) for x in np.unique(dst)))
    colors = [c for c in colors if c != 0]
    if len(colors) != 3:
        return None
    st = [c for c in colors if np.array_equal(src == c, dst == c)]
    if len(st) != 1:
        return None
    stable = st[0]
    counts_src = nonzero_counts(src)
    vars_ = [c for c in colors if c != stable]
    ranked = sorted(((counts_src[c], c) for c in vars_), reverse=True)
    if ranked[0][0] == ranked[1][0]:
        return None
    if counts_src[stable] >= ranked[1][0]:
        return None
    return stable, [ranked[0][1], ranked[1][1]]


def symmetries(arr: np.ndarray) -> list[np.ndarray]:
    outs = []
    for k in range(4):
        r = np.rot90(arr, k)
        outs.append(r)
        outs.append(np.fliplr(r))
    return outs


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


def endpoint_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    s = set((int(r), int(c)) for r, c in coords)
    keep = np.zeros_like(mask, dtype=bool)
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
    return keep


def fill_mask(mask: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    y = mask.copy()
    h, w = y.shape
    for r in range(h):
        nz = np.where(y[r, :])[0]
        if nz.size >= 2:
            cmin = int(nz.min())
            cmax = int(nz.max())
            for cc in range(cmin, cmax + 1):
                if not blocked[r, cc]:
                    y[r, cc] = True
    for c in range(w):
        nz = np.where(y[:, c])[0]
        if nz.size >= 2:
            rmin = int(nz.min())
            rmax = int(nz.max())
            for rr in range(rmin, rmax + 1):
                if not blocked[rr, c]:
                    y[rr, c] = True
    return y


def place_mask(base: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray | None:
    clash = mask & (base != 0) & (base != color)
    if np.any(clash):
        return None
    out = base.copy()
    out[mask] = color
    return out


def checker_parity_colors(arr: np.ndarray, stable: int) -> dict[int, int] | None:
    colors = [int(c) for c in np.unique(arr) if int(c) not in (0, stable)]
    if len(colors) != 2:
        return None
    mapping: dict[int, int] = {}
    for color in colors:
        parities = set(int((int(r) + int(c)) & 1) for r, c in np.argwhere(arr == color))
        if len(parities) != 1:
            return None
        parity = next(iter(parities))
        if parity in mapping:
            return None
        mapping[parity] = color
    if len(mapping) != 2:
        return None
    return mapping


def infer_template(src: np.ndarray, dst: np.ndarray):
    roles = extract_roles_from_pair(src, dst)
    if roles is None:
        return None
    stable, variables = roles
    stable_mask = src == stable
    if not np.array_equal(stable_mask, dst == stable):
        return None

    eligible = ~stable_mask
    src_mask = (src != 0) & eligible
    dst_mask = (dst != 0) & eligible
    rr, cc = np.indices(src.shape)
    if np.array_equal(dst_mask, eligible & ~src_mask):
        combined = np.where(src_mask, src, dst)
        even_cols = set(int(v) for v in combined[(eligible) & (((rr + cc) & 1) == 0)] if int(v) != 0)
        odd_cols = set(int(v) for v in combined[(eligible) & (((rr + cc) & 1) == 1)] if int(v) != 0)
        if len(even_cols) == 1 and len(odd_cols) == 1 and even_cols != odd_cols:
            return ("comp_checker", {})

    for idx in range(1, 8):
        pred = np.zeros_like(src)
        pred[stable_mask] = stable
        ok = True
        for color in variables:
            pred2 = place_mask(pred, symmetries(src == color)[idx], color)
            if pred2 is None:
                ok = False
                break
            pred = pred2
        if ok and np.array_equal(pred, dst):
            return ("sym", {"idx": idx})

    pred = np.zeros_like(src)
    pred[stable_mask] = stable
    ok = True
    for color in variables:
        pred2 = place_mask(pred, endpoint_mask(src == color), color)
        if pred2 is None:
            ok = False
            break
        pred = pred2
    if ok and np.array_equal(pred, dst):
        return ("endpoints", {})

    pred = np.zeros_like(src)
    pred[stable_mask] = stable
    blocked = stable_mask.copy()
    ok = True
    for color in variables:
        pred2 = place_mask(pred, fill_mask(src == color, blocked), color)
        if pred2 is None:
            ok = False
            break
        pred = pred2
    if ok and np.array_equal(pred, dst):
        return ("fill", {})

    return None


def apply_template(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    roles = extract_roles_from_sprite(x)
    if roles is None:
        return x.copy()
    stable, variables = roles
    stable_mask = x == stable

    if kind == "sym":
        y = np.zeros_like(x)
        y[stable_mask] = stable
        idx = params["idx"]
        for color in variables:
            pred = place_mask(y, symmetries(x == color)[idx], color)
            if pred is None:
                return x.copy()
            y = pred
        return y

    if kind == "endpoints":
        y = np.zeros_like(x)
        y[stable_mask] = stable
        for color in variables:
            pred = place_mask(y, endpoint_mask(x == color), color)
            if pred is None:
                return x.copy()
            y = pred
        return y

    if kind == "fill":
        y = np.zeros_like(x)
        y[stable_mask] = stable
        blocked = stable_mask.copy()
        for color in variables:
            pred = place_mask(y, fill_mask(x == color, blocked), color)
            if pred is None:
                return x.copy()
            y = pred
        return y

    if kind == "comp_checker":
        mapping = checker_parity_colors(x, stable)
        if mapping is None:
            return x.copy()
        rr, cc = np.indices(x.shape)
        parity = (rr + cc) & 1
        y = np.zeros_like(x)
        y[stable_mask] = stable
        eligible = ~stable_mask
        y[eligible & (x == 0) & (parity == 0)] = mapping[0]
        y[eligible & (x == 0) & (parity == 1)] = mapping[1]
        return y

    return x.copy()


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def choose_stable_cells(empty_mask: np.ndarray, rng: random.Random) -> list[tuple[int, int]]:
    cells = [(int(r), int(c)) for r, c in np.argwhere(empty_mask)]
    rng.shuffle(cells)
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            r1, c1 = cells[i]
            r2, c2 = cells[j]
            if max(abs(r1 - r2), abs(c1 - c2)) >= 2:
                return [cells[i], cells[j]]
    if len(cells) >= 2:
        return cells[:2]
    raise ValueError("not enough room for stable cells")


def add_stable_cells(src: np.ndarray, dst: np.ndarray, stable_color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    empty_mask = (src == 0) & (dst == 0)
    stable_cells = choose_stable_cells(empty_mask, rng)
    out_src = src.copy()
    out_dst = dst.copy()
    for r, c in stable_cells:
        out_src[r, c] = stable_color
        out_dst[r, c] = stable_color
    return out_src, out_dst


def random_rect(n: int, area_min: int, area_max: int, rng: random.Random, require_extra: bool) -> tuple[int, int, int, int]:
    choices = []
    for h in range(2, n + 1):
        for w in range(2, n + 1):
            area = h * w
            if area_min <= area <= area_max:
                if require_extra and area <= 4:
                    continue
                choices.append((h, w))
    if not choices:
        raise ValueError("no rectangle choices")
    h, w = rng.choice(choices)
    r0 = rng.randint(0, n - h)
    c0 = rng.randint(0, n - w)
    return r0, c0, h, w


def rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def non_corner_cell(r0: int, c0: int, h: int, w: int) -> tuple[int, int]:
    corners = {
        (r0, c0),
        (r0, c0 + w - 1),
        (r0 + h - 1, c0),
        (r0 + h - 1, c0 + w - 1),
    }
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if (r, c) not in corners:
                return r, c
    raise ValueError("rectangle has no extra cell")


def build_fill_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    stable, primary, secondary = colors
    for _ in range(100):
        p = random_rect(n, PRIMARY_FILL_AREA_MIN, PRIMARY_FILL_AREA_MAX, rng, require_extra=True)
        q = random_rect(n, SECONDARY_FILL_AREA_MIN, SECONDARY_FILL_AREA_MAX, rng, require_extra=False)
        p_cells = rect_cells(*p)
        q_cells = rect_cells(*q)
        if p_cells & q_cells:
            continue
        if len(p_cells) <= len(q_cells):
            continue
        src = np.zeros((n, n), dtype=int)
        dst = np.zeros_like(src)
        for color, rect, need_extra in ((primary, p, True), (secondary, q, False)):
            r0, c0, h, w = rect
            dst[r0 : r0 + h, c0 : c0 + w] = color
            corners = [(r0, c0), (r0, c0 + w - 1), (r0 + h - 1, c0), (r0 + h - 1, c0 + w - 1)]
            for r, c in corners:
                src[r, c] = color
            if need_extra:
                er, ec = non_corner_cell(r0, c0, h, w)
                src[er, ec] = color
        src, dst = add_stable_cells(src, dst, stable, rng)
        return src, dst
    raise ValueError("fill build failed")


def build_endpoints_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    stable, primary, secondary = colors
    for _ in range(100):
        src = np.zeros((n, n), dtype=int)
        dst = np.zeros_like(src)
        len1 = rng.randint(PRIMARY_LINE_MIN, n)
        len2 = rng.randint(SECONDARY_LINE_MIN, n - 1 if n > SECONDARY_LINE_MIN else n)
        if len1 <= len2:
            continue
        if rng.random() < 0.5:
            r1 = rng.randint(0, n - 1)
            c1 = rng.randint(0, n - len1)
            r2 = rng.randint(0, n - len2)
            c2 = rng.randint(0, n - len2)
            if abs(r1 - r2) <= 1 and not (c1 + len1 - 1 < c2 or c2 + len2 - 1 < c1):
                continue
            src[r1, c1 : c1 + len1] = primary
            src[r2, c2 : c2 + len2] = secondary
            dst[r1, c1] = primary
            dst[r1, c1 + len1 - 1] = primary
            dst[r2, c2] = secondary
            dst[r2, c2 + len2 - 1] = secondary
        else:
            c1 = rng.randint(0, n - 1)
            r1 = rng.randint(0, n - len1)
            c2 = rng.randint(0, n - 1)
            r2 = rng.randint(0, n - len2)
            if abs(c1 - c2) <= 1 and not (r1 + len1 - 1 < r2 or r2 + len2 - 1 < r1):
                continue
            src[r1 : r1 + len1, c1] = primary
            src[r2 : r2 + len2, c2] = secondary
            dst[r1, c1] = primary
            dst[r1 + len1 - 1, c1] = primary
            dst[r2, c2] = secondary
            dst[r2 + len2 - 1, c2] = secondary
        src, dst = add_stable_cells(src, dst, stable, rng)
        return src, dst
    raise ValueError("endpoints build failed")


def build_comp_checker_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    stable, primary, secondary = colors
    for _ in range(100):
        src = np.zeros((n, n), dtype=int)
        dst = np.zeros_like(src)
        stable_cells = choose_stable_cells(np.ones((n, n), dtype=bool), rng)
        stable_set = set(stable_cells)
        for r, c in stable_cells:
            src[r, c] = stable
            dst[r, c] = stable
        parity_to_color = {}
        if rng.random() < 0.5:
            parity_to_color[0] = primary
            parity_to_color[1] = secondary
        else:
            parity_to_color[0] = secondary
            parity_to_color[1] = primary
        parity_cells = {0: [], 1: []}
        for r in range(n):
            for c in range(n):
                if (r, c) in stable_set:
                    continue
                parity_cells[(r + c) & 1].append((r, c))
        src_counts = {}
        for parity, color in parity_to_color.items():
            pool = parity_cells[parity]
            if len(pool) < 5:
                break
            lo = 3
            hi = len(pool) - 1
            src_counts[color] = rng.randint(lo, hi)
        else:
            if src_counts[primary] == src_counts[secondary]:
                continue
            if src_counts[primary] < src_counts[secondary]:
                continue
            for parity, color in parity_to_color.items():
                pool = parity_cells[parity]
                chosen = set(rng.sample(pool, src_counts[color]))
                for r, c in pool:
                    if (r, c) in chosen:
                        src[r, c] = color
                    else:
                        dst[r, c] = color
            return src, dst
    raise ValueError("comp_checker build failed")


def random_sparse_pattern(n: int, primary: int, secondary: int, rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    k1 = rng.randint(5, min(8, n * n - 5))
    k2_hi = min(4, k1 - 1, n * n - k1 - 2)
    if k2_hi < 3:
        raise ValueError("bad sparse counts")
    k2 = rng.randint(3, k2_hi)
    cells = rng.sample([(r, c) for r in range(n) for c in range(n)], k1 + k2)
    for r, c in cells[:k1]:
        arr[r, c] = primary
    for r, c in cells[k1:]:
        arr[r, c] = secondary
    return arr


def build_sym_pair(n: int, colors: list[int], rng: random.Random, idx: int) -> tuple[np.ndarray, np.ndarray]:
    stable, primary, secondary = colors
    for _ in range(SYM_BUILD_RETRIES):
        dst = random_sparse_pattern(n, primary, secondary, rng)
        src = inverse_symmetry(dst, idx)
        if np.array_equal(src, dst):
            continue
        src, dst = add_stable_cells(src, dst, stable, rng)
        return src, dst
    raise ValueError("failed to build sym pair")


def choose_rule_setup(rng: random.Random) -> tuple[str, dict]:
    kind = rng.choice(RULES)
    params: dict = {}
    if kind == "sym":
        params["idx"] = rng.randint(1, 7)
    return kind, params


def build_training_pair(kind: str, params: dict, n: int, colors: list[int], rng: random.Random):
    if kind == "fill":
        return build_fill_pair(n, colors, rng)
    if kind == "endpoints":
        return build_endpoints_pair(n, colors, rng)
    if kind == "comp_checker":
        return build_comp_checker_pair(n, colors, rng)
    if kind == "sym":
        return build_sym_pair(n, colors, rng, params["idx"])
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


def draw_frame(grid: np.ndarray, r0: int, c0: int, s: int, frame_color: int, sprite: np.ndarray) -> None:
    r1, c1 = r0 + s - 1, c0 + s - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    nonblank_sigs = [tuple(colors_present(inner, exclude={0})) for inner in inners if not np.all(inner == 0)]
    if any(len(sig) != 3 for sig in nonblank_sigs):
        return False
    dup_count = sum(
        1 for i in range(len(nonblank_sigs)) for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False
    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _valid_sprite_roles(arr: np.ndarray) -> bool:
    roles = extract_roles_from_sprite(arr)
    return roles is not None


def _build_sprites(kind: str, setup_params: dict, n: int, p1: list[int], p2: list[int], rng: random.Random):
    train_src, train_dst = build_training_pair(kind, setup_params, n, p1, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred

    test_src, _ = build_training_pair(kind, setup_params, n, p2, rng)
    test_dst = apply_template(inferred_kind, inferred_params, test_src)

    if np.array_equal(test_dst, test_src):
        return None
    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nz(arr) or len(colors_present(arr, exclude={0})) != 3:
            return None
    if not _valid_sprite_roles(train_src) or not _valid_sprite_roles(test_src):
        return None
    train_sig = tuple(colors_present(train_src, exclude={0}))
    if train_sig != tuple(colors_present(train_dst, exclude={0})):
        return None
    if tuple(colors_present(test_src, exclude={0})) == train_sig:
        return None
    if infer_template(test_src, test_dst) is None:
        return None
    return train_src, train_dst, test_src, test_dst


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def _assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int]:
    order = sorted(range(4), key=lambda i: positions[i])
    train_slots = sorted(rng.sample(order, 2), key=order.index)
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    return train_slots, test_slot


def _render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    kind, setup_params = choose_rule_setup(rng)
    p1 = choose_palette(rng, 3, set())
    p2 = choose_palette(rng, 3, set(p1))

    built = _build_sprites(kind, setup_params, n, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)
    positions = quadrant_positions(s, rng)
    train_slots, test_slot = _assign_slots(positions, rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    grid = _render_scene(s, positions, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
