# Combo variant of 4c7dc4dd (index 11)
# combo_spec: D5/0+D12/2+D13/0
# Name: 三色精灵的角色交换形态学
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 画布、四个方框、一个空框、用同色集识别训练对的外层机制保持原题结构。
#    - 每个非空精灵现在都恰好有 3 种非黑色，而且在源精灵里像素数严格满足：
#      最少色 < 次多色 < 最多色。
#    - 按 D13 的三角色语言，这三色原本可解释为：稳定色 / 次变量色 / 主变量色。
#      但 D12/2 要求角色交换，所以真正发生形态学变化的是“原稳定色”
#      也就是源精灵里最少像素的那一色；另外两色虽然更大、更显眼，但在
#      源->目标中完全不动。
#    - D5/0 把原题多规则推理压缩成两种四邻域形态学：
#      1. dilate: 活跃色向上下左右扩展一格，但不能覆盖两种静止色。
#      2. erode: 活跃色里恰好只有一个四邻居的像素被删除。
#    - 小例子（5x5 内框）：
#      源：
#        2 2 2 2 2
#        0 0 4 0 0
#        0 0 1 1 0
#        0 0 1 0 0
#        3 3 3 3 0
#      这里 1 是最少像素色，因此虽然 2/3 更大，它们都保持不动；
#      若规则是 dilate，则目标中只有 1 会长粗，2/3 原地保留。
#      这和原始 D13 思路正好相反：不是大色变、小色稳，而是小色变、大色稳。
#
# b) 复杂度校准
#    - 只保留 dilate / erode 两档，避免训练对只有一例时出现过多模板分支。
#    - 三色角色靠“计数分层 + 训练对中是否变化”双重支撑，减少歧义：
#      人类先看到三色，再看到只有最少色在动，足以归纳出角色交换。
#    - 静止两色都选成长线段，保证如果错误地把它们当作旧变量色去做形态学，
#      结果会明显不同，从而让融合效果肉眼可见而不是退化成原版。
#
# c) 隐式假设审查
#    - 原 solver 的 `pick_var_color()` 假设“非稳定色只有一个”，在这里失效；
#      新版必须接受 3 色，并且允许 2 色完全不变、1 色变化。
#    - 原 D13 版本默认最少色稳定、其余两色变化；这里恰好反过来。
#      因此 `main()` 不能仅凭“最少色稳定”写死模板，必须先从训练对验证
#      “到底是哪一色发生了掩码变化”，再把这个角色迁移到测试图。
#    - 形态学膨胀若直接对整张非零掩码做，会把静止色错误吸收进去；
#      所以实现上必须只对活跃色的布尔掩码做运算，并把静止色视为不可覆盖障碍。
#    - 反向枚举训练对方向时，只有满足“变化色是源图最少色”的方向才应接受；
#      generator 用构造和回验保证这一点。
#
# d) 放置合法性
#    - 内框尺寸收紧到 5 或 6（外框 7 或 8），给三色和形态学留足空间。
#    - 先建活跃色源/目标，再只在二者都为空的位置搜索两条静止线段，
#      从而保证无重叠、无越界，且静止色在训练源/目标之间完全一致。
#    - 四个外框仍按四象限抖动放置，并在渲染后回验只检测到 4 个合法方框。
#
# e) 视觉可观察性
#    - 解题者能直接看到：两条较长的彩色线段在训练对中完全不变，
#      只有那个最小的小折线/小短线在长粗或收缩。
#    - 测试图也重复这种“少/中/多”三档分工，所以人可以先识别最少色，
#      再把同样的 dilate 或 erode 迁移过去。
#    - 规则不依赖颜色编号大小，也不依赖隐藏排序；全部线索都可从像素图直接观察。
#
# f) 可视化整洁度
#    - 静止色使用互不重叠的直线段，活跃色使用小路径；三层结构清晰。
#    - 静止色只放在活跃色源和目标都空的区域，避免“到底是哪一色在覆盖谁”的视觉噪音。
#    - 空框保持纯黑内区，背景噪声仍被方框隔离，不会干扰形态学观察。
#
# g) 变化的可推断性
#    - 没有任意颜色置换；颜色只承担角色。
#    - 训练对里恰好只有 1 种非黑色发生位置变化，而且它在源图里总是最少像素色。
#      这给“角色交换”提供了足够强的结构提示。
#
# h) 量化的融合核心行为
#    - 每个非空精灵恰好 3 色，且源图计数严格有序。
#    - 训练源->训练目标中，发生掩码变化的非背景颜色数恰好为 1。
#    - 该变化色在源图中是最少像素色。
#    - 若按“旧角色”误把较大的两色做同样形态学，输出必须与真实输出不同；
#      generator 会拒绝这种退化样本，后文统计也会量化这个差异。
#
# i) main() 的新增假设与 generator 保证
#    - 假设训练对共享同一三色集，测试图使用不相交三色集；generator 强制满足。
#    - 假设测试图也满足最少/次多/最多三档严格分层；generator 强制满足。
#    - 假设静止两色本身对同种形态学是“会变化的”，否则旧角色反事实可能碰巧
#      与真实输出相同；generator 明确把它们做成长线段并检查反事实差异。

import random

import numpy as np

from arc_utils import colors_present, neighbors4, paste


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
        if len(picked) >= 6:
            break
    return picked


def parse_scene(grid: np.ndarray):
    frames = find_square_frames(grid)
    if len(frames) < 3:
        return None

    inners = []
    for _, _, r0, c0, r1, c1 in frames:
        inners.append(grid[r0 + 1 : r1, c0 + 1 : c1])

    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = [(i, tuple(colors_present(inners[i], exclude={0}))) for i in nonblank]

    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] and sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        return None

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return None
    return {
        "frames": frames,
        "inners": inners,
        "blank_idx": blank_idx,
        "pair": pair,
        "test_idx": remaining[0],
    }


def nonzero_signature(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(colors_present(arr, exclude={0}))


def color_counts(arr: np.ndarray) -> dict[int, int]:
    return {int(c): int(np.sum(arr == c)) for c in np.unique(arr) if int(c) != 0}


def extract_roles_from_sprite(arr: np.ndarray):
    counts = color_counts(arr)
    if len(counts) != 3:
        return None
    items = sorted((cnt, color) for color, cnt in counts.items())
    if len({cnt for cnt, _ in items}) != 3:
        return None
    return {
        "active": items[0][1],
        "frozen": [items[2][1], items[1][1]],
        "counts": {color: cnt for cnt, color in items},
    }


def extract_roles_from_pair(src: np.ndarray, dst: np.ndarray):
    if nonzero_signature(src) != nonzero_signature(dst):
        return None
    src_roles = extract_roles_from_sprite(src)
    if src_roles is None:
        return None

    colors = list(nonzero_signature(src))
    changed = [c for c in colors if not np.array_equal(src == c, dst == c)]
    frozen = [c for c in colors if np.array_equal(src == c, dst == c)]
    if len(changed) != 1 or len(frozen) != 2:
        return None
    active = changed[0]
    if active != src_roles["active"]:
        return None
    return {
        "active": active,
        "frozen": frozen,
        "src_roles": src_roles,
    }


def dilate4_mask(mask: np.ndarray, blocked: np.ndarray | None = None) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w:
                if blocked is not None and bool(blocked[nr, nc]):
                    continue
                out[nr, nc] = True
    return out


def erode4_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    remove = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        deg = 0
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
                deg += 1
        if deg == 1:
            remove.append((rr, cc))
    for rr, cc in remove:
        out[rr, cc] = False
    return out


def apply_kind_to_active(kind: str, arr: np.ndarray, active_color: int, frozen_colors: list[int]) -> np.ndarray:
    active_mask = arr == active_color
    blocked = np.isin(arr, frozen_colors)
    if kind == "dilate":
        new_active = dilate4_mask(active_mask, blocked=blocked)
    elif kind == "erode":
        new_active = erode4_mask(active_mask)
    else:
        return arr.copy()

    out = arr.copy()
    out[active_mask] = 0
    out[new_active] = active_color
    for color in frozen_colors:
        out[arr == color] = color
    return out


def infer_template(src: np.ndarray, dst: np.ndarray):
    roles = extract_roles_from_pair(src, dst)
    if roles is None:
        return None

    active = roles["active"]
    frozen = roles["frozen"]
    matches = []
    for kind in ("dilate", "erode"):
        pred = apply_kind_to_active(kind, src, active, frozen)
        if not np.array_equal(pred, src) and np.array_equal(pred, dst):
            matches.append(kind)
    if len(matches) == 1:
        return matches[0], {}
    if len(matches) == 2:
        src_n = int(np.sum(src == active))
        dst_n = int(np.sum(dst == active))
        if dst_n > src_n:
            return "dilate", {}
        if dst_n < src_n:
            return "erode", {}
    return None


def apply_template(kind: str, arr: np.ndarray) -> np.ndarray:
    roles = extract_roles_from_sprite(arr)
    if roles is None:
        return arr.copy()
    return apply_kind_to_active(kind, arr, roles["active"], roles["frozen"])


def apply_counterfactual_old_roles(kind: str, arr: np.ndarray) -> np.ndarray:
    roles = extract_roles_from_sprite(arr)
    if roles is None:
        return arr.copy()
    fixed_color = roles["active"]
    moving_colors = roles["frozen"]
    out = np.zeros_like(arr)
    out[arr == fixed_color] = fixed_color
    occupied = out != 0
    for color in moving_colors:
        mask = arr == color
        if kind == "dilate":
            new_mask = dilate4_mask(mask, blocked=occupied)
        elif kind == "erode":
            new_mask = erode4_mask(mask)
        else:
            return arr.copy()
        clash = new_mask & occupied
        if np.any(clash):
            return arr.copy()
        out[new_mask] = color
        occupied |= new_mask
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a swapped-role morphology rule from one framed training pair, then solve the held-out framed sprite.

    Input structure:
    - The board contains four same-size hollow square frames on a noisy background.
    - One frame interior is blank, two interiors form the training pair, and the remaining non-blank interior is the test sprite.
    - Every non-blank interior uses exactly three non-black colors: one smallest-count color and two larger colors.

    Rule:
    - Find the training pair by matching non-zero color sets.
    - Determine whether the smallest-count color in the training source is the only color that changes by four-neighbor dilation or four-neighbor erosion, while the other two colors stay fixed.
    - Re-identify the smallest-count color inside the test sprite and apply the same morphology operator only to that color, preserving the other two colors.

    Key insight:
    - The transferable pattern is not merely “which morphology happened”, but also “which role receives it”: the color that would have been the stable accent in a three-color sprite is now the sole moving color.
    """
    g = np.asarray(grid, dtype=int)
    parsed = parse_scene(g)
    if parsed is None:
        return g.copy()

    src_a = parsed["inners"][parsed["pair"][0]]
    src_b = parsed["inners"][parsed["pair"][1]]
    test_inner = parsed["inners"][parsed["test_idx"]]

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
        active = roles["active"]
        src_n = int(np.sum(src == active))
        dst_n = int(np.sum(dst == active))
        test_n = int(np.sum(test_inner == test_roles["active"]))
        score = 100 - abs(src_n - test_n)
        if kind == "dilate" and dst_n > src_n:
            score += 20
        if kind == "erode" and dst_n < src_n:
            score += 20
        if best is None or score > best[0]:
            best = (score, kind, params)

    if best is None:
        return test_inner.copy()
    return apply_template(best[1], test_inner)


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

RULES = ["dilate", "erode"]
_GEN_RULE_INDEX = 0

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 10, 32
PATH_BUILD_RETRIES = 120
GEN_MAX_ATTEMPTS = 500


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def random_path_cells(n: int, length: int, rng: random.Random) -> list[tuple[int, int]] | None:
    for _ in range(PATH_BUILD_RETRIES):
        start = (rng.randrange(n), rng.randrange(n))
        path = [start]
        used = {start}
        while len(path) < length:
            r, c = path[-1]
            nbrs = []
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                if (nr, nc) in used:
                    continue
                ok = True
                for ar, ac in neighbors4(nr, nc):
                    if 0 <= ar < n and 0 <= ac < n and (ar, ac) in used and (ar, ac) != (r, c):
                        ok = False
                        break
                if ok:
                    nbrs.append((nr, nc))
            if not nbrs:
                break
            nxt = rng.choice(nbrs)
            path.append(nxt)
            used.add(nxt)
        if len(path) == length:
            return path
    return None


def build_active_pair(kind: str, n: int, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    max_len = max(3, n - 2)
    for _ in range(PATH_BUILD_RETRIES):
        length = rng.randint(3, max_len)
        cells = random_path_cells(n, length, rng)
        if cells is None:
            continue
        src = np.zeros((n, n), dtype=int)
        for r, c in cells:
            src[r, c] = color
        dst = apply_kind_to_active(kind, src, color, [])
        if np.array_equal(src, dst):
            continue
        src_n = int(np.sum(src == color))
        dst_n = int(np.sum(dst == color))
        if kind == "dilate" and dst_n <= src_n:
            continue
        if kind == "erode" and dst_n >= src_n:
            continue
        return src, dst
    return None


def line_mask(n: int, r0: int, c0: int, length: int, horizontal: bool) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    if horizontal:
        mask[r0, c0 : c0 + length] = True
    else:
        mask[r0 : r0 + length, c0] = True
    return mask


def all_line_candidates(n: int, min_length: int) -> list[dict]:
    out = []
    for length in range(min_length, n + 1):
        for r in range(n):
            for c in range(n - length + 1):
                out.append({"length": length, "mask": line_mask(n, r, c, length, True)})
        for c in range(n):
            for r in range(n - length + 1):
                out.append({"length": length, "mask": line_mask(n, r, c, length, False)})
    return out


def min_chebyshev_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    coords_a = np.argwhere(mask_a)
    coords_b = np.argwhere(mask_b)
    if coords_a.size == 0 or coords_b.size == 0:
        return 0
    best = None
    for r1, c1 in coords_a.tolist():
        for r2, c2 in coords_b.tolist():
            d = max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2)))
            if best is None or d < best:
                best = d
    return int(best)


def choose_frozen_masks(active_src: np.ndarray, active_dst: np.ndarray, rng: random.Random):
    n = active_src.shape[0]
    active_n = int(np.count_nonzero(active_src))
    occupied = (active_src != 0) | (active_dst != 0)
    candidates = []
    for cand in all_line_candidates(n, active_n + 1):
        mask = cand["mask"]
        if np.any(mask & occupied):
            continue
        candidates.append(cand)
    pairs = []
    active_dst_mask = active_dst != 0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            if a["length"] == b["length"]:
                continue
            if np.any(a["mask"] & b["mask"]):
                continue
            combo = a["mask"] | b["mask"]
            gap = min_chebyshev_distance(combo, active_dst_mask)
            score = gap * 100 + a["length"] + b["length"]
            pairs.append((score, a, b))
    if not pairs:
        return None
    pairs.sort(key=lambda t: t[0], reverse=True)
    top = pairs[: min(12, len(pairs))]
    _, a, b = rng.choice(top)
    if a["length"] > b["length"]:
        return a["mask"], b["mask"]
    return b["mask"], a["mask"]


def place_mask(arr: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray:
    out = arr.copy()
    out[mask] = color
    return out


def build_training_pair(kind: str, n: int, colors: list[int], rng: random.Random):
    active_color, major_color, minor_color = colors
    active_pair = build_active_pair(kind, n, active_color, rng)
    if active_pair is None:
        return None
    active_src, active_dst = active_pair
    frozen_masks = choose_frozen_masks(active_src, active_dst, rng)
    if frozen_masks is None:
        return None
    major_mask, minor_mask = frozen_masks
    src = active_src.copy()
    dst = active_dst.copy()
    src = place_mask(src, major_mask, major_color)
    src = place_mask(src, minor_mask, minor_color)
    dst = place_mask(dst, major_mask, major_color)
    dst = place_mask(dst, minor_mask, minor_color)
    return src, dst


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
    parsed = parse_scene(grid)
    if parsed is None:
        return False
    frames = parsed["frames"]
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    inners = parsed["inners"]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    nonblank_sigs = [nonzero_signature(inner) for inner in inners if np.any(inner != 0)]
    if any(len(sig) != 3 for sig in nonblank_sigs):
        return False
    dup_count = sum(
        1
        for i in range(len(nonblank_sigs))
        for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False

    train_a = inners[parsed["pair"][0]]
    train_b = inners[parsed["pair"][1]]
    forward = infer_template(train_a, train_b)
    reverse = infer_template(train_b, train_a)
    if (forward is None) == (reverse is None):
        return False
    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def role_swap_distance(kind: str, sprite: np.ndarray) -> int:
    intended = apply_template(kind, sprite)
    counter = apply_counterfactual_old_roles(kind, sprite)
    return int(np.sum(intended != counter))


def _build_sprites(kind: str, n: int, p1: list[int], p2: list[int], rng: random.Random):
    built = build_training_pair(kind, n, p1, rng)
    if built is None:
        return None
    train_src, train_dst = built
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    if infer_template(train_dst, train_src) is not None:
        return None

    built_test = build_training_pair(kind, n, p2, rng)
    if built_test is None:
        return None
    test_src, _ = built_test
    test_dst = apply_template(kind, test_src)

    if np.array_equal(test_dst, test_src):
        return None
    if role_swap_distance(kind, train_src) <= 0 or role_swap_distance(kind, test_src) <= 0:
        return None
    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nz(arr):
            return None
        if len(nonzero_signature(arr)) != 3:
            return None
    if extract_roles_from_sprite(train_src) is None or extract_roles_from_sprite(test_src) is None:
        return None
    if extract_roles_from_pair(train_src, train_dst) is None:
        return None
    if nonzero_signature(train_src) != nonzero_signature(train_dst):
        return None
    if nonzero_signature(test_src) == nonzero_signature(train_src):
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


def _next_rule() -> str:
    global _GEN_RULE_INDEX
    kind = RULES[_GEN_RULE_INDEX % len(RULES)]
    _GEN_RULE_INDEX += 1
    return kind


def _gen_attempt(kind: str) -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    p1 = choose_palette(rng, 3, set())
    p2 = choose_palette(rng, 3, set(p1))

    built = _build_sprites(kind, n, p1, p2, rng)
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
    kind = _next_rule()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(kind)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
