# Combo variant of 4c7dc4dd (index 21)
# combo_spec: D2/0+D13/2+D20/1
# Name: 单色边框配对纯几何
#
# Design analysis:
# a) concrete_behavior_and_trace:
#   - 画面仍是 30x30、4 个同尺寸空心方框、1 个空白内部、3 个非空精灵。
#   - 但三个非空精灵都严格只有 1 种非背景色，而且三者用的是同一个内部颜色。
#   - 因此旧逻辑若按“内部颜色集合相同”找训练对，会看到三个非空框签名都相同，
#     只能按扫描顺序误取前两个；新逻辑必须改看“哪两个非空框边框颜色相同”。
#   - 例：三个非空框内部都只用颜色 5；阅读顺序中的边框色若为 [2, 7, 2]，
#     则真正训练对是两个边框色 2 的框，而不是前两个颜色签名相同的非空框。
#   - 训练对内部再只在纯几何空间里推断规则：同色角点补成矩形(fill)、
#     线段只保留端点(endpoints)、或对四向镜像骨架做统一补满(sym_fill)。
# b) complexity_calibration:
#   - 没把题目降成单一操作，而是保留 3 个几何规则族。
#   - 同时去掉依赖多色或方向互逆天然歧义的规则，避免“单色后视觉上不同、
#     但 solver 仍需靠任意优先级打破平局”。
#   - 生成器强制：训练对在两个方向上合起来恰好只命中 1 个规则，且旧版按扫描顺序
#     取到的前两个非空框不能刚好就是训练对，这样 D2 与 D20 都真正生效。
# c) implicit_assumptions_review:
#   - 原 main() 依赖 `sigs` 中恰有一对颜色集合相同；融合后不成立，因为三个非空框
#     都共享同一单色签名，所以配对必须改为边框色分组。
#   - 原 `stable_colors/pick_var_color` 假设可能有“稳定色 + 变量色”；融合后所有精灵
#     都是单色，直接按占据几何推断即可，去掉颜色角色分析。
#   - 原 solver 通过多规则打分挑第一个可用规则；融合后这是新的隐式任意性来源，
#     因而改成“只接受唯一命中，否则拒绝推断”。
#   - 原 generator 允许测试精灵颜色集与训练对不同；融合后必须反过来固定为同一单色，
#     否则边框配对的重要性会减弱。
# d) placement_legality:
#   - 仍使用四象限抖动锚点；外框边长 6-8，天然不越界。
#   - 四个方框互不重叠；内部 sprite 只粘贴到各自方框内部。
# e) visual_observability:
#   - 人类先看边框颜色找到训练对，再看单色形状变化；颜色不再承担规则语义。
#   - 三个非空框内部同色，保证“边框配对”是可见且必要的，不依赖任何不可见编号。
# f) visual_cleanliness:
#   - 精灵仍在独立框内，背景与边框、内部颜色全部分离，减少叠盖噪声。
#   - sym_fill 固定保留一个四角零轨道，避免输出变成完全实心块而显得过闷或与 fill 混淆。
# g) inferability_of_change:
#   - 不做颜色映射，只有单色几何占据变化；2-3 组样例足以让人归纳
#     “看边框找训练对，再把训练对里的几何变换套到测试框”。

import numpy as np
import random
from arc_utils import colors_present, draw_line, paste


H = W = 30
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

RULES = ("fill", "endpoints", "sym_fill")

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
FILL_RECT_AREA_MIN, FILL_RECT_AREA_MAX = 6, 25
ENDPOINTS_LINE_LEN_MIN = 3
TEST_BUILD_RETRIES = 40
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


def mono_color(arr: np.ndarray) -> int | None:
    vals = [int(v) for v in np.unique(arr) if int(v) != 0]
    if len(vals) != 1:
        return None
    return vals[0]


def color_sig(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(colors_present(arr, exclude={0}))


def orbit_cells(n: int, r: int, c: int) -> list[tuple[int, int]]:
    cells = {
        (r, c),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (n - 1 - r, n - 1 - c),
    }
    return sorted(cells)


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


def apply_fill(arr: np.ndarray) -> np.ndarray:
    v = mono_color(arr)
    if v is None:
        return arr.copy()
    y = arr.copy()
    h, w = y.shape
    for r in range(h):
        nz = np.where(y[r, :] != 0)[0]
        if nz.size >= 2:
            y[r, int(nz.min()) : int(nz.max()) + 1] = v
    for c in range(w):
        nz = np.where(y[:, c] != 0)[0]
        if nz.size >= 2:
            y[int(nz.min()) : int(nz.max()) + 1, c] = v
    return y


def apply_endpoints(arr: np.ndarray) -> np.ndarray:
    v = mono_color(arr)
    if v is None:
        return arr.copy()
    coords = np.argwhere(arr == v)
    cells = set((int(r), int(c)) for r, c in coords)
    keep = np.zeros_like(arr, dtype=bool)
    for r, c in cells:
        deg = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in cells:
                    deg += 1
        if deg <= 1:
            keep[r, c] = True
    y = np.zeros_like(arr)
    y[keep] = v
    return y


def apply_sym_fill(arr: np.ndarray, keep0: np.ndarray) -> np.ndarray:
    v = mono_color(arr)
    if v is None:
        return arr.copy()
    base = sym_fill_base(arr)
    if base is None:
        return arr.copy()
    y = base.copy()
    y[(y == 0) & ~keep0] = v
    return y


def infer_fill(src: np.ndarray, dst: np.ndarray) -> dict | None:
    if mono_color(src) is None or color_sig(src) != color_sig(dst):
        return None
    pred = apply_fill(src)
    if np.array_equal(pred, dst) and not np.array_equal(src, dst):
        return {}
    return None


def infer_endpoints(src: np.ndarray, dst: np.ndarray) -> dict | None:
    if mono_color(src) is None or color_sig(src) != color_sig(dst):
        return None
    pred = apply_endpoints(src)
    if np.array_equal(pred, dst) and not np.array_equal(src, dst):
        return {}
    return None


def infer_sym_fill(src: np.ndarray, dst: np.ndarray) -> dict | None:
    v = mono_color(src)
    if v is None or color_sig(src) != color_sig(dst):
        return None
    base = sym_fill_base(src)
    if base is None:
        return None
    if not np.all(dst[base != 0] == v):
        return None
    bad = (base == 0) & (dst != 0) & (dst != v)
    if np.any(bad):
        return None
    keep0 = (base == 0) & (dst == 0)
    pred = apply_sym_fill(src, keep0)
    if np.array_equal(pred, dst) and not np.array_equal(src, dst):
        extra = (base == 0) & (dst != 0)
        if np.any(extra):
            return {"keep0": keep0}
    return None


def list_rule_matches(src: np.ndarray, dst: np.ndarray) -> list[tuple[str, dict]]:
    matches = []
    for kind, infer_fn in (
        ("fill", infer_fill),
        ("endpoints", infer_endpoints),
        ("sym_fill", infer_sym_fill),
    ):
        params = infer_fn(src, dst)
        if params is not None:
            matches.append((kind, params))
    return matches


def infer_unique_rule(src: np.ndarray, dst: np.ndarray) -> tuple[str, dict] | None:
    matches = list_rule_matches(src, dst)
    if len(matches) == 1:
        return matches[0]
    return None


def infer_unique_pair_rule(a: np.ndarray, b: np.ndarray) -> tuple[str, dict] | None:
    candidates = []
    for src, dst in ((a, b), (b, a)):
        tpl = infer_unique_rule(src, dst)
        if tpl is not None:
            candidates.append(tpl)
    if len(candidates) == 1:
        return candidates[0]
    return None


def apply_rule(kind: str, params: dict, arr: np.ndarray) -> np.ndarray:
    if kind == "fill":
        return apply_fill(arr)
    if kind == "endpoints":
        return apply_endpoints(arr)
    if kind == "sym_fill":
        return apply_sym_fill(arr, params["keep0"])
    return arr.copy()


def unique_border_pair(nonblank: list[int], frame_colors: list[int]) -> tuple[int, int] | None:
    by_border: dict[int, list[int]] = {}
    for idx in nonblank:
        by_border.setdefault(frame_colors[idx], []).append(idx)
    pairs = [tuple(idxs) for idxs in by_border.values() if len(idxs) == 2]
    if len(pairs) == 1 and len([idxs for idxs in by_border.values() if len(idxs) == 1]) == 1:
        return tuple(sorted(pairs[0]))
    return None


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def choose_palette_color(rng: random.Random, forbidden: set[int]) -> int:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if not pool:
        raise ValueError("not enough sprite colors")
    return rng.choice(pool)


def build_fill_pair(n: int, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    dst = np.zeros((n, n), dtype=int)
    while True:
        h = rng.randint(2, n)
        w = rng.randint(2, n)
        if FILL_RECT_AREA_MIN <= h * w <= FILL_RECT_AREA_MAX and h >= 2 and w >= 2:
            break
    r0 = rng.randint(0, n - h)
    c0 = rng.randint(0, n - w)
    dst[r0 : r0 + h, c0 : c0 + w] = color
    src = np.zeros_like(dst)
    src[r0, c0] = color
    src[r0, c0 + w - 1] = color
    src[r0 + h - 1, c0] = color
    src[r0 + h - 1, c0 + w - 1] = color
    return src, dst


def build_endpoints_pair(n: int, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
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
    draw_line(src, r1, c1, end_r=r1, end_c=c1 + len1 - 1, color=color)
    draw_line(src, r2, c2, end_r=r2, end_c=c2 + len2 - 1, color=color)
    dst[r1, c1] = color
    dst[r1, c1 + len1 - 1] = color
    dst[r2, c2] = color
    dst[r2, c2 + len2 - 1] = color
    return src, dst


def build_sym_fill_pair(n: int, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    src = np.zeros((n, n), dtype=int)
    keep0 = np.zeros((n, n), dtype=bool)
    for r, c in orbit_cells(n, 0, 0):
        keep0[r, c] = True

    orbits = []
    seen = set()
    for r in range(n):
        for c in range(n):
            orb = tuple(orbit_cells(n, r, c))
            if orb not in seen:
                seen.add(orb)
                orbits.append(list(orb))
    candidates = [orb for orb in orbits if not any(keep0[r, c] for r, c in orb)]
    candidates.sort(
        key=lambda orb: (
            -max(min(r, c, n - 1 - r, n - 1 - c) for r, c in orb),
            -len(orb),
            orb[0],
        )
    )
    picks = [orb for orb in candidates if len(orb) >= 2][:3]
    if len(picks) < 3:
        raise ValueError("not enough sym-fill orbits")
    for orb in picks:
        src[rng.choice(orb)] = color
    dst = apply_sym_fill(src, keep0)
    return src, dst


def build_training_pair(kind: str, n: int, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if kind == "fill":
        return build_fill_pair(n, color, rng)
    if kind == "endpoints":
        return build_endpoints_pair(n, color, rng)
    if kind == "sym_fill":
        return build_sym_fill_pair(n, color, rng)
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


def _assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int, int]:
    order = sorted(range(4), key=lambda i: positions[i])
    patterns = [
        ("blank", "test", "train_src", "train_dst"),
        ("test", "blank", "train_src", "train_dst"),
        ("train_src", "test", "blank", "train_dst"),
        ("train_src", "blank", "test", "train_dst"),
        ("test", "train_src", "blank", "train_dst"),
        ("blank", "train_src", "test", "train_dst"),
    ]
    labels = rng.choice(patterns)
    slots = {}
    for idx, label in zip(order, labels):
        slots[label] = idx
    return [slots["train_src"], slots["train_dst"]], slots["test"], slots["blank"]


def _choose_frame_colors_and_bg(used: set[int], rng: random.Random) -> tuple[int, int, int, list[int]]:
    pool = [c for c in ALL_COLORS if c not in used]
    train_frame_color, test_frame_color, blank_frame_color = rng.sample(pool, 3)
    bg_pool = [c for c in pool if c not in {train_frame_color, test_frame_color, blank_frame_color}]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return train_frame_color, test_frame_color, blank_frame_color, bg_palette


def _render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_colors: list[int],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite, frame_color in zip(positions, sprites, frame_colors):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


def _build_sprites(kind: str, n: int, color: int, rng: random.Random):
    train_src, train_dst = build_training_pair(kind, n, color, rng)
    pair_rule = infer_unique_pair_rule(train_src, train_dst)
    if pair_rule is None or pair_rule[0] != kind:
        return None
    for arr in (train_src, train_dst):
        if not _valid_nz(arr) or color_sig(arr) != (color,):
            return None
    for _ in range(TEST_BUILD_RETRIES):
        test_src, test_dst = build_training_pair(kind, n, color, rng)
        if any(not _valid_nz(arr) for arr in (test_src, test_dst)):
            continue
        if color_sig(test_src) != (color,) or color_sig(test_dst) != (color,):
            continue
        if np.array_equal(test_src, test_dst):
            continue
        if np.array_equal(test_src, train_src) or np.array_equal(test_src, train_dst):
            continue
        if infer_unique_pair_rule(train_src, train_dst) is None:
            continue
        inferred = infer_unique_rule(test_src, test_dst)
        if inferred is None or inferred[0] != kind:
            continue
        return train_src, train_dst, test_src, test_dst
    return None


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False

    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    blank_idxs = [i for i, inner in enumerate(inners) if np.all(inner == 0)]
    if len(blank_idxs) != 1:
        return False
    blank_idx = blank_idxs[0]

    nonblank = [i for i in range(4) if i != blank_idx]
    frame_colors = [int(col) for _, col, *_ in frames]
    pair = unique_border_pair(nonblank, frame_colors)
    if pair is None:
        return False
    if len({color_sig(inners[i]) for i in nonblank}) != 1:
        return False
    if any(len(color_sig(inners[i])) != 1 for i in nonblank):
        return False
    if set(nonblank[:2]) == set(pair):
        return False
    if infer_unique_pair_rule(inners[pair[0]], inners[pair[1]]) is None:
        return False
    return np.array_equal(main(grid), expected_output)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a monochrome geometric rule from a border-marked training pair and apply it to a test sprite.

    Input structure:
    - The grid contains four same-size hollow square frames on a noisy background.
    - One frame interior is blank, while the other three contain monochrome sprites that all use exactly one non-background color.
    - Among the non-blank frames, the two frames sharing a border color form the training pair; the remaining non-blank frame is the test input.

    Rule:
    - Ignore interior palette matching and pair the examples by repeated border color.
    - Infer a single pure geometric transformation from the training pair: rectangle filling, endpoint extraction, or symmetry completion with preserved zero holes.
    - Apply that same geometric rule to the test sprite and return only the transformed interior.

    Key insight: The task is still an analogy, but the pairing clue lives on the outer borders and the rule itself must be inferred only from monochrome shape geometry.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 3:
        return g.copy()

    inners = []
    frame_colors = []
    for _, col, r0, c0, r1, c1 in frames:
        inners.append(g[r0 + 1 : r1, c0 + 1 : c1])
        frame_colors.append(int(col))

    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    pair = unique_border_pair(nonblank, frame_colors)
    if pair is None:
        return inners[nonblank[0]].copy()

    test_candidates = [i for i in nonblank if i not in pair]
    if len(test_candidates) != 1:
        return inners[pair[0]].copy()
    test_inner = inners[test_candidates[0]]

    rule = infer_unique_pair_rule(inners[pair[0]], inners[pair[1]])
    if rule is None:
        return test_inner.copy()
    kind, params = rule
    return apply_rule(kind, params, test_inner)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2
    rule_pool = ["fill", "endpoints"] if n == 4 else list(RULES)
    kind = rng.choice(rule_pool)
    color = choose_palette_color(rng, set())

    built = _build_sprites(kind, n, color, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    train_frame_color, test_frame_color, blank_frame_color, bg_palette = _choose_frame_colors_and_bg(
        {color}, rng
    )
    positions = quadrant_positions(s, rng)
    train_slots, test_slot, blank_slot = _assign_slots(positions, rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    frame_colors = [blank_frame_color] * 4
    frame_colors[train_slots[0]] = train_frame_color
    frame_colors[train_slots[1]] = train_frame_color
    frame_colors[test_slot] = test_frame_color
    frame_colors[blank_slot] = blank_frame_color

    grid = _render_scene(s, positions, sprites, frame_colors, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
