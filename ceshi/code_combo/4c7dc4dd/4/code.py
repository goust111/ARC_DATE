# Combo variant of 4c7dc4dd (index 4)
# combo_spec: D6/2+D19/2
# Name: 缺口框架与提示线索
#
# Design analysis
# a) 具体化与行为追踪
#    - 四个对象仍然是 1 像素粗、同尺寸、轴对齐的正方形框，但每个框都带一个连续 1-2 像素缺口；
#      因此旧版“完整空心方框”检测会失败，solver 必须先识别“缺一小段边”的近似方框。
#    - 其中一个框不再是全黑空框，而是提示框：内部只放保留色 7 的 3 像素折角，像一段没画完的“转弯”符号。
#      这段提示像素只说明“训练关系属于四分之一旋转家族”，但并不直接给出顺/逆时针。
#    - 具体例子：若两个同配色训练框的缺口分别在 `top` 与 `right`，并且提示框的缺口也在 `top`，
#      那么提示框缺口先选出 `top` 缺口那一帧作为源帧；再由 `top -> right` 这个边方位变化读出
#      “顺时针 90 度”。旧逻辑会在完整框检测处失败，且即使忽略缺口，训练对本身被故意做成
#      “顺/逆时针都能得到同一个训练目标”，因此还需要提示框来消歧。
# b) 复杂度校准
#    - 只保留一个变换家族：90 度旋转；自由度仅有 `CW`/`CCW` 两档，避免把提示系统做成难以归纳的任意编码。
#    - 训练源图强制为 180 度对称、但不是 90 度对称，于是 `rot90` 与 `rot270` 在训练对上产生同一目标，
#      训练示例本身故意不暴露方向；测试图则拒绝这种对称性，让两种方向在输出上明显不同。
#    - 提示框内部像素只承担“这是旋转线索”的角色；真正的方向信息来自缺口，因此两个维度都必须参与。
# c) 隐式假设审查
#    - 旧 `find_square_frames()` 依赖四条边全部闭合；融合后必须改成“恰有一条边存在一个连续 1-2 格缺口”。
#    - 旧 `blank_idx` 假设存在唯一全零框；这里失效，必须改成通过保留色 7 的固定提示图案识别提示框。
#    - 旧代码把训练对方向交给模板匹配和打分启发式决定；这里不再可靠，因为训练对被构造成方向二义。
#      新 solver 必须先用提示框缺口选出源帧，再用源/目标框缺口的相对方位确定旋转方向。
#    - 旧模板库里有多种规则；融合后为了让提示真正必要，生成器只采样 `sym` 中的 90 度旋转两种成员。
# d) 放置合法性
#    - 仍使用 30x30 画布、4 个同尺寸方框、象限式布局，所有框互不重叠且不出界。
#    - 绘制时先清空整块区域，再画边框，再挖掉缺口，最后贴内部精灵；因此缺口总是开向黑色框内，
#      不会把背景噪声带进框里。
#    - 提示框缺口与训练源框缺口使用同一边向，确保源帧可被唯一定位；训练目标框缺口是源边向再旋转一步后的结果。
# e) 视觉可观察性
#    - 人眼会同时注意到两件事：四个框都不是完整方框；以及有一个框里只有保留色 7 的小折角提示。
#    - 训练对内部图像只告诉你“发生了 90 度转动，但方向不明”；提示框缺口标记源帧，训练框缺口差值再给出顺/逆。
#      这使得规则可以在单个 ARC 样例中被肉眼追踪，而不是依赖颜色 id 或隐藏排序。
# f) 可视化整洁度
#    - 提示色 7 不出现在训练/测试精灵、边框或背景中，提示框与普通精灵一眼可分。
#    - 缺口宽度只允许 1 或 2，且不吞掉角点，框轮廓依然稳定；内部提示图案只有 3 个像素，不会与精灵混淆。
# g) 变化的可推断性
#    - 颜色变化没有新增任意映射；训练/测试仍沿用“训练对同配色、测试配色不同”的原结构。
#    - 真正新增的是元线索：提示像素声明“看旋转”，缺口方位声明“旋转方向”。训练对提供形状一致性，
#      提示框提供方向消歧，三者组合后规则可推断。
# h) 融合行为量化指标
#    - 训练二义性指标：对每个样本都检查 `rot90(train_src) == rot270(train_src) == train_dst)`，确保训练对本身
#      不能独立区分顺/逆时针。
#    - 测试消歧指标：同时检查 `rot90(test_src)` 与 `rot270(test_src)` 必须不同，且只有由提示框解码出的那一个
#      才等于期望输出；否则样本会退化。
# i) 新 `main()` 假设复查
#    - 假设恰有一个提示框，且其内部严格等于保留色 7 的固定折角图案：`generate_input()` 强制满足。
#    - 假设训练对中恰有一帧的缺口边向与提示框相同，从而唯一标记源帧：生成器显式这样构造。
#    - 假设源/目标训练框的缺口边向相邻而非相对；否则无法读出 90 度转向：生成器由旋转规则直接派生目标缺口。

import numpy as np
import random
from arc_utils import colors_present, paste


H = W = 30
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
HINT_COLOR = 7

BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8
GAP_WIDTH_CHOICES = (1, 2)

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
SPARSE_COUNT_MAX = 10
ROTATION_IDXS = (2, 6)  # idx 2 = rot90 CCW, idx 6 = rot90 CW
GEN_MAX_ATTEMPTS = 400

GAP_SIDES = ("top", "right", "bottom", "left")
SIDE_TO_INDEX = {side: i for i, side in enumerate(GAP_SIDES)}


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


def symmetries(arr: np.ndarray) -> list[np.ndarray]:
    outs = []
    for k in range(4):
        r = np.rot90(arr, k)
        outs.append(r)
        outs.append(np.fliplr(r))
    return outs


def apply_sym_idx(arr: np.ndarray, idx: int) -> np.ndarray:
    return symmetries(arr)[idx].copy()


def rotate_gap_side(side: str, idx: int) -> str:
    step = 1 if idx == 6 else -1
    return GAP_SIDES[(SIDE_TO_INDEX[side] + step) % 4]


def gap_rotation_idx(src_side: str, dst_side: str) -> int | None:
    step = (SIDE_TO_INDEX[dst_side] - SIDE_TO_INDEX[src_side]) % 4
    if step == 1:
        return 6
    if step == 3:
        return 2
    return None


def color_signature(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(set(int(x) for x in arr.ravel() if int(x) != 0)))


def turn_hint_mask(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=int)
    r = n // 2
    c = n // 2
    mask[r, c] = 1
    mask[r, c - 1] = 1
    mask[r - 1, c] = 1
    return mask


def render_hint_sprite(n: int) -> np.ndarray:
    return turn_hint_mask(n) * HINT_COLOR


def is_turn_hint(inner: np.ndarray) -> bool:
    if inner.ndim != 2 or inner.shape[0] != inner.shape[1]:
        return False
    if np.any((inner != 0) & (inner != HINT_COLOR)):
        return False
    return np.array_equal((inner == HINT_COLOR).astype(int), turn_hint_mask(inner.shape[0]))


def identify_training_pair(
    frames: list[dict[str, object]],
    ignore_idx: int,
) -> tuple[int, int] | None:
    candidates = [i for i in range(len(frames)) if i != ignore_idx]
    sigs = [(i, color_signature(frames[i]["inner"])) for i in candidates]
    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] and sigs[a][1] == sigs[b][1]:
                if pair is not None:
                    return None
                pair = (sigs[a][0], sigs[b][0])
    return pair


def extract_scene(grid: np.ndarray) -> dict[str, object] | None:
    g = np.asarray(grid, dtype=int)
    frames_raw = find_gap_square_frames(g)
    if len(frames_raw) < 4:
        return None

    frames: list[dict[str, object]] = []
    for size, color, r0, c0, r1, c1, gap_side, gap_start, gap_width in frames_raw:
        frames.append(
            {
                "size": size,
                "color": color,
                "box": (r0, c0, r1, c1),
                "gap_side": gap_side,
                "gap_start": gap_start,
                "gap_width": gap_width,
                "inner": g[r0 + 1 : r1, c0 + 1 : c1],
            }
        )

    hint_idxs = [i for i, frame in enumerate(frames) if is_turn_hint(frame["inner"])]
    if len(hint_idxs) != 1:
        return None
    hint_idx = hint_idxs[0]

    pair = identify_training_pair(frames, hint_idx)
    if pair is None:
        return None

    remaining = [i for i in range(len(frames)) if i not in pair and i != hint_idx]
    if len(remaining) != 1:
        return None
    test_idx = remaining[0]

    hint_gap_side = str(frames[hint_idx]["gap_side"])
    src_matches = [i for i in pair if str(frames[i]["gap_side"]) == hint_gap_side]
    if len(src_matches) != 1:
        return None
    src_idx = src_matches[0]
    dst_idx = pair[1] if pair[0] == src_idx else pair[0]

    idx = gap_rotation_idx(str(frames[src_idx]["gap_side"]), str(frames[dst_idx]["gap_side"]))
    if idx not in ROTATION_IDXS:
        return None
    alt_idx = ROTATION_IDXS[0] if idx == ROTATION_IDXS[1] else ROTATION_IDXS[1]

    return {
        "grid": g,
        "frames": frames,
        "hint_idx": hint_idx,
        "src_idx": src_idx,
        "dst_idx": dst_idx,
        "test_idx": test_idx,
        "rotation_idx": idx,
        "alt_rotation_idx": alt_idx,
    }


def diagnose_fusion(grid: np.ndarray) -> dict[str, object] | None:
    scene = extract_scene(grid)
    if scene is None:
        return None
    frames = scene["frames"]
    src = frames[scene["src_idx"]]["inner"]
    dst = frames[scene["dst_idx"]]["inner"]
    test = frames[scene["test_idx"]]["inner"]
    idx = scene["rotation_idx"]
    alt_idx = scene["alt_rotation_idx"]
    return {
        "rotation_idx": idx,
        "alt_rotation_idx": alt_idx,
        "train_ambiguous": np.array_equal(apply_sym_idx(src, idx), dst)
        and np.array_equal(apply_sym_idx(src, alt_idx), dst),
        "test_disambiguated": not np.array_equal(apply_sym_idx(test, idx), apply_sym_idx(test, alt_idx)),
        "hint_gap_side": frames[scene["hint_idx"]]["gap_side"],
        "src_gap_side": frames[scene["src_idx"]]["gap_side"],
        "dst_gap_side": frames[scene["dst_idx"]]["gap_side"],
    }


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


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def random_halfturn_symmetric_pattern(n: int, colors: list[int], rng: random.Random) -> np.ndarray:
    orbits = []
    seen = set()
    for r in range(n):
        for c in range(n):
            orb = tuple(sorted({(r, c), (n - 1 - r, n - 1 - c)}))
            if orb not in seen:
                seen.add(orb)
                orbits.append(list(orb))

    for _ in range(120):
        arr = np.zeros((n, n), dtype=int)
        orbit_count = rng.randint(2, min(len(orbits), max(2, SPARSE_COUNT_MAX // 2 + 1)))
        picked = rng.sample(orbits, orbit_count)
        orbit_colors = [colors[0]] * orbit_count
        if len(colors) == 2 and orbit_count >= 2:
            for i in range(orbit_count):
                orbit_colors[i] = colors[rng.randint(0, 1)]
            orbit_colors[0] = colors[0]
            orbit_colors[1] = colors[1]
        for orb, col in zip(picked, orbit_colors):
            for r, c in orb:
                arr[r, c] = col
        if len(colors) == 2 and not all(np.any(arr == c) for c in colors):
            continue
        if not _valid_nz(arr):
            continue
        if np.array_equal(apply_sym_idx(arr, 2), arr):
            continue
        return arr
    raise ValueError("failed to build half-turn-symmetric sprite")


def build_ambiguous_training_pair(
    n: int,
    colors: list[int],
    rng: random.Random,
    idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(80):
        src = random_halfturn_symmetric_pattern(n, colors, rng)
        dst = apply_sym_idx(src, idx)
        alt_idx = ROTATION_IDXS[0] if idx == ROTATION_IDXS[1] else ROTATION_IDXS[1]
        if np.array_equal(dst, src):
            continue
        if not np.array_equal(apply_sym_idx(src, alt_idx), dst):
            continue
        return src, dst
    raise ValueError("failed to build ambiguous training pair")


def build_test_sprite(
    n: int,
    colors: list[int],
    rng: random.Random,
    idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    alt_idx = ROTATION_IDXS[0] if idx == ROTATION_IDXS[1] else ROTATION_IDXS[1]
    for _ in range(160):
        src = random_sparse_pattern(n, colors, rng)
        if len(colors) == 2 and not all(np.any(src == c) for c in colors):
            continue
        dst = apply_sym_idx(src, idx)
        alt = apply_sym_idx(src, alt_idx)
        if np.array_equal(dst, src) or np.array_equal(dst, alt):
            continue
        if not _valid_nz(src) or not _valid_nz(dst):
            continue
        return src, dst
    raise ValueError("failed to build test sprite")


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


def erase_gap(
    grid: np.ndarray,
    r0: int,
    c0: int,
    s: int,
    gap_side: str,
    gap_start: int,
    gap_width: int,
) -> None:
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


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used and c not in {0, HINT_COLOR}])
    bg_pool = [c for c in ALL_COLORS if c not in used and c not in {frame_color, HINT_COLOR}]
    feasible_periods = [p for p in BG_PERIOD_CHOICES if p <= len(bg_pool)]
    bg_palette = rng.sample(bg_pool, rng.choice(feasible_periods))
    return frame_color, bg_palette


def _assign_slots(rng: random.Random) -> tuple[int, int, int, int]:
    src_slot, dst_slot, test_slot, hint_slot = rng.sample(range(4), 4)
    return src_slot, dst_slot, test_slot, hint_slot


def _build_sprites(
    n: int,
    train_colors: list[int],
    test_colors: list[int],
    rng: random.Random,
    idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_src, train_dst = build_ambiguous_training_pair(n, train_colors, rng, idx)
    test_src, test_dst = build_test_sprite(n, test_colors, rng, idx)
    if color_signature(train_src) != color_signature(train_dst):
        raise ValueError("training palette drift")
    if color_signature(test_src) == color_signature(train_src):
        raise ValueError("test palette should differ from training palette")
    return train_src, train_dst, test_src, test_dst


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


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_gap_square_frames(grid)
    if len(frames) != 4:
        return False
    if len(find_complete_square_frames(grid)) != 0:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False

    scene = extract_scene(grid)
    if scene is None:
        return False
    info = diagnose_fusion(grid)
    if info is None or not info["train_ambiguous"] or not info["test_disambiguated"]:
        return False

    frames2 = scene["frames"]
    src = frames2[scene["src_idx"]]["inner"]
    dst = frames2[scene["dst_idx"]]["inner"]
    if not np.array_equal(apply_sym_idx(src, scene["rotation_idx"]), dst):
        return False
    return np.array_equal(main(grid), expected_output)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2
    idx = rng.choice(ROTATION_IDXS)
    color_count = rng.choice([1, 2])

    train_colors = choose_palette(rng, color_count, {HINT_COLOR})
    test_colors = choose_palette(rng, color_count, set(train_colors) | {HINT_COLOR})

    try:
        train_src, train_dst, test_src, test_dst = _build_sprites(n, train_colors, test_colors, rng, idx)
    except ValueError:
        return None

    frame_color, bg_palette = _choose_frame_and_bg(set(train_colors) | set(test_colors), rng)
    positions = quadrant_positions(s, rng)
    src_slot, dst_slot, test_slot, hint_slot = _assign_slots(rng)

    src_gap_side = rng.choice(GAP_SIDES)
    dst_gap_side = rotate_gap_side(src_gap_side, idx)
    hint_gap_side = src_gap_side
    test_gap_side = rng.choice(GAP_SIDES)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[src_slot] = train_src
    sprites[dst_slot] = train_dst
    sprites[test_slot] = test_src
    sprites[hint_slot] = render_hint_sprite(n)

    gap_sides = [None] * 4
    gap_sides[src_slot] = src_gap_side
    gap_sides[dst_slot] = dst_gap_side
    gap_sides[test_slot] = test_gap_side
    gap_sides[hint_slot] = hint_gap_side

    gaps = []
    for slot in range(4):
        width = 1 if slot == hint_slot else rng.choice(GAP_WIDTH_CHOICES)
        start = choose_gap_start(s, width, rng)
        gaps.append((gap_sides[slot], start, width))

    grid = _render_scene(s, positions, sprites, frame_color, bg_palette, gaps, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a directional quarter-turn rule from gapped frames, then apply it to the test sprite.

    Input structure:
    - The grid contains four non-overlapping square frames of the same size, and every frame has exactly one 1-2 cell gap on one side.
    - Two non-hint frames share the same non-zero color set and form the training pair; a third non-hint frame is the test sprite.
    - The remaining frame is a hint frame: inside it is a small color-7 bent mark, and its own gap side points to which matched training frame should be treated as the source.

    Rule:
    - Detect the incomplete frames, find the color-7 hint frame, and use its gap side to select the source frame from the matched training pair.
    - Compare the source frame's gap side with the destination frame's gap side; that edge-to-edge turn determines whether the hidden rotation is 90 degrees clockwise or counterclockwise.
    - Apply that same quarter-turn to the test sprite and return only the transformed test interior.

    Key insight:
    - The training sprites are intentionally direction-ambiguous: their pixels show that a quarter-turn happened, but not which way. The hint pixels say “read this as a turn”, and the missing border segment tells you which turn it is.
    """
    g = np.asarray(grid, dtype=int)
    scene = extract_scene(g)
    if scene is None:
        return g.copy()

    frames = scene["frames"]
    src = frames[scene["src_idx"]]["inner"]
    dst = frames[scene["dst_idx"]]["inner"]
    test = frames[scene["test_idx"]]["inner"]
    idx = scene["rotation_idx"]

    if not np.array_equal(apply_sym_idx(src, idx), dst):
        return test.copy()
    return apply_sym_idx(test, idx)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
