# Combo variant of 4c7dc4dd (index 30)
# combo_spec: D5/2+D17/0
# Name: 高密度边界提取

from collections import deque
import math
import random

import numpy as np

from arc_utils import colors_present, paste


# Design analysis
# design_analysis:
#   concretization:
#     - 仍然保留原题的 30x30 画布、4 个同尺寸空心方框、1 个 blank interior、2 个训练帧、1 个测试帧。
#     - D5/2 将变换固定为“2D 外边界提取”：把 sprite 的非零区域视作前景，只保留与外部零区域 4-邻接的前景像素，清掉内部像素。
#     - D17/0 再把可见对象限制为“高密度 blob”：训练源图和测试源图都必须占 inner n x n 面积的 60%-90%。
#       对 n=4/5/6，这对应 non-zero 计数分别在 [10,14] / [15,22] / [22,32]。
#   behavior_trace:
#     - 原题会在六类规则里猜一种；D5 单变体已经把它替换为固定 boundary trace。
#     - 现在还要区分“哪个训练图是源图”：只有高密度那一张才允许作为 src，dst 必须恰好等于 boundary_trace(src)。
#     - 具体例子：5x5 内部若 src 有 20 个像素，是一个几乎填满的厚块，dst 只剩外圈 14 个像素；
#       原 endpoints 规则只会保留线端，旧六选一逻辑也可能把稀疏图解释成别的规则，而这里明确要求“高密度厚区域 -> 外轮廓”。
#   complexity_calibration:
#     - 如果源图太稀，边界提取可能退化成恒等；如果太满，容易被误读成“总是挖空满方块”。
#     - 因此 generator 用“满方块减去 1-4 个贴边缺口”的方式造形：密度稳定落在 60%-90%，仍保留厚实内部，又不全是完美矩形。
#     - 颜色保持单色，训练对与测试对使用不同 palette，继续支撑原题的“按 palette 找训练对”机制，但不引入额外颜色映射负担。
#   implicit_assumptions_audit:
#     - 原 main 的找框、找 blank frame、按 non-zero palette 配训练对，这些前提仍然成立，所以保留。
#     - 原 main 里 stable_colors / pick_var_color / 六规则推断依赖“变量色”和“有限规则库”，在融合后全部失效，必须整体移除。
#     - 新 main 的新增假设有三条：
#       1) 训练对中存在且仅存在一个方向满足 high_density(src) 且 boundary_trace(src) == dst；
#       2) 测试图本身也是高密度厚形状；
#       3) 训练输出严格比训练输入更稀疏，至少删去若干内部像素。
#       generator 会显式构造并校验这三条。
#   placement_legality:
#     - 方框仍按四象限随机抖动放置，外框大小 S in {6,7,8}，inner side n=S-2，因此 sprite 永不越界。
#     - 4 个方框不重叠；渲染时先清 interior 为 0，再 paste sprite，所以不会和背景条纹混叠。
#     - 背景色、框色、训练 palette、测试 palette 全部互斥，降低误检额外方框的风险；最终仍由 validate_scene 二次筛样。
#   visual_observability:
#     - 训练示例里，输入是“接近填满”的块，输出是明显更薄的一圈外轮廓；这种变化在 4x4 到 6x6 小图上肉眼可见。
#     - generator 额外拒绝 boundary_output == old_endpoints_output 的样本，避免视觉上退回成“取端点”。
#     - 同时要求 interior_removed >= 2，让“高密度 -> 剥掉内部”在 2-3 个样例里足够稳定。
#   cleanliness_and_color_reasoning:
#     - 规则只依赖几何密度和外接触关系，不依赖颜色编号大小、黑色排序或其他人类不可见因素。
#     - 画面中的语义层次清楚：背景只是噪声，方框用于分组，真正变化只发生在黑底 interior 的单色 blob 上。
#   change_inferability:
#     - 人类可推断的核心线索是重复出现的“厚输入 / 薄轮廓输出”对，而不是某个特定矩形模板。
#     - 高密度约束在训练图和测试图上都重复出现，边界提取也在训练对里直接展示，因此两项变化是协同的，不是各自孤立发生。


H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
DENSE_RATIO_MIN = 0.60
DENSE_RATIO_MAX = 0.90
BOUNDARY_DST_MIN = 6
BOUNDARY_INTERIOR_REMOVED_MIN = 2
DENSE_BUILD_RETRIES = 200
NOTCH_COUNT_MIN, NOTCH_COUNT_MAX = 1, 4
GEN_MAX_ATTEMPTS = 300


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


def dense_nz_bounds(n: int) -> tuple[int, int]:
    area = n * n
    lo = max(SPRITE_NZ_MIN, int(math.ceil(DENSE_RATIO_MIN * area)))
    hi = min(SPRITE_NZ_MAX, int(math.floor(DENSE_RATIO_MAX * area)))
    return lo, hi


def is_high_density(arr: np.ndarray) -> bool:
    n = int(arr.shape[0])
    lo, hi = dense_nz_bounds(n)
    nz = int(np.count_nonzero(arr))
    return lo <= nz <= hi


def exterior_zero_mask(fg: np.ndarray) -> np.ndarray:
    fg = np.asarray(fg, dtype=bool)
    h, w = fg.shape
    ext = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def seed(r: int, c: int) -> None:
        if not fg[r, c] and not ext[r, c]:
            ext[r, c] = True
            q.append((r, c))

    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)
    for c in range(w):
        seed(0, c)
        seed(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not fg[nr, nc] and not ext[nr, nc]:
                ext[nr, nc] = True
                q.append((nr, nc))
    return ext


def has_hole(arr: np.ndarray) -> bool:
    fg = np.asarray(arr != 0, dtype=bool)
    ext = exterior_zero_mask(fg)
    return bool(np.any((~fg) & ~ext))


def connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    r0, c0 = map(int, coords[0])
    seen[r0, c0] = True
    q.append((r0, c0))
    reached = 1
    target = len(coords)
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
                reached += 1
    return reached == target


def outer_boundary_mask(arr: np.ndarray) -> np.ndarray:
    fg = np.asarray(arr != 0, dtype=bool)
    h, w = fg.shape
    if not np.any(fg):
        return np.zeros((h, w), dtype=bool)
    ext = exterior_zero_mask(fg)
    keep = np.zeros((h, w), dtype=bool)
    rr, cc = np.where(fg)
    for r, c in zip(rr.tolist(), cc.tolist()):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nr >= h or nc < 0 or nc >= w or ext[nr, nc]:
                keep[r, c] = True
                break
    return keep


def boundary_trace(arr: np.ndarray) -> np.ndarray:
    keep = outer_boundary_mask(arr)
    out = np.zeros_like(arr)
    out[keep] = arr[keep]
    return out


def old_endpoints_like(arr: np.ndarray) -> np.ndarray:
    fg = np.asarray(arr != 0, dtype=bool)
    h, w = fg.shape
    keep = np.zeros((h, w), dtype=bool)
    rr, cc = np.where(fg)
    for r, c in zip(rr.tolist(), cc.tolist()):
        deg = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and fg[nr, nc]:
                    deg += 1
        if deg <= 1:
            keep[r, c] = True
    out = np.zeros_like(arr)
    out[keep] = arr[keep]
    return out


def infer_boundary_direction(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = []
    for src, dst in ((a, b), (b, a)):
        pred = boundary_trace(src)
        src_n = int(np.count_nonzero(src))
        dst_n = int(np.count_nonzero(dst))
        removed = src_n - dst_n
        if not is_high_density(src):
            continue
        if removed < BOUNDARY_INTERIOR_REMOVED_MIN:
            continue
        if dst_n < BOUNDARY_DST_MIN:
            continue
        if np.array_equal(pred, dst):
            candidates.append((removed, src_n, src, dst))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2], candidates[0][3]


def extract_scene_roles(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 3:
        return None

    inners = [g[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]

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
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None and len(nonblank) >= 2:
        counts = [(i, int(np.count_nonzero(inners[i]))) for i in nonblank]
        counts.sort(key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a high-density boundary-stripping rule from a framed analogy scene and solve the test sprite.

    Input structure:
    - The grid contains four non-overlapping hollow square frames on a noisy background.
    - One frame interior is blank, two interiors form the training pair and share the same non-zero palette, and the remaining non-blank interior is the test source.
    - The informative sprites live in small 4x4 to 6x6 interiors and the source sprites are densely filled blobs rather than sparse line drawings.

    Rule:
    - Use the repeated non-zero palette to locate the training pair.
    - Find the direction where a high-density sprite occupying 60%-90% of its interior is transformed into only the 4-connected outer boundary of its non-zero region, preserving the kept color and clearing interior cells.
    - Apply that same boundary extraction to the test sprite.

    Key insight:
    - The visual cue is the combination of density and thinning: the training source is almost full, while the training target keeps only the exposed rim.
    - Once the dense blob is identified, the answer is obtained by peeling away its interior and keeping just the outside contour.
    """
    roles = extract_scene_roles(grid)
    if roles is None:
        return np.asarray(grid, dtype=int).copy()

    inners = roles["inners"]
    pair = roles["pair"]
    test_inner = inners[roles["test_idx"]]

    direction = infer_boundary_direction(inners[pair[0]], inners[pair[1]])
    if direction is None:
        return test_inner.copy()
    if not is_high_density(test_inner):
        return test_inner.copy()
    return boundary_trace(test_inner)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def carve_side_notch(fg: np.ndarray, rng: random.Random) -> None:
    n = fg.shape[0]
    side = rng.choice(["top", "bottom", "left", "right"])
    depth = rng.randint(1, max(1, n - 2))
    span = rng.randint(1, n - 1)
    start = rng.randint(0, n - span)
    if side == "top":
        fg[:depth, start : start + span] = False
    elif side == "bottom":
        fg[n - depth :, start : start + span] = False
    elif side == "left":
        fg[start : start + span, :depth] = False
    else:
        fg[start : start + span, n - depth :] = False


def random_dense_shape(n: int, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = dense_nz_bounds(n)
    for _ in range(DENSE_BUILD_RETRIES):
        fg = np.ones((n, n), dtype=bool)
        for _ in range(rng.randint(NOTCH_COUNT_MIN, NOTCH_COUNT_MAX)):
            carve_side_notch(fg, rng)

        src_n = int(np.count_nonzero(fg))
        if not (lo <= src_n <= hi):
            continue
        if not connected(fg):
            continue

        src = np.zeros((n, n), dtype=int)
        src[fg] = color
        if has_hole(src):
            continue

        dst = boundary_trace(src)
        dst_n = int(np.count_nonzero(dst))
        removed = src_n - dst_n
        if dst_n < BOUNDARY_DST_MIN:
            continue
        if removed < BOUNDARY_INTERIOR_REMOVED_MIN:
            continue
        if np.array_equal(dst, src):
            continue
        if np.array_equal(dst, old_endpoints_like(src)):
            continue
        if infer_boundary_direction(src, dst) is None:
            continue
        return src, dst
    raise ValueError("failed to build a high-density boundary sample")


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
    roles = extract_scene_roles(grid)
    if roles is None:
        return False
    frames = roles["frames"]
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False

    inners = roles["inners"]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False

    nonblank_sigs = [tuple(colors_present(inner, exclude={0})) for inner in inners]
    nonblank_sigs = [s for s in nonblank_sigs if s]
    dup_count = sum(
        1
        for i in range(len(nonblank_sigs))
        for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False

    pair = roles["pair"]
    direction = infer_boundary_direction(inners[pair[0]], inners[pair[1]])
    if direction is None:
        return False
    train_src, _train_dst = direction
    if not is_high_density(train_src):
        return False

    test_inner = inners[roles["test_idx"]]
    if not is_high_density(test_inner):
        return False
    if not np.array_equal(boundary_trace(test_inner), expected_output):
        return False
    return np.array_equal(main(grid), expected_output)


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

    train_palette = choose_palette(rng, 1, set())
    test_palette = choose_palette(rng, 1, set(train_palette))

    try:
        train_src, train_dst = random_dense_shape(n, train_palette[0], rng)
        test_src, test_dst = random_dense_shape(n, test_palette[0], rng)
    except ValueError:
        return None

    frame_color, bg_palette = _choose_frame_and_bg(set(train_palette) | set(test_palette), rng)
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
