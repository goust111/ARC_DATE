# Variant of 4c7dc4dd
# Dimension: D5 — 精灵变换操作集 (transform)
# Change: 六种规则之一: endpoints（取端点）、fill（填充间隙）、sym_fill（对称填充）、comp_checker（棋盘互补）、invert0（零与变量色互换）、sym（旋转/翻转） → 边界描绘

from collections import deque
import random

import numpy as np

from arc_utils import colors_present, paste


# Design analysis
# design_analysis:
#   concretization:
#     - 仍然是 30x30 画布、4 个同尺寸空心方框、1 个空白 interior、2 个训练帧、1 个测试帧。
#     - D5 现在不再在 6 种变换里择一，而是固定为“边界描绘”：
#       对 sprite 的整块非零前景取 4-连通外边界，只保留与外部零区域相邻的像素，内部像素清零，保留像素原色不变。
#     - 这里的“外边界”按 outside-connected zeros 定义，因此若出现孔洞，孔洞内侧边缘不应保留。
#       为了让规律更直观，generator 只生成无孔、厚实、至少含一个真实内部像素的 2D 形状。
#   behavior_trace:
#     - 原 endpoints 的局部逻辑是：在线段/细骨架上只保留低度数端点，结果通常只剩 1D 端点。
#     - 新逻辑是：对整个前景区域做 exterior flood fill，然后保留接触 exterior 的前景像素。
#     - 例子：3x4 实心矩形在旧 endpoints 下几乎全灭；在新规则下会变成一圈矩形边框。
#       差异在于旧规则关注“细线末端”，新规则关注“2D 厚区域的外轮廓”。
#   complexity_calibration:
#     - 若源 sprite 太薄，边界变换会退化成恒等；若全是满矩形，又容易被误读成“挖空矩形”。
#     - 因此使用“重叠矩形并集”生成厚实 blob：既有内部像素，视觉上又不总是规则矩形。
#     - 颜色保持 1 色，避免把问题转成颜色映射；训练对和测试对仍用不同调色板，维持原题的配对辨识机制。
#   implicit_assumptions_audit:
#     - 原 main 的找框、找空白帧、按 non-zero palette 配对，这些不依赖旧六选一规则，仍然成立。
#     - 原 main 的 stable_colors / pick_var_color / symmetry / fill / endpoints 推断都依赖“存在特殊变量色”或“规则属于有限旧集合”，
#       在新变体中已不成立，因此整体替换为“哪一方向满足 boundary_trace(src)==dst)”。
#     - 新 solver 的新增假设是：训练对中恰有一个方向是“厚形状 -> 边界”；generator 通过强制 src!=dst 且测试 palette 与训练对不同来保证。
#   placement_legality:
#     - 仍沿用原题的 quadrant 摆放和非重叠方框，interior 在方框内清为 0 后再 paste sprite，不会越界或互相覆盖。
#     - background palette 与 sprite/frame 颜色保持互斥，降低误检方框的概率；最终仍用 solver 自校验筛掉坏样本。
#   visual_observability:
#     - 训练输出一定比输入“更空”，但会保留完整 2D 轮廓而不是少数端点；这在 4x4 到 6x6 小画幅上仍然肉眼可见。
#     - generator 强制 interior_removed >= 2，并拒绝 boundary_output == old_endpoints_output，避免退化回旧 D5 行为。
#   cleanliness_and_color_reasoning:
#     - 规则完全由形状厚薄决定，不依赖颜色 id 排序或隐式颜色语义。
#     - 每个 sprite interior 只使用一种颜色，背景和边框颜色分离，避免多层叠加造成视觉杂讯。


# ------- CFG: constants governing sampling distribution -------
H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)  # inner side n = 4, 5, 6
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
BOUNDARY_SRC_MIN = 8
BOUNDARY_DST_MIN = 6
BOUNDARY_INTERIOR_REMOVED_MIN = 2
BOUNDARY_BUILD_RETRIES = 100
SECOND_RECT_PROB = 0.8
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


def exterior_zero_mask(fg: np.ndarray) -> np.ndarray:
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


def outer_boundary_mask(arr: np.ndarray) -> np.ndarray:
    fg = np.asarray(arr != 0, dtype=bool)
    h, w = fg.shape
    if not np.any(fg):
        return np.zeros_like(fg)
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


def infer_boundary_direction(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = []
    for src, dst in ((a, b), (b, a)):
        pred = boundary_trace(src)
        if np.array_equal(pred, dst):
            removed = int(np.count_nonzero(src)) - int(np.count_nonzero(dst))
            candidates.append((removed, src, dst))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], int(np.count_nonzero(t[1]))), reverse=True)
    return candidates[0][1], candidates[0][2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a contour-only sprite transformation from a framed training pair and apply it to the test sprite.

    Input structure:
    - The grid shows four non-overlapping hollow square frames on a noisy background.
    - One frame interior is blank, two interiors form the training source/target pair with the same non-zero color set, and the remaining non-blank interior is the test source.
    - The answer is the transformed inner sprite of the test frame, not the full 30x30 scene.

    Rule:
    - Match the training pair by identical non-zero palettes.
    - Determine which direction turns a thick non-zero sprite into only its 4-connected outer boundary pixels, keeping the original colors on kept cells and clearing interior cells.
    - Apply that same boundary extraction to the test sprite.

    Key insight:
    - The training output is not an endpoint set or a symmetry of the source; it is the exposed contour of the source foreground region.
    - Once the filled interior is mentally stripped away, the test answer is just the same sprite reduced to its outer outline.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
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
    sigs = [(i, tuple(colors_present(inners[i], exclude={0}))) for i in nonblank]

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
    test_inner = inners[remaining[0]]

    direction = infer_boundary_direction(inners[pair[0]], inners[pair[1]])
    if direction is None:
        return test_inner.copy()
    return boundary_trace(test_inner)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def old_endpoints_like(arr: np.ndarray) -> np.ndarray:
    fg = arr != 0
    h, w = arr.shape
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


def has_hole(arr: np.ndarray) -> bool:
    fg = arr != 0
    ext = exterior_zero_mask(fg)
    return bool(np.any((~fg) & ~ext))


def random_thick_shape(n: int, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(BOUNDARY_BUILD_RETRIES):
        arr = np.zeros((n, n), dtype=int)

        h1 = rng.randint(3, n)
        w1 = rng.randint(3, n)
        r1 = rng.randint(0, n - h1)
        c1 = rng.randint(0, n - w1)
        arr[r1 : r1 + h1, c1 : c1 + w1] = color

        if rng.random() < SECOND_RECT_PROB:
            for _ in range(30):
                h2 = rng.randint(2, n)
                w2 = rng.randint(2, n)
                r2 = rng.randint(0, n - h2)
                c2 = rng.randint(0, n - w2)
                rect2 = np.zeros((n, n), dtype=bool)
                rect2[r2 : r2 + h2, c2 : c2 + w2] = True
                base = arr != 0
                if np.any(base & rect2) and np.any(rect2 & ~base):
                    arr[rect2] = color
                    break

        src_n = int(np.count_nonzero(arr))
        if not (BOUNDARY_SRC_MIN <= src_n <= SPRITE_NZ_MAX):
            continue
        if has_hole(arr):
            continue

        dst = boundary_trace(arr)
        dst_n = int(np.count_nonzero(dst))
        if not (BOUNDARY_DST_MIN <= dst_n <= SPRITE_NZ_MAX):
            continue
        if np.array_equal(dst, arr):
            continue
        if src_n - dst_n < BOUNDARY_INTERIOR_REMOVED_MIN:
            continue
        if np.array_equal(dst, old_endpoints_like(arr)):
            continue
        if infer_boundary_direction(arr, dst) is None:
            continue
        return arr, dst
    raise ValueError("failed to build a non-degenerate boundary shape")


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
    nonblank_sigs = [tuple(colors_present(inner, exclude={0})) for inner in inners]
    nonblank_sigs = [s for s in nonblank_sigs if s]
    dup_count = sum(
        1 for i in range(len(nonblank_sigs)) for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
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

    train_src, train_dst = random_thick_shape(n, train_palette[0], rng)
    test_src, test_dst = random_thick_shape(n, test_palette[0], rng)

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
