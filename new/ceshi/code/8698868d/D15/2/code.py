# Variant of 8698868d
# Dimension: D15 — src-dst 基数关系 (multi_step)
# Change: 1:1 双射 — 每个精灵恰好匹配一个框架，反之亦然 → 一个精灵匹配多个框架（1:many）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
design_analysis:
  variant_goal:
    keep:
      - 仍然由大号框架网格和散落在外部的小号精灵构成
      - 仍然用“洞的 4 连通分量个数”作为匹配键
      - 仍然把匹配到的精灵塞进框架，并把精灵中的背景洞重着色成框架色
    change:
      - 原题每个洞数只出现一次，frame <-> sprite 是双射
      - 变体改成：一个洞数可在多个框架里重复出现，而外部只有一个对应精灵
      - 输出时该精灵会被复制到所有同洞数框架中，形成 broadcast 语义

  concrete_pixel_behavior:
    input:
      - 仍有 3-6 个 frame，紧密排成 1-3 x 1-3 网格
      - frame 尺寸仍为 6-8，sprite 尺寸仍为 frame_size-2
      - 只生成 2..min(4, N-1) 个 sprite 类别，每个类别对应一个唯一洞数
      - N 个 frame 的洞数从这些类别中取值，且每个类别至少出现一次，并强制至少一个类别出现 >= 2 次
    output:
      - 对每个 frame，按洞数查找到唯一 sprite 类别
      - 如果多个 frame 共享同一洞数，则它们内部出现同一 sprite 形状的不同着色版本

  behavior_trace:
    original_small_example:
      - frames 洞数为 [0, 1, 2]
      - sprites 洞数为 [2, 0, 1]
      - solver 用 used_sprites 消耗匹配，三个 sprite 各用一次
    variant_small_example:
      - frames 洞数为 [0, 1, 1, 2]
      - sprites 洞数为 [2, 0, 1]
      - 第一个洞数 1 的 frame 匹配到 sprite(1) 后，第二个洞数 1 的 frame 也必须继续使用同一个 sprite(1)
      - 因此 matching 必须从“逐个消费未使用 sprite”改成“洞数 -> sprite 的查表”
    difference:
      - 原代码会让第二个洞数 1 的 frame 匹配失败并保持纯框架，或者错误地依赖不存在的第二个 sprite(1)
      - 新代码会把 sprite(1) 复制到所有洞数 1 的 frame

  complexity_calibration:
    - 不把所有 frame 都做成同一个洞数；至少保留 2 个洞数类别，避免任务退化成“把唯一精灵复制 everywhere”
    - 也不让类别过多；最多 4 类，保证在 2-4 个样例里人类能看出“同洞数的框架共享同一外部精灵”
    - frame 颜色仍彼此不同，因此广播后的同一精灵会被不同框架色重着色，视觉上既能看出“同形状复制”，也能看出“嵌入的是对应框架”

  implicit_assumption_audit:
    - square-region 检测依赖“frame 比 sprite 大一档，且都是近似实心正方形”，该前提保持不变
    - frame grid 推断依赖“所有 frame 按固定步长排列”，该前提保持不变
    - frame 洞计数通过 interior != frame_color 统计，依赖 frame 内部只有背景洞；生成器继续保证
    - sprite 洞计数通过 sprite_region != sprite_color 统计，依赖 sprite 自身只有一种前景色；生成器继续保证
    - 原来的 used_sprites 依赖“每个洞数在 sprite 侧只消费一次”；该前提失效，必须删除
    - 新 solver 新增前提：“每个出现在 frame 中的洞数，在 sprite 侧至少出现一次且最好恰好一次”；生成器会显式保证

  placement_legality:
    - frame grid 仍整体放在大画布某处，不越界
    - sprite 仍只放在剩余背景区域，并使用 random_free_location 保证不与 frame 或其他 sprite 重叠
    - 因为 sprite 数少于 frame 数，放置只会更容易，不会破坏 solver 的区域识别

  visual_observability:
    - 输入里可直接看到：多个不同颜色框架拥有相同数量的内部洞，但外部只出现一个对应洞数的精灵
    - 输出里可直接看到：这些框架内部出现同一形状的复制版本
    - 为了避免偶然退化，生成器强制 N > sprite_count，且最大 frame 洞数重数 >= 2

  cleanliness:
    - 不在同一区域叠加额外标记；仍沿用原题整洁布局
    - 广播效果只体现在“同形状被多次嵌入”，不会引入新的视觉噪声

  color_inference:
    - 颜色规则不新增自由度
    - 仍然只是“精灵的洞被替换成所在 frame 的颜色”，这个结构性线索由嵌套关系直接给出
"""


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (3, 6)
UNIQUE_HOLE_K_RANGE = (2, 4)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 300


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Count frame holes and sprite holes as in the base task.
    - Match by connected-hole count.
    - Unlike the base task, the same sprite may serve multiple frames if that hole count
      appears multiple times among the frames.
    """
    g = np.array(grid, dtype=int)

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * min_fill_ratio):
            square_regions.append((h, comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(size, []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict[str, int]] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])

            inner = region[1:-1, 1:-1]
            hole_mask = inner != frame_color
            _lab, num_holes = label_mask(hole_mask, connectivity=4)

            frame_grid.append(
                {
                    "color": frame_color,
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "holes": int(num_holes),
                }
            )

    sprite_by_holes: dict[int, dict[str, np.ndarray | int]] = {}
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprite_by_holes.setdefault(
            int(num_holes),
            {
                "color": sprite_color,
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            },
        )

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for f in frame_grid:
        fr = int(f["out_r"])
        fc = int(f["out_c"])
        frame_color = int(f["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = sprite_by_holes.get(int(f["holes"]))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    inner_positions = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(SEED_RETRIES):
        random.shuffle(inner_positions)
        seeds: list[tuple[int, int]] = []
        for r, c in inner_positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == count:
                    return seeds
    return None


def _make_sprite_interior(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    sprite = np.full((size, size), sprite_color, dtype=int)
    if hole_count == 0:
        return sprite

    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{s} for s in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow):
            frontier = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_hole:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for j, other in enumerate(comps)
                        if j != ci
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    ):
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = BG

    _lab, num = label_mask(sprite != sprite_color, connectivity=4)
    if int(num) != int(hole_count):
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    if hole_count == 0:
        return frame

    inner_cells = (frame_size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = BG

    _lab, num = label_mask(frame != frame_color, connectivity=4)
    if int(num) != int(hole_count):
        return None
    return frame


def _assign_frame_hole_counts(
    unique_counts: list[int],
    frame_colors: list[int],
    sprite_color_by_k: dict[int, int],
) -> list[int] | None:
    counts = list(unique_counts)
    remaining = len(frame_colors) - len(unique_counts)
    for _ in range(remaining):
        counts.append(random.choice(unique_counts))

    for _ in range(100):
        trial = counts[:]
        random.shuffle(trial)
        freq = Counter(trial)
        ok = True
        for idx, hole_count in enumerate(trial):
            if freq[int(hole_count)] >= 2 and int(frame_colors[idx]) == int(sprite_color_by_k[int(hole_count)]):
                ok = False
                break
        if ok and max(freq.values()) >= 2:
            return trial
    return None


def _build_expected_output(
    frame_hole_counts: list[int],
    sprite_by_k: dict[int, np.ndarray],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    for idx, hole_count in enumerate(frame_hole_counts):
        ri, ci = divmod(idx, cols_grid)
        rr = ri * frame_size
        cc = ci * frame_size
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = sprite_by_k[int(hole_count)].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[frame_count])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, frame_count)

    sprite_count_max = min(UNIQUE_HOLE_K_RANGE[1], frame_count - 1)
    sprite_count = random.randint(UNIQUE_HOLE_K_RANGE[0], sprite_count_max)
    sprite_colors = random.sample(available, sprite_count)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - sprite_count)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    unique_hole_counts = list(range(start, start + sprite_count))
    random.shuffle(unique_hole_counts)

    sprite_color_by_k = {
        int(hole_count): int(sprite_colors[idx]) for idx, hole_count in enumerate(unique_hole_counts)
    }

    frame_hole_counts = _assign_frame_hole_counts(unique_hole_counts, frame_colors, sprite_color_by_k)
    if frame_hole_counts is None:
        return None

    sprite_by_k: dict[int, np.ndarray] = {}
    sprite_items: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(unique_hole_counts):
        patch = _make_sprite_interior(sprite_size, hole_count, sprite_colors[idx])
        if patch is None:
            return None
        sprite_by_k[int(hole_count)] = patch
        sprite_items.append((int(hole_count), patch))

    frame_items: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(frame_hole_counts):
        patch = _make_frame(frame_size, hole_count, frame_colors[idx])
        if patch is None:
            return None
        frame_items.append((int(hole_count), patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(0, canvas_h - grid_pixel_h)
    c0 = random.randint(0, canvas_w - grid_pixel_w)
    for idx, (_, frame_patch) in enumerate(frame_items):
        ri, ci = divmod(idx, cols_grid)
        rr = r0 + ri * frame_size
        cc = c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    for _, sprite_patch in sprite_items:
        try:
            r, c = random_free_location(
                canvas,
                sprite_patch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, sprite_patch, r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frame_hole_counts,
        sprite_by_k,
        frame_colors,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
    )
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
