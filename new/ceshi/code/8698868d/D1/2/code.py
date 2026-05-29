# Variant of 8698868d
# Dimension: D1 — 匹配条件：连通洞计数 (match_cond)
# Change: hole_counting — 精灵与框架各自内部的 4-连通背景色洞分量数目相等即配对... → 精灵洞总面积匹配框架洞数

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location

# Design analysis
# ===============
# spec_variant:
#   preserved:
#     - bg 固定为 8，frame 仍然是紧密方阵网格，sprite 仍然散落在外部背景区。
#     - 输出仍然只保留 frame 网格，并把匹配到的 sprite 填进 frame 内部。
#     - sprite 的洞像素仍会在输出时改成对应 frame 的颜色。
#   core_change:
#     - 原题匹配键:
#         frame_key = frame 内部 4-连通洞分量数
#         sprite_key = sprite 内部 4-连通洞分量数
#     - 变体匹配键:
#         frame_key = frame 内部洞数
#         sprite_key = sprite 内部所有洞像素的总面积
#   concrete_example:
#     - 一个 frame 内部有 3 个互不相邻的单像素洞, 所以 frame_key = 3。
#     - 一个 sprite 内部只有 1 个 4-连通洞块, 但这个洞块覆盖 3 个像素, 所以
#       原题 sprite_key = 1, 变体 sprite_key = 3。
#     - 原题会把它和 “1 洞 frame” 配对；本变体则把它和 “3 洞 frame” 配对。
#   complexity_calibration:
#     - 为了让“面积”与“数量”的差别肉眼可见，生成器默认让每个 sprite 只有一个
#       多像素洞块；这样旧规则下所有 sprite 大多只有 1 个洞分量，而新规则下它们
#       分别对应 2..N+1 个洞像素。
#     - 若继续允许 0 洞或 1 洞作为主流样本，新旧规则会在很多对象上重合，视觉差异
#       不明显，因此这里把 frame 洞数范围固定成连续的 2..N+1。
#   implicit_assumptions_review:
#     - `find_components(... monochromatic=True)` 依赖 sprite/frame 都是单色主体。
#       生成器保留这一点，只让洞使用背景色。
#     - 方形筛选 `area >= 0.5 * h * h` 依赖 sprite 虽然有洞但填充率不能过低。
#       生成器限制 sprite 洞面积 <= (sprite_size-2)^2，且 frame_size 取 6..8，
#       对应填充率始终 >= 0.55，满足该筛选。
#     - frame 仍通过 `label_mask(inner != frame_color)` 计数；这要求 frame 的洞必须
#       是彼此 4-分离的背景点。生成器继续用 isolated seeds 保证这一点。
#     - 新的 sprite_key 用 `hole_mask.sum()`，这要求所有非 sprite_color 像素都真的是
#       洞像素。生成器不引入第三种颜色，因此该假设成立。
#     - 匹配循环仍然是“第一个未使用且 key 相等的 sprite”。这要求 key 唯一。
#       生成器令 frame 洞数与 sprite 洞面积都恰好是连续整数 2..N+1 的一个排列，
#       从构造上保证一一对应。
#   placement_legality:
#     - frame 网格仍整体贴合、无间距地放在大画布里。
#     - sprite 仍使用 `random_free_location(..., padding=1)` 放到网格外背景区域，
#       保证不重叠、不断边，也不会接触 frame/sprite。
#   visual_observability:
#     - 人类在样例里会看到: frame 中是若干分散的单像素洞；sprite 中往往是一个
#       连成块的洞。输出说明配对依据不是“洞块个数”而是“洞像素个数”。
#   cleanliness:
#     - 输入中只有两类对象: 大 frame 网格 + 小 sprite 方块；洞都只出现在对象内部，
#       不与其他标记叠加，画面仍然整洁。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by sprite hole area versus frame hole count, then nest
    each sprite inside its frame and recolor the sprite holes to the frame color.
    """
    g = np.array(grid, dtype=int)
    BG = 8

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    # Find square regions (lower fill threshold to catch sprites with holes).
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
        sizes.setdefault(int(size), []).append(comp)

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

    frame_grid = []
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

    sprites = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        hole_area = int(np.sum(hole_mask))
        sprites.append(
            {
                "color": sprite_color,
                "hole_area": hole_area,
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    matches: dict[tuple[int, int], tuple[dict, dict]] = {}
    used_sprites: set[int] = set()
    for f in frame_grid:
        for i, s in enumerate(sprites):
            if i not in used_sprites and int(f["holes"]) == int(s["hole_area"]):
                matches[(int(f["out_r"]), int(f["out_c"]))] = (f, s)
                used_sprites.add(i)
                break

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for f in frame_grid:
        fr, fc = int(f["out_r"]), int(f["out_c"])
        frame_color = int(f["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        if (fr, fc) in matches:
            _f, sprite = matches[(fr, fc)]
            content = np.array(sprite["content"], dtype=int).copy()
            sprite_color = int(sprite["color"])
            content[content != sprite_color] = frame_color

            sh, sw = int(sprite["h"]), int(sprite["w"])
            out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
SEED_RETRIES = 50
GEN_RETRIES = 300
HOLE_AREA_START = 2

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` interior cells in a size x size patch, mutually 4-isolated."""
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


def _grow_connected_hole(size: int, hole_area: int) -> set[tuple[int, int]] | None:
    """Return one 4-connected interior hole of exact area `hole_area`."""
    inner_side = size - 2
    if hole_area < 0 or hole_area > inner_side * inner_side:
        return None
    if hole_area == 0:
        return set()

    cells = [(r, c) for r in range(inner_side) for c in range(inner_side)]
    seed = random.choice(cells)
    shape = {seed}

    while len(shape) < hole_area:
        frontier: list[tuple[int, int]] = []
        for r, c in shape:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < inner_side and 0 <= nc < inner_side and (nr, nc) not in shape:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        shape.add(random.choice(frontier))

    return {(r + 1, c + 1) for r, c in shape}


def _make_sprite(size: int, hole_area: int, sprite_color: int) -> np.ndarray | None:
    """
    Build a sprite with a solid border and a single 4-connected bg hole whose
    total pixel area equals `hole_area`.
    """
    sprite = np.full((size, size), sprite_color, dtype=int)
    hole_cells = _grow_connected_hole(size, hole_area)
    if hole_cells is None:
        return None
    for r, c in hole_cells:
        sprite[r, c] = BG

    actual_area = int(np.sum(sprite != sprite_color))
    _lab, num_holes = label_mask(sprite != sprite_color, connectivity=4)
    if actual_area != hole_area:
        return None
    if hole_area > 0 and num_holes != 1:
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Build a solid frame square with exactly `hole_count` isolated single-pixel holes."""
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    if hole_count == 0:
        return frame

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = BG

    _lab, num_holes = label_mask(frame != frame_color, connectivity=4)
    if num_holes != hole_count:
        return None
    return frame


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_area = {k: spatch for k, spatch in sprites}
    for idx, (k, _) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = ri * frame_size, ci * frame_size
        fc = frame_colors[idx]
        expected[rr : rr + frame_size, cc : cc + frame_size] = fc
        content = sprite_by_area[k].copy()
        content[content == BG] = fc
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _feasible_ns(frame_size: int) -> list[int]:
    sprite_size = frame_size - 2
    max_sprite_hole_area = (sprite_size - 2) ** 2
    max_n_from_sprite = max_sprite_hole_area - HOLE_AREA_START + 1
    return [n for n in sorted(_LAYOUTS_BY_N) if n <= max_n_from_sprite]


def _gen_attempt() -> np.ndarray | None:
    frame_size = random.choice(FRAME_SIZE_RANGE)
    feasible_ns = _feasible_ns(frame_size)
    if not feasible_ns:
        return None

    N = random.choice(feasible_ns)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[N])
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_H = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_W = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_H > CANVAS_MAX or min_W > CANVAS_MAX:
        return None
    H = random.randint(min_H, CANVAS_MAX)
    W = random.randint(min_W, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, N)
    sprite_colors = random.sample(available, N)

    hole_areas = list(range(HOLE_AREA_START, HOLE_AREA_START + N))
    frame_hole_perm = hole_areas[:]
    random.shuffle(frame_hole_perm)
    sprite_hole_perm = hole_areas[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for i, area in enumerate(sprite_hole_perm):
        patch = _make_sprite(sprite_size, area, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((area, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, count in enumerate(frame_hole_perm):
        patch = _make_frame(frame_size, count, frame_colors[i])
        if patch is None:
            return None
        frames.append((count, patch))

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, (_, fpatch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = r0 + ri * frame_size, c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = fpatch

    for _, spatch in sprites:
        try:
            r, c = random_free_location(canvas, spatch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, spatch, r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frames,
        sprites,
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
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
