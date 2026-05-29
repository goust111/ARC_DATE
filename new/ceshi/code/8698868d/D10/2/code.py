# Variant of 8698868d
# Dimension: D10 — 颜色分配规则 (meta)
# Change: 每个框架和精灵各自使用不同的非背景色；框架色和精灵色集合可重叠；总共 2*N 种颜色（4-12 种）... → 所有框架使用同一颜色

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location

# Design analysis
# ---
# source_task: None
#
# spec:
#   variant_core:
#     - 输入里的所有框架边框都改为同一个非背景色。
#     - 精灵仍然是一组较小的单色正方形，彼此颜色不同；为避免视觉歧义，
#       这里额外约束精灵颜色不等于统一框色。这样输出里整块框架网格的区分
#       只能依赖嵌入后的精灵颜色与洞形态，D10 变化肉眼可见。
#   concrete_behavior:
#     - 原题里“每个大框各自一种颜色”，solver 可以先找一批同尺寸单色正方形，
#       再把最大尺寸那组直接当作 frame 列表。
#     - 变体里所有框彼此接触且同色后，这些框会在 4 连通下合并成一个大单色组件。
#       因此 solver 不能再靠“每个框一个连通域”定位 frame。
#     - 新流程改为：先识别所有较小的单色正方形精灵 -> 由 sprite_size 反推
#       frame_size = sprite_size + 2 -> 再把最大单色组件当作整块 frame-grid，
#       按规则格子切分并统计每格内部洞数。
#   behavior_trace:
#     - 小例子：若有 1x3 个 6x6 框、统一为黑色，三个框的洞数分别是 0/1/2。
#       原逻辑的 `find_components(... monochromatic=True)` 会得到 3 个 6x6 框组件；
#       变体下它只会得到 1 个 6x18 的黑色组件。
#     - 原逻辑随后 `size_list[0] == 6`，能直接把这组 6x6 组件当 frames。
#     - 新逻辑则会从三个 4x4 精灵组件得到 `sprite_size=4`，推出 `frame_size=6`，
#       再把 6x18 大组件按宽度切成三个 6x6 frame slot，并分别统计洞数 0/1/2。
#   complexity_calibration:
#     - 若允许精灵也使用统一框色，则输出中可能出现“某一格内外同色”现象，
#       D10 的观察性下降，因此生成器禁止这种重色。
#     - 洞计数仍保持 0..N-1 的连续唯一值，保证匹配单射且不引入额外自由度。
#     - 保留 2-6 个框、6-8 的框边长、1-3 x 1-3 的紧密网格，使样例仍与原题同量级。
#   implicit_assumption_audit:
#     - 原 `square_regions.append(...)` 隐含前提：frame 也是独立正方形连通域。
#       变体下不成立，必须只把这一步用于 sprite 候选。
#     - 原 `sizes[max_size] -> frames` 隐含前提：最大正方形尺寸对应 frame。
#       变体下不成立，因为 frame-grid 可能是长方形甚至更大的正方形 bbox。
#     - 原 `frame_positions` 由各 frame 组件左上角推网格。变体下没有逐框组件，
#       改为由整块 frame-grid bbox 和已知 frame_size 直接推出行列数。
#     - 原输出时每格可读 `f["color"]`。变体下这个值对所有格相同，但该假设无害；
#       只要 generator 保证统一框色即可。
#   placement_legality:
#     - frame-grid 先整体放入画布，再用 `random_free_location(..., padding=1)` 放精灵，
#       保证精灵既不重叠，也不会与 frame-grid 4/8 邻接并意外并入大组件。
#     - frame-grid bbox 的宽高必须都是 `frame_size` 的整数倍，且乘积等于 sprite 数量。
#   visual_observability:
#     - 输入里，解题者能看到“所有大框同色、但内部洞数不同；外部散落的小精灵颜色不同”。
#     - 输出里，整块网格边框完全同色，不再靠框色区分格子，只能靠嵌入的精灵颜色。
#   cleanliness:
#     - 仍采用原题的紧密框网格与分散精灵；只减少框色变化，不增加额外标记层。
#   color_inference:
#     - 颜色变化规则非常结构化：所有 sprite 洞像素都重着色为统一框色。
#       因为所有框色一致，人类更容易直接看出“洞被补成边框色”这一规律。


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 300
MIN_FILL_RATIO = 0.5

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _count_holes_in_sprite(sprite_region: np.ndarray, sprite_color: int) -> int:
    hole_mask = sprite_region != int(sprite_color)
    _lab, num_holes = label_mask(hole_mask, connectivity=4)
    return int(num_holes)


def _square_sprite_candidates(comps: list[dict]) -> dict[int, list[dict]]:
    sizes: dict[int, list[dict]] = {}
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 8 and area >= int(h * h * MIN_FILL_RATIO):
            sizes.setdefault(h, []).append(comp)
    return sizes


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    assert comps

    # In this variant, only sprites remain independent square components.
    sprite_sizes = _square_sprite_candidates(comps)
    assert sprite_sizes
    sprite_size = max(sprite_sizes.items(), key=lambda item: len(item[1]))[0]
    sprites_raw = sprite_sizes[sprite_size]
    frame_size = int(sprite_size + 2)

    # All touching frames merge into one largest monochromatic component.
    frame_grid_comp = max(comps, key=lambda comp: int(comp["area"]))
    frame_color = int(frame_grid_comp["color"])
    frame_bbox = frame_grid_comp["bbox"]
    grid_h = int(bbox_h(frame_bbox))
    grid_w = int(bbox_w(frame_bbox))
    assert grid_h % frame_size == 0 and grid_w % frame_size == 0

    n_rows = int(grid_h // frame_size)
    n_cols = int(grid_w // frame_size)
    assert n_rows * n_cols == len(sprites_raw)

    frame_r0, frame_c0, _frame_r1, _frame_c1 = frame_bbox
    frame_grid = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(frame_r0 + ri * frame_size)
            inp_c = int(frame_c0 + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            assert region.shape == (frame_size, frame_size)

            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            border_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            assert border_color == frame_color

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
        num_holes = _count_holes_in_sprite(sprite_region, sprite_color)
        sprites.append(
            {
                "color": sprite_color,
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    sprite_by_holes = {int(sprite["holes"]): sprite for sprite in sprites}
    assert len(sprite_by_holes) == len(sprites)

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for frame in frame_grid:
        fr, fc = int(frame["out_r"]), int(frame["out_c"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = sprite_by_holes[int(frame["holes"])]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color

        sh, sw = int(sprite["h"]), int(sprite["w"])
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

    comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
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

    if _count_holes_in_sprite(sprite, sprite_color) != hole_count:
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

    inner = frame[1:-1, 1:-1]
    _lab, num = label_mask(inner != frame_color, connectivity=4)
    if int(num) != hole_count:
        return None
    return frame


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_color: int,
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_k = {int(k): spatch for k, spatch in sprites}
    for idx, (k, _frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = ri * frame_size, ci * frame_size
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = sprite_by_k[int(k)].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    N = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[N])
    frame_size = random.choice(FRAME_SIZE_RANGE)
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
    frame_color = random.choice(available)
    sprite_colors = random.sample([c for c in available if c != frame_color], N)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - N)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + N))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for i, hole_count in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, hole_count, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((int(hole_count), patch))

    frames: list[tuple[int, np.ndarray]] = []
    for hole_count in hole_counts:
        patch = _make_frame(frame_size, hole_count, frame_color)
        if patch is None:
            return None
        frames.append((int(hole_count), patch))

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, (_hole_count, frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = r0 + ri * frame_size, c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    for _hole_count, sprite_patch in sprites:
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
        frames,
        sprites,
        frame_color,
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


# New implicit assumptions after the D10 change:
# - The largest monochromatic component is the unified frame-grid. The generator
#   enforces this by using one frame color, keeping all sprites smaller, and
#   keeping sprites at least one padded cell away from the frame-grid.
# - All sprites share one common side length and remain independent square
#   components. The generator preserves the original `sprite_size = frame_size-2`
#   rule and never lets sprites touch.
# - Hole counts are unique across frames/sprites. The generator still uses one
#   consecutive range, so `holes -> sprite` remains a bijection.
