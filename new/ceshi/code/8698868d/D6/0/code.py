# Variant of 8698868d
# Dimension: D6 — 轨迹：精灵源位置的处理 (trajectory)
# Change: 默认消失 — 精灵不出现在输出中（输出仅为框架网格）... → 保留精灵在原位置

from __future__ import annotations

# Design analysis for D6 keep_source
# ----------------------------------
# variant_spec:
#   core_change:
#     original: 输出被裁成紧密框架网格；散落在外部的 sprites 不出现在输出里，看起来像被“移动”进框架。
#     new: 输出保留完整输入画布。框架位置仍要嵌套匹配 sprite，但外部散落 sprite 也原封不动保留，
#       因此视觉上是“复制到框架中”，不是“从源位置搬走”。
#   concrete_pixel_behavior:
#     input:
#       - 一组较大的 frame 方块紧密排成网格，位于画布某处。
#       - 一组较小的 sprite 方块散落在画布其他背景区域。
#     output:
#       - 画布尺寸与输入完全相同。
#       - 每个 frame 所在的方块区域被重建为纯 frame_color，并在内部嵌入按洞连通数匹配到的 sprite。
#       - sprite 的洞像素仍被重着色为对应 frame_color。
#       - 原始散落 sprite 不被删除，不被改色，也不被移动。
#   flow_trace:
#     example:
#       - 原逻辑在 build output 阶段新建一个 rows_grid*frame_size 的小画布，只画 frame grid；因此外部 sprite 自动消失。
#       - 新逻辑改为先复制整张输入画布，再在每个 frame 位置重画整块 frame，并把匹配 sprite 填入 interior。
#       - 差异集中在“输出承载介质”：从 cropped frame-only canvas 改成 full-canvas in-place rewrite。
#   complexity_calibration:
#     - 匹配键仍然只有“洞连通分量数”，不引入新匹配自由度，避免题意漂移。
#     - 视觉差异足够明显：输出里同时出现源 sprite 和框架内副本。
#     - 仍保留 2-6 个对象，且每个样例里有多个不同 hole counts，避免退化成单对象复制。
#   implicit_assumption_audit:
#     - find_components / 方块筛选 / 按尺寸区分 frames 与 sprites：仍成立，因为输入结构不变。
#     - frame_positions 推断紧密网格：仍成立，生成器继续保证无缝网格。
#     - 原代码里 out_h/out_w 使用紧密网格尺寸；这是旧 D6 的关键假设，新变体下失效，必须改成 out = g.copy()。
#     - 原代码里对 frame 区域先整体填 frame_color 再嵌套 sprite；这个步骤必须保留，否则输入 frame 里的 bg 洞会残留。
#     - 原代码不处理 sprite 源位置，因为输出裁剪后看不到它们；新变体下需要显式“什么都不做”，即不清除源 sprite。
#   placement_legality:
#     - sprites 仍必须放在 frame grid 之外，且不与任何非背景像素重叠。
#     - 使用 random_free_location(..., padding=1) 保证 sprite 与 frames / 彼此分离，避免颜色相同导致组件粘连。
#     - frame grid 仍完整位于画布内；输出只在这些 frame 方块位置覆写，因此不会越界。
#   visual_observability:
#     - 人类可直接比较 input/output：输入中散落 sprite 在输出里仍可见，同时 frame 内出现对应副本。
#     - 若 sprites 被放得离 grid 太近，复制关系会变乱；因此继续使用 padding 保持留白。
#   visual_cleanliness:
#     - 不在同一区域叠加超过两层信息：frame 区域只显示“重建后的 frame+嵌套 sprite”，外部区域只显示原始 sprites。
#     - 输出不再裁剪，但背景留白稳定，画面仍可读。
#   color_inference:
#     - 颜色规则不变：sprite 本体色保持自身颜色，sprite 洞被 frame_color 填补。
#     - 不引入任何基于数值大小的排序或颜色映射。
#   generator_checks:
#     - oracle 必须比较完整画布输出，而不是旧版紧密网格。
#     - 为防退化，后续测试统计“输出中位于 frame grid 外的非背景像素是否与输入一致且数量>0”；
#       这能量化 keep_source 是否真的发生。
#   new_main_assumptions_to_guarantee:
#     - 每个 frame hole count 都能在 sprites 中唯一匹配；生成器仍用 0..N-1 的双射保证。
#     - sprite 尺寸始终等于 frame_size-2；生成器继续按此构造，保证能无裁切嵌入。
#     - frame grid 区域外不存在需要被 solver 修改的对象；生成器只把 frame grid 作为唯一变换区。

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVERS_DIR = _ROOT / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVERS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then copy each sprite into its frame.

    D6 variant behavior:
    - The output keeps the full original canvas.
    - Sprites remain visible at their original scattered source positions.
    - Matching and nesting are unchanged; only the trajectory changes from move/disappear to copy/keep-source.
    """
    g = np.array(grid, dtype=int)
    BG = 8  # Azure background for this puzzle family

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
            square_regions.append((int(h), comp))

    assert square_regions

    # Group by size - frames are larger, sprites are smaller.
    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(int(frame_size), [])
    sprites_raw = sizes.get(int(sprite_size), [])
    assert frames

    # Determine frame grid structure from frame positions.
    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    # Build frame grid metadata (including the original input positions).
    frame_grid = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r:inp_r + frame_size, inp_c:inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])

            inner = region[1:-1, 1:-1]
            hole_mask = inner != int(frame_color)
            _lab, num_holes = label_mask(hole_mask, connectivity=4)

            frame_grid.append(
                {
                    "color": int(frame_color),
                    "inp_r": int(inp_r),
                    "inp_c": int(inp_c),
                    "holes": int(num_holes),
                }
            )

    # Process sprites - count holes in each.
    sprites = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != int(sprite_color)
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "color": int(sprite_color),
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    # Match frames to sprites by hole count.
    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for frame in frame_grid:
        for idx, sprite in enumerate(sprites):
            if idx not in used_sprites and int(frame["holes"]) == int(sprite["holes"]):
                matches[(int(frame["inp_r"]), int(frame["inp_c"]))] = sprite
                used_sprites.add(int(idx))
                break

    # D6 change: keep the original canvas and modify frame locations in place.
    out = g.copy()

    for frame in frame_grid:
        fr, fc = int(frame["inp_r"]), int(frame["inp_c"])
        frame_color = int(frame["color"])

        # Rebuild the whole frame square so original frame holes disappear.
        out[fr:fr + frame_size, fc:fc + frame_size] = int(frame_color)

        sprite = matches.get((int(fr), int(fc)))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = int(frame_color)

        sh, sw = int(sprite["h"]), int(sprite["w"])
        out[fr + 1:fr + 1 + sh, fc + 1:fc + 1 + sw] = content

    return out


# --------------- generator ---------------

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

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` interior cells that are pairwise 4-separated."""
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
    """Build a sprite with exactly `hole_count` 4-connected bg-hole components."""
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

    # Grow components without allowing two hole components to merge.
    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for comp_idx, comp in enumerate(comps):
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
                        for other_idx, other in enumerate(comps)
                        if other_idx != comp_idx
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
    if int(num) != hole_count:
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Build a frame square with isolated single-cell bg holes in the interior."""
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
    if int(num) != hole_count:
        return None
    return frame


def _build_expected_output(
    canvas: np.ndarray,
    frames: list[dict],
    sprites: list[tuple[int, np.ndarray]],
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    """Build the full-canvas expected output for the keep_source variant."""
    expected = np.array(canvas, dtype=int).copy()
    sprite_by_holes = {int(hole_count): patch for hole_count, patch in sprites}
    for frame in frames:
        rr = int(frame["r"])
        cc = int(frame["c"])
        holes = int(frame["holes"])
        frame_color = int(frame["color"])

        expected[rr:rr + frame_size, cc:cc + frame_size] = int(frame_color)
        content = np.array(sprite_by_holes[holes], dtype=int).copy()
        content[content == BG] = int(frame_color)
        expected[rr + 1:rr + 1 + sprite_size, cc + 1:cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    # 1. Pick N and grid layout.
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    # 2. Canvas size (must fit frame grid + room for sprites).
    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    height = random.randint(min_h, CANVAS_MAX)
    width = random.randint(min_w, CANVAS_MAX)

    # 3. Colors: N distinct non-bg per group; groups may overlap.
    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, n_frames)
    sprite_colors = random.sample(available, n_frames)

    # 4. Hole counts: consecutive range start..start+N-1.
    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - n_frames)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    # 5. Build sprites and frames.
    sprites: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, hole_count, sprite_colors[idx])
        if patch is None:
            return None
        sprites.append((int(hole_count), patch))

    frame_patches: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(hole_counts):
        patch = _make_frame(frame_size, hole_count, frame_colors[idx])
        if patch is None:
            return None
        frame_patches.append((int(hole_count), patch))

    # 6. Place frame grid on canvas.
    canvas = np.full((height, width), BG, dtype=int)
    r0 = random.randint(0, height - grid_pixel_h)
    c0 = random.randint(0, width - grid_pixel_w)
    frame_records: list[dict] = []
    for idx, (hole_count, frame_patch) in enumerate(frame_patches):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr:rr + frame_size, cc:cc + frame_size] = frame_patch
        frame_records.append(
            {
                "holes": int(hole_count),
                "color": int(frame_colors[idx]),
                "r": int(rr),
                "c": int(cc),
            }
        )

    # 7. Place sprites in remaining bg area (non-overlapping and separated).
    for _hole_count, sprite_patch in sprites:
        try:
            rr, cc = random_free_location(
                canvas,
                sprite_patch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, sprite_patch, rr, cc, transparent=BG)

    # 8. Oracle check against the full-canvas keep_source output.
    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(canvas, frame_records, sprites, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
