# Variant of 8698868d
# Dimension: D16 — 符号编码与图例 (symbol_instructions)
# Change: 无符号编码 — 配对规则完全从对象的结构属性（洞计数）中推断
#         → 颜色映射图例。输入顶边新增一个 2 行色带图例，定义 sprite_color -> frame_color。
#
# Design analysis
# ----------------
# spec:
#   core_change:
#     original: frame 和 sprite 通过“洞连通块数量相等”进行配对
#     variant: 先读取顶边图例中的颜色映射，再按颜色映射配对；洞数只保留为干扰信息
#   legend_geometry:
#     placement: 固定在输入最上边缘，占 2 行
#     entry_shape: 每个映射项是一个 2x2 竖向色带；上排两格 = sprite_color，下排两格 = frame_color
#     separation: 相邻映射项之间保留 1 列背景，避免色带粘连
#   affected_pipeline:
#     input_construction:
#       - 需要额外放置 legend band
#       - 需要让 legend 映射与洞数配对冲突，否则会退化回原题
#     solver:
#       - 原 main() 中“统计 frame/sprite 洞数后逐个匹配”的逻辑必须替换
#       - 仍然保留“识别大方框网格 + 小方形 sprite + 输出紧凑网格”的主干
#     output:
#       - 与原题相同：只输出紧凑 frame grid，不输出 legend
#   concrete_trace:
#     - 原题某个 frame 若有 2 个洞，就会拿到那个“也有 2 个洞”的 sprite
#     - 现在该 frame 只看自身边框颜色，例如边框是 red；顶边图例若显示 blue -> red，
#       那么无论 blue sprite 有几个洞，它都要被嵌入 red frame
#     - 差异点在匹配键：hole_count -> color legend
#   complexity_calibration:
#     - 若仍允许 5-6 对颜色，legend 会过长且颜色复用会削弱可推断性
#     - 因此将对象数收紧到 2-4，对应 4-8 个非背景色，保证 frame/sprite 颜色两两不同
#     - 洞数仍保持唯一且丰富，但只作为“旧规律会误导”的可见对照
#   implicit_assumption_review:
#     - 原代码把 sprite holes / frame holes 存进对象字典，并假定“存在唯一相等洞数对象”
#       这个前提在变体中失效，所以 main() 不能再依赖 holes 字段做匹配
#     - 原代码通过 size grouping 找 frame 和 sprite；legend 必须足够小，不能落入该筛选
#       因此 legend 只用 1x2 的单色连通块，不会被识别成 4x4 以上正方形
#     - 原代码默认 frame color 可从边框多数色读取，这一前提在变体中仍成立
#   placement_legality:
#     - legend 固定放在 row 0-1；frame grid 从 row >= 3 开始，避免覆盖 legend
#     - sprite 使用 random_free_location + padding=1，保证与 frame grid、legend 都不接触
#     - 输出仍按原题紧凑拼回 frame grid，尺寸合法
#   visual_observability:
#     - 顶边连续的小色带非常显眼，且每个输入都会出现
#     - 同时强制 legend pairing 与 hole pairing 为错排，确保“看 legend 才能做对”
#   cleanliness:
#     - legend 只占 2 行，不侵入主体区域；frame grid 与 sprites 仍分区清晰
#   color_inference:
#     - 每个样例只有 2-4 个映射项，且 sprite/frame 颜色不重名
#     - 上排颜色出现在散落 sprite 上，下排颜色出现在 frame 边框上，方向可由样例归纳

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location


def _read_legend_map(grid: np.ndarray, bg: int) -> dict[int, int]:
    """Read the 2-row top-edge legend: sprite_color -> frame_color."""
    H, W = grid.shape
    assert H >= 2

    band = (grid[:2, :] != int(bg)).any(axis=0)
    assert bool(band.any())

    mapping: dict[int, int] = {}
    c = 0
    while c < W:
        if not bool(band[int(c)]):
            c += 1
            continue
        c0 = int(c)
        while c + 1 < W and bool(band[int(c + 1)]):
            c += 1
        c1 = int(c)
        sub = grid[:2, c0 : c1 + 1]
        assert sub.shape[0] == 2
        assert bool((sub[0, :] != int(bg)).all())
        assert bool((sub[1, :] != int(bg)).all())
        top_vals = {int(v) for v in sub[0, :].tolist()}
        bottom_vals = {int(v) for v in sub[1, :].tolist()}
        assert len(top_vals) == 1 and len(bottom_vals) == 1
        sprite_color = next(iter(top_vals))
        frame_color = next(iter(bottom_vals))
        assert sprite_color not in mapping
        mapping[int(sprite_color)] = int(frame_color)
        c += 1

    assert mapping
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Read the top-edge legend to obtain sprite_color -> frame_color mapping.
    - Detect the large square frame grid and the smaller square sprites.
    - Place each sprite into the frame whose border color matches the legend target.
    - Recolor sprite hole pixels to the destination frame color, as in the original task.
    """
    g = np.array(grid, dtype=int)
    BG = 8

    legend_map = _read_legend_map(g, BG)
    frame_to_sprite = {int(frame_color): int(sprite_color) for sprite_color, frame_color in legend_map.items()}
    assert len(frame_to_sprite) == len(legend_map)

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    MIN_FILL_RATIO = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * MIN_FILL_RATIO):
            square_regions.append((int(h), comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    assert len(size_list) >= 2
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1])

    frames = sizes.get(int(frame_size), [])
    sprites_raw = sizes.get(int(sprite_size), [])
    assert frames and sprites_raw

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)

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
            frame_grid.append(
                {
                    "color": int(frame_color),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                }
            )

    sprites_by_color: dict[int, dict[str, np.ndarray | int]] = {}
    for comp in sprites_raw:
        b = comp["bbox"]
        sprite_region = crop_bbox(g, b)
        sprite_color = int(comp["color"])
        sprites_by_color[int(sprite_color)] = {
            "color": int(sprite_color),
            "content": sprite_region.copy(),
            "h": int(sprite_region.shape[0]),
            "w": int(sprite_region.shape[1]),
        }

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)

        sprite_color = frame_to_sprite.get(int(frame_color))
        assert sprite_color is not None
        sprite = sprites_by_color[int(sprite_color)]
        content = np.array(sprite["content"], dtype=int).copy()
        content[content != int(sprite_color)] = int(frame_color)
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


# --------------- generator ---------------

BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 4)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 300
LEGEND_H = 2
LEGEND_CELL_W = 2
LEGEND_GAP = 1
LEGEND_RESERVED_ROWS = 3

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _legend_width(count: int) -> int:
    return int(count * LEGEND_CELL_W + max(0, count - 1) * LEGEND_GAP)


def _random_derangement(values: list[int]) -> list[int] | None:
    if len(values) <= 1:
        return None
    if len(values) == 2:
        return [int(values[1]), int(values[0])]
    for _ in range(100):
        perm = values[:]
        random.shuffle(perm)
        if all(int(a) != int(b) for a, b in zip(values, perm)):
            return [int(v) for v in perm]
    return None


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
            frontier: list[tuple[int, int]] = []
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

    _, num = label_mask(sprite != sprite_color, connectivity=4)
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

    _, num = label_mask(frame != frame_color, connectivity=4)
    if int(num) != int(hole_count):
        return None
    return frame


def _make_legend(sprite_colors: list[int], frame_colors: list[int]) -> np.ndarray:
    width = _legend_width(len(sprite_colors))
    legend = np.full((LEGEND_H, width), BG, dtype=int)
    for i, (sprite_color, frame_color) in enumerate(zip(sprite_colors, frame_colors)):
        c0 = int(i * (LEGEND_CELL_W + LEGEND_GAP))
        legend[0, c0 : c0 + LEGEND_CELL_W] = int(sprite_color)
        legend[1, c0 : c0 + LEGEND_CELL_W] = int(frame_color)
    return legend


def _build_expected_output(
    frames: list[tuple[int, int, np.ndarray]],
    sprites: list[tuple[int, int, np.ndarray]],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    for idx, (frame_color, _frame_holes, _fpatch) in enumerate(frames):
        rr = int((idx // cols_grid) * frame_size)
        cc = int((idx % cols_grid) * frame_size)
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)
        sprite_color, _sprite_holes, spatch = sprites[idx]
        content = spatch.copy()
        content[content == BG] = int(frame_color)
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
        assert int(sprite_color) != int(frame_color)
    return expected


def _build_hole_matched_output(
    frames: list[tuple[int, int, np.ndarray]],
    sprites: list[tuple[int, int, np.ndarray]],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_holes = {int(sprite_holes): spatch for _sprite_color, sprite_holes, spatch in sprites}
    for idx, (frame_color, frame_holes, _fpatch) in enumerate(frames):
        rr = int((idx // cols_grid) * frame_size)
        cc = int((idx % cols_grid) * frame_size)
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)
        content = sprite_by_holes[int(frame_holes)].copy()
        content[content == BG] = int(frame_color)
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    N = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[N])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2
    legend_w = _legend_width(N)

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_H = max(CANVAS_MIN, LEGEND_RESERVED_ROWS + grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_W = max(CANVAS_MIN, max(grid_pixel_w, legend_w) + sprite_size + SPRITE_PADDING)
    if min_H > CANVAS_MAX or min_W > CANVAS_MAX:
        return None

    H = random.randint(min_H, CANVAS_MAX)
    W = random.randint(min_W, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    picked = random.sample(available, 2 * N)
    frame_colors = [int(c) for c in picked[:N]]
    sprite_colors = [int(c) for c in picked[N:]]

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    if max_hole < N:
        return None
    max_start = max(0, max_hole - N)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    frame_holes = list(range(start, start + N))
    random.shuffle(frame_holes)
    sprite_holes = _random_derangement(frame_holes)
    if sprite_holes is None:
        return None

    sprites: list[tuple[int, int, np.ndarray]] = []
    for i, hole_count in enumerate(sprite_holes):
        patch = _make_sprite_interior(sprite_size, int(hole_count), int(sprite_colors[i]))
        if patch is None:
            return None
        sprites.append((int(sprite_colors[i]), int(hole_count), patch))

    frames: list[tuple[int, int, np.ndarray]] = []
    for i, hole_count in enumerate(frame_holes):
        patch = _make_frame(frame_size, int(hole_count), int(frame_colors[i]))
        if patch is None:
            return None
        frames.append((int(frame_colors[i]), int(hole_count), patch))

    canvas = np.full((H, W), BG, dtype=int)

    legend = _make_legend(sprite_colors, frame_colors)
    legend_c = random.randint(0, W - legend_w)
    paste(canvas, legend, 0, legend_c, transparent=BG)

    r0 = random.randint(LEGEND_RESERVED_ROWS, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, (_frame_color, _hole_count, fpatch) in enumerate(frames):
        rr = int(r0 + (idx // cols_grid) * frame_size)
        cc = int(c0 + (idx % cols_grid) * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = fpatch

    for _sprite_color, _hole_count, spatch in sprites:
        try:
            r, c = random_free_location(
                canvas,
                spatch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, spatch, r, c, transparent=BG)

    expected = _build_expected_output(frames, sprites, rows_grid, cols_grid, frame_size, sprite_size)
    hole_expected = _build_hole_matched_output(frames, sprites, rows_grid, cols_grid, frame_size, sprite_size)
    if np.array_equal(expected, hole_expected):
        return None

    try:
        out = main(canvas)
    except Exception:
        return None

    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
