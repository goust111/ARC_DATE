# Combo variant of 8698868d (index 23)
# combo_spec: D7/2+D10/2
# Name: 标记隐形框架且全同框架色

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location

# Design analysis
# ---
# source_task: 8698868d
#
# spec:
#   variant_core:
#     - 所有普通框架都使用同一个可见框色；它们在紧密网格中相互接触时，会在
#       单色连通意义下合并成更大的组件。
#     - 原本应与背景同色的“隐形框架”不再真的隐形，而是用一个专用标记色画出。
#       这个标记色只在输入中作为提示存在；输出时这些框的边框与精灵洞像素都要
#       还原为背景色。
#     - 精灵仍然靠“洞的 4 连通分量个数”与框架配对，不能靠框色配对。
#   concrete_behavior:
#     - 原题可把最大尺寸的一批正方形单色组件直接当作 frames。这里不成立：
#       普通框因为全同色会彼此并块，标记框又可能只占少数格，frame-grid 不再
#       等于“一组同尺寸单色正方形”。
#     - 新 solver 先从较小的独立正方形组件找 sprite_size，再反推
#       frame_size = sprite_size + 2。随后改用 multicolor 前景连通组件抓取整块
#       frame-grid bbox，再按固定步长切成 frame slots。
#     - 每个 slot 的边框色只有两类：多数出现的统一框色，和少数出现的标记色。
#       标记色对应“输出改回背景色”的那类框。
#   behavior_trace:
#     - 例子：2x2 网格、frame_size=6、sprite_size=4。三个普通框都为黑色，
#       右下角原本隐形的框在输入里改成灰色。四个洞计数分别是 0/1/2/3。
#     - 原逻辑会尝试从 monochromatic components 中直接取最大正方形尺寸当 frames，
#       但这里三个黑框会连成一个 12x12 的 L 形大组件，灰框是一个单独 6x6 组件，
#       已经没有“四个独立 frame 组件”可拿。
#     - 新逻辑会先识别四个 4x4 精灵，推出 frame_size=6；再把整块 frame-grid 当作
#       最大 multicolor 前景组件，按 6x6 切成四格，分别统计各格相对自身边框色的洞数。
#       最后把灰色标记框对应的输出边框改成背景色，其余三格保留黑色。
#   complexity_calibration:
#     - 若继续允许 N=2，则“一格普通框 + 一格标记框”在输入中无法仅凭结构区分
#       哪个颜色应在输出中消失，规则会退化为不可判定。因此本变体把 frame 数下界
#       提高到 3，并强制普通框数量严格多于标记框数量。
#     - 标记框数量保持少数（1 到 floor((N-1)/2)），这样“多数统一框色 / 少数特殊标记色”
#       在 2-4 个样例中肉眼可归纳。
#     - 精灵颜色全部避开普通框色与标记色，减少同色遮蔽；洞计数仍是一组连续唯一值，
#       保持一一匹配且不引入额外自由度。
#   implicit_assumption_audit:
#     - 原 `sizes[max_size] -> frames` 假设 frame 是独立单色方块。D10 破坏该点，
#       融合后仍不成立，必须改成“最小正方形组件是 sprite，最大 multicolor 组件是 frame-grid”。
#     - 原 `frame_positions` 由逐框组件左上角反推出网格。融合后没有逐框组件列表，
#       改为使用 frame-grid bbox 和已知 frame_size 推出行列数。
#     - 原 `frame_color = border majority` 后默认此色也应进入输出。D7/2 破坏该点：
#       若边框色是特殊标记色，则输出色应为背景，而不是输入边框色。
#     - 原“每个框色不同”让 frame_color 自带身份信息；融合后完全不能靠框色配对，
#       只能靠洞计数，生成器因此继续保证洞计数唯一。
#   placement_legality:
#     - 先整体放置紧密 frame-grid，再用 `random_free_location(..., padding=1)` 放置精灵，
#       保证精灵既不重叠，也不会与 frame-grid 发生 4/8 邻接并并入大前景组件。
#     - frame-grid bbox 的宽高都必须是 `frame_size` 的整数倍，且分格数等于精灵数。
#     - 标记框和普通框都只是 1 像素厚边框的实心方块加洞，不引入额外重叠层。
#   visual_observability:
#     - 输入中可直接看到：一大片由同色普通框和少量异色标记框组成的规则网格，
#       外围散落着一组更小的彩色精灵。标记框不是随机杂色，因为它总出现在与普通框
#       同尺寸同结构的位置上。
#     - 输出中，普通框仍保留统一框色，而标记框整格边框会“消失”为背景色；同时该格
#       内嵌精灵的洞也被补成背景色。这让“标记色只用于输入提示”具备明确可见的结果。
#   cleanliness:
#     - 变化只发生在 frame-grid 这一层：没有额外图例、编号或覆盖纹理。
#       输入中两类框共享完全相同的几何骨架，只靠颜色角色区分，视觉上整洁。
#   color_inference:
#     - 颜色变化规则受位置与对象类型约束，而非颜色值本身：只有 frame-grid 里的少数
#       标记框会在输出中转为背景色；普通框和所有精灵主体色不会被任意重映射。


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (3, 6)
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


def _square_regions(comps: list[dict]) -> dict[int, list[dict]]:
    sizes: dict[int, list[dict]] = {}
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 12 and area >= int(h * h * MIN_FILL_RATIO):
            sizes.setdefault(h, []).append(comp)
    return sizes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the frame grid by matching each scattered sprite to the frame slot with the same hole-count signature.

    Input structure:
    - Background is fixed to azure (8). One region contains a tight grid of larger square frames; another region contains smaller square sprites.
    - Every ordinary frame border uses the same visible color. Frames that were originally invisible are shown instead with one special marker color.
    - Each sprite is a smaller monochrome square with zero or more hole components carved out of it.

    Rule:
    - Infer the common sprite size from the independent smaller square components, then deduce the larger frame size.
    - Locate the entire mixed-color frame grid as the largest foreground component, split it into equal frame slots, and count holes inside each slot relative to that slot's own border color.
    - Match each frame slot to the unique sprite with the same connected-hole count.
    - Output the tight frame grid only: ordinary slots keep the shared frame color, while marker-colored slots convert back to background; matched sprite holes are recolored to the slot's output border color.

    Key insight:
    - The uncommon frame color is not a second normal frame color. It is only an input-side marker for formerly invisible frames, so pairing still depends on hole counts, while the slot type decides whether the final border stays colored or becomes background.
    """
    g = np.array(grid, dtype=int)

    mono_comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    assert mono_comps

    square_by_size = _square_regions(mono_comps)
    assert square_by_size

    sprite_size = min(square_by_size)
    sprites_raw = square_by_size[sprite_size]
    frame_size = int(sprite_size + 2)

    fg_comps = find_components(g, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    assert fg_comps
    frame_grid_comp = max(fg_comps, key=lambda comp: int(comp["area"]))
    frame_bbox = frame_grid_comp["bbox"]
    grid_h = int(bbox_h(frame_bbox))
    grid_w = int(bbox_w(frame_bbox))
    assert grid_h % frame_size == 0 and grid_w % frame_size == 0

    n_rows = int(grid_h // frame_size)
    n_cols = int(grid_w // frame_size)
    assert n_rows * n_cols == len(sprites_raw)

    frame_r0, frame_c0, _frame_r1, _frame_c1 = frame_bbox
    frame_grid = []
    border_colors: list[int] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(frame_r0 + ri * frame_size)
            inp_c = int(frame_c0 + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            assert region.shape == (frame_size, frame_size)

            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            border_color = int(Counter(map(int, border.tolist())).most_common(1)[0][0])
            border_colors.append(border_color)

            inner = region[1:-1, 1:-1]
            hole_mask = inner != int(border_color)
            _lab, num_holes = label_mask(hole_mask, connectivity=4)
            frame_grid.append(
                {
                    "input_color": int(border_color),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "holes": int(num_holes),
                }
            )

    color_counts = Counter(border_colors)
    assert len(color_counts) == 2
    visible_frame_color, _ = color_counts.most_common(1)[0]

    sprites = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        num_holes = _count_holes_in_sprite(sprite_region, sprite_color)
        sprites.append(
            {
                "color": int(sprite_color),
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
        input_color = int(frame["input_color"])
        output_color = int(visible_frame_color) if input_color == int(visible_frame_color) else int(BG)
        out[fr : fr + frame_size, fc : fc + frame_size] = output_color

        sprite = sprite_by_holes[int(frame["holes"])]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = output_color

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


def _make_frame(frame_size: int, hole_count: int, input_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), input_color, dtype=int)
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
    _lab, num = label_mask(inner != input_color, connectivity=4)
    if int(num) != hole_count:
        return None
    return frame


def _build_expected_output(
    frames: list[dict],
    sprites: list[tuple[int, np.ndarray]],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_k = {int(k): spatch for k, spatch in sprites}
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = ri * frame_size, ci * frame_size
        output_color = int(frame["output_color"])
        expected[rr : rr + frame_size, cc : cc + frame_size] = output_color
        content = sprite_by_k[int(frame["holes"])].copy()
        content[content == BG] = output_color
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
    visible_frame_color, marker_color = random.sample(available, 2)
    sprite_palette = [c for c in available if c not in {visible_frame_color, marker_color}]
    sprite_colors = random.sample(sprite_palette, N)

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

    max_marked = max(1, (N - 1) // 2)
    marked_count = random.randint(1, max_marked)
    marked_indices = set(random.sample(range(N), marked_count))

    frames: list[dict] = []
    for idx, hole_count in enumerate(hole_counts):
        input_color = marker_color if idx in marked_indices else visible_frame_color
        output_color = BG if idx in marked_indices else visible_frame_color
        patch = _make_frame(frame_size, hole_count, input_color)
        if patch is None:
            return None
        frames.append(
            {
                "holes": int(hole_count),
                "input_color": int(input_color),
                "output_color": int(output_color),
                "patch": patch,
            }
        )

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = r0 + ri * frame_size, c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame["patch"]

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


# New implicit assumptions after the combo change:
# - The smallest valid square monochromatic components are the scattered sprites.
#   The generator enforces `frame_size = sprite_size + 2 >= sprite_size + 2`, keeps
#   sprite colors distinct from both frame colors, and prevents sprite touching.
# - The largest multicolor foreground component is the entire frame-grid. This holds
#   because the grid is placed contiguously and sprites are scattered with padding.
# - Exactly two frame-slot input colors appear: a majority visible frame color and a
#   minority marker color. The generator guarantees at least one of each and ensures
#   the visible frames are strictly more numerous, so `Counter(border_colors)` is stable.
# - Hole counts remain a bijection between frames and sprites. The generator preserves
#   one consecutive unique range, so `holes -> sprite` matching is deterministic.
