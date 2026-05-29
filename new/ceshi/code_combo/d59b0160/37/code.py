# Combo variant of d59b0160 (index 37)
# combo_spec: D13/2+D14/0
# Name: 保留分量移动到特定区域+裁剪输出

import math
import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, paste

# Design analysis for the combo variant
# - concrete_behavior:
#   输入仍是 16x16 背景网格，左上角仍有 4x4 的 L 形 legend，其他对象仍是与背景 4-分离的多色矩形分量。
#   原题只做“包含全部 legend 颜色就删除”；本融合版先做同样的删除，再把所有保留下来的 4-连通分量
#   依原输入中的阅读顺序搬到左上角，采用 1 像素间隔做紧凑货架式排布，最后仅输出该拼贴的紧框。
# - flow_trace:
#   例：legend 会留下一个 4x4 的边框分量和一个孤立色点，再加上 3 个不匹配矩形；原题输出仍是 16x16，
#   只是删除匹配矩形后的原位残留。融合后，同样先删掉匹配矩形，但 5 个幸存分量会被依次贴到
#   (0,0)、(0,5)、下一行左侧等位置，最终只保留覆盖这组拼贴的最小矩形。差异不仅是“删掉谁”，
#   还包括“保留下来的东西不再留在原地”以及“输出尺寸跟着新布局收紧”。
# - complexity_calibration:
#   只引入一种重排规则：按原 bbox 左上角排序，按固定宽度上限做左对齐换行排布。这样变体肉眼可见，
#   但不会出现依赖颜色 id、面积排序或多种不同摆放模板的不可归纳复杂度。generator 把 3 个保留对象
#   强制分散到不同象限，使移动效果稳定可见，并要求最终 combo 输出不能退化成“仅删除后裁剪”。
# - implicit_assumptions_review:
#   * `_bg_boundary` 仍成立，因为外边框仍以背景为主，generator 避免大对象占满边缘。
#   * `_find_legend` 的左上角 L 形检测不变；legend 仍固定在左上，且内部仍是 3 个目标颜色。
#   * `components_mask(g != bg, connectivity=4)` 仍是正确分量定义，因为删除判定和重排对象都基于同一批
#     非背景 4-连通分量。
#   * 新增的打包步骤依赖“每个幸存分量都可独立裁成 patch 再粘贴”；generator 保证输入分量彼此留有空隙。
#   * 新增的排序步骤依赖“阅读顺序有可见意义”；generator 用分区放置让幸存对象的先后顺序稳定清楚。
#   * 新增的裁剪步骤依赖“至少有一个幸存分量”；legend 被构造为两个不可能同时含齐全部目标色的分量，
#     因而输出绝不会空。
# - placement_legality:
#   生成器所有对象都通过候选位置枚举放置，要求 bbox 周围保留 1 像素背景 moat，避免重叠、越界或误连通。
#   3 个保留对象被放在互相分离的区域；删除对象再在剩余背景里安放，其中至少 1 个需要把输入幸存区域
#   的原位 bbox 向下或向右撑大，以保证“删除”在输入侧也有可观察痕迹。
# - visual_observability:
#   解题者在样例中会看到三件事同时发生：含齐图例颜色的对象消失；其余对象被集中搬到左上角；
#   输出画布只保留新拼贴的紧框。因为保留对象在输入里被故意打散，输出中的聚拢重排不会和单纯裁剪混淆。
# - neatness_and_inferability:
#   输出内部只使用统一的 1 像素间隔，不引入额外标记或颜色映射。排序依据是可见空间位置而非颜色编号；
#   颜色仍只承担“是否满足 legend 颜色全集”的逻辑角色，避免出现人类无法从少量样例归纳的隐藏规则。


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the top-left L-shaped legend and return (size, border_color, interior_colors).
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(x)
            return int(sz), int(x), colors
    raise AssertionError("no legend border found in grid")


def _component_record(g: np.ndarray, pts: np.ndarray) -> dict:
    """Package one component as coords + bbox + cropped patch + color set."""
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    return {
        "coords": pts,
        "bbox": (r0, c0, r1, c1),
        "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
        "colors": {int(g[r, c]) for r, c in pts},
    }


def _split_components(g: np.ndarray, bg: int, legend_colors: set[int]) -> tuple[list[dict], list[dict]]:
    """Split all non-background components into kept vs deleted groups."""
    kept: list[dict] = []
    deleted: list[dict] = []
    for pts in components_mask(g != bg, connectivity=4):
        rec = _component_record(g, pts)
        if legend_colors.issubset(rec["colors"]):
            deleted.append(rec)
        else:
            kept.append(rec)
    kept.sort(key=lambda rec: (rec["bbox"][0], rec["bbox"][1], rec["bbox"][2], rec["bbox"][3]))
    deleted.sort(key=lambda rec: (rec["bbox"][0], rec["bbox"][1], rec["bbox"][2], rec["bbox"][3]))
    return kept, deleted


def _pack_components(
    kept: list[dict],
    bg: int,
    width_limit: int,
    *,
    gap: int = 1,
    return_positions: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Pack kept component patches into a left-anchored shelf layout and crop the result.
    """
    if not kept:
        empty = np.full((0, 0), int(bg), dtype=int)
        return (empty, []) if return_positions else empty

    width_limit = int(width_limit)
    if width_limit <= 0:
        raise ValueError("width_limit must be positive")
    if any(rec["patch"].shape[1] > width_limit for rec in kept):
        raise ValueError("component wider than pack width")

    temp_h = sum(int(rec["patch"].shape[0]) for rec in kept) + gap * (len(kept) - 1)
    canvas = np.full((max(1, temp_h), width_limit), int(bg), dtype=int)
    positions: list[tuple[int, int]] = []

    row = 0
    col = 0
    row_h = 0
    for rec in kept:
        patch = rec["patch"]
        h, w = patch.shape
        if col > 0 and col + w > width_limit:
            row += row_h + gap
            col = 0
            row_h = 0
        paste(canvas, patch, row, col)
        positions.append((int(row), int(col)))
        col += w + gap
        row_h = max(row_h, int(h))

    packed_bbox = bbox(canvas != bg)
    if packed_bbox is None:
        out = np.full((0, 0), int(bg), dtype=int)
    else:
        out = crop_bbox(canvas, packed_bbox)
    return (out, positions) if return_positions else out


def _crop_after_deletion(g: np.ndarray, bg: int, legend_colors: set[int]) -> np.ndarray:
    """Reference D14-style result: delete matching components, keep survivors in place, then crop."""
    full = g.copy()
    _kept, deleted = _split_components(g, bg, legend_colors)
    for rec in deleted:
        for r, c in rec["coords"]:
            full[r, c] = bg
    keep_bbox = bbox(full != bg)
    if keep_bbox is None:
        return np.full((0, 0), int(bg), dtype=int)
    return crop_bbox(full, keep_bbox)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete legend-matching components, repack the survivors, and crop to the new collage.

    Input structure:
    - The canvas is mostly background.
    - A top-left legend is marked by a shared-color bottom edge and right edge; its interior displays the target palette.
    - All other foreground shapes are separate multicolor 4-connected components on the background.

    Rule:
    - Read the legend's target color set from the interior of the top-left marker.
    - Remove every non-background 4-connected component whose color set contains all target colors.
    - Keep every other component, sort the survivors by their original top-left position, and move them into a compact left-top packing with 1-cell gaps.
    - Return only the minimal bounding rectangle covering that packed collage.

    Key insight: The legend still decides which components survive, but surviving components are no longer copied in place; they are gathered into a tight upper-left collage, and the final answer is cropped to that rearranged layout.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    kept, _deleted = _split_components(g, bg, legend_colors)
    return _pack_components(kept, bg, g.shape[1])


# --------------- generator (combo D13/2 + D14/0) ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
KEEP_TALL_H = (5, 8)
KEEP_TALL_W = (2, 4)
KEEP_WIDE_H = (3, 5)
KEEP_WIDE_W = (5, 7)
KEEP_MED_H = (3, 6)
KEEP_MED_W = (2, 6)
DEL_TALL_H = (6, 10)
DEL_TALL_W = (2, 4)
DEL_WIDE_H = (3, 5)
DEL_WIDE_W = (5, 8)
DEL_MED_H = (3, 6)
DEL_MED_W = (2, 7)

# -- Component fill density (fraction of non-zero sprinkled colors within color-0 body) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Packing / placement controls --
PADDING = 1
PACK_GAP = 1
KEEP_ZONES = [
    {"row_lo": 1, "row_hi": 3, "col_lo": 8, "col_hi": 10},
    {"row_lo": 8, "row_hi": 10, "col_lo": 1, "col_hi": 3},
    {"row_lo": 8, "row_hi": 10, "col_lo": 8, "col_hi": 10},
]
MIN_TOTAL_SHIFT = 10
MAX_SURVIVOR_R1 = 13
MAX_SURVIVOR_C1 = 13

# -- Generation retry budget --
MAX_RETRIES = 1200


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend and place three target colors with one guaranteed isolated swatch."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    isolated = random.choice([(0, 1), (1, 0), (1, 1)])
    border_side = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        p for p in border_side
        if abs(p[0] - isolated[0]) + abs(p[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color
        out[r, c] = color


def _component_dims(*, deleted: bool) -> tuple[int, int]:
    """Sample component dimensions with slightly tighter kept-object bounds for reliable repacking."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if deleted:
        if mode == "tall":
            return random.randint(*DEL_TALL_H), random.randint(*DEL_TALL_W)
        if mode == "wide":
            return random.randint(*DEL_WIDE_H), random.randint(*DEL_WIDE_W)
        return random.randint(*DEL_MED_H), random.randint(*DEL_MED_W)
    if mode == "tall":
        return random.randint(*KEEP_TALL_H), random.randint(*KEEP_TALL_W)
    if mode == "wide":
        return random.randint(*KEEP_WIDE_H), random.randint(*KEEP_WIDE_W)
    return random.randint(*KEEP_MED_H), random.randint(*KEEP_MED_W)


def _component_patch(h: int, w: int, targets: list[int], deleted: bool) -> np.ndarray:
    """Build an h x w solid color-0 component with sprinkled palette cells."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if deleted:
        lo = max(lo, NUM_TARGETS)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if deleted:
        colors = list(targets) + [random.choice(PALETTE) for _ in range(n_colored - NUM_TARGETS)]
    else:
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        allowed = non_targets + random.sample(targets, used_target_count)
        colors = [random.choice(allowed) for _ in range(n_colored)]

    random.shuffle(colors)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _candidate_locations(
    canvas: np.ndarray,
    patch: np.ndarray,
    *,
    row_lo: int = 0,
    row_hi: int | None = None,
    col_lo: int = 0,
    col_hi: int | None = None,
    padding: int = PADDING,
    predicate=None,
) -> list[tuple[int, int]]:
    """Enumerate legal top-left locations under optional bounds plus a background moat."""
    h, w = patch.shape
    H, W = canvas.shape
    row_hi = H - h if row_hi is None else min(int(row_hi), H - h)
    col_hi = W - w if col_hi is None else min(int(col_hi), W - w)
    row_lo = max(0, int(row_lo))
    col_lo = max(0, int(col_lo))

    if row_lo > row_hi or col_lo > col_hi:
        return []

    out = []
    for r in range(row_lo, row_hi + 1):
        for c in range(col_lo, col_hi + 1):
            rr0 = max(0, r - padding)
            cc0 = max(0, c - padding)
            rr1 = min(H, r + h + padding)
            cc1 = min(W, c + w + padding)
            if not np.all(canvas[rr0:rr1, cc0:cc1] == BG):
                continue
            if predicate is not None and not predicate(r, c):
                continue
            out.append((int(r), int(c)))
    return out


def _place_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    *,
    row_lo: int = 0,
    row_hi: int | None = None,
    col_lo: int = 0,
    col_hi: int | None = None,
    padding: int = PADDING,
    predicate=None,
) -> tuple[int, int]:
    """Choose a random legal location and paste the patch there."""
    cands = _candidate_locations(
        canvas,
        patch,
        row_lo=row_lo,
        row_hi=row_hi,
        col_lo=col_lo,
        col_hi=col_hi,
        padding=padding,
        predicate=predicate,
    )
    if not cands:
        raise ValueError("no legal location")
    r, c = random.choice(cands)
    paste(canvas, patch, r, c)
    return r, c


def _place_kept_in_zone(
    inp: np.ndarray,
    survivor: np.ndarray,
    targets: list[int],
    zone: dict,
) -> tuple[int, int, tuple[int, int]]:
    """Place one kept component into a prescribed input zone."""
    for _ in range(40):
        h, w = _component_dims(deleted=False)
        patch = _component_patch(h, w, targets, deleted=False)
        try:
            r, c = _place_patch(
                inp,
                patch,
                row_lo=zone["row_lo"],
                row_hi=zone["row_hi"],
                col_lo=zone["col_lo"],
                col_hi=zone["col_hi"],
            )
        except ValueError:
            continue
        paste(survivor, patch, r, c)
        return r, c, (h, w)
    raise ValueError("failed to place kept component in zone")


def _place_deleted_component(
    inp: np.ndarray,
    targets: list[int],
    *,
    predicate=None,
) -> tuple[int, int, tuple[int, int]]:
    """Place one deleted component anywhere legal."""
    for _ in range(60):
        h, w = _component_dims(deleted=True)
        patch = _component_patch(h, w, targets, deleted=True)
        try:
            r, c = _place_patch(inp, patch, predicate=predicate)
        except ValueError:
            continue
        return r, c, (h, w)
    raise ValueError("failed to place deleted component")


def _translation_metric(kept: list[dict], packed_positions: list[tuple[int, int]]) -> int:
    """Sum of Manhattan shifts from original top-lefts to packed top-lefts."""
    total = 0
    for rec, (r_new, c_new) in zip(kept, packed_positions):
        r_old, c_old, _r1, _c1 = rec["bbox"]
        total += abs(int(r_old) - int(r_new)) + abs(int(c_old) - int(c_new))
    return int(total)


def _gen_attempt() -> np.ndarray | None:
    """Build one valid input for the combo behavior."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    survivor = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    legend_colors = set(targets)
    _place_legend(inp, survivor, targets)

    for zone in KEEP_ZONES:
        try:
            _place_kept_in_zone(inp, survivor, targets, zone)
        except ValueError:
            return None

    survivor_bbox = bbox(survivor != BG)
    if survivor_bbox is None:
        return None
    _sr0, _sc0, surv_r1, surv_c1 = survivor_bbox
    if surv_r1 > MAX_SURVIVOR_R1 or surv_c1 > MAX_SURVIVOR_C1:
        return None

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    num_deleted = component_count - NUM_KEPT

    anchor_deleted = False
    for idx in range(num_deleted):
        try:
            if idx == 0:
                def extends_survivor(r: int, c: int) -> bool:
                    return r > 4 and (r > surv_r1 - 1 or c > surv_c1 - 1)

                _place_deleted_component(inp, targets, predicate=extends_survivor)
                anchor_deleted = True
            else:
                _place_deleted_component(inp, targets)
        except ValueError:
            return None

    if not anchor_deleted:
        return None

    kept, _deleted = _split_components(survivor, BG, legend_colors)
    expected, packed_positions = _pack_components(
        kept,
        BG,
        GRID_W,
        gap=PACK_GAP,
        return_positions=True,
    )
    if expected.size == 0:
        return None

    combo = main(inp)
    if not np.array_equal(combo, expected):
        return None

    d14 = _crop_after_deletion(inp, BG, legend_colors)
    if np.array_equal(combo, d14):
        return None

    shift = _translation_metric(kept, packed_positions)
    if shift < MIN_TOTAL_SHIFT:
        return None

    if combo.shape == inp.shape:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
