# Variant of d59b0160
# Dimension: D14 — 网格输出形状 (grid_shape)
# Change: 输出与输入尺寸完全相同（same），删除的分量区域变为背景色
#   → 输出裁剪为保留分量的最小外接矩形

import math
import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, paste

# Design analysis for the D14 variant
# - concrete_behavior:
#   输入仍是 16x16 背景网格，左上角仍有 4x4 图例，其他对象仍是与背景 4-分离的多色实心矩形分量。
#   原题在删除“同时包含全部图例颜色”的分量后，保留整张 16x16 画布；本变体在同一步删除之后，
#   再把结果裁剪到“所有剩余非背景像素”的最小外接矩形。
# - flow_trace:
#   例：若 legend 在左上，三个保留矩形只占到第 11 行 / 第 12 列，但有一个待删除矩形延伸到第 15 行。
#   原逻辑：删除该矩形后，输出仍是 16x16，只是右下部分变成背景。
#   新逻辑：删除后先得到同样的全尺寸中间态，再把它裁成 0..11 行、0..12 列的紧框。
#   差异只发生在“最终输出形状”，删除判定本身不变。
# - complexity_calibration:
#   只改 D14，不改 legend 语义、不改颜色判定、不改连通性。为了让裁剪可观察，generator 保持 3 个保留分量，
#   并强制至少 1 个被删分量把输入前景边界向下或向右撑大；这样新旧规则在同一输入上会产生不同尺寸的输出。
# - implicit_assumptions_review:
#   * `_bg_boundary` 仍可用，因为输入外边框仍由背景主导。
#   * `_find_legend` 仍依赖左上角 L 形边框，这一点没有改变。
#   * `components_mask(g != bg, connectivity=4)` 仍正确，因为对象/图例都还是按“非背景整体前景”分量来判断。
#   * 原版 `return out` 隐含“输出与输入同尺寸”；这是 D14 中唯一必须替换的旧前提。
#   * 新版裁剪需要“删除后至少还有非背景像素”。generator 通过让 legend 至少保留一个与其余目标色分离的像素，
#     保证 legend 不会整体被删空，因此该假设成立。
# - placement_legality:
#   保留分量先放入左上 14x14 生存区，删除分量后放入完整 16x16 画布中，但至少一个删除分量必须越过生存区的
#   最终 bbox。所有对象保持 1 像素空隙，避免重叠和误连通。
# - visual_observability:
#   由于 legend 留在左上，裁剪主要体现在底边和右边收缩；删掉“边界撑开者”后，输出比输入明显更小，
#   且收缩量不是常数边框，而是由剩余分量的真实外接框决定。
# - neatness_and_color:
#   不引入新的颜色映射，也不按颜色 id 排序。颜色仍只承担“legend 指定目标集合”的可见角色。


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border (bg defined by the frame, not global frequency)."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    The legend is a top-left corner region whose bottom edge and right edge are the same color x.
    We detect the smallest size sz such that:
      - g[sz-1, 0:sz] are all x
      - g[0:sz, sz-1] are all x
    Then the legend color set is the set of colors in the interior (0:sz-1, 0:sz-1),
    excluding bg and excluding the border color x.
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
    assert False, "no legend border found in grid"


def _erase_matching_components(g: np.ndarray, bg: int, legend_colors: set[int]) -> np.ndarray:
    """Return the full-size post-deletion canvas before the D14 crop step."""
    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove palette-matching components, then tightly crop the survivors.

    Input structure:
    - Background fills most of the canvas.
    - A small legend sits in the top-left corner: its bottom and right edges share one border color, and its interior displays the target palette.
    - Elsewhere, the foreground consists of several multicolor 4-connected components on the background.

    Rule:
    - Read the target color set from the legend interior.
    - Delete every connected component whose color set contains all legend colors.
    - Keep every other non-background component unchanged.
    - After deletion, crop the result to the smallest rectangle covering all remaining non-background cells.

    Key insight: The legend still specifies which components get erased, but the final answer is no longer the whole canvas — it is the tight frame around whatever survives.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    full = _erase_matching_components(g, bg, legend_colors)
    keep_bbox = bbox(full != bg)
    if keep_bbox is None:
        return np.full((0, 0), int(bg), dtype=int)
    return crop_bbox(full, keep_bbox)


# --------------- generator (D14 variant) ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]            # medium, tall, wide
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

# -- Component fill density (fraction of non-zero pixels) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Cropping / placement controls --
SURVIVOR_ZONE_H = 14
SURVIVOR_ZONE_W = 14
PADDING = 1

# -- Generation retry budget --
MAX_RETRIES = 1000


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box (border color 3) at top-left and place target colors."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
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


def _component_dims() -> tuple[int, int]:
    """Sample component (h, w) according to shape-mode weights."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], deleted: bool) -> np.ndarray:
    """Build an h x w solid component colored 0 with sprinkled palette cells."""
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
    """Enumerate legal top-left locations with a background moat around the whole patch bbox."""
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
    """Choose a random legal location under optional row/col bounds and a custom predicate."""
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


def _gen_attempt() -> np.ndarray | None:
    """Build a valid input whose post-deletion output is tightly cropped."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    survivor = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, survivor, targets)

    for _ in range(NUM_KEPT):
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, deleted=False)
        try:
            r, c = _place_patch(
                inp,
                patch,
                row_hi=SURVIVOR_ZONE_H - h,
                col_hi=SURVIVOR_ZONE_W - w,
            )
        except ValueError:
            return None
        paste(survivor, patch, r, c)

    survivor_bbox = bbox(survivor != BG)
    if survivor_bbox is None:
        return None
    _r0, _c0, surv_r1, surv_c1 = survivor_bbox
    if surv_r1 >= GRID_H - 1 or surv_c1 >= GRID_W - 1:
        return None

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    num_deleted = component_count - NUM_KEPT

    anchor_deleted = False
    for idx in range(num_deleted):
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, deleted=True)
        try:
            if idx == 0:
                def extends_bbox(r: int, c: int, hh: int = h, ww: int = w) -> bool:
                    return r + hh - 1 > surv_r1 or c + ww - 1 > surv_c1

                _place_patch(inp, patch, predicate=extends_bbox)
                anchor_deleted = True
            else:
                _place_patch(inp, patch)
        except ValueError:
            return None

    if not anchor_deleted:
        return None

    expected_bbox = bbox(survivor != BG)
    if expected_bbox is None:
        return None
    expected = crop_bbox(survivor, expected_bbox)
    if expected.shape == inp.shape:
        return None

    input_bbox = bbox(inp != BG)
    if input_bbox is None:
        return None
    if input_bbox == expected_bbox:
        return None

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
