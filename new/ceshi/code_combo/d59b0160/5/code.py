# Combo variant of d59b0160 (index 5)
# combo_spec: D2/4+D14/0
# Name: 匹配分量压缩为单像素+裁剪输出

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox, components_mask, crop_bbox, paste

# Design analysis
# ===============
# - concrete_behavior_and_trace:
#   输入仍是 16x16、背景色 7、左上 4x4 图例、其余对象是与背景 4-分离的多色实心矩形。
#   设图例目标色为 {1,4,8}。原题里，一个同时含有 {1,4,8} 的 5x6 大矩形会被整块删成背景。
#   本融合版先把它压缩成其质心附近的 1 个像素，颜色取该分量的主色；然后不是保留整张 16x16，
#   而是把“图例 + 所有未命中矩形 + 所有压缩标记点”的并集紧裁成最小外接矩形。
#   因而同一对象会先经历“大块变单点”，再经历“画布跟着收缩”，最终 bbox 既小于输入 bbox，
#   又大于“如果把命中分量直接删空”的 bbox。
# - complexity_calibration:
#   仅凭裁剪不足以体现 D2，只有压缩也不足以体现 D14，所以 generator 强制至少 1 个命中分量满足：
#   其整块 bbox 把输入前景推得更远，但它压缩后的标记点仍落在保留对象 bbox 之外。
#   这样输出裁剪框由“保留对象 + 标记点”共同决定，而不是退化成单独某一个变体。
#   命中分量面积下限设为 12，且其彩色密度受限在 1/3 左右以内，使主色 0 几乎总是唯一众数，
#   单点压缩在视觉上清楚、颜色也不依赖不可见 tie-break。
# - implicit_assumption_audit:
#   `_bg_boundary` 仍依赖外边框主要是背景 7；generator 保持这一点。
#   `_find_legend` 仍依赖左上 L 形边框；图例几何未改。
#   `components_mask(g != bg)` 仍必须把每个对象矩形视为完整前景分量；因此 patch 继续用实心矩形。
#   原题循环里“匹配即整块填回 bg”的假设已失效，现需先记下分量的标记点和标记色，再清空原分量并回填单点。
#   新增裁剪步骤依赖“压缩后至少还有非背景像素”；图例始终保留且不会整体匹配，所以这一点成立。
#   质心四舍五入在任意稀疏形状上可能落到分量外，因此 solver 仍保留“回退到最近分量像素”的稳健处理；
#   generator 里的命中对象是矩形，样例中该点天然落在对象内部。
# - placement_legality:
#   所有对象 bbox 周围保留 1 像素背景缓冲，避免越界、重叠或误连通。
#   保留对象先放在较小的 survivor zone 中，再放命中对象；至少一个命中对象必须同时满足：
#   其 marker 超出 survivor bbox，但 marker 不能跑到整张 16x16 的最右/最下边，从而保证最终输出确实被裁小。
# - visual_observability:
#   人类可以先从图例读出目标色集合，再注意到“含全套目标色”的大矩形没有消失为洞，而是缩成单点。
#   随后又能看到输出尺寸并非原图大小，且裁剪边界不是只包住保留矩形，还会被这些单点继续向外撑开。
#   这种“先压缩、再以压缩结果参与裁剪”的联动，在 2-4 个样例中可稳定重复观察。
# - cleanliness_and_inference:
#   不引入新的颜色映射；标记颜色仍取对象内部频率最高的颜色，通常是显眼且稳定的 0。
#   图例固定在左上，输出主要在右边/下边收缩；对象彼此留白，标记点周围背景干净，不会出现难以辨认的叠层。


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the top-left legend whose bottom edge and right edge share one border color.
    Return (legend_size, border_color, target_color_set).
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


def _marker_position(pts: np.ndarray) -> tuple[int, int]:
    """Rounded centroid, snapped to the nearest component pixel if rounding lands outside."""
    mean_r = float(pts[:, 0].mean())
    mean_c = float(pts[:, 1].mean())
    rr = int(math.floor(mean_r + 0.5))
    cc = int(math.floor(mean_c + 0.5))

    diffs = pts - np.array([rr, cc], dtype=float)
    exact = np.where((diffs[:, 0] == 0) & (diffs[:, 1] == 0))[0]
    if exact.size:
        r, c = pts[int(exact[0])]
        return int(r), int(c)

    diffs = pts.astype(float) - np.array([mean_r, mean_c], dtype=float)
    dist2 = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    best = np.where(dist2 == dist2.min())[0]
    if best.size == 1:
        r, c = pts[int(best[0])]
        return int(r), int(c)

    tied = pts[best]
    order = np.lexsort((tied[:, 1], tied[:, 0]))
    r, c = tied[int(order[0])]
    return int(r), int(c)


def _marker_color(g: np.ndarray, pts: np.ndarray) -> int:
    """Dominant color of the component, with a geometry-based tie break."""
    cols = np.array([int(g[r, c]) for r, c in pts], dtype=int)
    vals, cnts = np.unique(cols, return_counts=True)
    best_count = int(cnts.max())
    tied = vals[cnts == best_count]
    if tied.size == 1:
        return int(tied[0])

    centroid = pts.astype(float).mean(axis=0)
    best_color = None
    best_metric = None
    best_anchor = None
    for color in tied:
        color_pts = pts[cols == int(color)].astype(float)
        d2 = ((color_pts - centroid) ** 2).sum(axis=1)
        metric = float(d2.mean())
        anchor = tuple(map(int, color_pts[np.argmin(d2)]))
        if (
            best_metric is None
            or metric < best_metric
            or (metric == best_metric and anchor < best_anchor)
        ):
            best_color = int(color)
            best_metric = metric
            best_anchor = anchor
    return int(best_color)


def _erase_matching_components(g: np.ndarray, bg: int, legend_colors: set[int]) -> np.ndarray:
    """Original-task post-state: matched components are removed completely."""
    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            out[pts[:, 0], pts[:, 1]] = bg
    return out


def _compress_matching_components(g: np.ndarray, bg: int, legend_colors: set[int]) -> np.ndarray:
    """Fused post-state before cropping: matched components collapse to single markers."""
    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if not legend_colors.issubset(cols):
            continue
        marker_r, marker_c = _marker_position(pts)
        marker_color = _marker_color(g, pts)
        out[pts[:, 0], pts[:, 1]] = bg
        out[marker_r, marker_c] = marker_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse palette-matching components to one pixel, then tightly crop the result.

    Input structure:
    - Most cells are background.
    - The top-left corner contains a small square legend whose bottom edge and right edge
      share one border color; the interior cells show the target color set.
    - Away from the legend, each object is a 4-connected non-background component.

    Rule:
    - Read the target color set from the legend interior.
    - Inspect every non-background component and test whether it contains all legend colors.
    - Components that fail the test stay unchanged.
    - Components that pass the test are cleared, replaced by one centroid-like marker pixel
      colored with the component's most frequent color, and then the whole board is cropped
      to the smallest rectangle covering every remaining non-background cell.

    Key insight:
    - The legend still selects components by color-set inclusion, but selected objects are no
      longer blanked away: they survive as single summary pixels, and those pixels still count
      when deciding the final cropped output frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    full = _compress_matching_components(g, bg, legend_colors)
    keep_bbox = bbox(full != bg)
    if keep_bbox is None:
        return np.full((0, 0), int(bg), dtype=int)
    return crop_bbox(full, keep_bbox)


# --------------- generator ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

DENSITY_LO = 0.10
DENSITY_HI = 0.55
MATCHED_DENSITY_HI = 0.33
MATCHED_MIN_AREA = 12

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

SURVIVOR_ZONE_H = 12
SURVIVOR_ZONE_W = 12
FINAL_MAX_R = GRID_H - 2
FINAL_MAX_C = GRID_W - 2
PADDING = 1

MAX_RETRIES = 1500


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the top-left legend and copy it into both working canvases."""
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


def _component_dims(*, matched: bool, max_h: int, max_w: int) -> tuple[int, int]:
    """Sample component dimensions under optional size caps."""
    for _ in range(100):
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h, w = random.randint(*TALL_H), random.randint(*TALL_W)
        elif mode == "wide":
            h, w = random.randint(*WIDE_H), random.randint(*WIDE_W)
        else:
            h, w = random.randint(*MED_H), random.randint(*MED_W)
        if h > max_h or w > max_w:
            continue
        if matched and h * w < MATCHED_MIN_AREA:
            continue
        return h, w
    fallback_h = min(max_h, 4 if matched else 3)
    fallback_w = min(max_w, 4 if matched else 2)
    return int(max(1, fallback_h)), int(max(1, fallback_w))


def _component_patch(h: int, w: int, targets: list[int], matched: bool) -> np.ndarray:
    """Build one solid rectangular component with sprinkled palette cells."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    patch = np.zeros((h, w), dtype=int)
    non_targets = [v for v in PALETTE if v not in targets]

    if matched:
        lo = max(lo, NUM_TARGETS)
        hi = min(hi, max(lo, math.floor(area * MATCHED_DENSITY_HI)))
        hi = min(hi, area - NUM_TARGETS)
        n_colored = random.randint(lo, hi)
        coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
        colors = list(targets) + [random.choice(PALETTE) for _ in range(n_colored - NUM_TARGETS)]
        random.shuffle(colors)
        for (r, c), color in zip(coords, colors):
            patch[r, c] = color
        return patch

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    used_target_count = random.randint(0, NUM_TARGETS - 1)
    allowed = non_targets + random.sample(targets, used_target_count)
    colors = [random.choice(allowed) for _ in range(n_colored)]
    for (r, c), color in zip(coords, colors):
        patch[r, c] = color
    return patch


def _patch_marker(patch: np.ndarray) -> tuple[int, int, int]:
    """Return the local marker row/col/color for a rectangular component patch."""
    pts = np.argwhere(np.ones(patch.shape, dtype=bool))
    r, c = _marker_position(pts)
    color = _marker_color(patch, pts)
    return int(r), int(c), int(color)


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
    """Enumerate legal top-left locations with a background moat around the patch bbox."""
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
    return int(r), int(c)


def _gen_attempt() -> np.ndarray | None:
    """Build one valid fused-variant input or return None on failure."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    survivor = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, survivor, targets)

    for _ in range(NUM_KEPT):
        placed = False
        for _ in range(60):
            h, w = _component_dims(
                matched=False,
                max_h=SURVIVOR_ZONE_H,
                max_w=SURVIVOR_ZONE_W,
            )
            patch = _component_patch(h, w, targets, matched=False)
            try:
                r, c = _place_patch(
                    inp,
                    patch,
                    row_hi=SURVIVOR_ZONE_H - h,
                    col_hi=SURVIVOR_ZONE_W - w,
                )
            except ValueError:
                continue
            paste(survivor, patch, r, c)
            placed = True
            break
        if not placed:
            return None

    survivor_bbox = bbox(survivor != BG)
    if survivor_bbox is None:
        return None
    _r0, _c0, surv_r1, surv_c1 = survivor_bbox

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    matched_count = component_count - NUM_KEPT
    if matched_count < 1:
        return None

    final_full = survivor.copy()
    anchor_placed = False

    for idx in range(matched_count):
        placed = False
        for _ in range(80):
            h, w = _component_dims(matched=True, max_h=GRID_H, max_w=GRID_W)
            patch = _component_patch(h, w, targets, matched=True)
            mr, mc, color = _patch_marker(patch)
            is_anchor = not anchor_placed

            if is_anchor:
                def predicate(r: int, c: int, hh: int = h, ww: int = w, rr: int = mr, cc: int = mc) -> bool:
                    marker_r = r + rr
                    marker_c = c + cc
                    end_r = r + hh - 1
                    end_c = c + ww - 1
                    marker_extends = marker_r > surv_r1 or marker_c > surv_c1
                    marker_stays_cropable = marker_r <= FINAL_MAX_R and marker_c <= FINAL_MAX_C
                    uncompressed_extends_farther = end_r > marker_r or end_c > marker_c
                    bbox_extends = end_r > surv_r1 or end_c > surv_c1
                    return (
                        marker_extends
                        and marker_stays_cropable
                        and uncompressed_extends_farther
                        and bbox_extends
                    )
            else:
                def predicate(r: int, c: int, rr: int = mr, cc: int = mc) -> bool:
                    return r + rr <= FINAL_MAX_R and c + cc <= FINAL_MAX_C

            try:
                r, c = _place_patch(inp, patch, predicate=predicate)
            except ValueError:
                continue

            final_full[r + mr, c + mc] = color
            if is_anchor:
                anchor_placed = True
            placed = True
            break
        if not placed:
            return None

    if not anchor_placed:
        return None

    input_bbox = bbox(inp != BG)
    delete_only = _erase_matching_components(inp, BG, set(targets))
    delete_bbox = bbox(delete_only != BG)
    final_bbox = bbox(final_full != BG)
    if input_bbox is None or delete_bbox is None or final_bbox is None:
        return None

    if not (final_bbox[2] < GRID_H - 1 and final_bbox[3] < GRID_W - 1):
        return None
    if final_bbox == delete_bbox:
        return None
    if final_bbox == input_bbox:
        return None
    if not (final_bbox[2] > delete_bbox[2] or final_bbox[3] > delete_bbox[3]):
        return None
    if not (input_bbox[2] > final_bbox[2] or input_bbox[3] > final_bbox[3]):
        return None

    expected = crop_bbox(final_full, final_bbox)
    if expected.shape == inp.shape:
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
