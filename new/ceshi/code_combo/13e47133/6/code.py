# Combo variant of 13e47133 (index 6)
# combo_spec: D4/0+D9/0
# Name: 深度上限+按面积排序处理

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
for _root in _HERE.parents:
    _solvers_dir = _root / "solvers"
    if _solvers_dir.exists():
        if str(_solvers_dir) not in sys.path:
            sys.path.append(str(_solvers_dir))
        break

from arc_utils import components_mask


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]

# Design analysis for the D4/0 + D9/0 combo
# -------------------------------------------
# a) 具体化与行为追踪
#    - 原题把每个非墙区域独立处理：marker 按离边界的切比雪夫深度排序成 palette，
#      然后整块区域都按 palette[depth % len(palette)] 填满。
#    - D4/0 把“填到中心”为“只填到 N 层”，其中 N = 该区域 marker 数；更深处保持 base。
#    - D9/0 再引入跨区域顺序：先按区域面积从小到大排序；较早区域真正用到的非 base 颜色会被
#      记录到一个全局 used set 中，后续区域构造 palette 时必须删掉这些已消耗颜色。
#    - 小例子：三个矩形区域面积依次为 40 < 60 < 90。
#      区域 A 有 1 个 marker [红]，输出只涂到 d<=1，并消耗红。
#      区域 B 有 2 个 marker [红, 蓝]。原题或纯 D4 都会使用红蓝交替；组合版先删掉已消耗的红，
#      剩余 palette 只有 [蓝]，所以 B 的 d=0,1,2 三层都变蓝，内部更深处回到 base。
#      区域 C 若 markers=[蓝, 绿]，则同理只能保留绿。差异不再是“每区单独读 marker 就够了”，
#      而是必须同时看 marker 数和前面较小区域已经拿走了哪些颜色。
# b) 复杂度校准
#    - 顺序变量只用“面积升序”这一种、颜色排除只用“非 base 颜色不可复用”这一种，避免规则爆炸。
#    - 生成器固定制造 3 个大矩形区域，并让后两个区域重复前一个区域的 marker 颜色，确保排除
#      规则在每个样本里至少触发两次，而不是偶然成立。
#    - marker 数只落在 1 或 2，且三个区域分别重复这些小整数，足以从少量样例中归纳“数量 = 深度上限”。
# c) 隐式假设审查
#    - 原代码默认“区域彼此独立”，所以在区域循环里立刻写 `out[r, c] = palette[...]`。组合版不再
#      成立；必须先从输入图抽取每个区域的静态信息，再按面积顺序串行应用排除。
#    - 原代码默认 palette 一旦由本区 marker 决定，就不会被别区影响。这里改成“从候选 palette 中
#      删除已消耗的非 base 颜色”，因此 palette 需要拆成“候选 palette”和“有效 palette”两层。
#    - D4/0 也破坏了 `out = g.copy()` 的安全性：若 deeper cells 不显式回写 base，原 marker 可能残留。
#      这里对每个区域格子都显式赋值：`d <= N` 时用有效 palette，否则用 base。
#    - 若 base 也参与全局消耗，则第一个区域之后所有区域都会坍缩，所以 solver 明确保留 base 永不排除；
#      生成器也保持所有区域共享同一个 base，和原题一致。
# d) 放置合法性
#    - 生成器使用一条整列墙加一条右半边整行墙，形成 T 形分割后的 3 个矩形区域。墙厚恒为 1，
#      区域互不重叠、4 连通、均在边界内。
#    - marker 都放在所属矩形某个角落沿对角线向内的连续深度上，不越界、不压墙、不互相重叠。
#    - 三个区域最小边长都至少为 7，因此 n=1 或 n=2 的 capped fill 都能留下可见的 base 内核。
# e) 视觉可观察性
#    - 面积大小、marker 数、marker 深度、颜色重复，都是输入中肉眼可见的结构；没有使用颜色编号、
#      扫描顺序之外的隐藏属性。
#    - 输出里最小区域先“拿走”一个颜色，较大区域虽然输入里也有该颜色 marker，但输出中该颜色缺席，
#      同时彩色层数又明显只到 marker 数对应的深度为止，二者可通过 2-3 组样例共同支撑。
# f) 可视化整洁度
#    - 区域始终是 3 个大矩形，墙是单一颜色，marker 只占极少数单像素，输出是整齐的同心矩形带加
#      base 内核；不存在多个后处理对象叠到同一区域里的噪声。
#    - 颜色排除只会缩短 palette，不会引入新颜色，因此输出仍保留原题“边框式填充”的整洁外观。
# g) 变化可推断性
#    - 颜色变化不是随意映射，而是“同色 marker 若已在更小区域的输出里出现，后续大区域就不能再用它”。
#      颜色复用链由输入中的重复 marker 直接提示，排除结果由输出中缺失的颜色直接体现。
# h) 新 solver 的额外输入假设
#    - 为避免“按面积排序”在同面积时需要隐藏 tie-break，生成器强制三个区域面积互不相同。
#    - 为避免 later region 全部颜色都被删光却没有任何新色留下，生成器固定使用颜色链
#      [A], [A,B], [B,C]，保证后两个区域既有被删颜色也有保留颜色。
#    - solver 允许 0-marker 区域并把它们视作纯 base 区，但生成器本组合始终让三个区域都带 marker，
#      从而稳定展示两条变体规则，而不是退化回原题或纯 D4。


def _cc4(mask: np.ndarray) -> list[np.ndarray]:
    return components_mask(mask, connectivity=4)


def _mode(arr: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev distance-to-boundary for True cells in component_mask.
    Boundary cells are those that touch False or the grid edge in their 8-neighborhood.
    Returns -1 outside the component.
    """
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = False
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not component_mask[rr, cc]:
                    is_boundary = True
                    break
            if is_boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _infer_bg_wall(grid: np.ndarray) -> tuple[int, int]:
    bg = _mode(grid)
    other = grid[grid != bg]
    wall = bg if other.size == 0 else _mode(other)
    return bg, wall


def _extract_regions(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, wall = _infer_bg_wall(g)
    regions: list[dict[str, object]] = []

    for pts in _cc4(g != wall):
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[int(r), int(c)] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)
        marker_count = 0
        dist_to_color: dict[int, int] = {}

        for r, c in pts:
            rr, cc = int(r), int(c)
            v = int(g[rr, cc])
            if v == base:
                continue
            marker_count += 1
            d = int(dist[rr, cc])
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = v

        if dist_to_color:
            ds = sorted(dist_to_color)
            palette: list[int] = []
            if ds[0] > 0:
                palette.append(int(base))
            for d in ds:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(base)]

        rows = pts[:, 0]
        cols = pts[:, 1]
        regions.append(
            {
                "pts": pts,
                "mask": comp_mask,
                "dist": dist,
                "base": int(base),
                "area": int(len(pts)),
                "bbox": (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
                "marker_count": int(marker_count),
                "palette": palette,
            }
        )

    return bg, regions


def _effective_palette(candidate: list[int], base: int, used_colors: set[int]) -> list[int]:
    palette = [int(col) for col in candidate if int(col) == int(base) or int(col) not in used_colors]
    if not palette:
        return [int(base)]
    return palette


def _ordered_regions(regions: list[dict[str, object]], *, by_area: bool) -> list[dict[str, object]]:
    if not by_area:
        return list(regions)
    return sorted(regions, key=lambda info: (int(info["area"]), *info["bbox"]))


def _transform(grid: np.ndarray, *, capped: bool, area_order: bool, exclude_used: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _bg, regions = _extract_regions(g)
    out = g.copy()
    used_colors: set[int] = set()

    for info in _ordered_regions(regions, by_area=area_order):
        pts = info["pts"]
        dist = info["dist"]
        base = int(info["base"])
        marker_count = int(info["marker_count"])
        candidate = list(info["palette"])
        palette = _effective_palette(candidate, base, used_colors) if exclude_used else candidate
        m = len(palette)

        for r, c in pts:
            rr, cc = int(r), int(c)
            d = int(dist[rr, cc])
            if (not capped) or d <= marker_count:
                out[rr, cc] = int(palette[d % m])
            else:
                out[rr, cc] = int(base)

        if exclude_used:
            used_colors.update(int(col) for col in palette if int(col) != int(base))

    return out


def _full_fill_main(grid: np.ndarray) -> np.ndarray:
    return _transform(grid, capped=False, area_order=False, exclude_used=False)


def _capped_independent_main(grid: np.ndarray) -> np.ndarray:
    return _transform(grid, capped=True, area_order=False, exclude_used=False)


def _combo_main(grid: np.ndarray) -> np.ndarray:
    return _transform(grid, capped=True, area_order=True, exclude_used=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-separated regions with truncated concentric bands whose reusable colors run out from small regions to large ones.

    Input structure:
    - A one-cell-thick wall color partitions the canvas into several rectangular non-wall regions.
    - Inside each region, the dominant base color is interrupted by a few isolated marker pixels placed at consecutive Chebyshev depths from one corner.

    Rule:
    - For each region, sort marker colors by boundary depth to build the same candidate palette as in the original task, prepending the base color when the shallowest marker is not on the boundary.
    - Process the regions from smaller area to larger area. Before painting a region, delete from its candidate palette every non-base color that has already appeared in an earlier processed region; the base color is never deleted.
    - Let N be that region's marker count. Paint only the cells with distance d <= N using the surviving palette cyclically, and leave any deeper cells as the base color.

    Key insight: Marker depths still determine palette order, but the final output now depends on two visible global cues at once: marker count caps how far the bands travel inward, and area order determines which repeated colors have already been consumed by smaller regions.
    """
    return _combo_main(np.asarray(grid, dtype=int))


# --------------- generator ---------------

H_MIN, H_MAX = 17, 30
W_MIN, W_MAX = 21, 30

MIN_REGION_SPAN = 7
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9
GEN_MAX_ATTEMPTS = 500

MIN_CAP_DELTA = 18
MIN_EXCLUDE_DELTA = 18
MIN_EXCLUDE_REGIONS = 2


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _pick_palette(rng: random.Random) -> tuple[int, int, list[int]]:
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    rng.shuffle(colors)
    return bg, wall, colors


def _make_t_layout(H: int, W: int, rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    if H < 2 * MIN_REGION_SPAN + 3 or W < 2 * MIN_REGION_SPAN + 3:
        return None
    c = rng.randint(MIN_REGION_SPAN, W - MIN_REGION_SPAN - 2)
    r = rng.randint(MIN_REGION_SPAN, H - MIN_REGION_SPAN - 2)

    left = (0, 0, H - 1, c - 1)
    top_right = (0, c + 1, r - 1, W - 1)
    bottom_right = (r + 1, c + 1, H - 1, W - 1)
    rects = [left, top_right, bottom_right]

    areas = [int((rr1 - rr0 + 1) * (cc1 - cc0 + 1)) for rr0, cc0, rr1, cc1 in rects]
    if len(set(areas)) != 3:
        return None

    for r0, c0, r1, c1 in rects:
        if min(r1 - r0 + 1, c1 - c0 + 1) < MIN_REGION_SPAN:
            return None

    wall_mask = np.zeros((H, W), dtype=int)
    wall_mask[:, c] = 1
    wall_mask[r, c:] = 1
    return wall_mask, rects


def _layout_options(max_d: int, n: int) -> list[int]:
    options: list[int] = []
    for start_d in range(2):
        if start_d + n - 1 > max_d:
            continue
        if start_d == 0 and max_d >= n + 1:
            options.append(start_d)
        elif start_d == 1 and max_d >= n + 2:
            options.append(start_d)
    return options


def _place_fixed_markers(
    rect: tuple[int, int, int, int], colors: list[int], rng: random.Random
) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = rect
    n = len(colors)
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    starts = _layout_options(max_d, n)
    if not starts:
        return []

    start_d = rng.choice(starts)
    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1

    markers = []
    for k, col in enumerate(colors):
        d = start_d + k
        markers.append((d, int(col), int(br + dr * d), int(bc + dc * d)))
    return markers


def _stamp_markers(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    bg: int,
    color_specs: list[list[int]],
    rng: random.Random,
) -> tuple[set[int], list[int]] | None:
    used: set[int] = set()
    counts: list[int] = []

    for rect, colors in zip(rects, color_specs):
        markers = _place_fixed_markers(rect, colors, rng)
        if not markers:
            return None
        counts.append(len(markers))
        for _d, col, rr, cc in markers:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None
            if int(grid[rr, cc]) != int(bg):
                return None
            grid[rr, cc] = int(col)
            used.add(int(col))

    return used, counts


def _validate_color_order(grid: np.ndarray, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != int(bg):
        return False
    return len(order) >= 2 and int(vals[order[1]]) == int(wall)


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    full = _full_fill_main(grid)
    capped = _capped_independent_main(grid)
    combo = _combo_main(grid)

    _bg, regions = _extract_regions(grid)
    area_rank = {
        tuple(int(x) for x in info["bbox"]): rank
        for rank, info in enumerate(_ordered_regions(regions, by_area=True))
    }
    exclude_changed = combo != capped
    changed_regions = set()
    for info in regions:
        pts = info["pts"]
        for r, c in pts:
            rr, cc = int(r), int(c)
            if bool(exclude_changed[rr, cc]):
                changed_regions.add(area_rank[tuple(int(x) for x in info["bbox"])])
                break

    return {
        "cap_delta": int(np.count_nonzero(combo != full)),
        "exclude_delta": int(np.count_nonzero(combo != capped)),
        "exclude_regions": len(changed_regions),
        "changed_late_regions": int(sum(rank > 0 for rank in changed_regions)),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    layout = _make_t_layout(H, W, rng)
    if layout is None:
        return None
    wall_mask, rects = layout

    bg, wall, marker_pool = _pick_palette(rng)
    color_chain = marker_pool[:3]
    color_specs = [
        [color_chain[0]],
        [color_chain[0], color_chain[1]],
        [color_chain[1], color_chain[2]],
    ]

    areas = [int((r1 - r0 + 1) * (c1 - c0 + 1)) for r0, c0, r1, c1 in rects]
    order = np.argsort(areas)
    ordered_rects = [rects[int(i)] for i in order]

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    stamped = _stamp_markers(grid, ordered_rects, bg, color_specs, rng)
    if stamped is None:
        return None
    used_colors, counts = stamped

    if counts != [1, 2, 2]:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used_colors) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        full = _full_fill_main(grid)
        capped = _capped_independent_main(grid)
        combo = _combo_main(grid)
    except Exception:
        return None

    if np.array_equal(grid, combo):
        return None
    if np.array_equal(combo, full):
        return None
    if np.array_equal(combo, capped):
        return None

    metrics = _combo_metrics(grid)
    if metrics["cap_delta"] < MIN_CAP_DELTA:
        return None
    if metrics["exclude_delta"] < MIN_EXCLUDE_DELTA:
        return None
    if metrics["exclude_regions"] < MIN_EXCLUDE_REGIONS:
        return None
    if metrics["changed_late_regions"] < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
