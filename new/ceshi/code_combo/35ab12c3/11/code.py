# Combo variant of 35ab12c3 (index 11)
# combo_spec: D2/0+D3/3+D10/1
# Name: 仅正交路径+星形连接+仅闭合多边形

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location

# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior + trace
#    - 原题把同色点按极角排成外轮廓，允许 H/V/45° 连接；2 点开放线也合法。
#    - 本融合先把同色点解释成“围绕一个整数质心的星形端点”，再把每个端点只用水平或垂直
#      线段连到该质心。与此同时，D10 的“仅闭合多边形”不能再直接沿用外轮廓闭环语义，
#      因为 D3 已经取消了 vertex-to-vertex 的外圈边。
#    - 因而这里把“闭合”具体化为：必须同时拥有上/下/左/右四个端点，也就是完整四向正交星。
#      2 臂直线和 3 臂 T 形都算开放结构，跳过不扩张。
#    - 小例子：输入顶点为 {(0,2),(2,0),(2,5),(4,2)}。
#      old: 原题尝试按外轮廓连顶点；该组会画成部分对角/外圈结构，或因不满足对角而残缺。
#      new: 先识别 hub=(2,2)，四个端点都与 hub 正交对齐，于是输出一朵四向十字星。
#      若顶点只剩 {(0,2),(2,0),(2,5)}，new 会直接跳过，因为少了“闭合”所需的第四个方向。
#
# b) Complexity calibration
#    - 若只做 D2+D3 而不管 D10，generator 很容易产出 2 臂或 3 臂星，视觉上更像线段/T 形，
#      会弱化“闭合”约束。
#    - 若强行保留任意多边形并改成星形，orth-only 下多数端点根本无法与同一 hub 直连。
#    - 所以融合后的形状族收缩为一种稳定而可观察的对象：四向正交星（上下左右四端点，
#      上下等长、左右等长）。它重复出现，足以让人类从 2-4 个样例看出“都向中心汇聚，
#      且必须是完整十字而不是开放线”。
#
# c) Hidden assumptions audit on main()
#    - 原题的 angle_cycle / wraparound edge 构造依赖“3+ 顶点组成外轮廓”；融合后不成立，
#      必须整体移除，改为 vertex -> hub。
#    - 原题 len==2 分支依赖开放线合法；融合后不成立，solver 必须要求恰好 4 个端点。
#    - 原题允许 H/V/45°；融合后所有主边与影子边都必须重新走 orth-only 判定。
#    - 星形 hub 不再天然是已有顶点，因此 solver 额外假设 hub 位置在输入里是背景。
#      generator 会显式保证这一点。
#    - shadow 逻辑仍以“某个 seed 端点旁的异色单像素”为偏移来源；因为四向星没有环序，
#      seed 只负责确定影子偏移，不再影响主结构的连接顺序。
#
# d) Placement legality
#    - 继续采用 output-first 的局部 patch 放置：先决定四向星与可选影子所覆盖的所有输出格，
#      再把输入退化为 4 个端点和 1 个影子 anchor。
#    - patch bbox 覆盖主星形与平移后的影子副本。全局放置仍用
#      random_free_location(..., padding=1, connectivity=8)，保证对象互不接触。
#    - 为了让 D2 与 D10 的限制可观察，generator 额外固定放入两类 input==output 的负样本：
#      一类是“对角四端点星”（若允许对角 spoke 就会扩张），一类是“开放 2/3 臂正交星”
#      （若允许开放结构就会扩张）。这两类都不会误触发当前 solver。
#
# e) Visual observability
#    - 训练样例里会反复出现：输入只有四个分散端点，输出却都向一个空白中心汇聚成十字花。
#    - 同时还会稳定出现两种“不变化”的对照：对角四端点组、以及开放的 2/3 臂正交组。
#      因而人类可以直接看出：不是所有“围着中心的点”都会连，必须既是正交的，又是闭合的。
#
# f) Visual cleanliness
#    - 只保留一种主对象族，避免把正交星、对角星、多边形边框混在一起造成语义噪声。
#    - 影子偏移只允许对角 8 邻域方向，这样与主 spoke 不共线，平移后的轮廓清晰可读。
#
# g) Change inferability
#    - 颜色没有隐藏映射：shape 色只负责主星形，shadow 色只负责对应的平移副本，
#      distractor 色保持原样。
#    - 融合的关键线索完全来自几何约束：四向完整性、正交汇聚、以及由 seed 邻居定义的
#      同偏移影子。
#
# New solver assumptions introduced by the combo
# ----------------------------------------------
# 1) 真正可扩张的 shape 色组恰好有 4 个顶点，并分别位于 hub 的上/下/左/右四个方向。
# 2) hub 是整数格且在输入里为空白；否则该色组不应被解释成星形。
# 3) 若存在 shadow，则其 anchor 只 8 邻接一个顶点，且 offset 与四条主 spoke 都不共线。

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
CARDINALS = {(-1, 0), (1, 0), (0, -1), (0, 1)}

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 4)
MAX_SHADOWS_CAP = 2
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

CROSS_ARM_RANGE = (2, 6)  # 2-5 inclusive
CROSS_EQUAL_ARMS_PROB = 0.35

WITNESS_ARM_RANGE = (2, 6)  # 2-5 inclusive

SHAPE_PLACEMENT_RETRIES = 50
GENERATE_RETRIES = 300


def _aligned_orth(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0)


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _clear_in_original(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _centroid_hub(points_rc: list[tuple[int, int]]) -> tuple[int, int] | None:
    n = len(points_rc)
    sum_r = sum(r for r, _ in points_rc)
    sum_c = sum(c for _, c in points_rc)
    if sum_r % n != 0 or sum_c % n != 0:
        return None
    return sum_r // n, sum_c // n


def _orth_star_signature(
    points_rc: list[tuple[int, int]],
    hub: tuple[int, int],
) -> dict[tuple[int, int], int] | None:
    if len(points_rc) != 4 or len(set(points_rc)) != 4:
        return None

    hr, hc = hub
    dirs_to_dist: dict[tuple[int, int], int] = {}
    for r, c in points_rc:
        if (r, c) == hub:
            return None
        if r != hr and c != hc:
            return None
        if r == hr:
            direction = (0, 1 if c > hc else -1)
            dist = abs(c - hc)
        else:
            direction = (1 if r > hr else -1, 0)
            dist = abs(r - hr)
        if dist <= 0 or direction in dirs_to_dist:
            return None
        dirs_to_dist[direction] = dist

    if set(dirs_to_dist) != CARDINALS:
        return None
    if dirs_to_dist[(-1, 0)] != dirs_to_dist[(1, 0)]:
        return None
    if dirs_to_dist[(0, -1)] != dirs_to_dist[(0, 1)]:
        return None
    return dirs_to_dist


def _find_seed_and_carried(
    pts_rc: list[tuple[int, int]],
    g: np.ndarray,
    bg: int,
    col: int,
) -> tuple[tuple[int, int], list[tuple[int, int, int]]]:
    h, w = g.shape
    seed_pts: list[tuple[int, int]] = []
    for r, c in pts_rc:
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    seed_pts.append((r, c))
                    break

    seed = min(seed_pts) if seed_pts else min(pts_rc)

    carried: list[tuple[int, int, int]] = []
    sr0, sc0 = seed
    for drn, dcn in DIR8:
        rr = sr0 + drn
        cc = sc0 + dcn
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != bg and v != col:
                carried.append((drn, dcn, v))
    return seed, carried


def _solve_star(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = sorted(tuple(map(int, p)) for p in np.argwhere(g == col))
        if len(pts_rc) != 4:
            continue

        hub = _centroid_hub(pts_rc)
        if hub is None:
            continue
        signature = _orth_star_signature(pts_rc, hub)
        if signature is None:
            continue

        hr, hc = hub
        if not (0 <= hr < h and 0 <= hc < w):
            continue
        if int(g[hr, hc]) != bg:
            continue

        _seed, carried = _find_seed_and_carried(pts_rc, g, bg, col)

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        valid_shape = True
        for r0, c0 in pts_rc:
            if not _aligned_orth(r0, c0, hr, hc):
                valid_shape = False
                break
            if not _clear_in_original(g, r0, c0, hr, hc, bg, allow={col}):
                valid_shape = False
                break
            drawable_edges.append(((r0, c0), (hr, hc)))
        if not valid_shape:
            continue

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _aligned_orth(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
    Expand only full four-direction orthogonal stars. A same-color group becomes a
    cross-like star when its four pixels sit north/south/east/west of one empty
    integer hub; an optional foreign 8-neighbor near one endpoint defines a shifted
    shadow copy along the same spokes.

    Input structure:
    - Background is the dominant color.
    - Each expandable shape color appears exactly four times as sparse endpoints.
    - The four endpoints are arranged around one empty hub in the four cardinal
      directions, with opposite arms equally long.
    - A shadow cue, if present, is a single different-color pixel 8-neighbor adjacent
      to one endpoint.
    - Negative witnesses may also appear: diagonal four-point stars and open 2/3-arm
      orthogonal stars. They stay unchanged.

    Rule:
    - For each non-background color, test whether its four pixels form a complete
      orthogonal star around an empty integer hub.
    - If so, draw the four endpoint-to-hub spokes using only horizontal/vertical
      segments.
    - If the chosen seed endpoint has a different-color neighbor, copy that color
      along a parallel translated version of every spoke using the same offset, while
      leaving main-shape pixels dominant on overlaps.
    - Colors that are not full four-direction stars remain unchanged.

    Key insight:
    The changed objects are not perimeter polygons anymore. They are recognized by a
    hidden empty center: four same-color endpoints complete a north/south/east/west
    pattern, so the output must converge to that center as a cross/star, not wrap
    around the outside. Diagonal-centered stars and incomplete orthogonal stars are
    intentionally left untouched, which reveals the D2 and D10 restrictions.
    """
    return _solve_star(grid)


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return list(_segment_cells(a[0], a[1], b[0], b[1]))


def _sparse_patch(points: list[tuple[int, int]], color: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.array(points, dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    patch = np.full((h, w), BG, dtype=int)
    for r, c in points:
        patch[r - r0, c - c0] = color
    return patch.copy(), patch.copy()


def _build_cross_spec() -> tuple[list[tuple[int, int]], tuple[int, int], dict[str, int]]:
    hub = (0, 0)
    if np.random.rand() < CROSS_EQUAL_ARMS_PROB:
        arm = int(np.random.randint(*CROSS_ARM_RANGE))
        lv = arm
        lh = arm
    else:
        lv = int(np.random.randint(*CROSS_ARM_RANGE))
        lh = int(np.random.randint(*CROSS_ARM_RANGE))
        if lv == lh:
            lh = CROSS_ARM_RANGE[0] if lh != CROSS_ARM_RANGE[0] else CROSS_ARM_RANGE[1] - 1
    vertices = [(-lv, 0), (lv, 0), (0, -lh), (0, lh)]
    return vertices, hub, {"lv": lv, "lh": lh}


def _shape_cells(vertices: list[tuple[int, int]], hub: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for vertex in vertices:
        cells.update(_gen_segment_cells(vertex, hub))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]], hub: tuple[int, int]) -> list[tuple[int, int]]:
    return [_step_dir(vertex[0], vertex[1], hub[0], hub[1]) for vertex in vertices]


def _choose_shadow(
    vertices: list[tuple[int, int]],
    hub: tuple[int, int],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    directions = _edge_dirs(vertices, hub)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    hub: tuple[int, int],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices + [hub]
    if shadow is not None:
        (_ar, _ac), (dr, dc) = shadow
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
        coords.append((hub[0] + dr, hub[1] + dc))
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(
    shape_color: int,
    shadow_color: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    vertices, hub, meta = _build_cross_spec()
    cells = _shape_cells(vertices, hub)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, hub, cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, hub, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    shifted_hub = (hub[0] - r0, hub[1] - c0)
    if inp[shifted_hub] != BG:
        return None

    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w):
            return None
        if inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)
        if sum(1 for rr, cc in neighbors8(anchor[0], anchor[1]) if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color) != 1:
            return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None
    if int(inp[shifted_hub]) != BG:
        return None
    return inp, out, meta


def _make_diagonal_witness_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    arm = int(np.random.randint(*WITNESS_ARM_RANGE))
    points = [(-arm, -arm), (-arm, arm), (arm, -arm), (arm, arm)]
    return _sparse_patch(points, color)


def _make_open_witness_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    hub = (0, 0)
    arms = {
        (-1, 0): int(np.random.randint(*WITNESS_ARM_RANGE)),
        (1, 0): int(np.random.randint(*WITNESS_ARM_RANGE)),
        (0, -1): int(np.random.randint(*WITNESS_ARM_RANGE)),
        (0, 1): int(np.random.randint(*WITNESS_ARM_RANGE)),
    }
    if np.random.rand() < 0.5:
        chosen_dirs = [((-1, 0), (1, 0)), ((0, -1), (0, 1))][int(np.random.randint(2))]
    else:
        missing = [(0, 1), (0, -1), (-1, 0), (1, 0)][int(np.random.randint(4))]
        chosen_dirs = [d for d in [(-1, 0), (1, 0), (0, -1), (0, 1)] if d != missing]
    points = [(hub[0] + dr * arms[(dr, dc)], hub[1] + dc * arms[(dr, dc)]) for dr, dc in chosen_dirs]
    return _sparse_patch(points, color)


def _sample_palette(shape_count: int) -> tuple[list[int], list[int], int, int] | None:
    reserved_witness_colors = 2
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - reserved_witness_colors)
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + reserved_witness_colors
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count:shape_count + shadow_count]
    diag_witness_color = int(palette[-2])
    open_witness_color = int(palette[-1])
    return shape_colors, shadow_colors, diag_witness_color, open_witness_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    try:
        loc = random_free_location(
            out,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count)
    if palette is None:
        return None
    shape_colors, shadow_colors, diag_witness_color, open_witness_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    total_colors = shape_count + len(shadow_colors) + 2
    witness_unequal = False

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            patch_in, patch_out, meta = patch
            if _try_place(inp, out, patch_in, patch_out):
                witness_unequal = witness_unequal or (meta["lv"] != meta["lh"])
                placed = True
                break
        if not placed:
            return None

    if not _try_place(inp, out, *_make_diagonal_witness_patch(diag_witness_color)):
        return None
    if not _try_place(inp, out, *_make_open_witness_patch(open_witness_color)):
        return None

    if not witness_unequal:
        return None
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
