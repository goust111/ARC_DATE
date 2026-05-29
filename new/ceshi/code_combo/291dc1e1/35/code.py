# Combo variant of 291dc1e1 (index 35)
# combo_spec: D4/0+D11/3+D13/2
# Name: 三band+band着色+band分离输出

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the fused variant
# a) 具体化与行为追踪:
#    原题输入有一组角点坐标轴：marker=0 在角上，blue=1 与 red=2 两条边线互相垂直；
#    所有矩形都平行于 blue 线，原 canonical 只有两条 band，db 分别为 1 和 4。
#    原 solver 会按 (db, dr, along_blue, along_red) 排序，若 blue 为竖直则先整体旋转每个矩形，
#    最后把全部矩形堆成一个单列输出。
#    本融合版把 canonical band 扩成三层，db 固定为 1/4/7；排序线索仍然来自同一套边线坐标系，
#    但输出阶段不再把三层混成一列，而是把每个 db 层单独堆成一根柱子，并按近→中→远从左到右并列。
#    同时每根柱子统一重着色：最近层用 blue(1)，最远层用 red(2)，中间层用 corner marker(0)。
#    这样输出的三个块同时编码了 band 分组与 band 身份。例：若三层矩形宽度分别是
#    near=[4,2], mid=[6,4], far=[2,8]，原题会输出一根 6 个矩形的总柱；融合后会输出三根并列柱，
#    左柱全 1，中柱全 0，右柱全 2。
# b) 复杂度校准:
#    - band 只增加到三层，不继续细分更多距离档位；这样变化明显，但规则仍可由 2-4 个样例归纳。
#    - 每层保留 2-4 个矩形，保证“同层独立堆叠”与“同层统一颜色”都有重复证据，不会只靠单个对象硬猜。
#    - 输出颜色只用 0/1/2 三种结构色，避免无结构的大量颜色映射。
# c) 隐式假设审查:
#    - 角点检测仍依赖“一个非背景角点，且与相邻边线颜色不同”；生成器保留该结构。
#    - 线遮罩仍假设整个角点行/列都是结构轴，不属于矩形；三 band 只是把矩形放得更远，不破坏此假设。
#    - 组件提取仍依赖每个对象 bbox 内无背景洞；生成器继续只放实心 2xh 矩形。
#    - 原题默认“所有矩形最终合并到一个输出序列”已失效；现在必须先按 db 分组再分别堆叠。
#    - 原题默认“颜色可原样复制”也失效；现在输出颜色由 band rank 决定。
#    - 新增假设是“恰有 3 个不同的 db 层，且每层非空”；generate_input() 用固定 3-band canonical 保证。
# d) 放置合法性:
#    - canonical short_dim 扩到 11 = 1 edge + 1 gap + 2 band0 + 1 gap + 2 band1 + 1 gap + 2 band2 + 1 border。
#      因而三条 band 的 db 正好是 1/4/7，互不接触，也不会压到边线。
#    - 同一 band 内矩形仍沿 blue 方向排列，矩形之间保留 2 格背景；各矩形宽度为 {2,4,6,8}。
#    - 整体旋转/翻转只改变朝向，不改变“不重叠、不越界、边线仍在角上”的合法性。
# e) 视觉可观察性:
#    - 输入中能肉眼看到 3 条平行 band；输出中能肉眼看到 3 根并列柱，且颜色与柱分组严格绑定。
#    - 因为每层有多个矩形，解题者能从几组例子看出：先在输入里按离 blue 线的距离分三层，
#      再各自按离 red 线的顺序向下堆叠。
# f) 可视化整洁度:
#    - 输入仍然是“角点 + 两条边线 + 三条整齐矩形带”，没有额外符号叠加。
#    - 输出三柱之间只放 1 列背景间隔，分组明确但不拥挤。
# g) 颜色变化可推断性:
#    - 为了让 0/1/2 的输出映射可读，生成器把输入矩形主色和杂色都限制在 {3,4,5,6,7,9}，
#      使 0/1/2 只承担结构角色。于是解题者能把三柱颜色稳定地关联到三条 band 的 rank。
#    - 中间 band 使用 marker 色 0，不是任意新颜色；它来自输入里第三个显著结构色，因此可推断。
# h) 退化检测指标:
#    - 结构指标: 抽取出的矩形必须恰有 3 个 db 层，且每层至少 2 个矩形。
#    - 行为指标: 融合输出必须同时不同于
#      1) 原题单柱原色输出，
#      2) 三柱但不重着色的输出，
#      3) 三色但仍单柱的输出。
#      只有这样，三 band、band 着色、band 分离三个改动才都真正生效。
# i) main() 的新增假设与 generator 保证:
#    - `_analyze_grid()` 假设提取到的对象确实是按 db 分成 3 层；generator 固定用 3 个 band 生成。
#    - `_solve()` 假设 band 的左右顺序由 db 升序决定；generator 让 db 只有 1/4/7 三档，因此稳定。
#    - 重着色逻辑假设输入矩形内部不使用 0/1/2 作为常规纹理颜色；generator 明确避免。

BG = 8
BLUE = 1
RED = 2
MARKER = 0
SHORT_DIM = 11
BAND_ROWS = (2, 5, 8)
BAND_OUTPUT_COLORS = (BLUE, MARKER, RED)
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)


def _locate_corner_and_axes(g: np.ndarray, bg: int) -> tuple[int, int, int, str, str]:
    H, W = g.shape
    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == H - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == W - 1:
            nbr.append(int(g[r, c - 1]))
        if nbr and all(int(n) != int(g[r, c]) for n in nbr):
            corner = (int(r), int(c))
            break
    assert corner is not None

    cr, cc = map(int, corner)
    marker = int(g[cr, cc])
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
        red_col = int(vcol if hcol == BLUE else hcol)
    else:
        hlen = 0
        c = int(cc + hstep)
        while 0 <= c < W and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < H and int(g[r, cc]) == int(vcol):
            vlen += 1
            r += int(vstep)
        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"
    return int(cr), int(cc), int(marker), blue_orient, red_orient


def _dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        if c0 > edge_idx:
            return int(c0 - edge_idx - 1)
        if c1 < edge_idx:
            return int(edge_idx - c1 - 1)
        return 0
    if r0 > edge_idx:
        return int(r0 - edge_idx - 1)
    if r1 < edge_idx:
        return int(edge_idx - r1 - 1)
    return 0


def _along_from_corner(rect: dict, orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _extract_rectangles(
    g: np.ndarray, bg: int, marker: int, cr: int, cc: int
) -> list[dict]:
    H, W = g.shape
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": box, "arr": sub.copy()})
    assert rects
    return rects


def _analyze_grid(grid: np.ndarray) -> tuple[int, str, list[dict], list[int]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, marker, blue_orient, red_orient = _locate_corner_and_axes(g, bg)
    rects = _extract_rectangles(g, bg, marker, cr, cc)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for rect in rects:
        rect["db"] = int(_dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect, red_orient, red_edge_idx))
        rect["ab"] = int(_along_from_corner(rect, blue_orient, cr, cc))
        rect["ar"] = int(_along_from_corner(rect, red_orient, cr, cc))

    rects.sort(key=lambda rect: (int(rect["db"]), int(rect["dr"]), int(rect["ab"]), int(rect["ar"])))
    band_levels = sorted({int(rect["db"]) for rect in rects})
    assert len(band_levels) == 3
    return bg, blue_orient, rects, band_levels


def _build_pillar(seq: list[np.ndarray], bg: int) -> np.ndarray:
    assert seq
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _prepare_patch(arr: np.ndarray, blue_orient: str, bg: int, target_color: int | None) -> np.ndarray:
    patch = np.array(arr, copy=True)
    if blue_orient == "v":
        patch = np.rot90(patch, k=1)
    if target_color is not None:
        patch = np.where(patch != int(bg), int(target_color), int(bg))
    return patch


def _solve(grid: np.ndarray, *, split_by_band: bool, recolor: bool) -> np.ndarray:
    bg, blue_orient, rects, band_levels = _analyze_grid(grid)
    band_rank = {int(db): idx for idx, db in enumerate(band_levels)}

    if split_by_band:
        pillars = []
        for db in band_levels:
            band = [rect for rect in rects if int(rect["db"]) == int(db)]
            assert band
            seq = []
            for rect in band:
                color = BAND_OUTPUT_COLORS[band_rank[int(db)]] if recolor else None
                seq.append(_prepare_patch(rect["arr"], blue_orient, bg, color))
            pillars.append(_build_pillar(seq, bg))

        gap = 1
        out_h = int(max(int(p.shape[0]) for p in pillars))
        out_w = int(sum(int(p.shape[1]) for p in pillars) + gap * (len(pillars) - 1))
        out = np.full((out_h, out_w), int(bg), dtype=int)
        out_c = 0
        for pillar in pillars:
            paste(out, pillar, 0, int(out_c), transparent=bg)
            out_c += int(pillar.shape[1]) + gap
        return out

    seq = []
    for rect in rects:
        color = BAND_OUTPUT_COLORS[band_rank[int(rect["db"])]] if recolor else None
        seq.append(_prepare_patch(rect["arr"], blue_orient, bg, color))
    return _build_pillar(seq, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Detect the corner-based blue/red coordinate system from the original task.
    - Extract solid rectangles and group them into 3 blue-distance bands.
    - Sort each band by the original red-distance / along-axis keys.
    - Rotate all rectangles 90 degrees CCW if the blue edge is vertical.
    - Recolor the near / middle / far bands to blue / marker / red respectively.
    - Stack each band into its own pillar and place the three pillars left-to-right.
    """
    return _solve(grid, split_by_band=True, recolor=True)


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return grid
        if corner == "tr":
            return np.fliplr(grid)
        if corner == "bl":
            return np.flipud(grid)
        return np.rot90(grid, 2)

    base = np.rot90(grid, 1)
    if corner == "bl":
        return base
    if corner == "tl":
        return np.flipud(base)
    if corner == "tr":
        return np.rot90(grid, 3)
    return np.flipud(np.rot90(grid, 3))


def _distribute_band_widths(count: int, used_target: int) -> list[int]:
    extra_units = (used_target - (4 * count - 2)) // 2
    order = np.random.permutation(count)
    units = np.zeros(count, dtype=int)
    remaining = int(extra_units)
    for i, idx in enumerate(order):
        slots_left = count - i - 1
        lo = max(0, remaining - 3 * slots_left)
        hi = min(3, remaining)
        take = int(np.random.randint(lo, hi + 1))
        units[int(idx)] = take
        remaining -= take
    return [2 + 2 * int(u) for u in units]


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.45:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _build_canonical() -> np.ndarray:
    long_dim = int(np.random.randint(12, 30))
    options: list[tuple[int, list[tuple[tuple[int, int, int], list[int]]]]] = []
    for trailing in (1, 2):
        max_used = long_dim - 2 - trailing
        if max_used < 6 or max_used % 2:
            continue
        max_count = min(4, (max_used + 2) // 4)
        triples = []
        for n0 in range(2, max_count + 1):
            for n1 in range(2, max_count + 1):
                for n2 in range(2, max_count + 1):
                    if n0 + n1 + n2 > 10:
                        continue
                    counts = (n0, n1, n2)
                    eligible = [idx for idx, n in enumerate(counts) if max_used <= 10 * n - 2]
                    if eligible:
                        triples.append((counts, eligible))
        if triples:
            options.append((max_used, triples))

    assert options
    max_used, triples = options[int(np.random.randint(len(options)))]
    counts, eligible = triples[int(np.random.randint(len(triples)))]
    anchor = int(np.random.choice(eligible))

    width_lists = []
    for idx, count in enumerate(counts):
        cap = min(max_used, 10 * count - 2)
        if idx == anchor:
            used = int(max_used)
        else:
            used = int(np.random.choice(np.arange(4 * count - 2, cap + 1, 2)))
        width_lists.append(_distribute_band_widths(count, used))

    # Keep the three pillars visually distinct often enough that the split is obvious.
    if width_lists[0] == width_lists[1] == width_lists[2]:
        width_lists[2] = list(reversed(width_lists[2]))

    grid = np.full((SHORT_DIM, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, widths in zip(BAND_ROWS, width_lists):
        c = 2
        for w in widths:
            color = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, int(w)), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)
            paste(grid, patch, int(r0), int(c))
            c += int(w) + 2
    return grid


def _variant_is_visible(grid: np.ndarray) -> bool:
    try:
        bg, _blue_orient, rects, band_levels = _analyze_grid(grid)
    except Exception:
        return False

    counts = {int(db): 0 for db in band_levels}
    for rect in rects:
        counts[int(rect["db"])] += 1
    if len(band_levels) != 3 or min(counts.values()) < 2:
        return False

    fused = _solve(grid, split_by_band=True, recolor=True)
    baseline = _solve(grid, split_by_band=False, recolor=False)
    no_recolor = _solve(grid, split_by_band=True, recolor=False)
    no_split = _solve(grid, split_by_band=False, recolor=True)

    if np.array_equal(fused, baseline):
        return False
    if np.array_equal(fused, no_recolor):
        return False
    if np.array_equal(fused, no_split):
        return False

    colors = set(map(int, fused.flatten()))
    if not {BLUE, MARKER, RED}.issubset(colors):
        return False

    gap = 1
    pillars = []
    band_rank = {int(db): idx for idx, db in enumerate(band_levels)}
    for db in band_levels:
        band = [rect for rect in rects if int(rect["db"]) == int(db)]
        seq = [
            _prepare_patch(rect["arr"], _blue_orient, bg, BAND_OUTPUT_COLORS[band_rank[int(db)]])
            for rect in band
        ]
        pillars.append(_build_pillar(seq, bg))
    expected_width = int(sum(int(p.shape[1]) for p in pillars) + gap * (len(pillars) - 1))
    return int(fused.shape[1]) == expected_width


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)
    try:
        _ = main(grid)
    except Exception:
        return None
    if not _variant_is_visible(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant 35")
