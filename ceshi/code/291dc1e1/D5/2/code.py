# Variant of 291dc1e1
# Dimension: D5 — 条件旋转规则 (transform)
# Change: 若蓝线为垂直方向，所有矩形做90度逆时针旋转；蓝线水平时不旋转... → 按band分组旋转——近蓝线的不旋转，远蓝线的旋转90度

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis
# source_task: none
# concrete_behavior:
#   - 输入仍然是“角点标记 + 两条边线 + 内部若干实心矩形”的原任务外观。
#   - 生成器仍使用两个与蓝线平行的矩形带：一个紧贴蓝线附近，一个更远。
#   - 输出仍按 (到蓝线距离, 到红线距离) 排序并自上而下堆叠。
#   - 唯一核心变化是旋转条件：以前由蓝线朝向控制，现在由矩形所在 blue-distance band 控制。
# behavior_trace:
#   - 例：蓝线在上边，近 band 的 db=1，远 band 的 db=4。原任务会在 blue_orient=="h" 时全部不旋转；
#     现在则是 db=1 的矩形保持 2xw，db=4 的矩形变成 wx2。
#   - 若整体输入再经旋转/翻转把蓝线移到左边，原任务会把全部矩形都旋转；现在仍然只有 db 更大的那一带旋转。
#   - 差异点因此落在 transform 环节，排序、识别矩形、堆叠方式都不变。
# complexity_calibration:
#   - band 只保留两档，避免规则自由度过高。
#   - 每档都有多个矩形，保证“近的不转、远的转”有重复支撑，不是单个特例。
#   - 远 band 只生成宽度 >=4 的 2xw 矩形，并加不对称内部点，确保旋转肉眼可见。
# hidden_assumptions_review:
#   - 角点探测依赖“角上的标记色与相邻两条边线颜色不同”；生成器保留这一结构。
#   - line_mask 覆盖整条角点所在行/列，依赖边线贯穿整行/整列；生成器保留原布局。
#   - `crop_bbox(...); if (sub == bg).any(): continue` 依赖对象是无孔矩形；生成器只在实心矩形内改颜色，不引入背景孔洞。
#   - 原代码 `if blue_orient == "v": rot90(...)` 依赖“同一实例里所有矩形共享同一旋转状态”。
#     变体下这个前提失效，必须改为逐矩形读取其 db 所在 band。
# placement_legality:
#   - 两个 band 分别放在 canonical 网格的第 2 行和第 5 行开始处，每个矩形高 2。
#   - 同一 band 内矩形之间留 2 列背景；band 之间留 1 行背景；因此不重叠、不越界。
#   - 再通过整网格的旋转/翻转把角点移动到四角，几何关系保持合法。
# visual_observability:
#   - 近 band 输出仍是横向 2xw，小而扁；远 band 输出变成纵向 wx2，又高又窄。
#   - 由于每个样例同时含多块近矩形和多块远矩形，人类能从 2-3 个样例看出“与蓝线距离决定旋转”。
# cleanliness:
#   - 矩形不接触边线，不互相贴边，线色不进入矩形内部，避免视觉混淆。
# color_inference:
#   - 颜色本身不编码规则，因此保持少量主体色并限制内部点颜色为非边线色。
#   - 旋转规律只由空间分区推断，不会与颜色映射混在一起。
# new_main_assumptions:
#   - 新 solver 用 `min(db)` 作为 near band，`db > min(db)` 作为 far band。
#   - 生成器保证总有两个 db 取值，且两档都非空；若外部测试只给一个 band，solver 会退化为“全不转”而非崩溃。


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

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

    if hcol == 1 or vcol == 1:
        blue_col = 1
        red_col = int(vcol if hcol == 1 else hcol)
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

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": sub.copy()})

    assert rects

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            col = int(edge_idx)
            if int(c0) > col:
                return int(c0 - col - 1)
            if int(c1) < col:
                return int(col - c1 - 1)
            return 0
        row = int(edge_idx)
        if int(r0) > row:
            return int(r0 - row - 1)
        if int(r1) < row:
            return int(row - r1 - 1)
        return 0

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    for rect in rects:
        rect["db"] = int(dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(dist_to_edge(rect, red_orient, red_edge_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            along_from_corner(rect, blue_orient),
            along_from_corner(rect, red_orient),
        )
    )

    return {
        "grid": g,
        "bg": bg,
        "blue_orient": blue_orient,
        "rects": rects,
    }


def _render_output(rects: list[dict], bg: int, blue_orient: str, rotate_mode: str) -> np.ndarray:
    near_db = min(int(rect["db"]) for rect in rects)
    seq = []
    for rect in rects:
        arr = rect["arr"]
        if rotate_mode == "band":
            if int(rect["db"]) > int(near_db):
                arr = np.rot90(arr, k=1)
        else:
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
        seq.append(arr)

    outW = int(max(int(arr.shape[1]) for arr in seq))
    outH = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((outH, outW), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((outW - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _solve(grid: np.ndarray, rotate_mode: str) -> np.ndarray:
    scene = _parse_scene(grid)
    return _render_output(scene["rects"], int(scene["bg"]), str(scene["blue_orient"]), rotate_mode)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to two colored edge lines.

    Variant rule:
    - Sort rectangles by (distance to blue edge, distance to red edge).
    - Rectangles in the nearest blue-distance band keep their orientation.
    - Rectangles in more distant blue-distance bands rotate 90° CCW.
    - Stack top-to-bottom, centered horizontally.
    """
    return _solve(grid, rotate_mode="band")


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)


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


def _band_used(widths: list[int]) -> int:
    return int(sum(int(w) for w in widths) + 2 * (len(widths) - 1))


def _pick_other(color: int) -> int:
    choices = [int(c) for c in DOMINANT_COLORS if int(c) != int(color)]
    return int(np.random.choice(choices))


def _make_patch(width: int, dominant: int) -> np.ndarray:
    patch = np.full((2, int(width)), int(dominant), dtype=int)

    # Fixed asymmetric motif so rotation remains visible even when colors repeat.
    patch[0, 0] = _pick_other(dominant)
    patch[1, min(int(width) - 1, 1)] = _pick_other(dominant)
    if int(width) >= 4:
        patch[0, int(width) - 1] = _pick_other(dominant)
    return patch


def _build_canonical() -> np.ndarray:
    for _ in range(400):
        n_near = int(np.random.randint(2, 5))
        n_far = int(np.random.randint(2, 5))
        if n_near + n_far > 7:
            continue

        near_widths = [int(w) for w in np.random.choice([2, 4, 6], size=n_near, replace=True)]
        far_widths = [int(w) for w in np.random.choice([4, 6, 8], size=n_far, replace=True)]

        used_near = _band_used(near_widths)
        used_far = _band_used(far_widths)
        max_used = max(int(used_near), int(used_far))
        trailing = int(np.random.randint(1, 4))
        long_dim = int(max_used + 2 + trailing)
        if long_dim < 9 or long_dim > 30:
            continue

        grid = np.full((8, long_dim), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        for r0, widths in ((2, near_widths), (5, far_widths)):
            c = 2
            for width in widths:
                dominant = int(np.random.choice(DOMINANT_COLORS))
                paste(grid, _make_patch(width, dominant), int(r0), int(c))
                c += int(width) + 2

        return grid
    raise RuntimeError("failed to build canonical input")


def _is_valid_variant_instance(grid: np.ndarray) -> bool:
    scene = _parse_scene(grid)
    db_values = sorted({int(rect["db"]) for rect in scene["rects"]})
    if len(db_values) != 2:
        return False

    counts = [sum(1 for rect in scene["rects"] if int(rect["db"]) == db) for db in db_values]
    if min(counts) < 2:
        return False

    new_out = _solve(grid, rotate_mode="band")
    old_out = _solve(grid, rotate_mode="blue_orient")
    return not np.array_equal(new_out, old_out)


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)
    if not _is_valid_variant_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            main(grid)
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D5 variant")
