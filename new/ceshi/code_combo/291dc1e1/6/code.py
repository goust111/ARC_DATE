# Combo variant of 291dc1e1 (index 6)
# combo_spec: D1/3+D4/0
# Name: 距离逆序+三band分布

from __future__ import annotations

import itertools

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis for this combo variant
# a) 具体化与行为追踪
#    - 原题 canonical 结构是两层矩形带, 到蓝线距离分别为 db=1 和 db=4, 每层内部再按到红线距离 dr 递增。
#      输出时按 (db, dr) 升序, 所以会先堆靠蓝线/靠角点的矩形。
#    - 本融合版把两处因素同时改掉:
#      1. band 数从两层扩成三层, canonical 中三层矩形带位于 db=1,4,7。
#      2. 排序主键前两项都反向, 变成先看 db 从远到近, 再看 dr 从远到近。
#    - 举例: 若三层 band 中各有 2 个矩形, 它们的 (db,dr) 可能依次是
#      [(1,1),(1,5),(4,1),(4,5),(7,1),(7,5)]。
#      原题输出顺序是 [(1,1),(1,5),(4,1),(4,5),(7,1),(7,5)]；
#      本题输出顺序则变成 [(7,5),(7,1),(4,5),(4,1),(1,5),(1,1)]。
#      所以解题者必须同时识别“多了一层更远 band”以及“排序方向反了”。
#    - 旋转规则不变: 若蓝线是竖直的, 仍对每个矩形做 90 度逆时针旋转后再纵向堆叠。
# b) 复杂度校准
#    - 三层 band 若每层只放 1 个矩形, 视觉上会退化成只按层排序, 看不出 dr 也被逆序；因此每层固定 2-5 个矩形。
#    - 但三层 * 5 会太拥挤, 因而总数额外限制到 6-10, 保持对象足够多又不至于糊成一排。
#    - 保留少量异色像素, 让同色同宽矩形在输出交换顺序时仍然肉眼可见。
# c) 隐式假设审查
#    - `corner` 定位依赖: 角点 marker 与相邻两条线颜色不同。生成器继续固定 marker=0, blue=1, red=2。
#    - `line_mask` 依赖: 角点整行整列可安全掩掉。三 band 仍全部位于内部, 与边线至少隔 1 格, 该前提不变。
#    - `components_mask` + bbox 提取依赖: 每个对象是 bbox 内无背景孔洞的实心矩形。生成器继续只放满矩形并仅撒少量异色像素。
#    - 排序代码原本把 `db/dr` 与 tie-breaker 一起正向排序。融合后只能反转 `db/dr`, 不能直接 `reverse=True`,
#      否则会错误反转 `along_from_corner` 的 tie-breaker, 造成额外规则漂移。
#    - 新增假设是输入里必须真的出现三种蓝线距离 {1,4,7}; 生成器在输出前显式验证, 防止退化回两 band。
# d) 放置合法性
#    - canonical short_dim 从 8 扩到 11: 1 条蓝线 + 1 间隔 + 2 高 band + 1 间隔 + 2 高 band + 1 间隔 + 2 高 band + 1 边框。
#      因此三层 band 的起始行固定为 2,5,8, 分别对应 db=1,4,7。
#    - 沿蓝线方向仍使用“矩形宽度为偶数 2/4/6/8, 相邻矩形间隔 2 格背景, 首个矩形距红线 1 格”。
#      `_distribute_band_widths()` 负责让每层总占用精确落在可行区间内, `_transform()` 只做翻转/旋转, 不会引入重叠或越界。
# e) 视觉可观察性
#    - 如果只扩成三层但仍升序, 解题者会得到“只是多了一排”的简单规则；如果只降序但保持两层, 变化太局部。
#      两者叠加后, 输出会明显从“近到远”改成“远到近”, 并且最先出现的不是原题已有的两层之一, 而是新出现的第三层。
#    - 生成器拒绝 `descending` 与 `ascending` 输出完全相同的样本, 保证 D1 改动可观察。
#    - 同时验证 `db` 精确为 {1,4,7}, 保证 D4 改动可观察。
# f) 可视化整洁度
#    - 输入仍只有四类视觉元素: 单色背景、两条边线、一个角点 marker、三排规整矩形。
#    - band 之间固定隔 1 行背景, 矩形之间固定隔 2 列背景, 不会出现贴边连块或难以分辨的接触。
# g) 颜色变化可推断性
#    - 颜色不参与新逻辑, 只用于区分边线/marker/矩形纹理。
#    - 异色像素数量被压到每个矩形最多 3 个, 且仅作为“矩形身份标签”, 不形成额外颜色映射任务。
# h) source_task
#    - D4/0 没有单独参考实现, 这里直接依据原题 spec 将 short_dim 规律性扩展到第三层 band。
# i) 新 solver 的额外假设审查
#    - `_solve()` 新增的逻辑只是对现有 `db/dr` 取负排序, 因而它额外依赖“矩形的 `db/dr` 可被可靠测出”。
#      生成器通过固定 band 与首列偏移确保这一点。
#    - `_band_metrics()` 假设输入确实只有矩形对象, 不含其他离散噪声；生成器满足该前提。

BG = 8
BLUE = 1
RED = 2
MARKER = 0
BAND_ROWS = (2, 5, 8)
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _parse_task(grid: np.ndarray) -> dict:
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
        arr = crop_bbox(g, box)
        if bool((arr == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": arr.copy()})

    assert rects
    return {
        "grid": g,
        "bg": bg,
        "corner": (cr, cc),
        "marker": marker,
        "blue_orient": blue_orient,
        "red_orient": red_orient,
        "rects": rects,
    }


def _dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
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


def _along_from_corner(rect: dict, orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _solve(grid: np.ndarray, *, descending: bool) -> np.ndarray:
    info = _parse_task(grid)
    bg = int(info["bg"])
    cr, cc = map(int, info["corner"])
    blue_orient = str(info["blue_orient"])
    red_orient = str(info["red_orient"])
    rects = [dict(rct) for rct in info["rects"]]

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rct in rects:
        rct["db"] = int(_dist_to_edge(rct, blue_orient, blue_edge_idx))
        rct["dr"] = int(_dist_to_edge(rct, red_orient, red_edge_idx))

    if descending:
        rects.sort(
            key=lambda r: (
                -int(r["db"]),
                -int(r["dr"]),
                _along_from_corner(r, blue_orient, cr, cc),
                _along_from_corner(r, red_orient, cr, cc),
            )
        )
    else:
        rects.sort(
            key=lambda r: (
                int(r["db"]),
                int(r["dr"]),
                _along_from_corner(r, blue_orient, cr, cc),
                _along_from_corner(r, red_orient, cr, cc),
            )
        )

    seq = []
    for rct in rects:
        arr = rct["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack solid rectangles taken from three parallel bands.

    Combo rule:
    - Rectangles occupy three bands at distances db in {1, 4, 7} from the blue edge.
    - Sort by (distance to blue edge, distance to red edge) in descending order.
    - Keep original tie-breakers along the edge directions.
    - Rotate each rectangle 90 degrees CCW if the blue edge is vertical.
    - Stack the resulting rectangles top-to-bottom, centered horizontally.
    """
    return _solve(grid, descending=True)


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
    if area < 8 or np.random.rand() < 0.55:
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


def _force_unique_signature(patch: np.ndarray, dominant: int, serial: int) -> np.ndarray:
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    h, w = map(int, patch.shape)
    positions = [(0, w - 1), (1, 0), (0, 0), (1, w - 1), (0, w // 2), (1, w // 2)]
    r0, c0 = positions[int(serial) % len(positions)]
    patch[int(r0), int(c0)] = minority_choices[int(serial) % len(minority_choices)]
    if w >= 4:
        r1 = 1 - int(r0)
        c1 = int((serial // len(minority_choices)) % w)
        if patch[int(r1), int(c1)] == dominant:
            patch[int(r1), int(c1)] = minority_choices[int((serial + 3) % len(minority_choices))]
    return patch


def _build_canonical() -> np.ndarray:
    long_dim = int(np.random.randint(13, 29))
    options = []
    for trailing in (1, 2):
        max_used = long_dim - 2 - trailing
        if max_used < 6 or max_used % 2:
            continue
        triples = []
        for counts in itertools.product(range(2, 6), repeat=3):
            if sum(counts) > 10:
                continue
            if any((4 * int(n) - 2) > max_used for n in counts):
                continue
            eligible = [i for i, n in enumerate(counts) if max_used <= 10 * int(n) - 2]
            if eligible:
                triples.append((tuple(map(int, counts)), eligible))
        if triples:
            options.append((max_used, triples))

    max_used, triples = options[int(np.random.randint(len(options)))]
    counts, eligible = triples[int(np.random.randint(len(triples)))]
    max_band = int(np.random.choice(eligible))

    band_widths = []
    for band_idx, count in enumerate(counts):
        hi = min(max_used, 10 * int(count) - 2)
        if band_idx == max_band:
            used_target = int(max_used)
        else:
            used_target = int(np.random.choice(np.arange(4 * int(count) - 2, hi + 1, 2)))
        band_widths.append(_distribute_band_widths(int(count), int(used_target)))

    grid = np.full((11, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    used_signatures: set[tuple[int, ...]] = set()
    color_pool = list(np.random.permutation(DOMINANT_COLORS))
    serial = 0

    for r0, widths in zip(BAND_ROWS, band_widths):
        c = 2
        for w in widths:
            if color_pool:
                color = int(color_pool.pop())
            else:
                color = int(np.random.choice(DOMINANT_COLORS))

            patch = np.full((2, int(w)), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)
            signature = tuple(int(v) for v in patch.reshape(-1))
            if signature in used_signatures:
                patch = _force_unique_signature(patch, color, serial)
                signature = tuple(int(v) for v in patch.reshape(-1))
            if signature in used_signatures:
                patch = _force_unique_signature(patch, color, serial + 11)
                signature = tuple(int(v) for v in patch.reshape(-1))
            used_signatures.add(signature)

            paste(grid, patch, int(r0), int(c))
            c += int(w) + 2
            serial += 1

    return grid


def _band_metrics(grid: np.ndarray) -> dict:
    info = _parse_task(grid)
    cr, cc = map(int, info["corner"])
    blue_orient = str(info["blue_orient"])
    red_orient = str(info["red_orient"])
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    rects = []
    for idx, rct in enumerate(info["rects"]):
        rect = dict(rct)
        rect["db"] = int(_dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect, red_orient, red_edge_idx))
        rect["idx"] = int(idx)
        rects.append(rect)

    counts_by_db = {}
    for rect in rects:
        counts_by_db[rect["db"]] = int(counts_by_db.get(rect["db"], 0) + 1)

    desc_ids = [
        rect["idx"]
        for rect in sorted(
            rects,
            key=lambda r: (
                -int(r["db"]),
                -int(r["dr"]),
                _along_from_corner(r, blue_orient, cr, cc),
                _along_from_corner(r, red_orient, cr, cc),
            ),
        )
    ]
    asc_ids = [
        rect["idx"]
        for rect in sorted(
            rects,
            key=lambda r: (
                int(r["db"]),
                int(r["dr"]),
                _along_from_corner(r, blue_orient, cr, cc),
                _along_from_corner(r, red_orient, cr, cc),
            ),
        )
    ]
    return {
        "db_values": sorted(int(v) for v in counts_by_db),
        "counts_by_db": {int(k): int(v) for k, v in counts_by_db.items()},
        "desc_ids": desc_ids,
        "asc_ids": asc_ids,
    }


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    try:
        metrics = _band_metrics(grid)
        new_out = _solve(grid, descending=True)
        old_out = _solve(grid, descending=False)
    except Exception:
        return None

    if metrics["db_values"] != [1, 4, 7]:
        return None
    if any(int(metrics["counts_by_db"].get(db, 0)) < 2 for db in (1, 4, 7)):
        return None
    if metrics["desc_ids"] == metrics["asc_ids"]:
        return None
    if np.array_equal(new_out, old_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo D1/3+D4/0")
