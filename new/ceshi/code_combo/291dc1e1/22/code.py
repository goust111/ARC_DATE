# Combo variant of 291dc1e1 (index 22)
# combo_spec: D2/0+D4/1
# Name: 忽略红线次键+单band

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the fused combo variant
# a) 具体化与行为追踪:
#    原题 canonical 视角是左上角 marker、顶边蓝线、左边红线，矩形分布在两个横向 band:
#    近 band 位于 rows 2:3，远 band 位于 rows 5:6；同 band 内再按距红线的远近排序。
#    这个融合版先应用 D4/1，把所有矩形压到同一个 2-cell 高的 band 中，因此 db 全相同，
#    原题的第一关键字失效。再应用 D2/0，把“同 band 内的次键”从 red distance 改为
#    “沿蓝线方向的位置”。若仍按角点侧量 along-blue，它和 dr 在单 band 内是仿射等价，
#    融合会退化回 D4/1；所以这里采用 D2 单变体里的可观察具体化: 沿蓝线的坐标从蓝线远端量起。
#    例: canonical 中 3 个矩形 A,B,C 从左到右放在同一 band 上。
#      原题 / 单band-red 版: A -> B -> C。
#      本融合版: C -> B -> A。
#    若蓝线为竖直，仍先对每个矩形逆时针旋转 90 度，再按这个顺序竖向堆叠。
#
# b) 复杂度校准:
#    单 band 若只有 1-2 个矩形会太弱，因此生成器使用 3-5 个矩形；宽度仍取 {2,4,6,8}，
#    并优先保证至少一块宽度 > 2、至少两种宽度出现。每个矩形使用互异 dominant color，
#    这样顺序翻转在输出中肉眼可见，不会因颜色重复而弱化。
#
# c) 隐式假设审查:
#    - corner 必须在四角之一；原 corner 检测仍可用。
#    - 两条边线仍占满 corner 所在整行/整列；line_mask 仍可正确抹掉坐标轴。
#    - 目标对象仍是 bbox 内无背景孔洞的实心矩形；components + crop_bbox 的过滤仍成立。
#    - 原 solver 的 `db` 作为主键假设存在多个 blue-band；在本变体上不再成立，因此主排序
#      不能再依赖 `db` 才能表达规则。
#    - 原 solver 的 `dr` 作为次键也不再允许使用；否则融合会退化成单band后按红线距离排序。
#      因此 combo solver 只按沿蓝线远端坐标排序，并保留 along-red 仅作异常 tie 的稳定器。
#
# d) 放置合法性:
#    canonical 网格固定为 5 行: 1 条蓝线 + 1 行间隔 + 2 行矩形 band + 1 行底边留白。
#    矩形从 red-line 旁的首个合法位置开始，沿蓝线方向依次放置，彼此间隔 2 个背景格；
#    长度总占用 `used = sum(widths) + 2*(n-1)`，并强制 `used <= 24`，再加 1-2 个 trailing，
#    所以 long_dim 始终不超过 28。之后的翻转/旋转保持不重叠且不越界。
#
# e) 视觉可观察性:
#    解题者会同时看到两个线索:
#    1. 所有源矩形都在同一条 band 上，不再有远近两个层带；
#    2. 输出并非从靠红线的一端开始堆，而是按沿蓝线方向的“远端优先”顺序堆。
#    因为颜色互异、宽度常不同，2-4 个样例足以看出这是“单band + 反向沿蓝线排序”。
#
# f) 可视化整洁度:
#    场景仍只有背景、corner marker、两条边线、若干分离矩形四类元素。矩形间保留固定 2 格空隙，
#    不引入遮挡或额外标记层，输入和输出都保持规整。
#
# g) 颜色变化可推断性:
#    不存在规则性的颜色映射；颜色只用于区分矩形身份。少量 minority 像素仅增强旋转/顺序可见性，
#    不承载额外语义，因此不会增加颜色归纳负担。
#
# h) 融合行为量化:
#    对合法样本，`n_db == 1` 应恒成立；同时 combo 排序相对原题排序应产生明显 inversion。
#    这里用 `inversion_ratio` 度量 combo 顺序相对原题顺序的逆序比例，并要求生成器只接受
#    `order_differs=True`、`output_differs=True`、`inversion_ratio>=0.99` 的样本，确保没有退化。
#
# i) 新 solver 假设与 generator 保证:
#    combo solver 假设每个矩形的 along-blue far-end 坐标可区分；generator 通过同向顺序放置、
#    正间隔、无重叠保证这一点。solver 中最终的 along-red tie-breaker 只是防御性代码，
#    对生成样本不会真正生效。


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _locate_structure(grid: np.ndarray) -> tuple[tuple[int, int], int, int, str, str, list[dict]]:
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
    return corner, marker, bg, blue_orient, red_orient, rects


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
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


def _along_from_corner(box: tuple[int, int, int, int], orient: str, corner: tuple[int, int]) -> int:
    cr, cc = map(int, corner)
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _along_from_far_end(box: tuple[int, int, int, int], orient: str, corner: tuple[int, int]) -> int:
    cr, cc = map(int, corner)
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(-r1 if cr == 0 else r0)
    return int(-c1 if cc == 0 else c0)


def _ordered_rects(grid: np.ndarray, *, combo: bool) -> tuple[list[dict], int, str]:
    corner, _marker, bg, blue_orient, red_orient, rects = _locate_structure(grid)
    cr, cc = map(int, corner)
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rect in rects:
        box = tuple(map(int, rect["box"]))
        rect["db"] = int(_dist_to_edge(box, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(box, red_orient, red_edge_idx))
        rect["ab_corner"] = int(_along_from_corner(box, blue_orient, corner))
        rect["ab_far"] = int(_along_from_far_end(box, blue_orient, corner))
        rect["ar_corner"] = int(_along_from_corner(box, red_orient, corner))

    if combo:
        rects.sort(key=lambda rect: (int(rect["ab_far"]), int(rect["ar_corner"])))
    else:
        rects.sort(
            key=lambda rect: (
                int(rect["db"]),
                int(rect["dr"]),
                int(rect["ab_corner"]),
                int(rect["ar_corner"]),
            )
        )
    return rects, bg, blue_orient


def _stack_rects(rects: list[dict], bg: int, blue_orient: str) -> np.ndarray:
    seq = []
    for rect in rects:
        arr = rect["arr"]
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
    Fused rule:
    - All source rectangles lie in a single band relative to the blue edge.
    - Ignore red-edge distance as an ordering key.
    - Order rectangles by position along the blue direction, measured from the far end.
    - If the blue edge is vertical, rotate every rectangle 90 degrees CCW.
    - Stack the ordered rectangles top-to-bottom, centered horizontally.
    """
    rects, bg, blue_orient = _ordered_rects(grid, combo=True)
    return _stack_rects(rects, bg, blue_orient)


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


def _sample_widths(count: int) -> list[int]:
    for _ in range(200):
        widths = [2 + 2 * int(np.random.randint(0, 4)) for _ in range(count)]
        used = int(sum(widths) + 2 * (count - 1))
        if used > 24:
            continue
        if max(widths) == 2:
            continue
        if len(set(widths)) < 2:
            continue
        return widths
    raise RuntimeError("failed to sample widths for combo variant")


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
    n_rects = int(np.random.randint(3, 6))
    widths = _sample_widths(n_rects)
    used = int(sum(widths) + 2 * (n_rects - 1))
    trailing_max = min(2, 28 - 2 - used)
    assert trailing_max >= 1
    trailing = int(np.random.randint(1, trailing_max + 1))
    long_dim = int(2 + used + trailing)

    grid = np.full((5, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    colors = np.random.choice(DOMINANT_COLORS, size=n_rects, replace=False)
    c = 2
    for width, color in zip(widths, colors):
        patch = np.full((2, int(width)), int(color), dtype=int)
        patch = _sprinkle_minorities(patch, int(color))
        paste(grid, patch, 2, c)
        c += int(width) + 2

    return grid


def _inversion_ratio(reference_boxes: list[tuple[int, int, int, int]], combo_boxes: list[tuple[int, int, int, int]]) -> float:
    n = len(reference_boxes)
    if n < 2:
        return 0.0
    pos = {box: idx for idx, box in enumerate(reference_boxes)}
    seq = [int(pos[box]) for box in combo_boxes]
    inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] > seq[j]:
                inv += 1
    max_inv = n * (n - 1) // 2
    return inv / max_inv


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    combo_rects, bg, blue_orient = _ordered_rects(grid, combo=True)
    original_rects, _bg2, _bo2 = _ordered_rects(grid, combo=False)

    combo_boxes = [tuple(map(int, rect["box"])) for rect in combo_rects]
    original_boxes = [tuple(map(int, rect["box"])) for rect in original_rects]
    combo_out = _stack_rects(combo_rects, bg, blue_orient)
    original_out = _stack_rects(original_rects, bg, blue_orient)

    db_values = {int(rect["db"]) for rect in combo_rects}
    return {
        "n_rects": len(combo_rects),
        "n_db": len(db_values),
        "order_differs": combo_boxes != original_boxes,
        "output_differs": not np.array_equal(combo_out, original_out),
        "inversion_ratio": float(_inversion_ratio(original_boxes, combo_boxes)),
    }


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    metrics = _behavior_metrics(grid)
    if int(metrics["n_db"]) != 1:
        return None
    if int(metrics["n_rects"]) < 3:
        return None
    if not bool(metrics["order_differs"]):
        return None
    if not bool(metrics["output_differs"]):
        return None
    if float(metrics["inversion_ratio"]) < 0.99:
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo variant input for 291dc1e1")
