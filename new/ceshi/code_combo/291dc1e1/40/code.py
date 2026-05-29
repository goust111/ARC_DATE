# Combo variant of 291dc1e1 (index 40)
# combo_spec: D1/3+D2/0+D4/1
# Name: 简化次键+单band+逆序

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# a) 具体化与行为追踪:
#    原题在 canonical 视图中有两条与蓝线平行的矩形 band: db=1 与 db=4。排序是
#    (db 升序, dr 升序, 沿蓝线坐标, 沿红线坐标)，所以会先输出靠蓝线近的一带，再在带内按靠红线近→远。
#    本融合把所有矩形收缩到同一条 band 上，因此所有对象的 db 完全相同；D1/3 又把主键方向改成远→近，
#    但在单 band 条件下这一项退化为常量；D2/0 则删去 dr，改为沿蓝线方向的位置作次键。
#    于是最终规则变成一维逆向位置排序: 只看矩形沿蓝线方向的位置，离角点远的先输出、近的后输出。
#    例: canonical 中同一 band 上从左到右有 A,B,C,D 四个矩形。原题输出是 A,B,C,D；融合后输出是 D,C,B,A。
#    若蓝线竖直，仍先对每个矩形做 90 度逆时针旋转，再按这个逆序堆叠。
# b) 复杂度校准:
#    单 band 会减少一个排序层级，因此 generator 保持 4-6 个矩形，并限制总长度不超过原题量级，
#    同时要求至少一个矩形宽度 > 2，让“旋转后尺寸变化”可见。矩形颜色优先取不同 dominant 色，
#    再少量加入 minority 像素，避免对象反转后视觉上不可区分。
# c) 隐式假设审查:
#    - corner 检测依赖 marker 在四角之一且与相邻两条边线颜色不同；生成器继续满足。
#    - line_mask 默认整条 corner row/col 都是边线；单 band 不会触碰它们，因此仍成立。
#    - components+bbox 依赖每个对象是 bbox 内无背景孔洞的实心矩形；生成器只生成满矩形。
#    - 原 solver 把 dr 当作同 band 内关键次序来源；融合后这个前提失效，因此排序必须显式忽略 dr。
#    - 由于单 band 下 db 恒定，D1/3 的“降序”不能再靠 `reverse=True` 粗暴实现，否则会连带反转额外 tie-break；
#      这里单独用 far-end 坐标来具体化“逆序”。
# d) 放置合法性:
#    canonical 网格固定为 5 行: 蓝线 1 行 + 间隔 1 行 + band 高 2 行 + 底边框 1 行。
#    所有矩形都放在同一条 band 上，彼此之间保留 2 格背景，并与边线至少隔 1 格；再通过翻转/旋转映射到
#    四个角与两种蓝线朝向，几何上不会产生重叠或越界。
# e) 视觉可观察性:
#    解题者会看到所有矩形都在同一条带上，而且输出顺序与输入沿蓝线方向的自然顺序完全相反。
#    这和原题“近 band 优先，再按另一条线距离排序”明显不同；有 4-6 个对象时，2-4 组样例足以看出该规律。
# f) 可视化整洁度:
#    输入只包含背景、marker、两条整边线和一排分离矩形，没有重叠层或局部噪声区域。
#    minority 像素数量很少，只用来打破内部对称，不会破坏矩形轮廓。
# g) 颜色变化可推断性:
#    颜色不承担逻辑角色，也没有颜色映射。dominant 色只负责区分对象，minority 像素只增强可辨识度。
# h) 融合行为验证指标:
#    用 `single_band_count == 1` 验证 D4/1，用 `variant_boxes == reversed(reference_boxes)` 验证“沿蓝线远→近”，
#    再用 `variant_output != reference_output` 排除视觉退化样本。
# i) 新 solver 的额外假设审查:
#    - `_along_from_far_end` 假设对象在蓝线方向上的投影互不重叠，否则会出现同键；生成器用顺序排布保证唯一。
#    - `main()` 仍允许非生成器输入存在多个 db 层，此时解释为“先按 db 降序，再在每层按沿蓝线远端坐标排序”；
#      生成器则强制只产出单 band 输入，确保融合语义稳定可见。


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _find_corner_and_orients(g: np.ndarray, bg: int) -> tuple[tuple[int, int], int, str, str]:
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
    return corner, marker, blue_orient, red_orient


def _extract_rects(g: np.ndarray, bg: int, corner: tuple[int, int], marker: int) -> list[dict]:
    H, W = g.shape
    cr, cc = map(int, corner)
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True
    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))

    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        arr = crop_bbox(g, box)
        if bool((arr == int(bg)).any()):
            continue
        rects.append({"box": box, "arr": arr.copy()})
    assert rects
    return rects


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        if int(c0) > int(edge_idx):
            return int(c0 - edge_idx - 1)
        if int(c1) < int(edge_idx):
            return int(edge_idx - c1 - 1)
        return 0
    if int(r0) > int(edge_idx):
        return int(r0 - edge_idx - 1)
    if int(r1) < int(edge_idx):
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


def _parse_rects(grid: np.ndarray) -> tuple[list[dict], int, tuple[int, int], str, str]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    corner, marker, blue_orient, red_orient = _find_corner_and_orients(g, bg)
    rects = _extract_rects(g, bg, corner, marker)

    cr, cc = map(int, corner)
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for rect in rects:
        box = tuple(map(int, rect["box"]))
        rect["db"] = int(_dist_to_edge(box, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(box, red_orient, red_edge_idx))
        rect["blue_near"] = int(_along_from_corner(box, blue_orient, corner))
        rect["blue_far"] = int(_along_from_far_end(box, blue_orient, corner))
        rect["red_near"] = int(_along_from_corner(box, red_orient, corner))
    return rects, bg, corner, blue_orient, red_orient


def _ordered_rects(grid: np.ndarray, *, variant: bool) -> tuple[list[dict], int, str]:
    rects, bg, _corner, blue_orient, _red_orient = _parse_rects(grid)
    rects = [dict(rect) for rect in rects]

    if variant:
        rects.sort(
            key=lambda rect: (
                -int(rect["db"]),
                int(rect["blue_far"]),
                int(rect["red_near"]),
            )
        )
    else:
        rects.sort(
            key=lambda rect: (
                int(rect["db"]),
                int(rect["dr"]),
                int(rect["blue_near"]),
                int(rect["red_near"]),
            )
        )
    return rects, int(bg), str(blue_orient)


def _stack_rects(rects: list[dict], bg: int, blue_orient: str) -> np.ndarray:
    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _solve(grid: np.ndarray, *, variant: bool) -> np.ndarray:
    rects, bg, blue_orient = _ordered_rects(grid, variant=variant)
    return _stack_rects(rects, bg, blue_orient)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Rectangles occupy a single band relative to the blue edge.
    - Ignore red-edge distance as a secondary key.
    - Order rectangles from the far end of the blue direction back toward the corner.
    - If the blue edge is vertical, rotate each rectangle 90 degrees CCW before stacking.
    """
    return _solve(grid, variant=True)


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


def _sample_widths() -> list[int]:
    while True:
        n_rects = int(np.random.randint(4, 7))
        widths = [2 * int(np.random.randint(1, 4)) for _ in range(n_rects)]
        used = int(sum(widths) + 2 * (n_rects - 1))
        if used <= 24 and max(widths) > 2:
            return widths


def _sprinkle_minorities(patch: np.ndarray, dominant: int, token: int) -> np.ndarray:
    area = int(patch.size)
    if area < 4:
        return patch

    if np.random.rand() < 0.45:
        return patch

    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != int(dominant)]
    color = int(minority_choices[int(token % len(minority_choices))])
    cell = int(token % area)
    flat = patch.reshape(-1)
    flat[cell] = color
    if area >= 8 and np.random.rand() < 0.5:
        flat[int((cell + area // 2) % area)] = int(minority_choices[int((token + 3) % len(minority_choices))])
    return patch


def _build_canonical() -> np.ndarray:
    widths = _sample_widths()
    used = int(sum(widths) + 2 * (len(widths) - 1))
    trailing = int(np.random.randint(1, 3))
    long_dim = int(used + trailing + 2)

    grid = np.full((5, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    color_pool = list(np.random.permutation(DOMINANT_COLORS)[: len(widths)])
    c = 2
    for idx, width in enumerate(widths):
        color = int(color_pool[idx])
        patch = np.full((2, int(width)), color, dtype=int)
        patch = _sprinkle_minorities(patch, color, idx + width)
        paste(grid, patch, 2, c)
        c += int(width) + 2
    return grid


def _behavior_metrics(grid: np.ndarray) -> dict:
    rects, _bg, _corner, blue_orient, red_orient = _parse_rects(grid)
    boxes = [tuple(map(int, rect["box"])) for rect in rects]
    db_values = {int(rect["db"]) for rect in rects}
    variant_rects, _, _ = _ordered_rects(grid, variant=True)
    reference_rects, _, _ = _ordered_rects(grid, variant=False)
    variant_boxes = [tuple(map(int, rect["box"])) for rect in variant_rects]
    reference_boxes = [tuple(map(int, rect["box"])) for rect in reference_rects]

    return {
        "n_rects": len(boxes),
        "n_db": len(db_values),
        "blue_orient": blue_orient,
        "red_orient": red_orient,
        "variant_boxes": variant_boxes,
        "reference_boxes": reference_boxes,
        "is_exact_reverse": variant_boxes == list(reversed(reference_boxes)),
        "output_differs": not np.array_equal(_solve(grid, variant=True), _solve(grid, variant=False)),
    }


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    metrics = _behavior_metrics(grid)
    if metrics["n_rects"] < 4:
        return None
    if metrics["n_db"] != 1:
        return None
    if not metrics["is_exact_reverse"]:
        return None
    if not metrics["output_differs"]:
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo variant input for 291dc1e1")
