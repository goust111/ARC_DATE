# Combo variant of 291dc1e1 (index 1)
# combo_spec: D1/4+D5/3
# Name: 尺寸驱动排序与旋转

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

"""
Design analysis for the combo variant

a) 具体化与行为追踪
- 输入外观仍是原题框架：角点 marker、两条边线、内部两条与蓝线平行的矩形带。
- 原题把矩形按 `(到蓝线距离, 到红线距离)` 排序；若蓝线竖直，则所有矩形统一逆时针旋转 90 度。
- 本融合版把“沿蓝线方向的长度 thick_dim”同时拿来做两件事：
  1. 排序：按 `thick_dim` 从短到长堆叠。
  2. 旋转：`2->0°`, `4->90°`, `6->180°`, `8->270°`。
- 具体例子：若四个矩形的输入位置顺序是 `[6, 2, 8, 4]`，原题会按 band/位置输出；
  本题会改成 `[2, 4, 6, 8]`，且这四块分别保留、转 90、转 180、转 270。
- 差异不是“先按尺寸排序，再随便旋转”，而是同一个尺寸变量同时控制顺序和朝向。

b) 复杂度校准
- 旋转编码只用四档 `thick_dim in {2,4,6,8}`，自由度有限，便于从少量样例归纳。
- 为了让排序不退化成大量 tie-break，本生成器每个样例固定放四个矩形，四档尺寸各出现一次。
- 为了让旋转映射可见，每个矩形都带有不对称内部花纹；否则 2x2 或纯色块会掩盖旋转差异。

c) 隐式假设审查
- 角点定位、蓝红线识别、line mask、连通域提取仍沿用原题前提，不受融合规则影响。
- 原题中 `dist_to_edge(...)` 和 `rects.sort((db, dr, ...))` 假设“位置距离是主键”；这里失效，
  必须改为读 `thick_dim`。
- 原题中 `if blue_orient == "v": rot90(...)` 假设“整张图里所有矩形共享同一旋转状态”；
  本题失效，必须改为逐矩形按尺寸查表。
- 新 solver 额外假设：每个矩形的 `thick_dim` 必须属于 `{2,4,6,8}`。生成器按 spec 保证这一点。

d) 放置合法性
- canonical 视角下蓝线在上、红线在左；矩形只放在第 2-3 行与第 5-6 行，两带之间隔 1 行。
- 同带矩形之间固定隔 2 格背景，第一块从离红线 1 格的位置开始，因此不重叠、不越界、不粘边线。
- 再做整图翻转或 90 度旋转到其他角落时，这些合法性保持不变。

e) 视觉可观察性
- 每个样例都同时包含 `2,4,6,8` 四档尺寸，因此“短到长”和“四档旋转码”能在同一输出里一起看到。
- 生成器显式拒绝“尺寸排序和原始位置排序恰好相同”的样例，否则人类会误以为仍是原题排序。
- 也拒绝“尺寸编码旋转输出与仅按尺寸排序但不编码旋转的输出相同”的退化样例。

f) 可视化整洁度
- 只保留原题已有元素：背景、边线、角点、矩形，不新增额外标记层。
- 内部花纹只出现在矩形内部且数量很少，用来显示朝向，不会破坏对象边界。

g) 颜色变化可推断性
- 颜色不参与规则编码。颜色只负责区分对象和让旋转更可见。
- 为避免把注意力拉到颜色映射上，主体色只从少量常用 ARC 颜色中选，内部异色也限制在同一小集合。

h) 量化验证指标
- `legacy_matches_size_order`: 原题排序序列是否与本题尺寸排序序列相同；应尽量为 False。
- `combo_matches_d1_output`: 本题输出是否与“只改排序、不做尺寸编码旋转”的输出相同；应为 False。
- `rotation_diff_count`: 相比 D1 参考旋转规则，有多少个矩形的旋转角不同；本生成器固定为 3。

i) 修改后 main() 的新隐式假设，以及 generator 如何保证
- 假设 1：矩形是无背景孔洞的实心轴对齐矩形。generator 只在矩形内部改颜色，不放背景洞。
- 假设 2：至少有一个矩形，且尺寸能从 bbox 沿蓝线方向直接测得。generator 固定生成 4 块矩形。
- 假设 3：尺寸排序应该能独立于位置被观察到。generator 用四个互异尺寸并打乱摆放来保证。
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
ROTATION_BY_SIZE = {2: 0, 4: 1, 6: 2, 8: 3}


def _locate_frame(g: np.ndarray) -> tuple[int, int, int, int, str, str]:
    h, w = g.shape
    bg = int(get_bg(g))

    corners = [(0, 0), (0, int(w - 1)), (int(h - 1), 0), (int(h - 1), int(w - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == h - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == w - 1:
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
        while 0 <= c < w and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < h and int(g[r, cc]) == int(vcol):
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
    return bg, marker, cr, cc, blue_orient, red_orient


def _extract_rectangles(g: np.ndarray, bg: int, marker: int, cr: int, cc: int) -> list[dict]:
    h, w = g.shape
    line_mask = np.zeros((h, w), dtype=bool)
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
    return rects


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


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


def _thick_dim(box: tuple[int, int, int, int], blue_orient: str) -> int:
    r0, c0, r1, c1 = map(int, box)
    if blue_orient == "h":
        return int(c1 - c0 + 1)
    return int(r1 - r0 + 1)


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg, marker, cr, cc, blue_orient, red_orient = _locate_frame(g)
    rects = _extract_rectangles(g, bg, marker, cr, cc)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for rect in rects:
        box = tuple(map(int, rect["box"]))
        rect["db"] = int(_dist_to_edge(box, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(box, red_orient, red_edge_idx))
        rect["ab"] = int(_along_from_corner(box, blue_orient, cr, cc))
        rect["ar"] = int(_along_from_corner(box, red_orient, cr, cc))
        rect["thick_dim"] = int(_thick_dim(box, blue_orient))

    return {
        "grid": g,
        "bg": bg,
        "cr": cr,
        "cc": cc,
        "blue_orient": blue_orient,
        "red_orient": red_orient,
        "rects": rects,
    }


def _sorted_rectangles(scene: dict, mode: str) -> list[dict]:
    rects = list(scene["rects"])
    if mode == "legacy":
        rects.sort(key=lambda r: (int(r["db"]), int(r["dr"]), int(r["ab"]), int(r["ar"])))
        return rects
    if mode == "size":
        rects.sort(key=lambda r: (int(r["thick_dim"]), int(r["ab"]), int(r["ar"])))
        return rects
    raise ValueError(f"unknown sort mode: {mode}")


def _rotation_k(thick_dim: int) -> int:
    assert int(thick_dim) in ROTATION_BY_SIZE
    return int(ROTATION_BY_SIZE[int(thick_dim)])


def _stack_arrays(arrays: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(arr.shape[1]) for arr in arrays))
    out_h = int(sum(int(arr.shape[0]) for arr in arrays))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in arrays:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _render(rects: list[dict], bg: int, blue_orient: str, rotate_mode: str) -> np.ndarray:
    legacy_k = 1 if blue_orient == "v" else 0
    seq = []
    for rect in rects:
        arr = rect["arr"]
        if rotate_mode == "legacy":
            k = int(legacy_k)
        elif rotate_mode == "combo":
            k = int(_rotation_k(int(rect["thick_dim"])))
        else:
            raise ValueError(f"unknown rotate mode: {rotate_mode}")
        if k:
            arr = np.rot90(arr, k=int(k))
        seq.append(arr)
    return _stack_arrays(seq, bg)


def _solve(grid: np.ndarray, sort_mode: str, rotate_mode: str) -> np.ndarray:
    scene = _parse_scene(grid)
    rects = _sorted_rectangles(scene, sort_mode)
    return _render(rects, int(scene["bg"]), str(scene["blue_orient"]), rotate_mode)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack rectangles in ascending order of their size measured parallel to the blue edge.

    The same size also encodes the per-rectangle rotation:
    - 2 -> 0 degrees
    - 4 -> 90 degrees CCW
    - 6 -> 180 degrees
    - 8 -> 270 degrees CCW
    """
    return _solve(grid, sort_mode="size", rotate_mode="combo")


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, sort_mode="legacy", rotate_mode="legacy")


def _d1_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, sort_mode="size", rotate_mode="legacy")


def variant_metrics(grid: np.ndarray) -> dict:
    scene = _parse_scene(grid)
    legacy_rects = _sorted_rectangles(scene, "legacy")
    size_rects = _sorted_rectangles(scene, "size")
    legacy_sizes = [int(rect["thick_dim"]) for rect in legacy_rects]
    size_sizes = [int(rect["thick_dim"]) for rect in size_rects]
    legacy_k = 1 if scene["blue_orient"] == "v" else 0

    combo_out = main(grid)
    d1_out = _d1_main(grid)
    legacy_out = _legacy_main(grid)

    return {
        "blue_orient": str(scene["blue_orient"]),
        "rect_count": int(len(scene["rects"])),
        "legacy_size_sequence": legacy_sizes,
        "size_sorted_sequence": size_sizes,
        "legacy_matches_size_order": bool(legacy_sizes == size_sizes),
        "combo_matches_d1_output": bool(np.array_equal(combo_out, d1_out)),
        "combo_matches_legacy_output": bool(np.array_equal(combo_out, legacy_out)),
        "rotation_keys": [int(_rotation_k(int(rect["thick_dim"]))) for rect in size_rects],
        "rotation_diff_count": int(
            sum(int(_rotation_k(int(rect["thick_dim"]))) != int(legacy_k) for rect in size_rects)
        ),
    }


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


def _pick_accents(dominant: int) -> tuple[int, int]:
    pool = [int(c) for c in DOMINANT_COLORS if int(c) != int(dominant)]
    a0, a1 = np.random.choice(pool, size=2, replace=False)
    return int(a0), int(a1)


def _make_patch(width: int, dominant: int) -> np.ndarray:
    patch = np.full((2, int(width)), int(dominant), dtype=int)
    a0, a1 = _pick_accents(int(dominant))

    # Fixed asymmetric motif so all four rotations remain visually distinct.
    patch[0, 0] = int(a0)
    patch[0, 1] = int(a1)
    patch[1, 0] = int(a0)
    if int(width) >= 4:
        patch[1, int(width) - 1] = int(a1)
    if int(width) >= 6:
        patch[0, int(width) - 2] = int(a0)
    return patch


def _build_canonical() -> np.ndarray:
    widths = [2, 4, 6, 8]
    order = list(map(int, np.random.permutation(len(widths))))
    shuffled = [int(widths[i]) for i in order]

    colors = list(map(int, np.random.choice(DOMINANT_COLORS, size=4, replace=False)))
    specs = list(zip(shuffled, colors))
    top_specs = specs[:2]
    bot_specs = specs[2:]

    used_top = int(sum(int(w) for w, _ in top_specs) + 2)
    used_bot = int(sum(int(w) for w, _ in bot_specs) + 2)
    trailing = int(np.random.randint(1, 3))
    long_dim = int(max(used_top, used_bot) + 2 + trailing)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, band in ((2, top_specs), (5, bot_specs)):
        c = 2
        for width, color in band:
            paste(grid, _make_patch(int(width), int(color)), int(r0), int(c))
            c += int(width) + 2

    return grid


def _is_valid_variant_instance(grid: np.ndarray) -> bool:
    metrics = variant_metrics(grid)
    if metrics["rect_count"] != 4:
        return False
    if metrics["size_sorted_sequence"] != [2, 4, 6, 8]:
        return False
    if metrics["legacy_matches_size_order"]:
        return False
    if metrics["combo_matches_d1_output"]:
        return False
    if metrics["rotation_keys"] != [0, 1, 2, 3]:
        return False
    if metrics["rotation_diff_count"] != 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)
    if not _is_valid_variant_instance(grid):
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 291dc1e1/1")
