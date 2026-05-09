# Combo variant of 291dc1e1 (index 37)
# combo_spec: D1/3+D4/0+D9/1
# Name: 逆序+三band+中部十字

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis for this combo variant
# a) 具体化与行为追踪
#    - 原任务: 蓝线和红线从角落出发，矩形只在十字的一侧，蓝距只有两档，按 (db, dr) 升序。
#      例如 canonical 中若蓝线在最上方、红线在最左侧，则矩形只会出现在右下区域；db=1 的整排先输出，
#      再输出 db=4 的整排；若蓝线改成竖直，所有矩形统一逆时针旋转 90 度。
#    - 本融合变体: 蓝线和红线移到网格中部，二者以 marker=0 在中心相交形成十字。矩形围绕十字分布，
#      相对蓝线的距离固定为三档 db in {1,4,7}，因此会出现上/下三层 band；同时矩形落在四个象限中。
#    - 排序改为按 (db, dr) 降序。也就是说，离蓝线最远的 band 先堆叠；同一 band 内，离红线更远的先堆叠。
#      这样输出顺序与原题的“近->远”完全反过来。若蓝线竖直，仍然统一旋转 90 度 CCW。
#    - 具体小例子: 若某输入有 6 个矩形，db 分别为 [1,1,4,4,7,7]，dr 分别为 [2,9,3,8,1,10]，
#      原题式升序会按 (1,2)->(1,9)->(4,3)->(4,8)->(7,1)->(7,10)；
#      本题会按 (7,10)->(7,1)->(4,8)->(4,3)->(1,9)->(1,2)。
#      中心十字只改变距离参照系，排序核心仍是“先按蓝距分层，再按红距排队”，只是方向翻转了。
# b) 复杂度校准
#    - 三档 db 明显强于原题的两档，且矩形必须覆盖四个象限，视觉上能直接看出“不是角落版原题”。
#    - 仍然只使用实心矩形、两条整线和一个 marker，没有引入额外对象类型，避免规则膨胀。
#    - 生成器固定至少 6 个矩形，并尽量让每个矩形签名不同；这样排序变化会在输出中肉眼可见。
# c) 隐式假设审查
#    - 原 solver 假设 marker 在四角之一；这里已失效，因此必须改成从整行/整列识别两条中部直线。
#    - 原 solver 只屏蔽 corner row/col；这里仍然屏蔽一整行一整列，但索引来自中心十字而不是角落。
#    - 原 solver 的 along_from_corner 依赖角点坐标；这里不再成立。新 solver 仅把它替换为稳定的全局
#      tie-break（优先更靠下、更靠右），并且生成器尽量避免出现大量完全同距的矩形。
#    - 原 solver 默认矩形都在红线/蓝线的一侧，因此 dr/db 通常天然区分对象；这里矩形分居四象限，
#      所以距离函数必须对中心线两侧都返回同一种“边缘到直线的像素间隔”。
# d) 放置合法性
#    - canonical 网格把蓝线固定在中部一行，红线固定在中部一列；db=1/4/7 的 2 高矩形分别落在
#      精确计算出的三层行带，天然不会与线重叠，也不会越界。
#    - 每个 band/半侧只放一个矩形，因此无需解决同带内碰撞；宽度由所选 dr 反推合法区间，保证不越界。
#    - 可选翻转/转置仅做刚体变换，不会破坏不重叠性质。
# e) 视觉可观察性
#    - 输入里中心十字和三层 band 很明显，解题者能先看出“以中心线为参考而不是角落”。
#    - 输出顺序会从最远 band 开始，和原题相反；生成器会拒绝与升序解完全相同的样本，避免退化。
#    - 少量异色像素只用于让矩形签名不完全重复，不承担逻辑角色。
# f) 可视化整洁度
#    - 整体仍是规则的几何布局: 一条横线、一条竖线、若干 2xn 实心块。对象之间有明确背景间隔。
#    - 中心只有一个 marker 像素，十字与矩形不重叠，不会出现多层覆盖难以分辨的问题。
# g) 颜色变化可推断性
#    - 颜色职责和原题一致: 蓝线=1、红线=2、marker=0、背景=8，其余颜色仅标识矩形内容。
#    - 没有额外的颜色映射或条件变色规则，因此样例中的可推断负担主要集中在空间关系和排序。
# h) source_task
#    - D1/3 的单变体给了“只反转主排序键”的实现提醒；本融合版沿用这个思路，但解析器和生成器都重写。
# i) 新 solver 的额外假设审查
#    - 新 parser 假设存在恰好一条几乎整行的蓝线和一条几乎整列的红线，且中心交点是 marker。
#      生成器固定满足该条件。
#    - 新排序假设矩形 bbox 内没有背景孔洞；生成器仍然只放满矩形，并只在内部撒少量异色像素。
#    - 新 tie-break 假设即便出现完全相同的 (db, dr)，用“更靠下、更靠右优先”也不会和主规则冲突；
#      生成器通过优先采样不同 dr 来降低该情况频率。

BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
BAND_DISTANCES = (1, 4, 7)


def _find_line(g: np.ndarray, color: int) -> tuple[str, int]:
    H, W = g.shape
    row_candidates = []
    for r in range(H):
        row = g[r, :]
        if np.all((row == int(color)) | (row == MARKER)) and int(np.count_nonzero(row == int(color))) >= W - 1:
            row_candidates.append(int(r))
    col_candidates = []
    for c in range(W):
        col = g[:, c]
        if np.all((col == int(color)) | (col == MARKER)) and int(np.count_nonzero(col == int(color))) >= H - 1:
            col_candidates.append(int(c))

    candidates = [("h", r) for r in row_candidates] + [("v", c) for c in col_candidates]
    assert len(candidates) == 1
    return candidates[0]


def _parse_task(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    blue_orient, blue_idx = _find_line(g, BLUE)
    red_orient, red_idx = _find_line(g, RED)
    assert blue_orient != red_orient

    cross_r = int(blue_idx if blue_orient == "h" else red_idx)
    cross_c = int(blue_idx if blue_orient == "v" else red_idx)
    assert int(g[cross_r, cross_c]) == MARKER

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cross_r, :] = True
    line_mask[:, cross_c] = True

    mask = (g != int(bg)) & (~line_mask) & (g != MARKER)
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
        "blue_orient": str(blue_orient),
        "red_orient": str(red_orient),
        "blue_idx": int(blue_idx),
        "red_idx": int(red_idx),
        "cross": (int(cross_r), int(cross_c)),
        "rects": rects,
    }


def _dist_to_line(rect: dict, orient: str, line_idx: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        if c0 > int(line_idx):
            return int(c0 - int(line_idx) - 1)
        if c1 < int(line_idx):
            return int(int(line_idx) - c1 - 1)
        return 0
    if r0 > int(line_idx):
        return int(r0 - int(line_idx) - 1)
    if r1 < int(line_idx):
        return int(int(line_idx) - r1 - 1)
    return 0


def _quadrant(rect: dict, cross_r: int, cross_c: int) -> tuple[str, str]:
    r0, c0, r1, c1 = map(int, rect["box"])
    v_side = "top" if r1 < int(cross_r) else "bottom"
    h_side = "left" if c1 < int(cross_c) else "right"
    return v_side, h_side


def _annotate_rects(info: dict) -> list[dict]:
    blue_orient = str(info["blue_orient"])
    red_orient = str(info["red_orient"])
    blue_idx = int(info["blue_idx"])
    red_idx = int(info["red_idx"])
    cross_r, cross_c = map(int, info["cross"])

    rects = []
    for base in info["rects"]:
        rect = dict(base)
        rect["db"] = int(_dist_to_line(rect, blue_orient, blue_idx))
        rect["dr"] = int(_dist_to_line(rect, red_orient, red_idx))
        rect["quadrant"] = _quadrant(rect, cross_r, cross_c)
        rects.append(rect)
    return rects


def _solve(grid: np.ndarray, *, descending: bool) -> np.ndarray:
    info = _parse_task(grid)
    bg = int(info["bg"])
    blue_orient = str(info["blue_orient"])
    rects = _annotate_rects(info)

    if descending:
        rects.sort(
            key=lambda r: (
                -int(r["db"]),
                -int(r["dr"]),
                -int(r["box"][2]),
                -int(r["box"][3]),
                -int(r["box"][0]),
                -int(r["box"][1]),
            )
        )
    else:
        rects.sort(
            key=lambda r: (
                int(r["db"]),
                int(r["dr"]),
                int(r["box"][0]),
                int(r["box"][1]),
                int(r["box"][2]),
                int(r["box"][3]),
            )
        )

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse a center cross, measure each rectangle's distance to the blue/red axes,
    sort by (db, dr) descending, rotate when the blue axis is vertical, and stack.
    """
    return _solve(grid, descending=True)


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
    flat = patch.reshape(-1)
    cells = np.random.choice(area, size=n_minor, replace=False)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _rect_rows(blue_row: int, db: int, v_side: str) -> tuple[int, int]:
    if v_side == "top":
        r1 = int(blue_row - db - 1)
        r0 = int(r1 - 1)
        return int(r0), int(r1)
    r0 = int(blue_row + db + 1)
    r1 = int(r0 + 1)
    return int(r0), int(r1)


def _rect_cols(red_col: int, dr: int, width: int, h_side: str) -> tuple[int, int]:
    if h_side == "left":
        c1 = int(red_col - dr - 1)
        c0 = int(c1 - width + 1)
        return int(c0), int(c1)
    c0 = int(red_col + dr + 1)
    c1 = int(c0 + width - 1)
    return int(c0), int(c1)


def _slot_layout(top_sides: list[str], bottom_sides: list[str]) -> list[tuple[int, str, str]]:
    slots = []
    for db, h_side in zip(BAND_DISTANCES, top_sides):
        slots.append((int(db), "top", str(h_side)))
    for db, h_side in zip(BAND_DISTANCES, bottom_sides):
        slots.append((int(db), "bottom", str(h_side)))
    return slots


def _choose_side_patterns() -> tuple[list[str], list[str]]:
    patterns = [
        (["left", "left", "right"], ["left", "right", "right"]),
        (["left", "right", "right"], ["left", "left", "right"]),
        (["left", "right", "left"], ["right", "left", "right"]),
        (["right", "left", "right"], ["left", "right", "left"]),
    ]
    top_sides, bottom_sides = patterns[int(np.random.randint(len(patterns)))]
    if np.random.rand() < 0.5:
        top_sides = ["left" if s == "right" else "right" for s in top_sides]
        bottom_sides = ["left" if s == "right" else "right" for s in bottom_sides]
    return list(top_sides), list(bottom_sides)


def _build_canonical() -> np.ndarray:
    H = 21
    W = int(np.random.choice([25, 27, 29]))
    blue_row = H // 2
    red_col = W // 2
    grid = np.full((H, W), BG, dtype=int)
    grid[blue_row, :] = BLUE
    grid[:, red_col] = RED
    grid[blue_row, red_col] = MARKER

    top_sides, bottom_sides = _choose_side_patterns()
    slots = _slot_layout(top_sides, bottom_sides)

    max_dr = int(min(red_col - 3, W - red_col - 4))
    assert max_dr >= len(slots)
    dr_values = list(np.random.choice(np.arange(1, max_dr + 1), size=len(slots), replace=False))
    np.random.shuffle(dr_values)

    used_signatures: set[tuple[int, ...]] = set()
    color_pool = list(np.random.permutation(DOMINANT_COLORS))
    rect_specs = []

    for slot_idx, (db, v_side, h_side) in enumerate(slots):
        dr = int(dr_values[slot_idx])
        max_width = min(8, 2 * ((red_col - dr - 1) // 2)) if h_side == "left" else min(8, 2 * ((W - red_col - dr - 2) // 2))
        legal_widths = [w for w in (2, 4, 6, 8) if w <= int(max_width)]
        assert legal_widths
        width = int(np.random.choice(legal_widths))

        r0, r1 = _rect_rows(blue_row, int(db), str(v_side))
        c0, c1 = _rect_cols(red_col, int(dr), int(width), str(h_side))
        assert 0 < r0 <= r1 < H - 1
        assert 0 < c0 <= c1 < W - 1

        if color_pool:
            color = int(color_pool.pop())
        else:
            color = int(np.random.choice(DOMINANT_COLORS))

        patch = np.full((2, int(width)), int(color), dtype=int)
        patch = _sprinkle_minorities(patch, int(color))
        signature = tuple(int(v) for v in patch.reshape(-1))
        if signature in used_signatures:
            choices = [int(v) for v in MINORITY_COLORS if int(v) != int(color)]
            patch[0, -1] = int(np.random.choice(choices))
            signature = tuple(int(v) for v in patch.reshape(-1))
        used_signatures.add(signature)
        rect_specs.append((r0, c0, patch))

    for r0, c0, patch in rect_specs:
        paste(grid, patch, int(r0), int(c0))

    return grid


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    if np.random.rand() < 0.5:
        out = np.flipud(out)
    if np.random.rand() < 0.5:
        out = np.fliplr(out)
    if np.random.rand() < 0.5:
        out = np.transpose(out)
    return out


def _sample_metrics(grid: np.ndarray) -> dict:
    info = _parse_task(grid)
    rects = _annotate_rects(info)
    band_set = tuple(sorted({int(r["db"]) for r in rects}))
    quadrants = {tuple(r["quadrant"]) for r in rects}
    cross_r, cross_c = map(int, info["cross"])
    H, W = grid.shape
    return {
        "band_set": band_set,
        "quadrant_count": int(len(quadrants)),
        "cross_centered": bool(2 <= cross_r <= H - 3 and 2 <= cross_c <= W - 3),
        "descending_differs": bool(not np.array_equal(_solve(grid, descending=True), _solve(grid, descending=False))),
    }


def _gen_attempt() -> np.ndarray | None:
    grid = _random_transform(_build_canonical())
    try:
        metrics = _sample_metrics(grid)
    except Exception:
        return None

    if tuple(metrics["band_set"]) != BAND_DISTANCES:
        return None
    if int(metrics["quadrant_count"]) < 4:
        return None
    if not bool(metrics["cross_centered"]):
        return None
    if not bool(metrics["descending_differs"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant 37")
