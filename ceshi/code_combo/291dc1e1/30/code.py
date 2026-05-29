# Combo variant of 291dc1e1 (index 30)
# combo_spec: D1/4+D5/3+D6/0
# Name: 尺寸驱动排序旋转+左对齐

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for the combo variant

a) 具体化与行为追踪
- 输入外观仍沿用原题框架：一个角上有 marker，两条边线从该角出发，内部有两条与蓝线平行的矩形带。
- 原题流程是：
  1. 识别角点、蓝线/红线方向；
  2. 按 (到蓝线距离, 到红线距离) 排序；
  3. 仅当蓝线竖直时，对全部矩形统一旋转 90 度 CCW；
  4. 输出时居中堆叠。
- 组合后流程是：
  1. 仍然先识别角点和边线；
  2. 不再看距离，改为按矩形沿蓝线方向的长边长度 thick_dim 排序；
  3. 每个矩形单独按 thick_dim 编码旋转：2->0 度，4->90 度，6->180 度，8->270 度；
  4. 仍是自上而下紧密堆叠，但统一左对齐。
- 小例子：输入中五个矩形的 thick_dim 依次散落为 [6, 2, 8, 4, 6]。
  原题会先输出靠近蓝线的上排，再输出下排；若蓝线水平则它们都不旋转，且居中堆叠。
  组合后会按 [2, 4, 6, 6, 8] 的尺寸顺序取出；其中 4 会转成竖条，6 会翻转 180 度，8 会转成另一方向的竖条，最后全部贴左边堆起来。
  差异不再是“谁离边线近”，而是“谁更长，以及长度决定了如何旋转”。

b) 复杂度校准
- D5 的旋转编码只有四档，所以生成器必须稳定地让四个尺寸 {2,4,6,8} 都出现；否则解题者看不到完整映射。
- 仅有四个尺寸档位，若所有矩形都只出现一次，尺寸排序容易看出来，但“编码”证据略弱；因此每个样例固定使用 5 个矩形：四种尺寸各一个，再额外重复 2/4/6 中的一种。
- 仍保留两条 band。这样输入里还保留了原题的空间坐标系线索，解题者必须主动放弃“按位置排序”的旧解释。
- 生成器会拒绝那些“尺寸排序碰巧等于原距离排序”的样本，也会拒绝“左对齐与居中看起来一样”的样本。

c) 隐式假设审查
- 原 main() 的 corner 检测、blue/red 推断、line_mask、bbox 提取都仍可复用，因为对象类型没变，仍是实心轴对齐矩形。
- 被破坏的原假设有三处：
  1. `db/dr` 是主排序线索：现在不成立，必须改为 `_thick_dim(...)`。
  2. `if blue_orient == "v": rotate 90 CCW`：现在不成立，旋转不再依赖全局朝向，而依赖单个矩形的尺寸。
  3. `out_c = (out_w - w) // 2`：现在不成立，必须固定为 `0`。
- 叠加后还出现了一个新前提：solver 必须能从 bbox 稳定读出 thick_dim，并且 thick_dim 必须落在 {2,4,6,8} 内，才能映射到 0/90/180/270。生成器继续保证矩形始终是 2 x even_length 的实心块，因此这一前提成立。

d) 放置合法性
- canonical 视角中蓝线固定在顶边，红线固定在左边，矩形只占第 2-3 行和第 5-6 行，和边线之间始终隔 1 格背景。
- 同一 band 内矩形之间固定留 2 格背景，不重叠；整张图宽度按两条 band 的较大占宽动态分配，尾部再留 1-2 格背景，因此不越界。
- 再对 canonical 网格做翻转/90 度旋转变换到任意 corner 和 blue_orient，非重叠与边界合法性保持不变。

e) 视觉可观察性
- 从少量样例里，人类应能同时看出三件事：
  1. 输出顺序不是按上下 band 位置，而是短条到长条；
  2. 不同尺寸在输出里出现不同旋转结果，例如 4 和 8 会变成竖条，6 会出现 180 度翻转后的内部花纹；
  3. 堆叠时左边齐平，而不是居中。
- 为了让 180 度旋转也可见，生成器给每个矩形放少量非对称彩色点，且不使用背景色，所以既能保留连通矩形，又能让旋转方向肉眼可见。

f) 可视化整洁度
- 结构层次仍只有：corner marker、两条整边线、背景、若干分离矩形。没有新增符号或覆盖层。
- 输入继续使用两条平行矩形带，输出是一个紧凑的单列堆栈。左对齐产生阶梯右边缘，但左边界整齐，整体仍易读。

g) 颜色变化可推断性
- 颜色不承担规则映射，仍只是对象外观。旋转编码完全由尺寸决定，而不是颜色决定。
- 少量 minority pixels 只用于暴露旋转，不引入新的颜色逻辑；并且每个矩形只改 1-3 个像素，主体颜色仍明显。

9) 修改后 solver 的新隐式假设，以及 generator 如何保证
- 新假设 1：每个目标对象都是无背景孔洞的单个矩形，bbox 就是对象真实范围。
  生成器保证所有 patch 只含非背景色，且轴对齐放置。
- 新假设 2：每个对象的 thick_dim 属于 {2,4,6,8}，否则尺寸编码旋转无定义。
  生成器只采样这四种尺寸。
- 新假设 3：尺寸排序应当真正改变顺序，而不是退回原题的距离排序。
  生成器显式比较 size-order 与 legacy-order，不同才接受。
- 新假设 4：左对齐应当带来可见差异。
  生成器要求 `alignment_signal > 0` 且左对齐输出不等于居中输出。
- 新假设 5：尺寸编码旋转应当真的起作用，而不是和原题统一旋转等价。
  生成器要求至少一个矩形在新旧旋转规则下产生不同结果，并进一步拒绝整体输出与原题输出完全相同的样本。
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
ACCENT_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
ROTATION_K_BY_SIZE = {2: 0, 4: 1, 6: 2, 8: 3}


def _locate_frame(g: np.ndarray) -> tuple[int, int, int, int, str, str]:
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
    return bg, marker, cr, cc, blue_orient, red_orient


def _extract_rectangles(g: np.ndarray, bg: int, marker: int, cr: int, cc: int) -> list[dict]:
    H, W = g.shape
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


def _rotation_k(thick_dim: int, blue_orient: str, rotate_mode: str) -> int:
    if rotate_mode == "legacy":
        return 1 if blue_orient == "v" else 0
    assert rotate_mode == "combo"
    assert int(thick_dim) in ROTATION_K_BY_SIZE
    return int(ROTATION_K_BY_SIZE[int(thick_dim)])


def _prepare_rects(g: np.ndarray) -> tuple[int, str, str, int, int, list[dict]]:
    bg, marker, cr, cc, blue_orient, red_orient = _locate_frame(g)
    rects = _extract_rectangles(g, bg, marker, cr, cc)
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for rect in rects:
        rect["size"] = int(_thick_dim(rect["box"], blue_orient))
        rect["db"] = int(_dist_to_edge(rect["box"], blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect["box"], red_orient, red_edge_idx))
        rect["ab"] = int(_along_from_corner(rect["box"], blue_orient, cr, cc))
        rect["ar"] = int(_along_from_corner(rect["box"], red_orient, cr, cc))
    return bg, blue_orient, red_orient, cr, cc, rects


def _ordered_rects(g: np.ndarray, *, sort_mode: str) -> tuple[int, str, list[dict]]:
    bg, blue_orient, _red_orient, _cr, _cc, rects = _prepare_rects(g)
    if sort_mode == "legacy":
        rects.sort(key=lambda r: (int(r["db"]), int(r["dr"]), int(r["ab"]), int(r["ar"])))
    else:
        assert sort_mode == "combo"
        rects.sort(key=lambda r: (int(r["size"]), int(r["ab"]), int(r["ar"])))
    return bg, blue_orient, rects


def _transform_rect(arr: np.ndarray, thick_dim: int, blue_orient: str, rotate_mode: str) -> np.ndarray:
    return np.rot90(arr, k=_rotation_k(int(thick_dim), blue_orient, rotate_mode))


def _solve(grid: np.ndarray, *, sort_mode: str, rotate_mode: str, left_align: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, blue_orient, rects = _ordered_rects(g, sort_mode=sort_mode)

    seq = [_transform_rect(rect["arr"], int(rect["size"]), blue_orient, rotate_mode) for rect in rects]
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = 0 if left_align else int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - sort rectangles by thick_dim (their long side parallel to the blue edge);
    - rotate each rectangle according to that thick_dim:
      2 -> 0 deg, 4 -> 90 deg, 6 -> 180 deg, 8 -> 270 deg;
    - stack the transformed rectangles top-to-bottom with shared left alignment.
    """
    return _solve(grid, sort_mode="combo", rotate_mode="combo", left_align=True)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, sort_mode="legacy", rotate_mode="legacy", left_align=False)


def _combo_centered(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, sort_mode="combo", rotate_mode="combo", left_align=False)


def _transform_grid(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
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


def _decorate_patch(thick_dim: int, dominant: int) -> np.ndarray:
    patch = np.full((2, int(thick_dim)), int(dominant), dtype=int)
    accents = [int(c) for c in ACCENT_COLORS if int(c) != int(dominant)]
    a = int(np.random.choice(accents))
    b_choices = [c for c in accents if int(c) != int(a)]
    b = int(np.random.choice(b_choices if b_choices else accents))

    patch[0, 0] = int(a)
    if int(thick_dim) >= 4:
        patch[1, 1] = int(b)
    if int(thick_dim) >= 6:
        patch[0, 2] = int(a)
    return patch


def _sample_sizes() -> list[int]:
    base = [2, 4, 6, 8]
    extra = int(np.random.choice([2, 4, 6]))
    sizes = base + [extra]
    return list(map(int, np.random.permutation(sizes)))


def _build_canonical() -> np.ndarray:
    sizes = _sample_sizes()
    colors = [int(np.random.choice(DOMINANT_COLORS)) for _ in sizes]

    top_count = int(np.random.choice([2, 3]))
    top_specs = list(zip(sizes[:top_count], colors[:top_count]))
    bottom_specs = list(zip(sizes[top_count:], colors[top_count:]))

    used_top = int(sum(size for size, _ in top_specs) + 2 * (len(top_specs) - 1))
    used_bottom = int(sum(size for size, _ in bottom_specs) + 2 * (len(bottom_specs) - 1))
    trailing = int(np.random.randint(1, 3))
    long_dim = int(max(used_top, used_bottom) + 2 + trailing)
    assert 9 <= int(long_dim) <= 28

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, band_specs in ((2, top_specs), (5, bottom_specs)):
        c = 2
        for size, dominant in band_specs:
            paste(grid, _decorate_patch(int(size), int(dominant)), int(r0), int(c))
            c += int(size) + 2

    return grid


def variant_metrics(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    _bg, blue_orient, rects = _ordered_rects(g, sort_mode="combo")
    legacy_boxes = [tuple(map(int, rect["box"])) for rect in _ordered_rects(g, sort_mode="legacy")[2]]
    combo_boxes = [tuple(map(int, rect["box"])) for rect in rects]

    rotation_changes = 0
    widths = []
    size_counts: dict[int, int] = {}
    for rect in rects:
        thick_dim = int(rect["size"])
        size_counts[thick_dim] = int(size_counts.get(thick_dim, 0) + 1)
        new_arr = _transform_rect(rect["arr"], thick_dim, blue_orient, "combo")
        old_arr = _transform_rect(rect["arr"], thick_dim, blue_orient, "legacy")
        rotation_changes += int(new_arr.shape != old_arr.shape or not np.array_equal(new_arr, old_arr))
        widths.append(int(new_arr.shape[1]))

    max_width = int(max(widths))
    alignment_signal = int(sum((max_width - width) // 2 for width in widths))
    combo = main(g)
    centered = _combo_centered(g)
    legacy = _legacy_main(g)

    return {
        "blue_orient": blue_orient,
        "rect_count": len(rects),
        "size_counts": dict(sorted(size_counts.items())),
        "size_order_matches_legacy": bool(combo_boxes == legacy_boxes),
        "rotation_changes": int(rotation_changes),
        "alignment_signal": int(alignment_signal),
        "left_equals_centered": bool(np.array_equal(combo, centered)),
        "combo_equals_legacy": bool(np.array_equal(combo, legacy)),
        "final_widths": widths,
    }


def _is_informative(grid: np.ndarray) -> bool:
    metrics = variant_metrics(grid)
    if metrics["size_order_matches_legacy"]:
        return False
    if metrics["rotation_changes"] <= 0:
        return False
    if metrics["alignment_signal"] <= 0:
        return False
    if metrics["left_equals_centered"]:
        return False
    if metrics["combo_equals_legacy"]:
        return False
    if sorted(metrics["size_counts"]) != [2, 4, 6, 8]:
        return False
    return True


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform_grid(canonical, blue_orient, corner)


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        try:
            if not _is_informative(grid):
                continue
            _ = main(grid)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant 30")
