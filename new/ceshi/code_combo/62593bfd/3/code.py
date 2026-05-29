# Combo variant of 62593bfd (index 3)
# combo_spec: D1/4+D5/4
# Name: 凸性分组 + unique擦除
#
# Design analysis
# ===============
# 1) 具体化与行为追踪
#    原题先按 D4-canonical shape 统计“是否有同形伙伴”，shared -> 顶边，unique -> 底边。
#    本融合版改成两步联动：
#    - 分组维度改为凸性。这里采用离散、可见、可操作的定义：
#      一个对象若完全填满自己的 tight bounding box，则视为 convex；
#      若 bbox 内仍有背景缺口，则视为 non-convex。
#    - 变换维度改为 erase。convex 对象保留并平移到顶边；non-convex 对象不再去底边，而是直接消失。
#    小例子：输入含一个 2x4 实心矩形、一个 L 形、一个 T 形。
#    - 原题：三者若 canonical shape 都唯一，则全部去底边。
#    - 融合版：2x4 矩形因为 bbox 全满，保留到 row 0；L 与 T 因 bbox 有缺口，被完全擦除。
#    差异不仅是“底边变擦除”，还包括“谁被保留”不再由跨对象频次决定，而由对象自身几何性质决定。
#
# 2) 复杂度校准
#    - convex 若允许任意“看起来差不多凸”的离散形状，人眼在 2-4 个样例内难稳定归纳。
#      因此生成器只造显眼的实心矩形作为 convex。
#    - non-convex 只从 L / T / plus / staircase / C-like 等 bbox 内有明显缺口的形状中选。
#    - 每个样例固定包含 2-3 个 convex 和 2-3 个 non-convex，共 4-6 个对象，保证
#      “哪些对象被保留”和“哪些对象消失”都有重复支撑，而不是单个偶发现象。
#
# 3) 隐式假设审查
#    - find_components(..., monochromatic=False) 依赖不同颜色对象之间不能 8 邻接接触，
#      否则会被错误并成同一组件。生成器因此使用 no-touch 放置，而不是仅防 overlap。
#    - main() 里 crop_bbox + c0 提取依赖组件 bbox 是紧的；生成器不会引入额外噪声像素，
#      所以“bbox 是否全满”是稳定的。
#    - 新规则不再依赖 Counter(shape_key)，因此必须完全移除原来的形状频次分支；
#      否则会把本应保留的 unique-convex 对象误删或误送到底边。
#    - paste 到顶边时要求保留对象在输出中彼此不重叠也不接触；output-first 放置显式保证这点。
#
# 4) 放置合法性
#    - 输出只放 convex 对象，统一贴到 row 0，保留它们的 c0。
#    - non-convex 对象不出现在输出，所以不会和顶部对象发生冲突。
#    - 输入阶段把所有对象散到 interior rows，convex 保持输出时的 c0，non-convex 可自由选 c0；
#      所有放置都要求不越界、不重叠、且不存在 8 邻接接触。
#
# 5) 视觉可观察性与整洁度
#    - 解题者会看到：输出顶部只剩一排实心矩形，输入中的带缺口对象全部消失。
#    - 颜色只区分对象，不编码规则；样例间颜色洗牌也不影响人类归纳。
#    - 输入散布避免 row 0 和最底贴边，输出则只在顶部出现保留对象，版式差异清晰。
#
# 6) 防退化设计
#    - 生成器强制所有对象的 D4 canonical shape 全部互不相同。
#      这样原题逻辑和单独的 D5“unique 擦除”逻辑都会把所有对象删光；
#      只有本融合版会保留 convex 对象到顶边，说明“凸性分组”确实在起作用。
#    - 同时每个样例都至少有 2 个 convex，因此输出不会退化成空网格。
#
# 7) 新 main() 的额外输入假设
#    - 假设每个对象都是单色、8 连通、bbox 尺寸有限，生成器满足这一点。
#    - 假设对象之间不 8 接触，避免 monochromatic=False 的组件串联，生成器满足这一点。
#    - 假设 convex 的离散定义就是“tight bbox 全满”；生成器只造实心矩形作正例，只造有缺口
#      形状作反例，因此训练样例与 solver 的判据一致且可被人眼读出。

from __future__ import annotations

import random

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def _canonical_shape(coords):
    """Canonical D4-invariant key used only for generator anti-degeneracy checks."""
    pts = [(int(r), int(c)) for r, c in coords]

    def _norm(ps):
        mr = min(r for r, c in ps)
        mc = min(c for r, c in ps)
        return tuple(sorted((r - mr, c - mc) for r, c in ps))

    cur = list(pts)
    best = None
    for _ in range(4):
        n = _norm(cur)
        best = n if best is None or n < best else best
        n = _norm([(-r, c) for r, c in cur])
        best = n if n < best else best
        cur = [(c, -r) for r, c in cur]
    return best


def _mask_from_coords(coords, shape):
    mask = np.zeros(shape, dtype=bool)
    h, w = map(int, shape)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < h and 0 <= cc < w:
            mask[rr, cc] = True
    return mask


def _is_convex_patch(patch, bg):
    """Operational convexity: the tight bounding box contains no background holes."""
    return bool(np.all(np.array(patch, dtype=int) != int(bg)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep convex objects and erase non-convex ones.

    - Extract each 8-connected object as a tight patch.
    - If the tight bbox is fully filled, the object is treated as convex and moved to row 0.
    - Otherwise the object is treated as non-convex and erased.
    - Retained objects preserve shape, color, orientation, and left column c0.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert comps

    out = np.full((h, w), bg, dtype=int)
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]

        if not _is_convex_patch(pat, bg):
            continue
        paste(out, pat, 0, int(c0), transparent=bg)

    return out


_RECT_DIMS = [
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 3),
    (3, 4),
    (4, 4),
]

_NONCONVEX_BASES = [
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (2, 1)],
]

_CLASS_CONFIGS = [(2, 2), (2, 3), (3, 2), (3, 3)]


def _rectangle_coords(h, w):
    return [(r, c) for r in range(int(h)) for c in range(int(w))]


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, c in pts)
    mc = min(c for r, c in pts)
    return [(r - mr, c - mc) for r, c in pts]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, c in coords)
    mc = max(c for r, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _no_touch(grid, patch, r0, c0, bg):
    """Reject overlap and any 8-neighbor contact with existing foreground."""
    ph, pw = map(int, patch.shape)
    gh, gw = map(int, grid.shape)
    if r0 < 0 or c0 < 0 or r0 + ph > gh or c0 + pw > gw:
        return False

    mask = patch != int(bg)
    if not np.any(mask):
        return False
    if np.any(grid[r0 : r0 + ph, c0 : c0 + pw][mask] != int(bg)):
        return False

    for pr, pc in np.argwhere(mask):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        rr0 = max(0, rr - 1)
        rr1 = min(gh, rr + 2)
        cc0 = max(0, cc - 1)
        cc1 = min(gw, cc + 2)
        if np.any(grid[rr0:rr1, cc0:cc1] != int(bg)):
            return False
    return True


def _interior_rows(grid_h, obj_h):
    max_r0 = int(grid_h - obj_h)
    rows = list(range(1, max_r0))
    return rows if rows else list(range(0, max_r0 + 1))


def _pick_convex_objects(n_convex, colors, canon_used):
    dims = _RECT_DIMS[:]
    random.shuffle(dims)
    objs = []
    for h, w in dims:
        if len(objs) >= n_convex:
            break
        coords = _rectangle_coords(h, w)
        canon = _canonical_shape(coords)
        if canon in canon_used:
            continue
        canon_used.add(canon)
        objs.append(
            {
                "coords": coords,
                "color": int(colors[len(objs)]),
                "is_convex": True,
            }
        )
    if len(objs) != n_convex:
        raise ValueError("not enough convex shapes")
    return objs


def _pick_nonconvex_objects(n_nonconvex, color_offset, colors, canon_used):
    bases = _NONCONVEX_BASES[:]
    random.shuffle(bases)
    objs = []
    for base in bases:
        if len(objs) >= n_nonconvex:
            break
        canon = _canonical_shape(base)
        if canon in canon_used:
            continue
        canon_used.add(canon)
        coords = _transform_shape(base, random.randint(0, 3), random.choice([False, True]))
        objs.append(
            {
                "coords": coords,
                "color": int(colors[color_offset + len(objs)]),
                "is_convex": False,
            }
        )
    if len(objs) != n_nonconvex:
        raise ValueError("not enough non-convex shapes")
    return objs


def _place_output_convex(out, objs, bg):
    for obj in objs:
        patch = obj["patch"]
        _h, w = map(int, patch.shape)
        choices = list(range(out.shape[1] - w + 1))
        random.shuffle(choices)
        placed = False
        for c0 in choices:
            if _no_touch(out, patch, 0, c0, bg):
                paste(out, patch, 0, c0, transparent=bg)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("convex output placement failed")


def _scatter_with_fixed_c0(inp, patch, c0, bg):
    rows = _interior_rows(inp.shape[0], patch.shape[0])
    random.shuffle(rows)
    for r0 in rows:
        if _no_touch(inp, patch, r0, int(c0), bg):
            paste(inp, patch, r0, int(c0), transparent=bg)
            return
    raise ValueError("fixed-c0 scatter failed")


def _scatter_freely(inp, patch, bg):
    h, w = map(int, patch.shape)
    rows = _interior_rows(inp.shape[0], h)
    row_choices = rows[:]
    random.shuffle(row_choices)
    for r0 in row_choices:
        c_choices = list(range(inp.shape[1] - w + 1))
        random.shuffle(c_choices)
        for c0 in c_choices:
            if _no_touch(inp, patch, int(r0), int(c0), bg):
                paste(inp, patch, int(r0), int(c0), transparent=bg)
                return
    raise ValueError("free scatter failed")


def _gen_attempt():
    h = random.randint(19, 30)
    w = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)

    n_convex, n_nonconvex = random.choice(_CLASS_CONFIGS)
    canon_used = set()
    convex_objs = _pick_convex_objects(n_convex, colors, canon_used)
    nonconvex_objs = _pick_nonconvex_objects(n_nonconvex, n_convex, colors, canon_used)
    objs = convex_objs + nonconvex_objs

    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)

    out = np.full((h, w), bg, dtype=int)
    random.shuffle(convex_objs)
    _place_output_convex(out, convex_objs, bg)

    inp = np.full((h, w), bg, dtype=int)
    all_objs = objs[:]
    random.shuffle(all_objs)
    for obj in all_objs:
        if obj["is_convex"]:
            _scatter_with_fixed_c0(inp, obj["patch"], obj["c0"], bg)
        else:
            _scatter_freely(inp, obj["patch"], bg)

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo D1/4+D5/4")
