# Variant of 62593bfd
# Dimension: D1 — 分组判据——形状等价类的分类规则 (match_cond)
# Change: 计算每个对象在D4二面体群（4旋转x2翻转=8变换）下的canonical形状，统计各canonical形状的出现频次；... → 按凸性分组
#
# Design analysis
# ===============
# variant_spec:
#   source_task: none
#   concrete_behavior:
#     - 保留原题的宏观版式：输入中有 4-6 个单色 8 连通对象，输出只做纵向平移，保留每个对象的左边界 c0。
#     - 新的分组判据不再比较对象之间是否“同形”，而是单独判断每个对象是否“凸”。
#     - 为了把“凸性”落实到离散像素网格并保证人眼可见，这里采用可操作定义：
#       对象若完全填满自己的 tight bounding box，则记为 convex；否则记为 non_convex。
#       于是凸组就是实心矩形，非凸组就是 L / T / 十字 / 阶梯等 bbox 内存在缺口的形状。
#     - 固定映射为：convex -> TOP，non_convex -> BOTTOM。
#   behavior_trace:
#     example:
#       - 输入里有一个 2x4 的实心矩形和一个 T 形，二者 canonical shape 都各自唯一。
#       - 原题逻辑：二者都属于 unique，都会下移到底边。
#       - 变体逻辑：矩形因为 bbox 被完全填满，移到顶边；T 形因为 bbox 内有缺口，移到底边。
#       - 差异发生在“为对象决定目标边缘”的判定环节，平移方式本身不变。
#   complexity_calibration:
#     - 若把“凸”定义得过于宽泛，少量样例里不容易肉眼稳定归纳；若定义得过于隐晦，则会退化成不可见规则。
#     - 因此生成器只造两类非常显眼的对象：
#       convex: 2x2, 2x3, 2x4, 2x5, 3x3, 3x4, 4x4 等实心矩形
#       non_convex: L / T / plus / staircase / C-like 等 bbox 有明显缺口的形状
#     - 每个样例至少包含 2 个 convex 和 2 个 non_convex，对分组有重复支撑。
#   implicit_assumptions_review:
#     - find_components(..., monochromatic=False):
#       这依赖“不同颜色对象之间不能发生 8 邻接接触”，否则它们会被误并成一个组件。
#       原题生成器只防 overlap，不防接触；本变体生成器显式加入 8 邻域间隔约束，保证该前提成立。
#     - patch / c0 提取逻辑：
#       依赖对象经过 tight crop 后，其 bbox 仍然是需要保留的水平锚点；凸性分组不改变这一点。
#     - paste 到 top / bottom：
#       依赖同组对象在对应边缘上不重叠；生成器继续采用 output-first 放置并校验。
#     - 旧代码里的 shape_key / Counter:
#       完全依赖“按 D4 等价类统计频次”的旧属性；在变体中必须移除，否则会把对象误分到底部。
#   legality:
#     - 输出阶段先把 convex 放在 row 0，把 non_convex 放在 row H-h；水平位置随机但保持可放置。
#     - 输入阶段只重新随机 r0，保持输出时选定的 c0，不越界、不重叠、也不发生 8 邻接接触。
#   observability:
#     - 看到样例时，人眼会直接观察到：所有实心矩形都被吸到上边，所有带缺口的形状都落到下边。
#     - 颜色只用于区分对象，不参与规则；不同样例里颜色会洗牌，因此不会引入不可见编码。
#   anti_degeneracy:
#     - 生成器强制所有对象的 D4 canonical shape 互不相同。
#       因而在同一输入上，原题 solver 会把所有对象都判成 unique 并全部送到底部；
#       新 solver 则会按 convex / non_convex 分成上下两组，确保变体行为确实发生。

from __future__ import annotations

import random

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def _canonical_shape(coords):
    """Canonical D4-invariant key, retained for generator anti-degeneracy checks."""
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
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _is_convex_patch(patch, bg):
    """Operational convexity for this variant: the tight bbox has no missing cell."""
    return bool(np.all(np.array(patch, dtype=int) != int(bg)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to top or bottom edge by convexity.

    Rule used in this variant:
    - Extract each 8-connected object as a tight patch.
    - If the object's tight bounding box is fully filled (no bg cell inside the patch),
      treat it as convex and move it to the TOP.
    - Otherwise treat it as non-convex and move it to the BOTTOM.
    - Horizontal position (left column c0 of bounding box) is preserved.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=False, return_format="dict"
    )
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]

        extracted.append(
            {
                "c0": int(c0),
                "pat": pat,
                "is_convex": _is_convex_patch(pat, bg),
            }
        )

    out = np.full((H, W), int(bg), dtype=int)
    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if d["is_convex"] else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    return out


# ---------------------- generator ----------------------

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


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, c in pts)
    mc = min(c for r, c in pts)
    return [(r - mr, c - mc) for r, c in pts]


def _rectangle_coords(h, w):
    return [(r, c) for r in range(int(h)) for c in range(int(w))]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, c in coords)
    mc = max(c for r, c in coords)
    p = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        p[int(r), int(c)] = int(color)
    return p


def _can_place(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    mask = patch != bg
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False

    for pr, pc in cells:
        gr = int(r0 + pr)
        gc = int(c0 + pc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = gr + dr
                cc = gc + dc
                if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != int(bg):
                    return False
    return True


def _pick_convex_objects(n_convex, colors, canon_used):
    rects = _RECT_DIMS[:]
    random.shuffle(rects)
    objs = []
    for h, w in rects:
        if len(objs) >= n_convex:
            break
        coords = _rectangle_coords(h, w)
        if h != w and random.choice([False, True]):
            coords = _rectangle_coords(w, h)
        canon = _canonical_shape(coords)
        if canon in canon_used:
            continue
        canon_used.add(canon)
        objs.append({"coords": coords, "color": colors[len(objs)], "is_convex": True})
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
                "color": colors[color_offset + len(objs)],
                "is_convex": False,
            }
        )
    if len(objs) != n_nonconvex:
        raise ValueError("not enough non-convex shapes")
    return objs


def _scatter_row(inp, patch, c0, bg):
    h, _w = patch.shape
    H = inp.shape[0]
    low = 1
    high = H - h - 1
    if high < low:
        raise ValueError("no interior rows")
    tries = list(range(low, high + 1))
    random.shuffle(tries)
    for r0 in tries:
        if _can_place(inp, patch, r0, c0, bg):
            return r0
    return None


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)

    n_convex, n_nonconvex = random.choice(_CLASS_CONFIGS)
    total = n_convex + n_nonconvex

    canon_used = set()
    convex_objs = _pick_convex_objects(n_convex, avail, canon_used)
    nonconvex_objs = _pick_nonconvex_objects(n_nonconvex, n_convex, avail, canon_used)
    objs = convex_objs + nonconvex_objs
    assert len(objs) == total
    random.shuffle(objs)

    out = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = patch.shape
        dest_r0 = 0 if obj["is_convex"] else H - h

        choices = list(range(W - w + 1))
        random.shuffle(choices)
        placed = False
        for c0 in choices:
            if _can_place(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                obj["patch"] = patch
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    random.shuffle(objs)
    for obj in objs:
        patch = obj["patch"]
        c0 = int(obj["c0"])
        r0 = _scatter_row(inp, patch, c0, bg)
        if r0 is None:
            raise ValueError("input scatter failed")
        paste(inp, patch, r0, c0, transparent=bg)

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd convexity variant")
