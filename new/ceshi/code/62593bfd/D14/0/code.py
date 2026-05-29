# Variant of 62593bfd
# Dimension: D14 — 干扰项与额外结构 (src)
# Change: 网格中仅包含背景色和待处理的连通分量对象，无干扰项或额外标记
#   → 增加不参与分组的固定边框线
#
# Design analysis
# ===============
# a) 具体化与行为追踪
# - 将“固定边框线”具体化为一条单色、高 1、贯穿整行的横线，位置只允许在最顶行或最底行。
#   这样它确实贴着某条边，而且会真实占据原题的目标边界，迫使 solver 先识别并绕开它。
# - 输入与输出都保留这条线，线本身不参与形状分组，也不会被平移。
# - 小例子：
#   原题里，若某个 3 像素竖条是唯一形状，solver 会把它移到底边。
#   现在如果顶边是一条整行装饰线，旧逻辑会把它当成普通对象参与 top/bottom 放置；
#   变体逻辑则先识别“整行且贴顶/贴底”的固定线，保留它原位，再把共享形状对象放到离顶部最近的空闲行，
#   把唯一形状对象放到离底部最近的空闲行。
# - 变化影响的环节：
#   1. 输入构造：新增一条边线，并保证其与普通对象分离。
#   2. solver 处理：先过滤固定线，再统计其余对象的 canonical shape。
#   3. 输出生成：边线原位复制，其他对象保持原题的纵向平移规则。
#
# b) 复杂度校准
# - 干扰结构只有一种：整行边线。它足够醒目，不会与普通 2-5 尺寸对象混淆。
# - 边线位置只在顶边/底边二选一，既有分支变化，又不会引入过多自由度。
# - 线的颜色使用一个未分配给对象的非背景色，避免和对象在同色下发生连通歧义。
#
# c) 对原 main() 的隐式假设审查
# - `find_components(... monochromatic=False)` 旧前提：所有非背景前景块都应参与分组。
#   变体下失效，因为边线也是前景，且可能与邻近异色对象在 8 连通意义下混成一块。
#   新版改为 `monochromatic=True`，按单色组件处理，这也更贴合原 spec 中“单色对象”的定义。
# - `for comp in comps: ... shape_key = _canonical_shape(rel)` 旧前提：每个组件都是可移动对象。
#   变体下失效，因为整行边线不应获得 shape_key，更不应被搬到顶部/底部。
# - `dest_r0 = 0 if freq[...] >= 2 else H - h` 旧前提：顶部/底部边界始终空闲。
#   变体下失效；若边线占据顶行，则共享对象必须落到 `row 1`；若边线占据底行，则唯一对象必须落到 `row H-h-1`。
#
# d) 放置合法性
# - 生成器先决定边线在顶/底哪一侧，再把该行保留出来，不允许普通对象占用。
# - 输出仍然采用 output-first：先把共享形状对象放到顶部最近空闲行、唯一形状对象放到底部最近空闲行，
#   同时保留边线，再按相同 c0 把对象散射回输入。
# - 由于边线只占 1 行，且对象 bbox 最多 5 行，19-30 的高度足以留出安全空间。
#
# e) 视觉可观察性
# - 人眼能直接看到一整行“装饰线”在输入输出中都不动，而其他小对象会上下整理。
# - 顶/底两种边线位置都会出现，因此解题者能从少量样例中归纳“这条线不参与搬运，但会占据一个目标边界”。
#
# f) 可视化整洁度
# - 边线独占最外侧一行，普通对象留在内部行区，避免元素互相覆盖。
# - 输出仍保留原题的上/下两层布局，整体比把干扰项散布在内部更整洁。
#
# g) 颜色变化可推断性
# - 本变体不改变对象颜色；边线颜色只是额外装饰色，不编码额外映射关系。
# - 颜色数量增加 1，但该颜色始终对应“固定边线”，结构线索非常明确。
#
# h) 变体后的新增假设
# - 若存在固定线，它必须是一个单色组件，bbox 为 `(side, 0, side, W-1)`，面积恰好为 `W`。
# - 普通对象不会也是“整行贴边线”；generator 通过面积 3-12、bbox <= 5 直接保证这一点。
# - 普通对象不占用边线那一行，因此固定线在输入输出中都保持独立组件。

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def _canonical_shape(coords):
    """Canonical shape key invariant under the D4 dihedral group."""
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


def _is_fixed_border_line(comp, H, W):
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return (
        int(comp["area"]) == int(W)
        and c0 == 0
        and c1 == W - 1
        and r0 == r1
        and r0 in (0, H - 1)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Same grouping rule as the original task, except that a full-width top/bottom
    border line is treated as fixed decoration and copied unchanged.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=True, return_format="dict"
    )
    assert comps

    fixed_lines = []
    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))

        if _is_fixed_border_line(comp, H, W):
            fixed_lines.append({"r0": r0, "c0": c0, "pat": patch})
            continue

        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]
        extracted.append(
            {
                "c0": int(c0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    top_blocked = any(line["r0"] == 0 for line in fixed_lines)
    bottom_blocked = any(line["r0"] == H - 1 for line in fixed_lines)

    for line in fixed_lines:
        paste(out, line["pat"], line["r0"], line["c0"], transparent=int(bg))

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        if int(freq[d["shape_key"]]) >= 2:
            dest_r0 = 1 if top_blocked else 0
        else:
            dest_r0 = int(H - h - 1) if bottom_blocked else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    return out


# ────────────────────── generator ──────────────────────


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly *area* pixels."""
    deltas = (
        [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        if connectivity == 8
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells:
                    continue
                all_r = [rr for rr, _ in cells] + [nr]
                all_c = [cc for _, cc in cells] + [nc]
                if max(all_r) - min(all_r) < max_dim and max(all_c) - min(all_c) < max_dim:
                    frontier.append((nr, nc))
        if not frontier:
            raise ValueError("cannot grow shape")
        cells.add(random.choice(frontier))
    mr = min(r for r, _c in cells)
    mc = min(c for _r, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, _c in pts)
    mc = min(c for _r, c in pts)
    return [(r - mr, c - mc) for r, c in pts]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    p = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        p[r, c] = color
    return p


def _no_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = patch != bg
    return not np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != bg)


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
]


def _place_border_line(grid, color, side):
    row = 0 if side == "top" else grid.shape[0] - 1
    grid[row, :] = int(color)
    return row


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)

    objs = []
    canon_used: set = set()

    for gsz in group_sizes:
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        canon = _canonical_shape(base)
        if canon in canon_used:
            raise ValueError("duplicate canonical")
        canon_used.add(canon)
        for _ in range(gsz):
            transformed = _transform_shape(
                base, random.randint(0, 3), random.choice([False, True])
            )
            objs.append({"coords": transformed, "color": avail[ci], "shared": True})
            ci += 1

    for _ in range(n_unique):
        for _ in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon not in canon_used:
                canon_used.add(canon)
                objs.append({"coords": shape, "color": avail[ci], "shared": False})
                ci += 1
                break
        else:
            raise ValueError("unique shape not found")

    line_color = avail[ci]
    line_side = random.choice(["top", "bottom"])

    out = np.full((H, W), bg, dtype=int)
    line_row = _place_border_line(out, line_color, line_side)
    top_blocked = line_row == 0
    bottom_blocked = line_row == H - 1

    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = patch.shape
        if obj["shared"]:
            dest_r0 = 1 if top_blocked else 0
        else:
            dest_r0 = H - h - 1 if bottom_blocked else H - h
        placed = False
        for _ in range(300):
            c0 = random.randint(0, W - w)
            if _no_overlap(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = c0
                obj["patch"] = patch
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    _place_border_line(inp, line_color, line_side)
    for obj in objs:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = obj["c0"]
        placed = False
        for _ in range(300):
            r0 = random.randint(0, H - h)
            if _no_overlap(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D14")
