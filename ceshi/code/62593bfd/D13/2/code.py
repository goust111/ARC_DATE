# Variant of 62593bfd
# Dimension: D13 — 对象面积与形状复杂度约束 (meta)
# Change: 对象面积3-12像素，外接矩形限制在5x5以内，使用8连通随机生长生成形状... → 固定面积——所有对象面积相同

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


# Design analysis for the D13 variant
# -----------------------------------
# spec_variant:
#   preserved_rule:
#     - 仍然按 8 连通单色对象的 D4 canonical 形状分组。
#     - canonical 形状出现 >= 2 次的对象上移到顶边；只出现 1 次的对象下移到底边。
#     - 每个对象只做竖直平移，保留其 bbox 左列 c0。
#   D13_change:
#     - 原题中 area 在 [3, 12] 随机变化；这里改为“整题固定面积”。
#     - 为了让面积线索彻底失效，本实现把每个样例中的所有对象固定为 5 像素。
#     - 5 像素在 5x5 内仍有充足的 8 连通非同构形状，足以同时生成重复组和唯一组。
#   concrete_behavior_trace:
#     - 例：原题可同时出现一个 3 像素 L 形和一个 8 像素十字形，解题者可能先靠面积粗分。
#     - 变体里若 5 个对象都恰好是 5 像素，则“面积=5”对所有对象都相同，无法提供任何分组信息。
#     - solver 仍然只能比较 canonical 形状：两个 D4 等价的 5 像素对象归为 shared，其余 5 像素形状归为 unique。
#   complexity_calibration:
#     - 固定 5 像素而不是 3 像素，避免形状库太小、过于对称或视觉差异不明显。
#     - 继续保留 4-6 个对象、至少一个 shared group 和至少一个 unique shape，防止任务退化成全同或全异。
#     - 仍然使用 distinct non-bg colors，颜色只用于分离对象，不参与任何逻辑。
#   hidden_assumption_review:
#     - 原代码 `find_components(..., monochromatic=False)` 依赖“不同颜色对象永不接触”，否则会把接触对象合并成一个 multicolor component。
#     - 这不是题意要求，只是原 generator 侥幸回避；变体里我将 solver 改为 `monochromatic=True`，与“单色对象”规格一致。
#     - 原代码对 area 没有显式依赖；D13 只影响 generator，不需要在 solver 中引入面积判断。
#   placement_legality:
#     - output 仍然先构造，再按相同 c0 散射成 input。
#     - 由于 solver 使用 8 连通组件，generator 不能只避免像素重叠；还必须避免对象间 8 邻域接触。
#     - 本实现为 input/output 放置都加入一圈安全间距，从而保证对象始终被独立识别。
#   visual_observability:
#     - 人类在 2-4 个样例中可以直接看到：所有对象大小相同，但只有某些轮廓互为旋转/翻转同形。
#     - 输出又只按“形状是否有同类”分成上下两带，因此 D13 的核心变化是可见的。
#   neatness:
#     - 对象之间保留空白，不发生粘连；顶部组和底部组分层清晰。
#   new_solver_assumptions:
#     - solver 假设每个前景对象是单色且 8 连通。
#     - solver 不要求面积相同；这是 generator 强化出的元约束。
#     - generator 通过固定 5 像素面积、distinct colors、8 邻域留白来满足这些假设。


def _canonical_shape(coords):
    """Canonical key invariant under the D4 dihedral group."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group 8-connected monochromatic objects by D4-canonical shape.

    Shared shapes (freq >= 2) move to row 0; unique shapes move to row H-h.
    Horizontal bbox position is preserved.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
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
                "shape_key": _canonical_shape(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), bg, dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=bg)

    return out


# ---------------------- generator ----------------------

_FIXED_AREA = 5
_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
]


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


def _all_unique_transforms(coords):
    seen = set()
    out = []
    for rot in range(4):
        for flip in (False, True):
            transformed = tuple(sorted(_transform_shape(coords, rot, flip)))
            if transformed not in seen:
                seen.add(transformed)
                out.append([tuple(p) for p in transformed])
    return out


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    p = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        p[int(r), int(c)] = int(color)
    return p


def _shape_dims(coords):
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _is_valid_shape(coords):
    h, w = _shape_dims(coords)
    return 2 <= h <= 5 and 2 <= w <= 5 and len(coords) == _FIXED_AREA


def _clear_of_neighbors(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    local_cells = {(int(r), int(c)) for r, c in np.argwhere(patch != bg)}
    global_cells = {(r0 + r, c0 + c) for r, c in local_cells}

    for gr, gc in global_cells:
        if int(grid[gr, gc]) != bg:
            return False

    for gr, gc in global_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = gr + dr, gc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in global_cells:
                    continue
                if int(grid[nr, nc]) != bg:
                    return False
    return True


def _random_shape(canon_used, *, require_multi_view=False):
    for _ in range(400):
        shape = _grow_shape(_FIXED_AREA, connectivity=8, max_dim=5)
        canon = _canonical_shape(shape)
        if canon in canon_used:
            continue
        if not _is_valid_shape(shape):
            continue
        transforms = _all_unique_transforms(shape)
        if require_multi_view and len(transforms) < 2:
            continue
        return shape, canon, transforms
    raise ValueError("shape not found")


def _place_objects(objs, H, W, bg):
    grid = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["patch"]
        h, w = patch.shape
        dest_r0 = int(obj["dest_r0"])

        candidates = list(range(0, W - w + 1))
        random.shuffle(candidates)
        placed = False
        for c0 in candidates:
            if _clear_of_neighbors(grid, patch, dest_r0, c0, bg):
                paste(grid, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("placement failed")
    return grid


def _scatter_input(objs, H, W, bg):
    grid = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        top_r = int(obj["dest_r0"])
        bottom_r = int(H - h)

        candidates = [r for r in range(1, H - h) if r not in {top_r, bottom_r}]
        random.shuffle(candidates)
        placed = False
        for r0 in candidates:
            if _clear_of_neighbors(grid, patch, r0, c0, bg):
                paste(grid, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")
    return grid


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)
    objs = []
    canon_used = set()

    for gsz in group_sizes:
        base, canon, transforms = _random_shape(canon_used, require_multi_view=(gsz >= 2))
        canon_used.add(canon)
        random.shuffle(transforms)
        for idx in range(gsz):
            coords = transforms[idx % len(transforms)]
            objs.append(
                {
                    "coords": coords,
                    "color": int(avail[ci]),
                    "shared": True,
                }
            )
            ci += 1

    for _ in range(n_unique):
        shape, canon, _transforms = _random_shape(canon_used, require_multi_view=False)
        canon_used.add(canon)
        objs.append(
            {
                "coords": shape,
                "color": int(avail[ci]),
                "shared": False,
            }
        )
        ci += 1

    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, _w = obj["patch"].shape
        obj["dest_r0"] = 0 if obj["shared"] else int(H - h)

    out = _place_objects(objs, H, W, bg)
    inp = _scatter_input(objs, H, W, bg)

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D13 variant")
