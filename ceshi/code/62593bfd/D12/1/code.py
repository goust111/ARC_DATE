# Variant of 62593bfd
# Dimension: D12 — 颜色分配策略 (meta)
# Change: 每个对象使用不同的非背景色；颜色通常各不相同但允许不同对象使用相同颜色... → 颜色作为反向线索——同色对象形状必然不同

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


# Design analysis for the D12 variant
# ----------------------------------
# spec_variant:
#   unchanged_rule:
#     - 仍然只按 8 连通对象的 canonical shape（D4 等价）分组。
#     - 形状频次 >= 2 的对象移到顶边；频次 == 1 的对象移到底边。
#     - 每个对象只做竖直平移，c0 保持不变。
#   d12_change:
#     - 原题里颜色通常是“每个对象一个颜色”，因此颜色虽然无用，但也不形成干扰。
#     - 变体中强制制造一个误导信号：至少有一对对象使用同一种非背景色，
#       但这两个对象必须属于不同 canonical shape 组。
#     - 为了让误导在样例里肉眼可见，这一对同色对象进一步强制跨越两类命运：
#       一个来自 duplicated-shape 组（会移到顶部），一个来自 unique-shape 组（会移到底部）。
#   concrete_behavior_trace:
#     - 例：输入里有一个红色 L 形，它和另一个蓝色 L 形同组，所以输出去顶部；
#       同时还有一个红色 T 形，它没有同形伙伴，所以输出去底部。
#     - 原题生成器通常不会出现“同色但不同形”的对，因此人可能下意识把颜色当成对象身份。
#     - 变体要求同一个红色在样例中同时出现在“去顶部”和“去底部”两侧，
#       直接打破“同色=同组/同命运”的直觉，迫使解题者回到形状本身。
#   complexity_calibration:
#     - 只引入一组重复颜色（其余对象颜色仍各不相同），避免颜色重复太多导致视觉过乱。
#     - 4-6 个对象、每个 bbox 仍在 5x5 内，保持与原题相同的对象复杂度。
#     - 共享颜色只在两个对象间复用，且这两个对象 canonical shape 必不同，
#       所以不会把“颜色冲突”扩展成大规模组合歧义。
#   implicit_assumption_audit_main:
#     - get_bg(g): 假设背景色仍是全局最多颜色。生成器保持大网格+少量小对象，因此成立。
#     - find_components(..., monochromatic=False):
#         这行已经说明 solver 本来就把“所有非背景像素的 8 连通块”当对象，
#         完全不依赖每个对象颜色唯一。D12 变化下这仍然是正确做法。
#         新风险是：若两个对象在 8 邻域接触，即使颜色不同也会被合并成一个对象。
#         原 generator 只防 overlap，不防 touch；在颜色复用后这会更难肉眼分辨。
#         因此 generator 额外保证对象之间至少留 1 格 8 邻域间隔。
#     - pat[m] = patch[m]:
#         这行保留对象内部颜色。因为对象仍是纯色，重复颜色不会破坏 patch 提取。
#     - shape_key = _canonical_shape(rel):
#         只依赖坐标形状，不依赖颜色。D12 下正是我们要保留的核心。
#     - dest_r0 = 0 / H-h:
#         仍然只由 shape frequency 决定。颜色重复不会改变该逻辑。
#   placement_legality:
#     - 输出仍先构造 top/bottom 目标，再反推输入。
#     - 新增 8 邻域 clearance 检查，保证对象彼此不重叠、不接触、不越界。
#     - 输入散布时仍固定 c0，只随机 r0；并优先避开已经是最终行的位置，使移动更可见。
#   visual_observability:
#     - 每个样例至少出现一对同色对象，且一个最终到顶部、一个最终到底部。
#     - 因此从 2-3 组样例中，人类可以直接观察到：
#       “颜色重复并不代表它们会被归到一起，真正决定去向的是形状是否有同形伙伴。”
#   cleanliness:
#     - 同色复用只做一对，避免网格里出现大量同色对象造成噪声。
#     - 对象之间加入空隙，避免同色对象在视觉上挤成一团。
#   color_inference:
#     - 这里没有颜色映射变换，颜色只承担“反向线索”的角色。
#     - 可推断性要求变成：样例必须稳定展示“同色也会分开处理”，
#       所以生成器对重复颜色的存在与跨 top/bottom 分布都做强约束，而不是低概率出现。


def _canonical_shape(coords):
    """Canonical D4-invariant shape key, intentionally ignoring color."""
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
    Sort objects by shape-sharing status only.

    D12 note:
    - Colors are intentionally unreliable metadata in this variant.
    - The solver therefore keeps the original color-agnostic extraction:
      connected components are found over all non-background pixels, and only
      canonical shape frequency decides whether an object goes to the top or
      bottom edge.
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


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly `area` cells."""
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
                if (
                    max(all_r) - min(all_r) < max_dim
                    and max(all_c) - min(all_c) < max_dim
                ):
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
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _can_place_with_gap(grid, patch, r0, c0, bg, gap=1):
    """Disallow overlap and 8-neighborhood touching for cleaner objects."""
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    fg = np.argwhere(patch != bg)
    for pr, pc in fg:
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        for nr in range(max(0, rr - gap), min(H, rr + gap + 1)):
            for nc in range(max(0, cc - gap), min(W, cc + gap + 1)):
                if grid[nr, nc] != bg:
                    return False
    return True


def _assign_variant_colors(objs, bg):
    """
    Assign colors so that:
    - at least one color is reused,
    - reused color spans one shared-shape object and one unique-shape object,
    - same-color objects never share the same canonical shape.
    """
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)

    shared_ids = [i for i, obj in enumerate(objs) if obj["shared"]]
    unique_ids = [i for i, obj in enumerate(objs) if not obj["shared"]]
    if not shared_ids or not unique_ids:
        raise ValueError("need both shared and unique objects")

    dup_shared = random.choice(shared_ids)
    dup_unique = random.choice(unique_ids)
    dup_color = avail[0]

    objs[dup_shared]["color"] = dup_color
    objs[dup_unique]["color"] = dup_color

    color_iter = iter(avail[1:])
    for obj in objs:
        if "color" not in obj:
            obj["color"] = next(color_iter)

    color_to_shapes = {}
    for obj in objs:
        color_to_shapes.setdefault(int(obj["color"]), set()).add(obj["shape_key"])
    for color, shape_keys in color_to_shapes.items():
        color_count = sum(1 for obj in objs if int(obj["color"]) == color)
        if len(shape_keys) != color_count:
            raise ValueError("same-color objects must have different canonical shapes")


def _candidate_input_rows(H, h, dest_r0):
    rows = [r for r in range(1, H - h) if r != dest_r0]
    if rows:
        return rows
    rows = [r for r in range(0, H - h + 1) if r != dest_r0]
    return rows if rows else [dest_r0]


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
]


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])

    group_sizes, n_unique = random.choice(_CONFIGS)

    objs = []
    canon_used = set()

    for gsz in group_sizes:
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        canon = _canonical_shape(base)
        if canon in canon_used:
            raise ValueError("duplicate shared canonical")
        canon_used.add(canon)
        for _ in range(gsz):
            transformed = _transform_shape(
                base, random.randint(0, 3), random.choice([False, True])
            )
            objs.append(
                {
                    "coords": transformed,
                    "shape_key": canon,
                    "shared": True,
                }
            )

    for _ in range(n_unique):
        for _attempt in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon in canon_used:
                continue
            canon_used.add(canon)
            objs.append(
                {
                    "coords": shape,
                    "shape_key": canon,
                    "shared": False,
                }
            )
            break
        else:
            raise ValueError("unique shape not found")

    _assign_variant_colors(objs, bg)
    random.shuffle(objs)

    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)

    out = np.full((H, W), bg, dtype=int)
    placement_order = sorted(objs, key=lambda obj: obj["patch"].shape[1], reverse=True)
    for obj in placement_order:
        patch = obj["patch"]
        h, w = map(int, patch.shape)
        dest_r0 = 0 if obj["shared"] else int(H - h)
        cols = list(range(0, W - w + 1))
        random.shuffle(cols)
        for c0 in cols:
            if _can_place_with_gap(out, patch, dest_r0, c0, bg, gap=1):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                break
        else:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    scatter_order = sorted(objs, key=lambda obj: obj["patch"].shape[0], reverse=True)
    for obj in scatter_order:
        patch = obj["patch"]
        h, _w = map(int, patch.shape)
        dest_r0 = 0 if obj["shared"] else int(H - h)
        rows = _candidate_input_rows(H, h, dest_r0)
        random.shuffle(rows)
        c0 = int(obj["c0"])
        for r0 in rows:
            if _can_place_with_gap(inp, patch, int(r0), c0, bg, gap=1):
                paste(inp, patch, int(r0), c0, transparent=bg)
                break
        else:
            raise ValueError("input scatter failed")

    result = main(inp)
    if not np.array_equal(result, out):
        raise AssertionError("round-trip check failed")

    # D12 safeguard: every sample must visibly contain a duplicated color, and
    # that duplicated color must cross the top/bottom split induced by shape.
    dup_colors = [color for color, cnt in Counter(obj["color"] for obj in objs).items() if cnt >= 2]
    if len(dup_colors) != 1:
        raise AssertionError("expected exactly one duplicated color")
    dup_color = dup_colors[0]
    dup_objs = [obj for obj in objs if obj["color"] == dup_color]
    if len(dup_objs) != 2 or dup_objs[0]["shared"] == dup_objs[1]["shared"]:
        raise AssertionError("duplicated color must span shared and unique objects")
    if dup_objs[0]["shape_key"] == dup_objs[1]["shape_key"]:
        raise AssertionError("same-color objects must differ in canonical shape")

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError, StopIteration):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D12 variant")
