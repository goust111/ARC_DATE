# Variant of 62593bfd
# Dimension: D2 — 分组后的方向分配规则 (match_cond)
# Change: duplicated形状（频次>=2）的对象移到顶部（row 0），unique形状（频次==1）的对象移到底部（row... → duplicated移左边缘，unique移右边缘（水平分流）

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste

# Design analysis for the D2 variant
# ----------------------------------
# spec:
#   core_change:
#     - 原题按“形状是否重复”做上下分流，且保留列位置。
#     - 变体改为左右分流：重复形状 -> 左边缘 col 0；唯一形状 -> 右边缘 col W-w；保留行位置。
#   concrete_behavior:
#     - 输入仍然是 4-6 个单色 8 连通对象，形状按 D4 canonical key 分组。
#     - 输出中每个对象的像素图案保持不变，只做整体平移。
#     - 共享形状对象全部贴左边，唯一形状对象全部贴右边；对象在输入中的顶部行 r0 需要原样保留。
#   trace_example:
#     - 若某个 shared 对象 bbox 为 (r0=7, c0=11, h=3, w=4)，原题输出会去 (0, 11)，
#       本变体输出应去 (7, 0)。
#     - 若某个 unique 对象 bbox 为 (r0=12, c0=3, h=4, w=5)，原题输出会去 (H-4, 3)，
#       本变体输出应去 (12, W-5)。
#   main_dependency_audit:
#     - 原代码提取对象时只保存 c0，这隐含了“保留列、改变行”的旧前提；变体必须改存 r0。
#     - 原 `dest_r0 = 0 / H-h` 这一分支直接编码了上下分流；变体必须改为
#       `dest_c0 = 0 / W-w`。
#     - `paste()` 默认允许覆盖，原题之所以安全，是因为生成器保证路由后不碰撞。
#       新生成器同样必须保证输出中的对象不会因左右归边而连成一个 8 连通体。
#   generator_adaptation:
#     - 继续沿用 output-first。
#     - 先造形状与颜色，再根据所有 patch 的高度选择足够大的 H。
#     - 输出阶段把所有对象放进互不相邻的行带中（至少隔 1 行背景），这样即使
#       `find_components(..., monochromatic=False)` 也不会把异色对象并成一个大组件。
#     - 输入阶段保持这些行带不变，只随机改列位置，从而让“保留行、水平归边”在
#       样例里可直接观察。
#   observability:
#     - 至少要求一个 shared 对象在输入时不在左边缘，且至少一个 unique 对象不在右边缘，
#       否则水平移动会退化得不明显。
#     - 行位置在 input/output 中保持一致，解题者会看到“对象横向吸附到左右边缘”，
#       与原题的上下吸附形成清晰差异。


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to left or right edge by whether their shape (up to
    rotation / reflection) is shared with another object.

    Rule:
    - Compute the canonical shape of each 8-connected component
      (invariant under D4: 0°/90°/180°/270° rotations × optional
      horizontal reflection).
    - Shared shapes  (canonical shape freq ≥ 2) → translate to LEFT  (col 0).
    - Unique shapes  (freq == 1)                → translate to RIGHT (col W−w).
    - Vertical position (top row r0 of bounding box) is preserved.
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
                "r0": int(r0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, w = map(int, pat.shape)
        dest_r0 = int(d["r0"])
        dest_c0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(W - w)
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))

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
    mr = min(r for r, c in cells)
    mc = min(c for r, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


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
    p = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        p[r, c] = color
    return p


def _pack_rows_with_gaps(heights, H, gap=1):
    """Pack objects into non-touching row bands."""
    n = len(heights)
    required = sum(int(h) for h in heights) + max(0, n - 1) * int(gap)
    if required > H:
        raise ValueError("insufficient height for row packing")

    slack = int(H - required)
    outer_gaps = [0] * (n + 1)
    for _ in range(slack):
        outer_gaps[random.randrange(n + 1)] += 1

    rows = []
    cur = outer_gaps[0]
    for i, h in enumerate(heights):
        rows.append(int(cur))
        cur += int(h)
        if i < n - 1:
            cur += int(gap) + outer_gaps[i + 1]
        else:
            cur += outer_gaps[-1]
    return rows


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
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)

    objs = []
    canon_used = set()

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

    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = patch.shape
        obj["patch"] = patch
        obj["h"] = int(h)
        obj["w"] = int(w)

    min_H = max(19, sum(obj["h"] for obj in objs) + len(objs) - 1)
    if min_H > 30:
        raise ValueError("objects too tall for clean horizontal routing")
    H = random.randint(min_H, 30)
    W = random.randint(19, 30)

    random.shuffle(objs)
    packed_rows = _pack_rows_with_gaps([obj["h"] for obj in objs], H, gap=1)

    out = np.full((H, W), bg, dtype=int)
    for obj, r0 in zip(objs, packed_rows):
        patch = obj["patch"]
        dest_c0 = 0 if obj["shared"] else W - obj["w"]
        paste(out, patch, int(r0), int(dest_c0), transparent=bg)
        obj["r0"] = int(r0)
        obj["dest_c0"] = int(dest_c0)

    inp = np.full((H, W), bg, dtype=int)
    shared_shifted = False
    unique_shifted = False
    for obj in objs:
        patch = obj["patch"]
        c0 = random.randint(0, W - obj["w"])
        paste(inp, patch, obj["r0"], int(c0), transparent=bg)
        if obj["shared"] and c0 != 0:
            shared_shifted = True
        if (not obj["shared"]) and c0 != W - obj["w"]:
            unique_shifted = True

    if not shared_shifted or not unique_shifted:
        raise ValueError("scatter did not visibly exercise both horizontal branches")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D2")
