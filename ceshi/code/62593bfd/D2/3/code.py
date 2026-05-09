# Variant of 62593bfd
# Dimension: D2 — 分组后的方向分配规则 (match_cond)
# Change: duplicated形状（频次>=2）的对象移到顶部（row 0），unique形状（频次==1）的对象移到底部（row... → 多数组移顶部，少数组移底部

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste

# Design analysis for the D2 variant
# ----------------------------------
# a) 具体化与行为追踪
# - 输入仍然是 19..30 的网格，背景上散落 5..6 个单色 8 连通对象；输出仍只做竖直平移，保留每个对象的 bbox 左边界 c0。
# - 形状分组仍按 D4 canonical shape（旋转/翻转等价）进行。变化只在“哪些组去顶部”：
#   原规则: 频次 >= 2 的组全部去顶部，频次 == 1 的组去底部。
#   新规则: 只把对象数最多的组移到顶部；其他所有组都移到底部。
# - 具体例子: 组大小 [3, 2, 1] 时，原规则会把 3 组和 2 组都放到顶部，1 组放到底部；
#   新规则只把 3 组放顶部，而 2 组与 1 组都放到底部。可见差异是“重复形状现在也可能去底部”。
#
# b) 复杂度校准
# - 若生成 [2,1,1] 之类样本，则新旧规则输出完全一致，变体不可观察。
# - 因此生成器只使用带“唯一最大组 + 至少一个较小重复组”的配置: [3,2], [3,2,1], [4,2]。
#   这样每个样本都至少有 2 个“旧规则会上移、新规则会下移”的对象，差异稳定可见。
#
# c) 原 main() 的隐式假设审查
# - `shape_key = _canonical_shape(rel)` 仍然成立；D2 不改变形状等价定义。
# - `freq = Counter(...)` 仍然成立，但 `freq >= 2` 这个阈值假设不再成立，必须替换为“是否等于全局最大组大小”。
# - `dest_c0 = c0` 仍然成立，因为变体没有改水平对齐规则。
# - `paste(...)` 默认 overwrite=True；这要求输出布局中对象绝不重叠。生成器必须继续保证这一点。
# - `find_components(... monochromatic=False)` 在原题中之所以无害，是因为对象之间本来就不 8 邻接；
#   若不同颜色对象接触会被误并成一个组件。新生成器显式保证对象之间有 1 格 8 邻域间隔，维持这个前提。
#
# d) 放置合法性
# - 仍采用 output-first：先在输出网格的 top/bottom 行放好对象，再按相同 c0 把它们散射回输入。
# - 由于对象 bbox 高宽 <= 5，且 H >= 19，顶部 row 0 与底部 row H-h 之间天然有大间隔，不会跨区碰撞。
# - 新的 `_no_contact` 要求对象之间不仅不重叠，还不能 8 邻接，这样 solver 的连通域提取始终稳定。
#
# e) 视觉可观察性
# - 人类可见的核心信号是：有些“出现了不止一次的形状”并没有去顶部，而是和真正少数派一起去了底部。
# - 为了让这一点在 2..4 个样例中足够明确，生成器保证每个样本都含两个重复组且组大小不同。
#
# f) 可视化整洁度
# - 对象仍是纯色、不重叠、横向错开；顶部与底部各成一排，视觉上清楚。
# - 额外的 8 邻域间隔避免了斜对角粘连造成的“看起来像一个大块”的噪声。
#
# g) 颜色变化可推断性
# - 本变体不引入颜色映射。颜色只用于区分对象，不参与排序或比较。
#
# h) 变体验证指标
# - 量化指标选用 `demoted_duplicates`: 频次 >= 2 但又不是最大组的对象数。
# - 对原题该值恒为 0；对本变体的生成样本应恒 >= 2。后续测试脚本据此检查是否退化回原规则。
#
# i) main() 修改后的新隐式假设
# - solver 允许最大组并列：所有 `freq == max_freq` 的组都去顶部。这是最稳妥的定义。
# - 但 generator 为了可观察性，始终构造“唯一最大组”，避免并列样本让规则显得含糊。


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


def _extract_objects(grid: np.ndarray):
    g = np.array(grid, dtype=int)
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
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
        extracted.append(
            {
                "c0": int(c0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )
    return g, bg, extracted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move only the globally largest canonical-shape group to the top edge.

    Rule:
    - Compute each 8-connected component's canonical shape under D4.
    - Count how many objects belong to each canonical-shape group.
    - Groups with the maximum count go to TOP (row 0).
    - All other groups go to BOTTOM (row H-h).
    - Horizontal position (bbox left edge c0) is preserved.
    """
    g, bg, extracted = _extract_objects(grid)
    H, W = g.shape

    freq = Counter(d["shape_key"] for d in extracted)
    max_freq = max(int(v) for v in freq.values())
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h = int(pat.shape[0])
        dest_r0 = 0 if int(freq[d["shape_key"]]) == max_freq else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    return out


# ────────────────────── generator ──────────────────────

def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly `area` pixels within max_dim."""
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
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _no_contact(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    fg = np.argwhere(patch != bg)
    for pr, pc in fg:
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        rs = max(0, rr - 1)
        re = min(H, rr + 2)
        cs = max(0, cc - 1)
        ce = min(W, cc + 2)
        if np.any(grid[rs:re, cs:ce] != bg):
            return False
    return True


_CONFIGS = [
    [3, 2],
    [3, 2, 1],
    [4, 2],
]


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    color_idx = 0

    group_sizes = random.choice(_CONFIGS)
    max_group = max(group_sizes)
    if sum(1 for g in group_sizes if g == max_group) != 1:
        raise ValueError("config must have a unique largest group")

    objs = []
    canon_used: set = set()

    for gsz in group_sizes:
        for _ in range(200):
            area = random.randint(3, 12)
            base = _grow_shape(area, 8)
            canon = _canonical_shape(base)
            if canon not in canon_used:
                canon_used.add(canon)
                break
        else:
            raise ValueError("group base shape not found")

        for _ in range(gsz):
            transformed = _transform_shape(
                base, random.randint(0, 3), random.choice([False, True])
            )
            objs.append(
                {
                    "coords": transformed,
                    "color": avail[color_idx],
                    "group_size": int(gsz),
                }
            )
            color_idx += 1

    out = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = patch.shape
        dest_r0 = 0 if int(obj["group_size"]) == max_group else int(H - h)
        placed = False
        for _ in range(500):
            c0 = random.randint(0, W - w)
            if _no_contact(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                obj["patch"] = patch
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["patch"]
        h, w = patch.shape
        c0 = int(obj["c0"])
        placed = False
        for _ in range(500):
            r0 = random.randint(0, H - h)
            if _no_contact(inp, patch, r0, c0, bg):
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
    raise RuntimeError("failed to generate input for 62593bfd D2 variant")
