# Combo variant of 800d221b (index 7)
# combo_spec: D1/3+D6/1
# Name: 颜色交换+长度排序匹配
#
# Design analysis:
# - 具体化与行为追踪:
#   原题按“物理接触”给每条 branch 着色，seed 内部不变；这里改成两步联动：
#   先把 4 条 branch 按 path-only 长度排序，再把 4 个 seed 按 interior 面积排序，最长 branch
#   对应最大 seed 的原始多数色；同时每个 seed 内部把多数色和少数色整体互换。
#   例如某条最短 branch 实际连着一个 12 格大 seed，原题会拿这个大 seed 的颜色；本变体若最短
#   branch 排到最后，则它改拿最小 seed 的多数色，而自己身边那个大 seed 在输出里还会发生颜色交换。
# - 复杂度校准:
#   仍保留 hub + 4 个边缘 seed + 4 条 branch 的原始视觉家族，但固定为 4 对对象，并强制“面积全异、
#   长度全异”，避免排序歧义。seed 只使用 2 种内部颜色，且每格样例都同时出现“排序匹配”和“颜色交换”。
# - 隐式假设审查:
#   main() 新增的关键前提是：每个 seed 恰有一个 disjoint 的 path-only branch；seed interior 恰有两色且
#   严格多数；branch ranking 与 seed ranking 可稳定排序。generator 通过模板化几何、严格不同的面积/长度、
#   以及样本级退化检查来保证这些前提。
# - 放置合法性:
#   4 个 seed 占据四个边角边缘区，border 只出现在非贴边一侧；paths 走互不相交的象限折线；hub 固定在中心，
#   因此不会越界、重叠，也不会生成 T-junction。
# - 视觉可观察性:
#   解题者能直接看到四个 seed 大小不同、四条 branch 长短不同，而且 branch 输出颜色经常不等于其物理相邻
#   seed 的输入多数色；同时 seed 内部颜色整体翻转，形成第二条稳定线索。
# - 可视化整洁度:
#   路径、border、seed interior 三层结构保持分离；每个象限只有一条 branch 和一个 seed，不会出现难以辨认
#   的多对象叠压。
# - 变化的可推断性:
#   只用两种 seed 内部色，减少颜色映射自由度；排序依据只使用人类可见的长度/面积，不依赖数值 id。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


GRID_SHAPE = (27, 27)
HUB_SIZE = 3
HUB_R0 = 12
HUB_C0 = 12
CARDINALS = ("N", "E", "S", "W")
SEED_DIMS = ((2, 3), (2, 4), (3, 3), (3, 4))  # areas 6, 8, 9, 12
MINORITY_COUNTS = {6: 2, 8: 2, 9: 3, 12: 4}
GEN_RETRIES = 200
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for idx, ((r0, c0), (r1, c1)) in enumerate(zip(points, points[1:])):
        if r0 != r1 and c0 != c1:
            raise ValueError(f"non-orthogonal segment: {(r0, c0)} -> {(r1, c1)}")
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            seg = [(r0, c) for c in range(c0, c1 + step, step)]
        else:
            step = 1 if r1 >= r0 else -1
            seg = [(r, c0) for r in range(r0, r1 + step, step)]
        if idx:
            seg = seg[1:]
        out.extend(seg)
    return out


def _comp_deg_score(mask: np.ndarray) -> float:
    comps = components_mask(mask, connectivity=4)
    if not comps:
        return -1e9
    best_pts = max(comps, key=lambda coords: int(coords.shape[0]))
    best_sz = int(best_pts.shape[0])
    H, W = mask.shape
    m = np.zeros(mask.shape, dtype=bool)
    m[best_pts[:, 0], best_pts[:, 1]] = True
    deg2 = 0
    deg4 = 0
    for r, c in best_pts:
        r = int(r)
        c = int(c)
        d = 0
        if r > 0 and bool(m[r - 1, c]):
            d += 1
        if r + 1 < H and bool(m[r + 1, c]):
            d += 1
        if c > 0 and bool(m[r, c - 1]):
            d += 1
        if c + 1 < W and bool(m[r, c + 1]):
            d += 1
        if d == 2:
            deg2 += 1
        elif d == 4:
            deg4 += 1
    return float((deg2 - deg4) * best_sz / max(1, best_sz))


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors, "grid needs at least one non-bg color"
    return int(max(colors, key=lambda c: _comp_deg_score(g == int(c))))


def _largest_solid_square(mask: np.ndarray) -> tuple[int, int, int]:
    H, W = mask.shape
    ps = mask.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[r1, c1])
        if r0 > 0:
            s -= int(ps[r0 - 1, c1])
        if c0 > 0:
            s -= int(ps[r1, c0 - 1])
        if r0 > 0 and c0 > 0:
            s += int(ps[r0 - 1, c0 - 1])
        return int(s)

    best = None
    for k in range(2, min(H, W) + 1):
        found = None
        for r0 in range(0, H - k + 1):
            r1 = r0 + k - 1
            for c0 in range(0, W - k + 1):
                c1 = c0 + k - 1
                if rect_sum(r0, c0, r1, c1) == k * k:
                    found = (k, r0, c0)
        if found is not None:
            best = found
    assert best is not None, "hub square not found"
    return best


def _major_minor(vals: list[int]) -> tuple[int, int]:
    cnt = Counter(int(v) for v in vals)
    assert len(cnt) == 2, f"expected 2 seed colors, got {cnt}"
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    assert items[0][1] != items[1][1], f"seed majority must be strict, got {cnt}"
    return int(items[0][0]), int(items[1][0])


def _seed_border_cells(seed_lab: np.ndarray, sid: int, p_mask: np.ndarray) -> set[tuple[int, int]]:
    pts = np.argwhere(seed_lab == int(sid))
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    H, W = seed_lab.shape
    border = _border_from_bbox(r0, c0, r1, c1, H, W)
    return {(r, c) for r, c in border if bool(p_mask[r, c])}


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    P = _infer_path_color(g, bg)
    k, r0, c0 = _largest_solid_square(g == int(P))
    assert k == HUB_SIZE, f"expected {HUB_SIZE}x{HUB_SIZE} hub, got {k}"
    hub = np.zeros((H, W), dtype=bool)
    hub[r0 : r0 + k, c0 : c0 + k] = True

    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) == 4, f"expected 4 seed regions, got {seed_n}"

    seeds: list[dict] = []
    border_masks = np.zeros((H, W), dtype=bool)
    p_mask = g == int(P)
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        major, minor = _major_minor(vals)
        border = _seed_border_cells(seed_lab, sid, p_mask)
        assert border, f"seed {sid} missing path border"
        for r, c in border:
            border_masks[r, c] = True
        seeds.append(
            {
                "sid": int(sid),
                "pts": [(int(r), int(c)) for r, c in pts],
                "area": int(len(pts)),
                "major": int(major),
                "minor": int(minor),
                "border": border,
                "anchor": min((int(r), int(c)) for r, c in pts),
            }
        )

    path_only_mask = (g == int(P)) & (~hub) & (~border_masks)
    branch_lab, branch_n = label_mask(path_only_mask, connectivity=4)
    assert int(branch_n) == len(seeds), f"expected {len(seeds)} branches, got {branch_n}"

    branches: list[dict] = []
    used_seed_ids: set[int] = set()
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        cell_set = {(int(r), int(c)) for r, c in pts}
        attached = []
        for seed in seeds:
            touches = any(
                (int(r + dr), int(c + dc)) in seed["border"]
                for r, c in cell_set
                for dr, dc in DIRS4
            )
            if touches:
                attached.append(seed["sid"])
        assert len(attached) == 1, f"branch {bid} should attach to exactly one seed, got {attached}"
        sid = int(attached[0])
        assert sid not in used_seed_ids, f"multiple branches attached to seed {sid}"
        used_seed_ids.add(sid)
        hub_touch = sum(
            1
            for r, c in cell_set
            if any(
                0 <= r + dr < H and 0 <= c + dc < W and bool(hub[r + dr, c + dc])
                for dr, dc in DIRS4
            )
        )
        assert hub_touch > 0, f"branch {bid} must touch the hub"
        branches.append(
            {
                "bid": int(bid),
                "pts": cell_set,
                "length": int(len(cell_set)),
                "physical_sid": sid,
                "hub_touch": hub_touch,
                "anchor": min(cell_set),
            }
        )

    seeds.sort(key=lambda s: (-s["area"], s["anchor"]))
    branches.sort(key=lambda b: (-b["length"], b["anchor"]))

    physical_major = {
        branch["bid"]: next(seed["major"] for seed in seeds if seed["sid"] == branch["physical_sid"])
        for branch in branches
    }
    sorted_major = {
        branch["bid"]: seeds[idx]["major"]
        for idx, branch in enumerate(branches)
    }

    return {
        "grid": g,
        "bg": bg,
        "path_color": P,
        "hub_mask": hub,
        "hub_center": (r0 + k // 2, c0 + k // 2),
        "seeds": seeds,
        "branches": branches,
        "physical_major": physical_major,
        "sorted_major": sorted_major,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branch network by ranked branch/seed pairing, while swapping seed interiors.

    Input structure:
    - The most common color is background.
    - One non-bg color forms a thin orthogonal path network with a 3x3 solid hub.
    - Four edge-touching seed rectangles contain exactly two interior colors plus a 1-pixel path-colored frame on non-edge sides.

    Rule:
    - Infer the path color and isolate the hub, the seed interiors, the seed borders, and the path-only branch chains.
    - Sort the four branches by length and sort the four seed interiors by area; pair them rank-to-rank from largest to smallest.
    - Recolor each physical branch together with its local seed border to the paired seed's original majority color, then swap that seed interior's majority/minority colors in place.
    - Recolor the hub center to the branch color that appears most often on branch pixels adjacent to the hub.

    Key insight: Adjacency no longer decides branch color; the visible clues are branch length, seed area, and the simultaneous color swap inside every seed.
    """
    scene = _analyze_scene(grid)
    g = scene["grid"]
    out = g.copy()

    for seed in scene["seeds"]:
        major = int(seed["major"])
        minor = int(seed["minor"])
        for r, c in seed["pts"]:
            val = int(out[r, c])
            if val == major:
                out[r, c] = minor
            elif val == minor:
                out[r, c] = major

    for idx, branch in enumerate(scene["branches"]):
        matched_seed = scene["seeds"][idx]
        color = int(matched_seed["major"])
        physical_seed = next(seed for seed in scene["seeds"] if seed["sid"] == branch["physical_sid"])
        for r, c in branch["pts"]:
            out[r, c] = color
        for r, c in physical_seed["border"]:
            out[r, c] = color

    touch_cols = Counter()
    hub = scene["hub_mask"]
    H, W = g.shape
    for branch in scene["branches"]:
        for r, c in branch["pts"]:
            if any(0 <= r + dr < H and 0 <= c + dc < W and bool(hub[r + dr, c + dc]) for dr, dc in DIRS4):
                touch_cols[int(out[r, c])] += 1
    assert touch_cols, "hub must touch recolored branches"
    cr, cc = scene["hub_center"]
    out[cr, cc] = int(touch_cols.most_common(1)[0][0])
    return out


def _seed_bbox(slot: str, h: int, w: int, H: int, W: int) -> tuple[int, int, int, int]:
    if slot == "N":
        return 0, 3, h - 1, 3 + w - 1
    if slot == "E":
        return 4, W - w, 4 + h - 1, W - 1
    if slot == "S":
        return H - h, 19, H - 1, 19 + w - 1
    if slot == "W":
        return 19, 0, 19 + h - 1, w - 1
    raise ValueError(slot)


def _border_from_bbox(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if r0 > 0:
        for c in range(c0, c1 + 1):
            cells.add((r0 - 1, c))
    if r1 < H - 1:
        for c in range(c0, c1 + 1):
            cells.add((r1 + 1, c))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            cells.add((r, c0 - 1))
    if c1 < W - 1:
        for r in range(r0, r1 + 1):
            cells.add((r, c1 + 1))
    if r0 > 0 and c0 > 0:
        cells.add((r0 - 1, c0 - 1))
    if r0 > 0 and c1 < W - 1:
        cells.add((r0 - 1, c1 + 1))
    if r1 < H - 1 and c0 > 0:
        cells.add((r1 + 1, c0 - 1))
    if r1 < H - 1 and c1 < W - 1:
        cells.add((r1 + 1, c1 + 1))
    return cells


def _branch_end(slot: str, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if slot == "N":
        return r1 + 2, (c0 + c1) // 2
    if slot == "E":
        return (r0 + r1) // 2, c0 - 2
    if slot == "S":
        return r0 - 2, (c0 + c1) // 2
    if slot == "W":
        return (r0 + r1) // 2, c1 + 2
    raise ValueError(slot)


def _branch_path(slot: str, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    end_r, end_c = _branch_end(slot, bbox)
    if slot == "N":
        points = [(11, 13), (6, 13), (6, end_c), (end_r, end_c)]
    elif slot == "E":
        points = [(13, 15), (13, 18), (7, 18), (7, end_c), (end_r, end_c)]
    elif slot == "S":
        points = [(15, 13), (21, 13), (21, 16), (19, 16), (19, 18), (end_r, 18), (end_r, end_c)]
    elif slot == "W":
        points = [(13, 11), (13, 7), (18, 7), (18, end_c), (end_r, end_c)]
    else:
        raise ValueError(slot)
    return _polyline(points)


def _paint_seed_interior(
    g: np.ndarray,
    cells: list[tuple[int, int]],
    major: int,
    minor: int,
    rng: random.Random,
) -> None:
    area = len(cells)
    n_minor = int(MINORITY_COUNTS[area])
    chosen = cells[:]
    rng.shuffle(chosen)
    for r, c in chosen[:n_minor]:
        g[r, c] = minor
    for r, c in chosen[n_minor:]:
        g[r, c] = major


def _branch_length_rank(branch_paths: dict[str, list[tuple[int, int]]]) -> list[str]:
    return sorted(branch_paths, key=lambda slot: (-len(branch_paths[slot]), slot))


def _seed_area_rank(seed_dims_by_slot: dict[str, tuple[int, int]]) -> list[str]:
    return sorted(seed_dims_by_slot, key=lambda slot: (-(seed_dims_by_slot[slot][0] * seed_dims_by_slot[slot][1]), slot))


def _build_input(rng: random.Random) -> tuple[np.ndarray, dict]:
    H, W = GRID_SHAPE
    bg, P, color_a, color_b = rng.sample(list(range(10)), 4)
    g = np.full((H, W), bg, dtype=int)
    g[HUB_R0 : HUB_R0 + HUB_SIZE, HUB_C0 : HUB_C0 + HUB_SIZE] = P

    dims_perm = list(SEED_DIMS)
    rng.shuffle(dims_perm)
    seed_dims_by_slot = {slot: dims_perm[idx] for idx, slot in enumerate(CARDINALS)}

    branch_paths: dict[str, list[tuple[int, int]]] = {}
    border_by_slot: dict[str, set[tuple[int, int]]] = {}
    interior_by_slot: dict[str, list[tuple[int, int]]] = {}
    for slot in CARDINALS:
        h, w = seed_dims_by_slot[slot]
        bbox = _seed_bbox(slot, h, w, H, W)
        r0, c0, r1, c1 = bbox
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        border = _border_from_bbox(r0, c0, r1, c1, H, W)
        path = _branch_path(slot, bbox)
        interior_by_slot[slot] = cells
        border_by_slot[slot] = border
        branch_paths[slot] = path

    occupied: set[tuple[int, int]] = set()
    for border in border_by_slot.values():
        assert occupied.isdisjoint(border), "seed borders overlap"
        occupied |= border
    for slot, path in branch_paths.items():
        pset = set(path)
        assert occupied.isdisjoint(pset), f"path overlaps seed structure for {slot}"
        occupied |= pset
    hub_cells = {(r, c) for r in range(HUB_R0, HUB_R0 + HUB_SIZE) for c in range(HUB_C0, HUB_C0 + HUB_SIZE)}
    assert occupied.isdisjoint(hub_cells), "path/seed overlaps hub"

    odd_slot = rng.choice(CARDINALS)
    major_by_slot = {slot: (color_b if slot == odd_slot else color_a) for slot in CARDINALS}
    minor_by_slot = {slot: (color_a if slot == odd_slot else color_b) for slot in CARDINALS}

    for slot in CARDINALS:
        _paint_seed_interior(g, interior_by_slot[slot], major_by_slot[slot], minor_by_slot[slot], rng)
        for r, c in border_by_slot[slot]:
            g[r, c] = P
        for r, c in branch_paths[slot]:
            g[r, c] = P

    meta = {
        "bg": bg,
        "path_color": P,
        "major_by_slot": major_by_slot,
        "minor_by_slot": minor_by_slot,
        "seed_dims_by_slot": seed_dims_by_slot,
        "branch_paths": branch_paths,
        "odd_slot": odd_slot,
    }
    return g, meta


def _fusion_mismatch_count(scene: dict) -> int:
    return sum(
        1
        for branch in scene["branches"]
        if int(scene["physical_major"][branch["bid"]]) != int(scene["sorted_major"][branch["bid"]])
    )


def _all_lengths_unique(scene: dict) -> bool:
    lengths = [int(branch["length"]) for branch in scene["branches"]]
    return len(lengths) == len(set(lengths))


def _all_areas_unique(scene: dict) -> bool:
    areas = [int(seed["area"]) for seed in scene["seeds"]]
    return len(areas) == len(set(areas))


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp, _meta = _build_input(random.Random())
        try:
            scene = _analyze_scene(inp)
            _ = main(inp)
        except Exception:
            continue
        if not _all_lengths_unique(scene):
            continue
        if not _all_areas_unique(scene):
            continue
        if _fusion_mismatch_count(scene) < 2:
            continue
        return inp
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
