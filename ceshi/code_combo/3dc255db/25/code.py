# Combo variant of 3dc255db (index 25)
# combo_spec: D8/0+D11/0
# Name: 距离排序+路径穿越
#
# Design analysis
# ===============
# a) 具体化与行为追踪
# - 原题中每个杯都可独立处理：识别尾刺 -> 擦除本杯内容 -> 沿尾刺方向连续画 `content_count`
#   个像素。原 generator 还保证路径无障碍，因此多个杯之间没有真实交互。
# - 本融合变体同时改两处：
#   1. 多杯时，按杯 bbox 中心到整张网格中心的距离，从近到远处理。
#   2. 射线遇到任何杯的边框像素时，不涂色、不消耗配额，但继续向前；遇到先前杯已经画出的
#      线段时，也不停下，而是直接覆盖后继续。
# - 具体例子：右半侧一行上有两个相向的杯，左边杯 A 更靠近网格中心、尾刺向右；右边杯 B 更远、
#   尾刺向左。A 先画线，线会穿过 B 的左边框/内部继续前进；随后 B 再画线，它会穿过 A 的右边框，
#   并在走廊重叠区覆盖 A 先前留下的颜色。和原题相比，差异不再是“是否画得到”，而是
#   “是否跳过边框继续”和“重叠处最终保留谁的颜色”。
# b) 复杂度校准
# - 若只保留排序、不制造重叠，则视觉上几乎退化成 D11 单变体；若只保留穿越、不让后发射覆盖前线，
#   又退化成顺序无关。为避免退化，generator 强制生成一对同轴相向杯，使它们：
#   1. 都会实际使用至少一次“跳过异杯边框”；
#   2. 都会在可涂色单元上产生真实重叠，保证处理顺序影响最终输出。
# - 第三个杯若存在，则放到清晰通道中，不再制造第二组交互，避免整张图变成多层覆盖谜题。
# c) 原 solver 中被破坏的隐式假设
# - 原版 `for t in range(...): out[rr, cc] = color` 依赖“路径上每格都可直接涂色”。融合后不成立：
#   边框格必须跳过，已有线段也不能阻塞。
# - 原版逐杯“擦内容再立刻画线”也不成立。若 A 先穿过 B 的嘴部区域，而 B 的内容还没先清掉，
#   A 会错误改写 B 的输入内容，随后 B 再擦除时又会把 A 的线抹掉。解决方式是：先基于静态输入识别
#   所有杯，再全局擦除所有杯内容，最后按距离顺序画线。
# - 原版默认多杯无顺序依赖；融合后顺序决定重叠像素最终颜色，因此 solver 必须显式排序。
# d) 放置合法性
# - 杯 bbox 仍保持至少 1 格间距，保证连通块解析稳定。
# - 交互 pair 必须同轴相向、距离顺序唯一，并满足两个条件：射线都能越过对方至少一个边框像素，
#   且两条射线在非边框单元上有重叠。
# - 所有杯内容仍放在各自 mouth 内部，且避开最终会被任意射线涂色的单元，以免输入视觉过脏。
# e) 视觉可观察性
# - 排序依据只使用“离网格中心远近”，这是人类可见的几何关系。
# - 穿越现象通过“线在异色杯边框处断一格但随后继续”体现；顺序现象通过“重叠走廊最终保留较远杯颜色”
#   体现。二者都能从 2-4 组样例中肉眼观察。
# f) 可视化整洁度
# - 只强制一组主交互 pair，额外杯保持独立；关闭 stray pixel，避免额外孤立噪声干扰“穿杯+覆盖”主规律。
# - 边框颜色和内容颜色尽量彼此不同，重叠区域的颜色归属更清楚。
# g) 颜色变化可推断性
# - 颜色语义不变：边框色仍是边框，内容色仍是喷射线颜色。没有隐藏的颜色值映射，也不依赖颜色大小排序。
# 新 main() 的隐式假设审查
# - 假设 1：每个杯在原输入中仍能被唯一检测为一个边框连通块，mouth 内主要只有一种本杯内容色。
#   generator 继续保证这一点。
# - 假设 2：排序 tie 需要稳定且最好少见。solver 用 `(distance, r0, c0, border_color)` 兜底，
#   generator 额外避免距离完全相同。
# - 假设 3：需要跳过的障碍恰好就是“所有检测到的杯边框”。generator 不在路径上放 stray，也不让
#   其它静态对象混入，因此该假设成立。
# - 假设 4：后发射线覆盖前发射线是允许的，且不会再被任何残留内容擦掉。generator 通过“先散内容、
#   solver 全局擦内容后再画线”与之匹配。

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood within bbox from one edge, blocked by border cells."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _cup_order_key(bbox: tuple[int, int, int, int], H: int, W: int) -> tuple[int, int, int]:
    """Visible sort key: bbox center distance to grid center, then top-left for stability."""
    r0, c0, r1, c1 = bbox
    dr2 = (r0 + r1) - (H - 1)
    dc2 = (c0 + c1) - (W - 1)
    return dr2 * dr2 + dc2 * dc2, r0, c0


def _detect_cups(grid: np.ndarray) -> tuple[int, list[dict]]:
    """Detect cups from the static input grid."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cups: list[dict] = []

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 5:
                continue

            r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
            bbox = (r0, c0, r1, c1)
            border_set = {(int(r), int(c)) for r, c in border_coords}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

            edges = (
                ("U", [p for p in border_set if p[0] == r0]),
                ("D", [p for p in border_set if p[0] == r1]),
                ("L", [p for p in border_set if p[1] == c0]),
                ("R", [p for p in border_set if p[1] == c1]),
            )
            candidates = [
                (d, cells[0]) for d, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for d, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[d])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((d, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]
            cups.append(
                {
                    "border_color": int(border_col),
                    "bbox": bbox,
                    "border_set": border_set,
                    "tail_dir": tail_dir,
                    "spike": spike,
                    "content_col": content_col,
                    "content_cells": list(content_cells),
                }
            )

    return bg, cups


def _ejection_cells_from_ray(
    full_ray: list[tuple[int, int]],
    border_union: set[tuple[int, int]],
    content_count: int,
) -> tuple[list[tuple[int, int]], int]:
    """Take the first `content_count` paintable cells, skipping border cells without consuming."""
    ejection: list[tuple[int, int]] = []
    used_skips = 0
    for cell in full_ray:
        if len(ejection) >= content_count:
            break
        if cell in border_union:
            used_skips += 1
            continue
        ejection.append(cell)
    return ejection, used_skips


def _ordered_ejection_cells(
    H: int,
    W: int,
    spike: tuple[int, int],
    tail_dir: str,
    content_count: int,
    border_union: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    """Ray to the edge: border cells are skipped, prior lines are overwritten."""
    dr, dc = DIR4[tail_dir]
    rr, cc = spike[0] + dr, spike[1] + dc
    full_ray: list[tuple[int, int]] = []
    while 0 <= rr < H and 0 <= cc < W:
        full_ray.append((rr, cc))
        rr += dr
        cc += dc
    return _ejection_cells_from_ray(full_ray, border_union, content_count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fusion rule:
    1. Detect all cups from the original input.
    2. Erase every cup's content first.
    3. Process cups from near-center to far-center.
    4. Each ray skips cup-border cells but continues; existing rays are overwritten, not blocking.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, cups = _detect_cups(g)
    out = g.copy()

    border_union: set[tuple[int, int]] = set()
    for cup in cups:
        border_union |= cup["border_set"]

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

    cups.sort(key=lambda cup: (*_cup_order_key(cup["bbox"], H, W), int(cup["border_color"])))
    for cup in cups:
        ejection_cells, _ = _ordered_ejection_cells(
            H,
            W,
            cup["spike"],
            cup["tail_dir"],
            len(cup["content_cells"]),
            border_union,
        )
        for r, c in ejection_cells:
            out[r, c] = int(cup["content_col"])

    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
PRIMARY_NUM_CUPS_CHOICES = (2, 3)
FALLBACK_NUM_CUPS_CHOICES = (2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.0
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical Y/funnel cup whose tail points up."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    g = np.zeros((h, w), dtype=int)
    g[0, L] = color
    g[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    g[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        g[L + 1:L + hL + 1, 0] = color
    if hR:
        g[R + 1:R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the patch mouth interior from the opposite edge."""
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:
        seeds = [(r, w - 1) for r in range(h) if not border_mask[r, w - 1]]
    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate canonical (tail=U) data to the target tail direction."""
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for (r, c) in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for (r, c) in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for (r, c) in mouth]
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    """Sample one cup shape that fits the orientation-specific bbox limits."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    lo, hi = ARM_LEN_RANGE
    L_arm = int(rng.integers(lo, hi + 1))
    R_arm = int(rng.integers(lo, hi + 1))
    while L_arm + R_arm + 1 > max_w_canon:
        if L_arm >= R_arm:
            L_arm -= 1
        else:
            R_arm -= 1

    wlo, whi = WALL_EXT_RANGE
    hL = int(rng.integers(wlo, whi + 1))
    hR = int(rng.integers(wlo, whi + 1))
    if hL + hR == 0:
        hL = 1
    while max(L_arm + hL, R_arm + hR) + 1 > max_h_canon:
        if L_arm + hL >= R_arm + hR and hL > 0:
            hL -= 1
        elif hR > 0:
            hR -= 1
        else:
            break
    if hL + hR == 0:
        return None
    return L_arm, R_arm, hL, hR


def _build_cup_spec(
    rng: np.random.Generator,
    border_color: int,
    content_color: int,
    direction: str,
) -> dict | None:
    """Sample one cup shape with fixed direction."""
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape
    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 3:
        return None
    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    return {
        "border_color": int(border_color),
        "content_color": int(content_color),
        "direction": str(direction),
        "patch": patch,
        "spike_rel": spike,
        "mouth_rel": mouth,
        "ph": int(patch.shape[0]),
        "pw": int(patch.shape[1]),
        "mouth_size": len(mouth),
    }


def _ray_cells_to_edge(H: int, W: int, spike: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    """All cells beyond the spike up to the grid edge."""
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    ray: list[tuple[int, int]] = []
    while 0 <= rr < H and 0 <= cc < W:
        ray.append((rr, cc))
        rr += dr
        cc += dc
    return ray


def _finalize_cup(spec: dict, r0: int, c0: int, H: int, W: int) -> dict:
    """Attach absolute coordinates to a sampled cup."""
    spike_abs = (r0 + int(spec["spike_rel"][0]), c0 + int(spec["spike_rel"][1]))
    patch = spec["patch"]
    ph, pw = patch.shape
    border_abs = {
        (r0 + int(r), c0 + int(c))
        for r, c in np.argwhere(patch != 0)
    }
    return {
        "border_color": int(spec["border_color"]),
        "content_color": int(spec["content_color"]),
        "direction": str(spec["direction"]),
        "patch": patch.copy(),
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "spike_abs": spike_abs,
        "spike_rel": tuple(spec["spike_rel"]),
        "border_abs": border_abs,
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in spec["mouth_rel"]],
        "content_cap": min(CONTENT_COUNT_MAX, len(spec["mouth_rel"])),
        "full_ray": _ray_cells_to_edge(H, W, spike_abs, str(spec["direction"])),
    }


def _ray_profile(full_ray: list[tuple[int, int]], border_union: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], int | None]:
    """Paintable cells in order, plus min content count that forces at least one border skip."""
    paintable: list[tuple[int, int]] = []
    min_skip_count: int | None = None
    for idx, cell in enumerate(full_ray):
        if cell in border_union:
            if min_skip_count is None and any(later not in border_union for later in full_ray[idx + 1:]):
                min_skip_count = len(paintable) + 1
            continue
        paintable.append(cell)
    return paintable, min_skip_count


def _distance_value(bbox: tuple[int, int, int, int], H: int, W: int) -> int:
    return _cup_order_key(bbox, H, W)[0]


def _find_interacting_pair_placement(
    rng: np.random.Generator,
    H: int,
    W: int,
    spec_a: dict,
    spec_b: dict,
    layout: str,
) -> tuple[dict, dict] | None:
    """
    Place a near-center cup A and a farther cup B on the same axis.
    Both rays must skip foreign border cells and also overlap on paintable cells.
    """
    ph_a, pw_a = int(spec_a["ph"]), int(spec_a["pw"])
    ph_b, pw_b = int(spec_b["ph"]), int(spec_b["pw"])
    sr_a, sc_a = spec_a["spike_rel"]
    sr_b, sc_b = spec_b["spike_rel"]
    candidates: list[tuple[tuple[int, int, int], dict, dict]] = []

    def consider(r_a: int, c_a: int, r_b: int, c_b: int) -> None:
        cup_a = _finalize_cup(spec_a, r_a, c_a, H, W)
        cup_b = _finalize_cup(spec_b, r_b, c_b, H, W)
        dist_a = _distance_value(cup_a["bbox"], H, W)
        dist_b = _distance_value(cup_b["bbox"], H, W)
        if dist_a >= dist_b:
            return

        border_union = cup_a["border_abs"] | cup_b["border_abs"]
        paintable_a, min_skip_a = _ray_profile(cup_a["full_ray"], border_union)
        paintable_b, min_skip_b = _ray_profile(cup_b["full_ray"], border_union)
        if min_skip_a is None or min_skip_b is None:
            return
        cap_a = int(cup_a["content_cap"])
        cap_b = int(cup_b["content_cap"])
        if min_skip_a > cap_a or min_skip_b > cap_b:
            return

        set_b_cache: dict[int, set[tuple[int, int]]] = {}
        best_local: tuple[int, int, int] | None = None
        best_counts: tuple[int, int] | None = None
        for count_a in range(min_skip_a, cap_a + 1):
            cells_a = set(paintable_a[:count_a])
            for count_b in range(min_skip_b, cap_b + 1):
                cells_b = set_b_cache.setdefault(count_b, set(paintable_b[:count_b]))
                overlap = len(cells_a & cells_b)
                if overlap == 0:
                    continue
                _, skips_a = _ejection_cells_from_ray(cup_a["full_ray"], border_union, count_a)
                _, skips_b = _ejection_cells_from_ray(cup_b["full_ray"], border_union, count_b)
                score = (overlap, skips_a + skips_b, -(count_a + count_b))
                if best_local is None or score > best_local:
                    best_local = score
                    best_counts = (count_a, count_b)

        if best_counts is None:
            return

        cup_a["content_count"] = int(best_counts[0])
        cup_b["content_count"] = int(best_counts[1])
        cup_a["pair_role"] = "near"
        cup_b["pair_role"] = "far"
        candidates.append((best_local, cup_a, cup_b))

    if layout in ("right", "left"):
        row_lo = max(int(sr_a), int(sr_b))
        row_hi = min(H - ph_a + int(sr_a), H - ph_b + int(sr_b))
        rows = list(range(row_lo, row_hi + 1))
        rng.shuffle(rows)
        for spike_row in rows:
            r_a = spike_row - int(sr_a)
            r_b = spike_row - int(sr_b)
            cols_a = list(range(W - pw_a + 1))
            cols_b = list(range(W - pw_b + 1))
            rng.shuffle(cols_a)
            rng.shuffle(cols_b)
            for c_a in cols_a:
                for c_b in cols_b:
                    if layout == "right":
                        if c_b < c_a + pw_a + 1:
                            continue
                    else:
                        if c_a < c_b + pw_b + 1:
                            continue
                    consider(r_a, c_a, r_b, c_b)
    else:
        col_lo = max(int(sc_a), int(sc_b))
        col_hi = min(W - pw_a + int(sc_a), W - pw_b + int(sc_b))
        cols = list(range(col_lo, col_hi + 1))
        rng.shuffle(cols)
        for spike_col in cols:
            c_a = spike_col - int(sc_a)
            c_b = spike_col - int(sc_b)
            rows_a = list(range(H - ph_a + 1))
            rows_b = list(range(H - ph_b + 1))
            rng.shuffle(rows_a)
            rng.shuffle(rows_b)
            for r_a in rows_a:
                for r_b in rows_b:
                    if layout == "down":
                        if r_b < r_a + ph_a + 1:
                            continue
                    else:
                        if r_a < r_b + ph_b + 1:
                            continue
                    consider(r_a, c_a, r_b, c_b)

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    _, cup_a, cup_b = best[int(rng.integers(0, len(best)))]
    return cup_a, cup_b


def _try_place_clear_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    spec: dict,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Place an extra cup whose full ray stays clear of existing cups and rays."""
    ph, pw = int(spec["ph"]), int(spec["pw"])
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)
    coords = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords)

    for r0, c0 in coords:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        trial = _finalize_cup(spec, r0, c0, H, W)
        full_ray = trial["full_ray"]
        if not full_ray:
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for (r, c) in full_ray):
            continue
        trial["content_count"] = int(rng.integers(1, int(trial["content_cap"]) + 1))
        trial["pair_role"] = "clear"
        return trial
    return None


def _draw_borders(canvas: np.ndarray, cups_meta: list[dict]) -> None:
    for meta in cups_meta:
        for r, c in meta["border_abs"]:
            canvas[r, c] = int(meta["border_color"])


def _prepare_paths(cups_meta: list[dict], require_interaction: bool) -> dict | None:
    """Build border-union-aware rays and reject degenerate instances."""
    border_union = set().union(*(meta["border_abs"] for meta in cups_meta))
    for meta in cups_meta:
        meta["border_union"] = border_union
        ejection_cells, used_skips = _ejection_cells_from_ray(
            meta["full_ray"],
            border_union,
            int(meta["content_count"]),
        )
        meta["ejection_cells"] = ejection_cells
        meta["used_skip_count"] = int(used_skips)

    cell_counts = Counter(cell for meta in cups_meta for cell in meta["ejection_cells"])
    overlap_cells = {cell for cell, cnt in cell_counts.items() if cnt > 1}
    all_ejection_cells = set(cell_counts)
    total_used_skips = sum(int(meta["used_skip_count"]) for meta in cups_meta)

    for meta in cups_meta:
        available = [cell for cell in meta["mouth_abs"] if cell not in all_ejection_cells]
        if len(available) < int(meta["content_count"]):
            return None
        meta["available_mouth_abs"] = available

    if require_interaction and (total_used_skips == 0 or len(overlap_cells) == 0):
        return None

    return {
        "border_union": border_union,
        "overlap_cells": overlap_cells,
        "all_ejection_cells": all_ejection_cells,
        "total_used_skips": int(total_used_skips),
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's content inside mouth cells that remain visually clean."""
    for meta in cups_meta:
        mouth_abs = meta["available_mouth_abs"]
        count = int(meta["content_count"])
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        chosen = [mouth_abs[int(i)] for i in idxs]
        meta["content_abs"] = chosen
        for r, c in chosen:
            canvas_in[r, c] = int(meta["content_color"])
    return True


def _simulate_output(canvas_in: np.ndarray, cups_meta: list[dict], reverse_order: bool) -> np.ndarray:
    """Simulate the fusion solver using metadata, optionally reversing the sorted order."""
    out = np.array(canvas_in, dtype=int)
    H, W = out.shape
    bg = int(get_bg(out))
    cups = list(cups_meta)
    cups.sort(key=lambda cup: (*_cup_order_key(cup["bbox"], H, W), int(cup["border_color"])))
    if reverse_order:
        cups.reverse()

    for meta in cups_meta:
        for r, c in meta["content_abs"]:
            out[r, c] = bg

    for meta in cups:
        for r, c in meta["ejection_cells"]:
            out[r, c] = int(meta["content_color"])
    return out


def _distance_keys_are_unique(cups_meta: list[dict], H: int, W: int) -> bool:
    keys = [_distance_value(meta["bbox"], H, W) for meta in cups_meta]
    return len(keys) == len(set(keys))


def _instance_metrics(canvas_in: np.ndarray) -> dict:
    """Quantify whether a generated instance truly uses the fused behavior."""
    g = np.array(canvas_in, dtype=int)
    H, W = g.shape
    bg, cups = _detect_cups(g)
    border_union = set().union(*(cup["border_set"] for cup in cups))
    overlap_counter: Counter[tuple[int, int]] = Counter()
    total_skips = 0
    for cup in cups:
        ejection_cells, used_skips = _ordered_ejection_cells(
            H,
            W,
            cup["spike"],
            cup["tail_dir"],
            len(cup["content_cells"]),
            border_union,
        )
        total_skips += int(used_skips)
        overlap_counter.update(ejection_cells)

    ordered_out = main(g)
    reverse_out = np.array(g, dtype=int)
    for cup in cups:
        for r, c in cup["content_cells"]:
            reverse_out[r, c] = bg
    cups_sorted = sorted(cups, key=lambda cup: (*_cup_order_key(cup["bbox"], H, W), int(cup["border_color"])))
    for cup in reversed(cups_sorted):
        ejection_cells, _ = _ordered_ejection_cells(
            H,
            W,
            cup["spike"],
            cup["tail_dir"],
            len(cup["content_cells"]),
            border_union,
        )
        for r, c in ejection_cells:
            reverse_out[r, c] = int(cup["content_col"])

    return {
        "num_cups": len(cups),
        "used_border_skips": int(total_skips),
        "overlap_cells": sum(1 for cnt in overlap_counter.values() if cnt > 1),
        "order_diff_pixels": int(np.count_nonzero(ordered_out != reverse_out)),
    }


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Build one fused instance with both distance-order and border-skip interaction."""
    if num_cups is None:
        num_cups = int(rng.choice(FALLBACK_NUM_CUPS_CHOICES))
    if num_cups < 2:
        return None

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    palette = list(range(1, 10))
    rng.shuffle(palette)
    if len(palette) < 2 * num_cups:
        return None

    colors = [(int(palette[2 * i]), int(palette[2 * i + 1])) for i in range(num_cups)]
    layout_to_dirs = {
        "right": ("R", "L"),
        "left": ("L", "R"),
        "down": ("D", "U"),
        "up": ("U", "D"),
    }
    layout = str(rng.choice(list(layout_to_dirs)))
    dir_a, dir_b = layout_to_dirs[layout]

    spec_a = _build_cup_spec(rng, colors[0][0], colors[0][1], dir_a)
    spec_b = _build_cup_spec(rng, colors[1][0], colors[1][1], dir_b)
    if spec_a is None or spec_b is None:
        return None

    pair = _find_interacting_pair_placement(rng, H, W, spec_a, spec_b, layout)
    if pair is None:
        return None
    cups_meta = [pair[0], pair[1]]

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    for meta in cups_meta:
        r0, c0, r1, c1 = meta["bbox"]
        occ_cup[r0:r1 + 1, c0:c1 + 1] = True
        for r, c in meta["full_ray"]:
            occ_path[r, c] = True

    for idx in range(2, num_cups):
        for _ in range(50):
            direction = str(rng.choice(["U", "D", "L", "R"]))
            spec = _build_cup_spec(rng, colors[idx][0], colors[idx][1], direction)
            if spec is None:
                continue
            extra = _try_place_clear_cup(rng, H, W, spec, occ_cup, occ_path)
            if extra is None:
                continue
            cups_meta.append(extra)
            r0, c0, r1, c1 = extra["bbox"]
            occ_cup[r0:r1 + 1, c0:c1 + 1] = True
            for r, c in extra["full_ray"]:
                occ_path[r, c] = True
            break
        else:
            return None

    if len(cups_meta) != num_cups:
        return None
    if not _distance_keys_are_unique(cups_meta, H, W):
        return None

    prep = _prepare_paths(cups_meta, require_interaction=True)
    if prep is None:
        return None

    canvas_in = np.zeros((H, W), dtype=int)
    _draw_borders(canvas_in, cups_meta)
    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    ordered_out = _simulate_output(canvas_in, cups_meta, reverse_order=False)
    reverse_out = _simulate_output(canvas_in, cups_meta, reverse_order=True)
    if np.array_equal(ordered_out, reverse_out):
        return None

    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, ordered_out):
        return None

    metrics = _instance_metrics(canvas_in)
    if metrics["used_border_skips"] <= 0 or metrics["order_diff_pixels"] <= 0 or metrics["overlap_cells"] <= 0:
        return None
    return canvas_in


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    num_cups = int(rng.choice(PRIMARY_NUM_CUPS_CHOICES))
    for _ in range(GEN_RETRIES_PRIMARY):
        result = _gen_attempt(rng, num_cups=num_cups)
        if result is not None:
            return result
    for _ in range(GEN_RETRIES_FALLBACK):
        result = _gen_attempt(rng, num_cups=int(rng.choice(FALLBACK_NUM_CUPS_CHOICES)))
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
