# Variant of 3dc255db
# Dimension: D11 — 杯的数量与空间关系 (meta)
# Change: 1-3个杯，不重叠(bbox间至少1格间距)，弹射路径无障碍... → 嵌套杯
#
# Design analysis
# ===============
# a) 具体化与行为追踪
# - 变体固定至少出现 1 对嵌套杯：一个大杯和一个小杯。小杯完全位于大杯的 mouth
#   interior 内部，但两者边框颜色不同、边框互不接触。
# - 为了让“内杯先弹射，可能影响外杯内容”这一点既可见又不破坏外杯边框，嵌套对采用
#   相反的尾方向：外杯朝某一方向收尖，内杯朝相反方向收尖。于是内杯的弹射线会沿着
#   外杯的开口方向穿出，穿过外杯 mouth interior 的一部分内容像素，但不会撞上外杯边框。
# - 小例子（外杯 tail=U, 内杯 tail=D）：
#   1. 原始逻辑：把每个杯当作独立对象，从输入静态地数 mouth 里的内容数；外杯会把
#      所有自己的内容都向上弹出。
#   2. 变体逻辑：先处理内杯，内杯把自己颜色的直线向下画出，这条线会覆盖外杯口内
#      的若干外杯内容；再处理外杯时，外杯只能看到“剩下还没被覆盖的外杯内容”，
#      因此外杯最终弹出的线更短。
#   3. 差异：处理顺序不再可交换；同一输入下，若忽略嵌套顺序，输出会不同。
# - 受影响的环节：
#   1. solver 不能再“按颜色组件独立立即执行”，必须先收集所有杯，再按嵌套深度排序。
#   2. generator 必须构造真正发生覆盖的样例，而不是仅仅把一个小杯随便放进大杯内部。
#
# b) 复杂度校准
# - 只引入一种新的空间关系：单层嵌套对。样例中总杯数为 2 或 3：1 对嵌套杯 + 可选 1 个
#   独立杯。这样既能稳定展示 D11 的变化，又不会引入多级树状层级导致推理过难。
# - 嵌套对的方向关系固定为相反方向，减少自由度；但外杯整体朝向仍随机 U/D/L/R，避免
#   样例退化成单一朝向模板。
#
# c) 原 solver 的隐式假设审查
# - `for border_col in colors_present(g, exclude={bg})`: 原题默认杯彼此独立，边框检测后即可
#   立即执行。变体中仍可用来找候选边框，但不能边找边改输出，否则外杯会在内杯之前执行。
# - `passing` 使用 mouth 中的非 bg / 非 border 颜色来确认 tail：原题 mouth 中只有本杯内容。
#   变体中外杯 mouth 内还会出现内杯的边框、内杯内容和内杯弹射线，直接依赖 mouth 颜色会混淆。
#   因此此处改为主要依赖几何：canonical 生成族保证只有一个 bbox edge 是单像素 spike。
# - `content_cells = [p for p in mouth if g[p] == content_col]`: 原题默认 mouth 中的非本边框像素
#   都可以视为本杯内容候选。变体中这不成立，因为外杯 mouth 里还包含内杯边框和内杯弹射线。
#   新 solver 需显式排除：
#   1. 其他杯的边框像素。
#   2. 已经由更内层杯生成的弹射线像素。
# - `out[rr, cc] = content_col`: 原题无覆盖顺序问题。变体里保留覆盖行为是必要的，因为它正是
#   外杯内容被影响的机制；generator 需要保证这类覆盖落在外杯内容上而不是外杯边框上。
#
# d) 放置合法性
# - 外杯先放，内杯在外杯 mouth interior 内部搜索合法偏移。
# - 内杯的 border 和 mouth 全部必须位于外杯 mouth interior 内。
# - 内杯弹射路径在外杯 bbox 内部经过的所有格都必须仍属于外杯 mouth interior，从而保证
#   它穿过的是外杯内部/开口，而不是外杯边框。
# - 外杯与独立杯仍保持原题的 bbox 间隔和路径不冲突；嵌套对作为一个整体占用外杯 bbox。
#
# e) 视觉可观察性
# - 变体的可见线索是：一个小杯真正“坐在”大杯口内，而且 output 中内杯的线会穿过大杯口，
#   同时外杯自己的弹射线比静态数出的内容更短。
# - generator 强制至少 1 个被覆盖的外杯内容像素，并拒绝 `new_main(input) == flat_main(input)`
#   的退化样例。
#
# f) 可视化整洁度
# - 取消 stray pixel；嵌套已经提供足够的复杂度，额外噪声会让杯内外关系更难辨认。
# - 所有嵌套干扰都限制在外杯 mouth 区域内，不允许破坏外杯边框。
#
# g) 颜色变化可推断性
# - 不增加新的颜色映射规则。每个杯仍是“边框色 + 内容色”两色对象。
# - 生成器默认给嵌套对使用互不相同的 4 个颜色，避免外杯内容与内杯边框/内杯线颜色混淆。
#
# h) main() 的新增假设
# - 每个杯仍来自原 canonical 生成族，因此 tail spike 对应的 bbox 边是唯一单像素边。
# - 嵌套关系是单层包含，按“某杯边框整体位于另一杯 mouth interior 内”来判定。
# - 祖先杯的真实内容只会出现在“自身 mouth 区域减去后代边框与后代已发射线”后的剩余格内；
#   generator 按此构造。
#
# i) 生成器如何保证这些假设
# - 外杯 shape 要足够大，内杯 shape 要足够小。
# - 内杯方向固定为外杯方向的 opposite。
# - 至少 1 个外杯内容像素被内杯前缀弹射线覆盖，且外杯仍保留至少 2 个内容像素用于后续弹射。

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-conn flood within bbox, seeded from the requested bbox side, stopping at border cells."""
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


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical funnel/Y cup with tail pointing UP."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    g = np.zeros((h, w), dtype=int)
    g[0, L] = color
    g[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    g[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        g[L + 1 : L + hL + 1, 0] = color
    if hR:
        g[R + 1 : R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
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
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
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


def _build_cup_instance(rng: np.random.Generator, direction: str, border_color: int, role: str) -> dict | None:
    for _ in range(80):
        shape = _sample_cup_shape(rng, direction)
        if shape is None:
            continue
        L_arm, R_arm, hL, hR = shape
        patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
        mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
        patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
        ph, pw = patch.shape
        border_rel = [(int(r), int(c)) for r, c in np.argwhere(patch != 0)]
        footprint_rel = set(border_rel) | set(mouth)

        if role == "outer":
            if ph < 5 or pw < 5 or len(mouth) < 10:
                continue
        else:
            if ph > 5 or pw > 5 or len(mouth) < 3 or len(mouth) > 6:
                continue

        return {
            "patch": patch,
            "spike": spike,
            "mouth_rel": list(mouth),
            "border_rel": border_rel,
            "footprint_rel": footprint_rel,
            "shape": (ph, pw),
            "direction": direction,
            "border_color": border_color,
        }
    return None


def _full_path_from_spike(H: int, W: int, spike: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    path = []
    while 0 <= rr < H and 0 <= cc < W:
        path.append((rr, cc))
        rr += dr
        cc += dc
    return path


def _add_abs(offset: tuple[int, int], coords: list[tuple[int, int]] | set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0 = offset
    return [(r0 + r, c0 + c) for (r, c) in coords]


def _detect_cups(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    cups: list[dict] = []

    for border_col in colors_present(grid, exclude={bg}):
        for border_coords in components_mask(grid == int(border_col), connectivity=8):
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
            candidates = [(d, cells[0]) for d, cells in edges if len(cells) == 1 and cells[0] not in corners]
            if len(candidates) != 1:
                continue

            tail_dir, spike = candidates[0]
            mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
            if len(mouth) < 3:
                continue

            cups.append(
                {
                    "border_color": int(border_col),
                    "bbox": bbox,
                    "border_set": border_set,
                    "mouth": mouth,
                    "tail_dir": tail_dir,
                    "spike": spike,
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )

    for i, cup in enumerate(cups):
        parents = []
        for j, other in enumerate(cups):
            if i == j:
                continue
            if cup["border_set"].issubset(other["mouth"]):
                parents.append(j)
        cup["parents"] = parents

    def depth(idx: int) -> int:
        parents = cups[idx]["parents"]
        if not parents:
            return 0
        return 1 + max(depth(p) for p in parents)

    for i, cup in enumerate(cups):
        cup["depth"] = depth(i)

    return cups


def _processing_order(cups: list[dict]) -> list[int]:
    return sorted(range(len(cups)), key=lambda i: (cups[i]["depth"], -cups[i]["area"]), reverse=True)


def _apply_nested_logic(grid: np.ndarray, nested: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    cups = _detect_cups(g)
    out = g.copy()

    if not cups:
        return out

    all_borders = set().union(*(cup["border_set"] for cup in cups))
    emitted_cells: set[tuple[int, int]] = set()
    order = _processing_order(cups) if nested else list(range(len(cups)))

    for idx in order:
        cup = cups[idx]
        blocked = (all_borders - cup["border_set"]) | emitted_cells if nested else set()
        eligible = [p for p in cup["mouth"] if p not in blocked]

        cnt = Counter()
        for r, c in eligible:
            v = int(out[r, c] if nested else g[r, c])
            if v != bg and v != cup["border_color"]:
                cnt[v] += 1
        if not cnt:
            continue

        content_col = int(cnt.most_common(1)[0][0])
        source = out if nested else g
        content_cells = [p for p in eligible if int(source[p[0], p[1]]) == content_col]

        for r, c in content_cells:
            out[r, c] = bg

        dr, dc = DIR4[cup["tail_dir"]]
        sr, sc = cup["spike"]
        for t in range(1, len(content_cells) + 1):
            rr = sr + dr * t
            cc = sc + dc * t
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[rr, cc] = content_col
            if nested:
                emitted_cells.add((rr, cc))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Nested-cup variant of 3dc255db.

    Rule change:
    - Detect every cup border first.
    - If one cup lies inside another cup's mouth interior, process the inner cup first.
    - Descendant emission lines are treated as transient foreign pixels when processing ancestors,
      so an inner cup can reduce the amount of ancestor content that survives to be ejected.
    """
    return _apply_nested_logic(grid, nested=True)


def _flat_main(grid: np.ndarray) -> np.ndarray:
    """Baseline close to the original independent-cup behavior, used to reject degenerate samples."""
    return _apply_nested_logic(grid, nested=False)


def _variant_metrics(grid: np.ndarray) -> dict:
    cups = _detect_cups(np.array(grid, dtype=int))
    nested_edges = sum(len(cup["parents"]) for cup in cups)
    if not cups:
        return {"nested_edges": 0, "max_loss": 0, "diff_vs_flat": False}

    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    all_borders = set().union(*(cup["border_set"] for cup in cups))
    descendant_footprints: list[set[tuple[int, int]]] = []
    for i, cup in enumerate(cups):
        desc = set()
        for j, other in enumerate(cups):
            if i == j:
                continue
            if i in other["parents"]:
                desc |= other["border_set"]
                desc |= other["mouth"]
        descendant_footprints.append(desc)

    emitted: set[tuple[int, int]] = set()
    max_loss = 0
    for idx in _processing_order(cups):
        cup = cups[idx]
        exclusive = [p for p in cup["mouth"] if p not in descendant_footprints[idx]]
        raw_cnt = Counter()
        for r, c in exclusive:
            v = int(g[r, c])
            if v != bg and v != cup["border_color"]:
                raw_cnt[v] += 1
        raw_best = raw_cnt.most_common(1)[0][1] if raw_cnt else 0

        blocked = (all_borders - cup["border_set"]) | emitted
        eligible = [p for p in cup["mouth"] if p not in blocked]
        cnt = Counter()
        for r, c in eligible:
            v = int(out[r, c])
            if v != bg and v != cup["border_color"]:
                cnt[v] += 1
        effective = cnt.most_common(1)[0][1] if cnt else 0
        max_loss = max(max_loss, max(0, raw_best - effective))

        if cnt:
            color = int(cnt.most_common(1)[0][0])
            cells = [p for p in eligible if int(out[p[0], p[1]]) == color]
            for r, c in cells:
                out[r, c] = bg
            sr, sc = cup["spike"]
            dr, dc = DIR4[cup["tail_dir"]]
            for t in range(1, len(cells) + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                    break
                out[rr, cc] = color
                emitted.add((rr, cc))

    diff_vs_flat = not np.array_equal(main(g), _flat_main(g))
    return {"nested_edges": nested_edges, "max_loss": int(max_loss), "diff_vs_flat": diff_vs_flat}


def _try_place_independent_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_color: int,
    canvas: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    direction = str(rng.choice(["U", "D", "L", "R"]))
    inst = _build_cup_instance(rng, direction, border_color, role="inner")
    if inst is None:
        return None

    ph, pw = inst["shape"]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)
    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        if occ_path[r0 : r0 + ph, c0 : c0 + pw].any():
            continue

        spike_abs = (r0 + inst["spike"][0], c0 + inst["spike"][1])
        path_cells = _full_path_from_spike(H, W, spike_abs, direction)
        if len(path_cells) < 2:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        border_abs = _add_abs((r0, c0), inst["border_rel"])
        mouth_abs = _add_abs((r0, c0), inst["mouth_rel"])
        count_cap = min(CONTENT_COUNT_MAX, len(mouth_abs), len(path_cells))
        if count_cap < 1:
            continue
        content_count = int(rng.integers(1, count_cap + 1))
        chosen = rng.choice(len(mouth_abs), size=content_count, replace=False)
        content_cells = [mouth_abs[int(i)] for i in chosen]

        for r, c in border_abs:
            canvas[r, c] = border_color
        for r, c in content_cells:
            canvas[r, c] = content_color
        occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True
        for r, c in path_cells:
            occ_path[r, c] = True

        return {
            "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
            "border_abs": border_abs,
            "path_cells": path_cells,
        }

    return None


def _try_place_nested_pair(
    rng: np.random.Generator, H: int, W: int, outer_border: int, outer_content: int, inner_border: int, inner_content: int
) -> dict | None:
    outer_direction = str(rng.choice(["U", "D", "L", "R"]))
    inner_direction = OPPOSITE[outer_direction]
    outer = _build_cup_instance(rng, outer_direction, outer_border, role="outer")
    inner = _build_cup_instance(rng, inner_direction, inner_border, role="inner")
    if outer is None or inner is None:
        return None

    oph, opw = outer["shape"]
    iph, ipw = inner["shape"]
    outer_offsets = [(r0, c0) for r0 in range(H - oph + 1) for c0 in range(W - opw + 1)]
    rng.shuffle(outer_offsets)

    for or0, oc0 in outer_offsets:
        outer_border_abs = set(_add_abs((or0, oc0), outer["border_rel"]))
        outer_mouth_abs = set(_add_abs((or0, oc0), outer["mouth_rel"]))
        outer_path = _full_path_from_spike(H, W, (or0 + outer["spike"][0], oc0 + outer["spike"][1]), outer_direction)
        if len(outer_path) < 3:
            continue

        inner_offsets = [(ir0, ic0) for ir0 in range(or0, or0 + oph - iph + 1) for ic0 in range(oc0, oc0 + opw - ipw + 1)]
        rng.shuffle(inner_offsets)
        for ir0, ic0 in inner_offsets:
            inner_border_abs = set(_add_abs((ir0, ic0), inner["border_rel"]))
            inner_mouth_abs = set(_add_abs((ir0, ic0), inner["mouth_rel"]))
            inner_footprint_abs = set(_add_abs((ir0, ic0), inner["footprint_rel"]))
            if not inner_footprint_abs.issubset(outer_mouth_abs):
                continue

            inner_spike_abs = (ir0 + inner["spike"][0], ic0 + inner["spike"][1])
            inner_path = _full_path_from_spike(H, W, inner_spike_abs, inner_direction)
            if len(inner_path) < 2:
                continue

            inside_outer = [p for p in inner_path if or0 <= p[0] <= or0 + oph - 1 and oc0 <= p[1] <= oc0 + opw - 1]
            if not inside_outer:
                continue
            if any(p not in outer_mouth_abs for p in inside_outer):
                continue

            count_cap_inner = min(CONTENT_COUNT_MAX, len(inner_mouth_abs), len(inner_path))
            if count_cap_inner < 1:
                continue

            non_overlap_outer = [p for p in outer_mouth_abs if p not in inner_footprint_abs]
            if len(non_overlap_outer) < 3:
                continue

            inner_count_options = list(range(1, count_cap_inner + 1))
            rng.shuffle(inner_count_options)
            for inner_count in inner_count_options:
                emitted_inside_outer = [p for p in inner_path[:inner_count] if p in outer_mouth_abs and p not in inner_footprint_abs]
                if not emitted_inside_outer:
                    continue
                overlap_count = int(rng.integers(1, min(2, len(emitted_inside_outer)) + 1))
                overlap_cells = emitted_inside_outer[:overlap_count]
                remaining_candidates = [p for p in non_overlap_outer if p not in overlap_cells]
                max_remaining = min(4, len(remaining_candidates), len(outer_path) - overlap_count)
                if max_remaining < 2:
                    continue
                remaining_count = int(rng.integers(2, max_remaining + 1))
                outer_count = overlap_count + remaining_count
                if outer_count > min(CONTENT_COUNT_MAX, len(non_overlap_outer), len(outer_path)):
                    continue

                inner_mouth_list = sorted(inner_mouth_abs)
                outer_rest = list(remaining_candidates)
                rng.shuffle(inner_mouth_list)
                rng.shuffle(outer_rest)
                outer_content_cells = list(overlap_cells) + outer_rest[:remaining_count]
                inner_content_cells = inner_mouth_list[:inner_count]

                return {
                    "outer": {
                        "border_color": outer_border,
                        "content_color": outer_content,
                        "bbox": (or0, oc0, or0 + oph - 1, oc0 + opw - 1),
                        "border_abs": sorted(outer_border_abs),
                        "content_abs": outer_content_cells,
                        "path_cells": outer_path,
                    },
                    "inner": {
                        "border_color": inner_border,
                        "content_color": inner_content,
                        "bbox": (ir0, ic0, ir0 + iph - 1, ic0 + ipw - 1),
                        "border_abs": sorted(inner_border_abs),
                        "content_abs": inner_content_cells,
                        "path_cells": inner_path,
                    },
                    "overlap_cells": overlap_cells,
                }

    return None


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    if num_cups == 2 and len(palette) < 4:
        return None
    if num_cups == 3 and len(palette) < 6:
        return None

    outer_border, outer_content, inner_border, inner_content = [int(palette.pop()) for _ in range(4)]
    nested = _try_place_nested_pair(rng, H, W, outer_border, outer_content, inner_border, inner_content)
    if nested is None:
        return None

    for meta in (nested["outer"], nested["inner"]):
        for r, c in meta["border_abs"]:
            canvas[r, c] = meta["border_color"]
        for r, c in meta["content_abs"]:
            canvas[r, c] = meta["content_color"]

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    or0, oc0, or1, oc1 = nested["outer"]["bbox"]
    occ_cup[or0 : or1 + 1, oc0 : oc1 + 1] = True
    for path_cells in (nested["outer"]["path_cells"], nested["inner"]["path_cells"]):
        for r, c in path_cells:
            occ_path[r, c] = True

    if num_cups == 3:
        extra_border = int(palette.pop())
        extra_content = int(palette.pop())
        extra = _try_place_independent_cup(rng, H, W, extra_border, extra_content, canvas, occ_cup, occ_path)
        if extra is None:
            return None

    try:
        new_out = main(canvas)
        flat_out = _flat_main(canvas)
    except Exception:
        return None

    metrics = _variant_metrics(canvas)
    if metrics["nested_edges"] < 1 or metrics["max_loss"] < 1:
        return None
    if np.array_equal(new_out, flat_out) or not metrics["diff_vs_flat"]:
        return None
    return canvas


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    for _ in range(GEN_RETRIES_PRIMARY):
        result = _gen_attempt(rng, num_cups=num_cups)
        if result is not None:
            return result
    for _ in range(GEN_RETRIES_FALLBACK):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
