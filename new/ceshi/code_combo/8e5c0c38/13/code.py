# Combo variant of 8e5c0c38 (index 13)
# combo_spec: D3/2+D6/0
# Name: 多色四重对称修复

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis
# a) 具体化与行为追踪
#    - 现在每个对象是一个 4 连通的双色精灵，而不是“一个颜色层 = 一个对象”。
#      clean 输出里的每个精灵同时满足水平轴和垂直轴镜像；两种颜色都在这对轴下保持不变。
#    - 小例子：某个精灵的联合形状是四重对称，主色形成外框，次色只占中心附近 4 个格子。
#      输入里额外插入一对次色竖向噪声点，它们左右成对，但只出现在水平轴上方。
#      原题逻辑只按颜色层找竖轴，会把这对次色点保留下来；只做 D3 的共享竖轴版本也会保留它们。
#      融合版先把整个双色连通精灵当作一个对象，再搜索共享的横纵轴对，并要求四重轨道里的所有位置都存在且同色，
#      所以这对缺少上下镜像的次色点会被删除。
# b) 复杂度校准
#    - 继续保留 22x22、3 个分离 slot、output-first 生成，避免引入额外布局变量。
#    - 每个精灵恰好 2 色，6 个对象色全局互不复用，减少跨对象颜色混淆。
#    - 次色优先分配到靠近中心、面积较小的四重轨道上，使“共享轴”和“颜色也必须匹配”都能从样例中被看见。
#    - 噪声不仅有单点，也强制至少出现一次“左右成对但上下不全”的碎片，确保 D6 的增强是可观察的。
# c) 隐式假设审查（对应原 main 的关键前提）
#    - `colors_present(...)` 不再能直接代表对象集合；融合后必须先按非背景 4 连通分量分对象。
#    - 原来 `pts = np.argwhere(g == col)` 默认“同色像素共享一个对象和一条轴”。现在同一对象里有两色，所以轴搜索必须基于整个连通精灵。
#    - 原逻辑只枚举竖轴，并只检查一个左右镜像点；融合后必须枚举 `(ky, kx)`，并检查完整四重轨道。
#    - 原逻辑把“镜像位置有同色像素”当成充分条件；融合后这只满足一半，另一半还要求水平镜像也成立。
#    - 新增假设是：每个对象真的是单个连通精灵、颜色不跨对象复用、真实轴对落在该对象 bbox 范围内。
#      生成器会分别保证这三点。
# d) 放置合法性
#    - 生成器仍使用相互分离的 3 个 slot，在局部画布上先生成 clean 精灵，再整体粘贴到全局。
#    - clean 联合 mask 强制 4 连通；噪声只能放在该 mask 的 4 邻接背景里，并且 slot 间距足够大，因此不会桥接对象。
#    - 噪声轨道必须与 clean 轨道完全不相交，避免噪声借用真像素凑成合法轨道。
# e) 视觉可观察性
#    - 人眼能同时看到两个信号：同一精灵有两层颜色，而且有些多余像素即便左右成对也会被删除，因为它们没有上下同色对应。
#    - 至少一个样本会包含次色噪声，否则“噪声可落在两种颜色里”只停留在规格里而不体现在数据上。
# f) 可视化整洁度
#    - 主色做大块外形，次色集中在中心或内部轨道，层次稳定，不会出现六种颜色在同一局部打架的效果。
#    - 三个对象保持大间隔，噪声紧贴所属精灵，观察者可以明确判断哪些点是附着在哪个对象上的。
# g) 变化的可推断性
#    - 不做颜色映射，颜色 id 本身没有语义；颜色只表示“同一精灵里的两种材料”。
#    - 规则恒定为：围绕同一对轴，只有颜色也匹配的完整四重轨道能保留。样例之间不会切换到别的对称类型。
# h) 反退化目标
#    - 生成器拒绝两类退化样本：`D3-only` 共享竖轴清理器也能完全做对，或 `D6-only` 的逐色四重清理器也能完全做对。
#      这样保留下来的输入必须同时依赖“双色共享对象”与“四重对称”两个维度。


GRID_SIZE = 22
NUM_SPRITES = 3
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 8
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
WIDTH_MIN = 3
WIDTH_MAX = 12
DENSITY_LO = 0.35
DENSITY_HI = 0.89
ACCENT_FRAC_LO = 0.14
ACCENT_FRAC_HI = 0.34

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12
NOISE_PLAN_RETRIES = 100

# ── CFG: score gap thresholds ──
CLEAN_PAIR_GAP_MIN = 4
FINAL_PAIR_GAP_MIN = 3

# ── CFG: retry limits ──
SPRITE_RETRIES = 280
GEN_RETRIES = 900

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return len(_connected_components(mask)) == 1


def _orbit_cells(
    r: int, c: int, ky: int, kx: int, height: int, width: int
) -> set[tuple[int, int]] | None:
    rows = {int(r), int(ky) - int(r)}
    cols = {int(c), int(kx) - int(c)}
    orbit: set[tuple[int, int]] = set()
    for rr in rows:
        for cc in cols:
            if not (0 <= rr < height and 0 <= cc < width):
                return None
            orbit.add((int(rr), int(cc)))
    return orbit


def _axis_ranges(coords: list[tuple[int, int]], height: int, width: int) -> tuple[range, range]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    ky_lo = max(0, 2 * min(rows))
    ky_hi = min(2 * (height - 1), 2 * max(rows))
    kx_lo = max(0, 2 * min(cols))
    kx_hi = min(2 * (width - 1), 2 * max(cols))
    return range(ky_lo, ky_hi + 1), range(kx_lo, kx_hi + 1)


def _color_complete_orbit(
    g: np.ndarray,
    orbit: set[tuple[int, int]] | None,
    pt_set: set[tuple[int, int]],
    color: int,
) -> bool:
    if orbit is None or not (orbit <= pt_set):
        return False
    return all(int(g[rr, cc]) == int(color) for rr, cc in orbit)


def _component_fourfold_score(
    g: np.ndarray, coords: list[tuple[int, int]], ky: int, kx: int, height: int, width: int
) -> int:
    pt_set = {(int(r), int(c)) for r, c in coords}
    score = 0
    for r, c in pt_set:
        orbit = _orbit_cells(r, c, ky, kx, height, width)
        if _color_complete_orbit(g, orbit, pt_set, int(g[r, c])):
            score += 1
    return score


def _best_component_axes(
    g: np.ndarray, coords: list[tuple[int, int]], height: int, width: int
) -> tuple[int, int, int]:
    ky_range, kx_range = _axis_ranges(coords, height, width)
    best_score = -1
    best_ky = 0
    best_kx = 0
    for ky in ky_range:
        for kx in kx_range:
            score = _component_fourfold_score(g, coords, ky, kx, height, width)
            if score > best_score:
                best_score = int(score)
                best_ky = int(ky)
                best_kx = int(kx)
    return best_ky, best_kx, best_score


def _second_best_component_score(
    g: np.ndarray,
    coords: list[tuple[int, int]],
    true_pair: tuple[int, int],
    height: int,
    width: int,
) -> int:
    ky_range, kx_range = _axis_ranges(coords, height, width)
    best_other = -1
    for ky in ky_range:
        for kx in kx_range:
            if (ky, kx) == true_pair:
                continue
            score = _component_fourfold_score(g, coords, ky, kx, height, width)
            if score > best_other:
                best_other = int(score)
    return best_other


def _per_color_fourfold_score(
    pts: set[tuple[int, int]], ky: int, kx: int, height: int, width: int
) -> int:
    score = 0
    for r, c in pts:
        orbit = _orbit_cells(r, c, ky, kx, height, width)
        if orbit is not None and orbit <= pts:
            score += 1
    return score


def _best_axes_for_points(
    pts: set[tuple[int, int]], height: int, width: int
) -> tuple[int, int, int]:
    coords = sorted(pts)
    ky_range, kx_range = _axis_ranges(coords, height, width)
    best_score = -1
    best_ky = 0
    best_kx = 0
    for ky in ky_range:
        for kx in kx_range:
            score = _per_color_fourfold_score(pts, ky, kx, height, width)
            if score > best_score:
                best_score = int(score)
                best_ky = int(ky)
                best_kx = int(kx)
    return best_ky, best_kx, best_score


def _best_vertical_axis(coords: list[tuple[int, int]], width: int) -> int:
    cols = [int(c) for _, c in coords]
    kx_range = range(max(0, 2 * min(cols)), min(2 * (width - 1), 2 * max(cols)) + 1)
    pt_set = {(int(r), int(c)) for r, c in coords}
    best_score = -1
    best_kx = 0
    for kx in kx_range:
        score = 0
        for r, c in pt_set:
            mc = int(kx) - int(c)
            if 0 <= mc < width and (int(r), mc) in pt_set:
                score += 1
        if score > best_score:
            best_score = int(score)
            best_kx = int(kx)
    return best_kx


def _legacy_color_vertical_clean(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _, width = g.shape
    out = g.copy()
    for col in colors_present(g, exclude={bg}):
        coords = [tuple(x) for x in np.argwhere(g == int(col))]
        if not coords:
            continue
        best_kx = _best_vertical_axis(coords, width)
        pt_set = {(int(r), int(c)) for r, c in coords}
        for r, c in coords:
            mc = int(best_kx) - int(c)
            if not (0 <= mc < width and (int(r), mc) in pt_set):
                out[int(r), int(c)] = int(bg)
    return out


def _legacy_multicolor_vertical_clean(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _, width = g.shape
    out = g.copy()
    comps = _connected_components(g != bg)
    for comp in comps:
        best_kx = _best_vertical_axis(comp, width)
        pt_set = {(int(r), int(c)) for r, c in comp}
        for r, c in comp:
            mc = int(best_kx) - int(c)
            if not (
                0 <= mc < width
                and (int(r), mc) in pt_set
                and int(g[int(r), mc]) == int(g[int(r), int(c)])
            ):
                out[int(r), int(c)] = int(bg)
    return out


def _legacy_per_color_fourfold_clean(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape
    out = g.copy()
    for col in colors_present(g, exclude={bg}):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(col))}
        if not pts:
            continue
        best_ky, best_kx, _ = _best_axes_for_points(pts, height, width)
        for r, c in pts:
            orbit = _orbit_cells(r, c, best_ky, best_kx, height, width)
            if orbit is None or not (orbit <= pts):
                out[int(r), int(c)] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each connected two-color sprite by enforcing one shared fourfold symmetry pair.

    Input structure:
    - The background fills most of the 22x22 grid. Three separated non-background connected components act as sprites.
    - Each sprite uses exactly two colors that belong to the same object, and a few extra pixels of either color may break the symmetry.

    Rule:
    - For each connected sprite, search for the horizontal axis and vertical axis that maximize same-color complete fourfold orbits over the whole two-color component.
    - Delete every pixel whose orbit under that shared axis pair is incomplete or contains a different color at any mirrored position.

    Key insight:
    - The axes belong to the combined sprite rather than to individual colors, but survival is stricter than occupancy symmetry alone: every kept pixel needs full fourfold support in the same color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    comps = _connected_components(g != bg)
    assert comps, "grid must contain at least one non-background sprite"

    for comp in comps:
        pt_set = {(int(r), int(c)) for r, c in comp}
        best_ky, best_kx, _ = _best_component_axes(g, comp, height, width)
        for r, c in comp:
            orbit = _orbit_cells(r, c, best_ky, best_kx, height, width)
            color = int(g[int(r), int(c)])
            if not _color_complete_orbit(g, orbit, pt_set, color):
                out[int(r), int(c)] = int(bg)

    return out


def _canonical_cells(top_h: int, left_w: int) -> set[tuple[int, int]]:
    forced = {(0, 0)}
    if left_w > 1:
        forced.add((0, left_w - 1))
    if top_h > 1:
        forced.add((top_h - 1, 0))
    if top_h > 1 and left_w > 1:
        forced.add((top_h - 1, left_w - 1))

    cells = set(forced)
    fill_prob = random.uniform(0.22, 0.72)
    for r in range(top_h):
        for c in range(left_w):
            if (r, c) in forced:
                continue
            if random.random() < fill_prob:
                cells.add((int(r), int(c)))

    if len(cells) == len(forced) and top_h * left_w > len(forced):
        pool = [(r, c) for r in range(top_h) for c in range(left_w) if (r, c) not in cells]
        cells.add(random.choice(pool))
    return cells


def _mirror_canonical(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for r, c in cells:
        rows = {int(r), int(h - 1 - r)}
        cols = {int(c), int(w - 1 - c)}
        for rr in rows:
            for cc in cols:
                pts.add((int(rr), int(cc)))
    return pts


def _build_clean_mask(max_h: int, max_w: int) -> tuple[np.ndarray, int, int]:
    for _ in range(SPRITE_RETRIES):
        h = random.randint(HEIGHT_MIN, min(HEIGHT_MAX, max_h))
        w = random.randint(WIDTH_MIN, min(WIDTH_MAX, max_w))
        top_h = (h + 1) // 2
        left_w = (w + 1) // 2
        canonical = _canonical_cells(top_h, left_w)
        pts = _mirror_canonical(canonical, h, w)

        count = len(pts)
        area = h * w
        if not (PIXEL_COUNT_MIN <= count <= PIXEL_COUNT_MAX):
            continue
        if not (math.ceil(DENSITY_LO * area) <= count <= math.floor(DENSITY_HI * area)):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            mask[int(r), int(c)] = True

        if not _is_connected(mask):
            continue

        coords = [tuple(x) for x in np.argwhere(mask)]
        true_pair = (h - 1, w - 1)
        blank = np.zeros((h, w), dtype=int)
        blank[mask] = 1
        if _second_best_component_score(blank, coords, true_pair, h, w) > count - CLEAN_PAIR_GAP_MIN:
            continue
        return mask, h - 1, w - 1
    raise ValueError("failed to build clean fourfold mask")


def _symmetry_groups(mask: np.ndarray, true_ky: int, true_kx: int) -> list[dict]:
    h, w = mask.shape
    pts = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    groups: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for r, c in sorted(pts):
        if (r, c) in seen:
            continue
        orbit = _orbit_cells(r, c, true_ky, true_kx, h, w)
        if orbit is None or not (orbit <= pts):
            return []
        cells = sorted(orbit)
        for cell in cells:
            seen.add(cell)
        boundary = False
        for rr, cc in cells:
            for nr, nc in _neighbors4(rr, cc):
                if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                    boundary = True
                    break
            if boundary:
                break
        dist = min(abs(2 * rr - true_ky) + abs(2 * cc - true_kx) for rr, cc in cells)
        groups.append(
            {
                "cells": cells,
                "boundary": boundary,
                "dist": dist,
                "size": len(cells),
                "rand": random.random(),
            }
        )
    return groups


def _pick_accent_groups(groups: list[dict], accent_min: int, accent_max: int) -> list[dict] | None:
    ordered = sorted(groups, key=lambda g: (g["boundary"], g["dist"], g["size"], g["rand"]))
    dp: dict[int, list[int]] = {0: []}
    for idx, group in enumerate(ordered):
        size = int(group["size"])
        updates: dict[int, list[int]] = {}
        for total, chosen in list(dp.items()):
            new_total = total + size
            if new_total > accent_max or new_total in dp or new_total in updates:
                continue
            updates[new_total] = chosen + [idx]
        dp.update(updates)

    feasible = [total for total in dp if accent_min <= total <= accent_max]
    if not feasible:
        return None

    target_center = random.randint(accent_min, accent_max)
    feasible.sort(key=lambda total: (abs(total - target_center), total))
    top = feasible[: min(5, len(feasible))]
    target = random.choice(top)
    return [ordered[idx] for idx in dp[target]]


def _paint_multicolor_sprite(
    mask: np.ndarray, colors: tuple[int, int], true_ky: int, true_kx: int
) -> tuple[np.ndarray, int] | None:
    base_color, accent_color = colors
    total = int(mask.sum())
    accent_min = max(2, math.ceil(ACCENT_FRAC_LO * total))
    accent_max = min(total - 2, max(accent_min, math.floor(ACCENT_FRAC_HI * total)))
    if accent_min > accent_max:
        return None

    groups = _symmetry_groups(mask, true_ky, true_kx)
    if not groups:
        return None

    chosen = _pick_accent_groups(groups, accent_min, accent_max)
    if not chosen:
        return None

    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = int(base_color)
    accent_count = 0
    for group in chosen:
        for r, c in group["cells"]:
            sprite[int(r), int(c)] = int(accent_color)
            accent_count += 1

    present = colors_present(sprite, exclude={0})
    if sorted(present) != sorted([base_color, accent_color]):
        return None
    return sprite, accent_count


def _build_clean_sprite(colors: tuple[int, int], max_h: int, max_w: int) -> tuple[np.ndarray, int, int, int]:
    for _ in range(SPRITE_RETRIES):
        mask, true_ky, true_kx = _build_clean_mask(max_h, max_w)
        painted = _paint_multicolor_sprite(mask, colors, true_ky, true_kx)
        if painted is None:
            continue
        sprite, accent_count = painted
        coords = [tuple(x) for x in np.argwhere(sprite != 0)]
        count = len(coords)
        true_pair = (true_ky, true_kx)
        if _second_best_component_score(sprite, coords, true_pair, *sprite.shape) > count - CLEAN_PAIR_GAP_MIN:
            continue
        return sprite, true_ky, true_kx, accent_count
    raise ValueError("failed to build clean multicolor sprite")


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(out, sprite, r0, c0, transparent=0)
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
        spec["true_ky"] = int(spec["axis_ky_local"] + 2 * r0)
        spec["true_kx"] = int(spec["axis_kx_local"] + 2 * c0)
        spec["clean_count"] = int(np.count_nonzero(sprite))
        placed.append(spec)
    return out, placed


def _adjacent_mask(clean_mask: np.ndarray) -> np.ndarray:
    adj = np.zeros_like(clean_mask, dtype=bool)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    return adj & ~clean_mask


def _fragment_candidates(clean_mask: np.ndarray, true_ky: int, true_kx: int) -> dict[str, list[dict]]:
    height, width = clean_mask.shape
    adj = _adjacent_mask(clean_mask)
    clean_pts = {tuple(x) for x in np.argwhere(clean_mask)}

    vertical: dict[tuple[tuple[int, int], ...], dict] = {}
    horizontal: dict[tuple[tuple[int, int], ...], dict] = {}
    singletons: dict[tuple[tuple[int, int], ...], dict] = {}

    for r, c in np.argwhere(adj):
        r = int(r)
        c = int(c)
        orbit = _orbit_cells(r, c, true_ky, true_kx, height, width)
        if orbit is None or orbit & clean_pts:
            continue
        mr = int(true_ky) - r
        mc = int(true_kx) - c
        if mr == r or mc == c:
            continue

        vpair = tuple(sorted(((r, c), (r, mc))))
        if 0 <= mc < width and all(adj[rr, cc] for rr, cc in vpair):
            vertical[vpair] = {
                "kind": "vertical_pair",
                "cells": vpair,
                "orbit": frozenset(orbit),
            }

        hpair = tuple(sorted(((r, c), (mr, c))))
        if 0 <= mr < height and all(adj[rr, cc] for rr, cc in hpair):
            horizontal[hpair] = {
                "kind": "horizontal_pair",
                "cells": hpair,
                "orbit": frozenset(orbit),
            }

        key = ((r, c),)
        singletons[key] = {
            "kind": "singleton",
            "cells": key,
            "orbit": frozenset(orbit),
        }

    return {
        "vertical_pair": list(vertical.values()),
        "horizontal_pair": list(horizontal.values()),
        "singleton": list(singletons.values()),
    }


def _choose_noise_fragments(
    candidates: dict[str, list[dict]],
    lo: int,
    hi: int,
    require_vertical_pair: bool,
) -> list[dict] | None:
    minimum = max(lo, 2 if require_vertical_pair else lo)
    if minimum > hi:
        return None

    targets = list(range(minimum, hi + 1))
    random.shuffle(targets)
    all_candidates = candidates["vertical_pair"] + candidates["horizontal_pair"] + candidates["singleton"]

    for target in targets:
        for _ in range(NOISE_PLAN_RETRIES):
            chosen: list[dict] = []
            used_orbit_cells: set[tuple[int, int]] = set()
            remaining = int(target)

            if require_vertical_pair:
                starters = [cand for cand in candidates["vertical_pair"] if len(cand["cells"]) <= remaining]
                if not starters:
                    return None
                first = random.choice(starters)
                chosen.append(first)
                used_orbit_cells |= set(first["orbit"])
                remaining -= len(first["cells"])

            while remaining > 0:
                viable = [
                    cand
                    for cand in all_candidates
                    if len(cand["cells"]) <= remaining and set(cand["orbit"]).isdisjoint(used_orbit_cells)
                ]
                if not viable:
                    break

                exact = [cand for cand in viable if len(cand["cells"]) == remaining]
                if exact:
                    cand = random.choice(exact)
                elif remaining % 2 == 1:
                    ones = [cand for cand in viable if len(cand["cells"]) == 1]
                    if not ones:
                        break
                    cand = random.choice(ones)
                else:
                    twos = [cand for cand in viable if len(cand["cells"]) == 2]
                    cand = random.choice(twos if twos and random.random() < 0.7 else viable)

                chosen.append(cand)
                used_orbit_cells |= set(cand["orbit"])
                remaining -= len(cand["cells"])

            if remaining == 0:
                return chosen
    return None


def _colorize_fragments(
    fragments: list[dict], colors: tuple[int, int], *, force_accent_fragment: bool
) -> tuple[list[tuple[int, int, int]], bool]:
    base_color, accent_color = colors
    colored: list[tuple[int, int, int]] = []
    used_accent = False
    for idx, fragment in enumerate(fragments):
        if force_accent_fragment and idx == 0:
            color = int(accent_color)
        elif force_accent_fragment and not used_accent and idx == len(fragments) - 1:
            color = int(accent_color)
        else:
            color = int(accent_color if random.random() < 0.4 else base_color)
        if color == int(accent_color):
            used_accent = True
        for r, c in fragment["cells"]:
            colored.append((int(r), int(c), int(color)))
    return colored, used_accent


def _add_noise(
    inp: np.ndarray,
    out: np.ndarray,
    spec: dict,
    *,
    require_vertical_pair: bool,
    force_accent_noise: bool,
) -> bool:
    true_ky = int(spec["true_ky"])
    true_kx = int(spec["true_kx"])
    colors = tuple(int(x) for x in spec["colors"])
    clean_mask = np.isin(out, list(colors))
    clean_coords = [tuple(x) for x in np.argwhere(clean_mask)]
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    candidates = _fragment_candidates(clean_mask, true_ky, true_kx)
    fragments = _choose_noise_fragments(candidates, lo, hi, require_vertical_pair)
    if not fragments:
        return False

    colored_noise, used_accent = _colorize_fragments(
        fragments, colors, force_accent_fragment=force_accent_noise
    )
    if force_accent_noise and not used_accent:
        return False

    trial = out.copy()
    for r, c, color in colored_noise:
        trial[int(r), int(c)] = int(color)

    trial_coords = [tuple(x) for x in np.argwhere(np.isin(trial, list(colors)))]
    if clean_count - _second_best_component_score(trial, trial_coords, (true_ky, true_kx), GRID_SIZE, GRID_SIZE) < FINAL_PAIR_GAP_MIN:
        return False

    for r, c, color in colored_noise:
        inp[int(r), int(c)] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(range(10))
    palette = random.sample([c for c in range(1, 10) if c != bg], 2 * NUM_SPRITES)
    color_pairs = [(palette[2 * i], palette[2 * i + 1]) for i in range(NUM_SPRITES)]
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    try:
        for colors, slot in zip(color_pairs, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_ky_local, axis_kx_local, accent_count = _build_clean_sprite(colors, max_h, max_w)
            sprite_specs.append(
                {
                    "colors": colors,
                    "sprite": sprite,
                    "axis_ky_local": axis_ky_local,
                    "axis_kx_local": axis_kx_local,
                    "accent_count": accent_count,
                }
            )
    except ValueError:
        return None

    result = _slot_place_sprites(bg, sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs = result

    if len(_connected_components(out != bg)) != NUM_SPRITES:
        return None

    eligible = list(placed_specs)
    random.shuffle(eligible)
    required_spec = None
    for spec in eligible:
        clean_mask = np.isin(out, list(spec["colors"]))
        candidates = _fragment_candidates(clean_mask, int(spec["true_ky"]), int(spec["true_kx"]))
        if candidates["vertical_pair"] and int(spec["accent_count"]) <= max(10, int(spec["clean_count"]) // 3):
            required_spec = spec
            break
    if required_spec is None:
        return None

    inp = out.copy()
    used_required_accent = False
    for spec in placed_specs:
        is_required = spec is required_spec
        ok = _add_noise(
            inp,
            out,
            spec,
            require_vertical_pair=is_required,
            force_accent_noise=is_required,
        )
        if not ok:
            return None
        used_required_accent |= is_required

    if not used_required_accent:
        return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_legacy_multicolor_vertical_clean(inp), out):
        return None
    if np.array_equal(_legacy_per_color_fourfold_clean(inp), out):
        return None
    if np.array_equal(_legacy_color_vertical_clean(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
