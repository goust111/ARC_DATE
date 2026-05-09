# Combo variant of 8e5c0c38 (index 15)
# combo_spec: D3/2+D12/0
# Name: 共享颜色多色精灵修复

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis
# a) 具体化与行为追踪
#    - 融合后每个精灵不再是“单色连通块”，也不再是“双色但整体 4 连通”的简单情况，而是一个局部区域里的
#      两个同轴部件：外层大壳一种颜色，内部悬浮小块另一种颜色。二者共享同一条垂直对称轴，但中间留有背景缝，
#      所以若只看 4 连通，它们会被拆成两个部件。
#    - 同时，三个精灵里有两个会复用同一种外壳颜色。于是“按颜色收集全部像素”会把两个远处精灵混在一起；
#      “按所有非背景做连通域”又会把单个精灵拆成两个部件。正确对象边界必须先把复用颜色按空间切开，再把同一
#      局部里的双色部件重新配对。
#    - 小例子追踪：
#      旧版原题逻辑：对共享外壳色 S 的所有像素一起找轴，会把左上外壳和右上外壳混成一层；对内部小块色单独找轴，
#      稀疏小块加噪声后又容易围绕错误轴保留噪声。
#      D3-only 逻辑：把每个非背景连通块各自修复，会把“外壳”和“内部小块”当成两个精灵，外壳能修好，但小块因
#      稀疏且带噪声而常选错轴。
#      融合后逻辑：先按单色 4 连通得到 6 个部件；再利用局部邻近关系把每个外壳与附近内部小块合成一个双色精灵；
#      对合并后的联合形状找共享轴，再按“镜像位置必须是同色”删噪声。这样共享颜色与多色同轴两个维度同时起作用。
# b) 复杂度校准
#    - 仍保持 22x22、3 个局部区域、删除型输出，避免引入额外颜色映射或重绘操作。
#    - 前景颜色总数控制在 5：两只精灵共享 1 个外壳色，另外 1 个外壳色和 3 个内部色各自独立。共享关系清楚，
#      但不会因为颜色种类过多而失去可读性。
#    - 每个精灵固定由“1 个外壳部件 + 1 个内部部件”组成，且内部部件很小、轴歧义高；这让“先配对再修轴”成为
#      明确需求，而不是可有可无的实现细节。
# c) 隐式假设审查（对应原 main 的关键写法）
#    - `colors = colors_present(...)`：原本可把颜色直接当对象 id。融合后失效，因为一个精灵有两色，且一种颜色会在
#      两个不同精灵里重复出现。
#    - `pts = np.argwhere(g == col)`：原本默认“同色 = 同一精灵”。融合后同色可能对应两个远处分离的外壳，所以必须先
#      按单色连通拆开。
#    - 若改成 `find_components(g != bg)` 直接取多色连通域：在本变体中也失效，因为同一精灵的两色部件故意被背景缝隔开，
#      一个真实精灵会被拆成两个连通块。
#    - 轴搜索本身仍可沿用“枚举所有垂直轴并最大化自镜像匹配”的思路，但输入点集必须来自“局部双色配对后的联合对象”。
#    - 删除阶段仍是 delete-only；不过判断条件必须是“镜像位置存在且颜色相同”，避免不同颜色在共享轴下互相充当配对。
# d) 放置合法性
#    - 生成器继续使用 3 个远离的 slot，保证不同精灵整体区域不重叠。
#    - 每个局部 sprite 先在本地画布上生成完整输出，再随机放入 slot；外壳只占边框，内部色只放在中间腔体里，因此
#      两部件不会越界或意外接触。
#    - 外壳噪声只允许长在 bbox 外侧，内部噪声只允许长在腔体内部且远离外壳；这样输入中每个精灵仍然恰好对应两个
#      单色连通部件，不会被噪声桥接成一块，也不会和别的精灵相连。
# e) 视觉可观察性
#    - 人类能直接看到：同一种外壳色在两个远处区域重复出现，但每个区域里都还有一个不同的内部色小块；输出是把每个
#      局部双色组合分别修成同轴对称。
#    - 内部小块比外壳稀疏得多，单独看会比较不稳定；配对后共享轴明显。这能从少量样例中支撑“先看局部双色搭配，再
#      按共享轴删噪声”的归纳。
# f) 可视化整洁度
#    - 三个精灵分散在原题同款大间隔 slot 中；每个局部结构固定为“外框 + 中央小块”，层次简单，视觉上不会杂乱。
#    - 不使用隐藏排序规则，也不让多种标记堆在同一区域。共享色只重复两次，且始终是完整外壳，易于一眼识别。
# g) 变化的可推断性
#    - 没有按颜色 id 做映射，也没有“某颜色总要变成另一颜色”的不可见规则。颜色只承担可见的结构角色：
#      两个远处外壳共享一种颜色，内部色各自不同，而修复操作始终是删除不满足同色镜像的噪声。


GRID_SIZE = 22
NUM_SPRITES = 3
NUM_FOREGROUND_COLORS = 5
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

HEIGHT_MIN = 6
HEIGHT_MAX = 9
WIDTH_MIN = 6
WIDTH_MAX = 9
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46

ACCENT_PIXELS_MIN = 2
ACCENT_PIXELS_MAX = 4
ACCENT_AXIS_GAP_MAX = 2

SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

SPRITE_RETRIES = 220
GEN_RETRIES = 900
ACCENT_RETRIES = 240
FLIP_PROB = 0.5


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components_from_mask(mask)) == 1


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
        scores[k] = score
    return scores


def _best_axis_for_coords(coords: list[tuple[int, int]], width: int) -> int:
    scores = _axis_scores(coords, width)
    best_k = 0
    best_score = -1
    for k, score in scores.items():
        if score > best_score:
            best_score = int(score)
            best_k = int(k)
    return best_k


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int) -> int:
    scores = _axis_scores(coords, width)
    return max(score for k, score in scores.items() if k != true_k)


def _coords_to_mask(coords: list[tuple[int, int]] | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _expand_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    for _ in range(radius):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _component_groups(comps: list[dict], shape: tuple[int, int]) -> list[list[int]]:
    if not comps:
        return []
    masks = [_coords_to_mask(comp["coords"], shape) for comp in comps]
    expanded = [_expand_mask(mask, radius=2) for mask in masks]
    graph = {i: set() for i in range(len(comps))}
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if int(comps[i]["color"]) == int(comps[j]["color"]):
                continue
            if np.any(expanded[i] & expanded[j]):
                graph[i].add(j)
                graph[j].add(i)

    groups: list[list[int]] = []
    seen: set[int] = set()
    for i in range(len(comps)):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        group: list[int] = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        groups.append(sorted(group))
    return groups


def _group_union_coords(comps: list[dict], group: list[int]) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for idx in group:
        coords.extend((int(r), int(c)) for r, c in comps[idx]["coords"])
    return coords


def _legacy_d3_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _, w = g.shape
    out = g.copy()
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    for comp in comps:
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        if not coords:
            continue
        pts = set(coords)
        k = _best_axis_for_coords(coords, w)
        for r, c in coords:
            cc = k - c
            if not (0 <= cc < w and (r, cc) in pts and int(g[r, cc]) == int(g[r, c])):
                out[r, c] = bg
    return out


def _legacy_d12_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _, w = g.shape
    out = g.copy()
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    for comp in comps:
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        if not coords:
            continue
        pts = set(coords)
        k = _best_axis_for_coords(coords, w)
        for r, c in coords:
            cc = k - c
            if not (0 <= cc < w and (r, cc) in pts):
                out[r, c] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each shared-color multicolor sprite by pairing its local color parts and enforcing one vertical axis.

    Input structure:
    - The background fills most cells. Three noisy sprites are present, but each sprite is split into two 4-connected
      monochromatic parts: a large outer shell and a small inner part with a different color.
    - Two distant sprites reuse the same shell color, so one foreground color appears in two disconnected places.

    Rule:
    - First split the grid into monochromatic 4-connected components, then merge nearby different-color components
      that belong to the same local sprite.
    - For each merged two-color sprite, find the vertical axis that best fits the union of both parts, and delete any
      pixel whose mirror across that axis is missing or has a different color.

    Key insight:
    - Neither color-only grouping nor whole-foreground connectivity is sufficient here: one color is reused across
      sprites, while each true sprite is itself broken into two disconnected color parts. The correct axis appears only
      after those local two-color parts are reassembled.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, w = g.shape

    out = g.copy()
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "grid must contain at least one non-background component"

    for group in _component_groups(comps, g.shape):
        coords = _group_union_coords(comps, group)
        if not coords:
            continue
        pts = set(coords)
        k = _best_axis_for_coords(coords, w)
        for r, c in coords:
            cc = k - c
            if not (0 <= cc < w and (r, cc) in pts and int(g[r, cc]) == int(g[r, c])):
                out[r, c] = bg

    return out


def _symmetry_groups(h: int, w: int) -> list[dict]:
    true_k = w - 1
    groups: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            mc = true_k - c
            if not (0 <= mc < w):
                continue
            if mc == c:
                cells = [(r, c)]
            else:
                cells = sorted([(r, c), (r, mc)])
            for cell in cells:
                seen.add(cell)
            groups.append(
                {
                    "cells": cells,
                    "size": len(cells),
                    "dist": min(abs(2 * cc - true_k) for _, cc in cells),
                }
            )
    return groups


def _touches_mask(mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    if not mask.any():
        return True
    h, w = mask.shape
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _build_accent_mask(h: int, w: int) -> tuple[np.ndarray, int]:
    true_k = w - 1
    groups = _symmetry_groups(h, w)
    max_pixels = min(ACCENT_PIXELS_MAX, h * w)
    for _ in range(ACCENT_RETRIES):
        target = random.randint(ACCENT_PIXELS_MIN, max_pixels)
        order = sorted(groups, key=lambda g: (g["dist"], g["size"], random.random()))
        mask = np.zeros((h, w), dtype=bool)
        chosen: set[int] = set()
        count = 0

        starts = [idx for idx, g in enumerate(order) if g["size"] <= target]
        if not starts:
            continue
        start_idx = random.choice(starts[: min(4, len(starts))])
        for r, c in order[start_idx]["cells"]:
            mask[r, c] = True
        chosen.add(start_idx)
        count += order[start_idx]["size"]

        while count < target:
            frontier: list[int] = []
            for idx, group in enumerate(order):
                if idx in chosen or count + group["size"] > target:
                    continue
                if _touches_mask(mask, group["cells"]):
                    frontier.append(idx)
            if not frontier:
                break
            frontier.sort(key=lambda idx: (order[idx]["dist"], order[idx]["size"], random.random()))
            pick = random.choice(frontier[: min(4, len(frontier))])
            for r, c in order[pick]["cells"]:
                mask[r, c] = True
            chosen.add(pick)
            count += order[pick]["size"]

        if count != target or not _is_connected(mask):
            continue
        coords = [tuple(x) for x in np.argwhere(mask)]
        if _second_best_score(coords, true_k, w) < len(coords) - ACCENT_AXIS_GAP_MAX:
            continue
        return mask, true_k
    raise ValueError("failed to build ambiguous accent mask")


def _build_clean_sprite(colors: tuple[int, int], max_h: int, max_w: int) -> tuple[np.ndarray, dict]:
    shell_color, accent_color = colors
    h_choices = [h for h in range(HEIGHT_MIN, min(HEIGHT_MAX, max_h) + 1)]
    w_choices = [w for w in range(WIDTH_MIN, min(WIDTH_MAX, max_w) + 1)]
    if not h_choices or not w_choices:
        raise ValueError("slot too small for fused sprite")

    for _ in range(SPRITE_RETRIES):
        h = random.choice(h_choices)
        w = random.choice(w_choices)
        sprite = np.zeros((h, w), dtype=int)
        shell_mask = np.zeros((h, w), dtype=bool)
        shell_mask[0, :] = True
        shell_mask[-1, :] = True
        shell_mask[:, 0] = True
        shell_mask[:, -1] = True

        accent_h = h - 4
        accent_w = w - 4
        if accent_h < 2 or accent_w < 2:
            continue
        accent_mask_local, accent_k_local = _build_accent_mask(accent_h, accent_w)
        accent_mask = np.zeros((h, w), dtype=bool)
        accent_mask[2 : 2 + accent_h, 2 : 2 + accent_w] = accent_mask_local

        if np.any(shell_mask & accent_mask):
            continue

        sprite[shell_mask] = shell_color
        sprite[accent_mask] = accent_color
        coords = [tuple(x) for x in np.argwhere(sprite != 0)]
        true_k = w - 1
        if not (PIXEL_COUNT_MIN <= len(coords) <= PIXEL_COUNT_MAX):
            continue
        if _second_best_score(coords, true_k, w) > len(coords) - SPRITE_AXIS_GAP_MIN:
            continue
        meta = {
            "axis_k_local": true_k,
            "shell_mask_local": shell_mask,
            "accent_mask_local": accent_mask,
        }
        return sprite, meta
    raise ValueError("failed to build clean fused sprite")


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

        shell_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        accent_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        shell_local = spec["shell_mask_local"]
        accent_local = spec["accent_mask_local"]
        shell_mask[r0 : r0 + h, c0 : c0 + w] = shell_local
        accent_mask[r0 : r0 + h, c0 : c0 + w] = accent_local

        cavity_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        cavity_mask[r0 + 2 : r0 + h - 2, c0 + 2 : c0 + w - 2] = True
        bbox_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        bbox_mask[r0 : r0 + h, c0 : c0 + w] = True

        spec["r0"] = r0
        spec["c0"] = c0
        spec["true_k"] = spec["axis_k_local"] + 2 * c0
        spec["shell_mask"] = shell_mask
        spec["accent_mask"] = accent_mask
        spec["cavity_mask"] = cavity_mask
        spec["bbox_mask"] = bbox_mask
        spec["union_coords"] = [tuple(x) for x in np.argwhere(shell_mask | accent_mask)]
        placed.append(spec)
    return out, placed


def _noise_candidates(
    component_mask: np.ndarray,
    occupied_mask: np.ndarray,
    allowed_mask: np.ndarray,
    true_k: int,
    out_grid: np.ndarray,
    color: int,
    bg: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    adj = np.zeros_like(component_mask)
    adj[1:, :] |= component_mask[:-1, :]
    adj[:-1, :] |= component_mask[1:, :]
    adj[:, 1:] |= component_mask[:, :-1]
    adj[:, :-1] |= component_mask[:, 1:]
    adj &= allowed_mask
    adj &= ~occupied_mask

    rows, cols = np.where(adj)
    candidates: list[tuple[int, int]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if (r, c) in forbidden:
            continue
        mirror_c = true_k - c
        if mirror_c == c:
            continue
        if 0 <= mirror_c < GRID_SIZE:
            if int(out_grid[r, mirror_c]) == int(color):
                continue
            if int(out_grid[r, mirror_c]) != int(bg):
                continue
            if (r, mirror_c) in forbidden:
                continue
        candidates.append((r, c))
    return candidates


def _pick_noise_cell(
    candidates: list[tuple[int, int]], current_union: list[tuple[int, int]], true_k: int
) -> tuple[int, int] | None:
    if not candidates:
        return None
    scored: list[tuple[int, float, tuple[int, int]]] = []
    for cell in candidates:
        alt = _second_best_score(current_union + [cell], true_k, GRID_SIZE)
        scored.append((alt, random.random(), cell))
    scored.sort(key=lambda x: (x[0], x[1]))
    best_alt = scored[0][0]
    pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
    return random.choice(pool)


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict, bg: int) -> bool:
    shell_color, accent_color = spec["colors"]
    true_k = int(spec["true_k"])
    current_union = list(spec["union_coords"])
    forbidden: set[tuple[int, int]] = set()
    chosen: list[tuple[int, int, int]] = []

    accent_coords = [tuple(x) for x in np.argwhere(spec["accent_mask"])]
    accent_noise_target = 1 if len(accent_coords) <= 2 else random.randint(1, 2)
    shell_noise_target = random.randint(0, 1)

    for _ in range(accent_noise_target):
        candidates = _noise_candidates(
            spec["accent_mask"],
            inp != bg,
            spec["cavity_mask"] & ~spec["accent_mask"],
            true_k,
            out,
            accent_color,
            bg,
            forbidden,
        )
        cell = _pick_noise_cell(candidates, current_union, true_k)
        if cell is None:
            return False
        r, c = cell
        chosen.append((r, c, accent_color))
        current_union.append(cell)
        forbidden.add(cell)
        mirror = (r, true_k - c)
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden.add(mirror)

    for _ in range(shell_noise_target):
        candidates = _noise_candidates(
            spec["shell_mask"],
            inp != bg,
            ~spec["bbox_mask"],
            true_k,
            out,
            shell_color,
            bg,
            forbidden,
        )
        cell = _pick_noise_cell(candidates, current_union, true_k)
        if cell is None:
            return False
        r, c = cell
        chosen.append((r, c, shell_color))
        current_union.append(cell)
        forbidden.add(cell)
        mirror = (r, true_k - c)
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden.add(mirror)

    if not chosen:
        return False
    if len(spec["union_coords"]) - _second_best_score(current_union, true_k, GRID_SIZE) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c, color in chosen:
        inp[r, c] = color
    return True


def _color_assignment() -> tuple[int, list[tuple[int, int]]]:
    bg = random.choice(range(10))
    fg = random.sample([c for c in range(10) if c != bg], NUM_FOREGROUND_COLORS)
    shared_shell = fg[0]
    unique_shell = fg[1]
    accent_colors = fg[2:]

    shell_colors = [shared_shell, shared_shell, unique_shell]
    random.shuffle(shell_colors)
    pairs = [(shell_colors[i], accent_colors[i]) for i in range(NUM_SPRITES)]
    return bg, pairs


def fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    groups = _component_groups(comps, g.shape)
    color_hist: dict[int, int] = {}
    for comp in comps:
        col = int(comp["color"])
        color_hist[col] = color_hist.get(col, 0) + 1
    return {
        "mono_components": len(comps),
        "sprite_groups": len(groups),
        "group_sizes": sorted(len(group) for group in groups),
        "duplicated_component_colors": sorted(col for col, n in color_hist.items() if n > 1),
        "foreground_colors": sorted(colors_present(g, exclude={bg})),
    }


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg, pairs = _color_assignment()
    slots = random.choice(LAYOUTS)

    sprite_specs: list[dict] = []
    try:
        for colors, slot in zip(pairs, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, meta = _build_clean_sprite(colors, max_h, max_w)
            sprite_specs.append({"colors": colors, "sprite": sprite, **meta})
    except ValueError:
        return None

    result = _slot_place_sprites(bg, sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs = result

    if len(find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")) != 6:
        return None
    if fusion_metrics(out)["group_sizes"] != [2, 2, 2]:
        return None

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out, spec, bg):
            return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    metrics = fusion_metrics(inp)
    if metrics["group_sizes"] != [2, 2, 2]:
        return None
    if len(metrics["duplicated_component_colors"]) != 1:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_legacy_d3_main(inp), out):
        return None
    if np.array_equal(_legacy_d12_main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
