# Combo variant of 8e5c0c38 (index 36)
# combo_spec: D3/2+D11/1+D12/0
# Name: 共享颜色多色精灵优先级修复

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis
# a) 具体化与行为追踪
#    - 融合后仍是 22x22 画布上的 3 个“接近垂直对称”的对象，但对象不再是单色。每个精灵恰好 2 色，
#      且其中 2 个精灵共享同一种主色；第 3 个精灵完全由独占颜色组成。
#    - 一个具体追踪：
#      精灵 A 用 {共享色 S, 独占色 a}，精灵 B 用 {S, 独占色 b}，精灵 C 用 {c, d}。
#      原题代码会把 S 的所有像素全并在一起找一条轴，A 与 B 的共享色会互相干扰；而 a / b / c / d
#      又会被当成 4 个独立单色对象，各自单独定轴，也会丢失“同一精灵内两色共轴”的信息。
#      融合后应先利用颜色频率识别“谁是共享色”：S 在全局出现最多，因此延后处理。
#      先处理不含 S 的精灵 C，可以直接把它整块修复并占用其空间；再处理含 S 的两个精灵，
#      只在剩余区域里从各自独占色锚点出发恢复整块双色对象，然后按共享轴删去任一颜色上的单边噪声。
#    - 差异总结：
#      原逻辑是“按颜色并行、每色单独定轴”；融合逻辑是“先按颜色频率识别共享色，再按精灵是否含共享色分阶段，
#      且轴属于整个双色精灵而不是单一颜色层”。
# b) 复杂度校准
#    - 颜色系统控制在 5 个前景色：2 个共享色精灵共用 1 色，各自再有 1 个独占辅色，第 3 个精灵用 2 个独占色。
#      这样共享关系肉眼可见，但不会出现太多颜色导致难以归纳。
#    - 仍保持 3 个远距 slot，避免对象重叠；每个对象依旧是 delete-only 的“去噪恢复对称”，没有额外颜色映射。
#    - 共享色被设计成两个精灵的主色，因此其全局像素数稳定高于其他颜色；“按频率延后处理”可从样例中反复出现。
# c) 隐式假设审查（针对原 main 的关键前提）
#    - `for col in colors`：原本默认“一个颜色就是一个对象”。融合后失效，因为一个对象含 2 色，且共享色跨两个对象重复出现。
#    - `pts = np.argwhere(g == col)`：原本默认“同色全局收点后仍对应一个局部结构”。融合后共享色 S 会覆盖两个对象，不能这样用。
#    - 每个颜色自己找轴：原本默认轴只属于该颜色层。融合后轴属于整个双色精灵，必须在两色并集上找。
#    - 并行独立处理颜色：原本默认颜色之间无依赖。融合后需要先识别共享色，再延后含共享色的对象。
#    - 新增假设：
#      1) 含共享色的精灵各自至少有一个独占色连通锚点；
#      2) 完整精灵在非背景 4 连通意义下仍是一整块；
#      3) 共享色是全局出现次数唯一最多的前景色。
#      生成器会显式构造并验证这三点。
# d) 放置合法性
#    - 仍使用原题的 3 个大间隔 slot，所有 clean bbox 和加噪后 bbox 都不会互相接触，更不会越界。
#    - 噪声只加在所属精灵的 4 邻接背景上，且镜像位置不能已有同色像素，也不允许噪声把两个 slot 桥接起来。
# e) 视觉可观察性
#    - 观察输入输出时，人类能直接看到：有两个对象含相同主色，但它们都和各自的第二颜色一起恢复同一条竖直对称轴；
#      第三个对象没有共享色，因此总是先被“单独修好”这一类情况在多组样例里会重复出现。
#    - 频率规则不依赖颜色 id，而依赖“哪个颜色明显出现在两个对象里、像素更多”，属于可视信号。
# f) 可视化整洁度
#    - 双色对象采用“主色外层 + 辅色中轴附近少量成组区域”的稳定结构，能看出是一个对象而不是两个混杂块。
#    - slot 间距沿用原题，避免共享色跨对象造成视觉拥挤。
# g) 变化的可推断性
#    - 没有做任意颜色替换；颜色只承担“共享主色”或“对象内部辅色”的角色。
#    - 共享关系只有 1 组，且在每个样例里恰好重复两次，便于从少量样例归纳“共享色后处理、但仍按局部双色对象修复”。


GRID_SIZE = 22
NUM_SPRITES = 3
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

PIXEL_COUNT_MIN = 8
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
ODD_SIDE_W_RANGE = range(1, 6)
EVEN_SIDE_W_RANGE = range(2, 7)
DENSITY_LO = 0.35
DENSITY_HI = 0.89

ACCENT_FRAC_SHARED_LO = 0.14
ACCENT_FRAC_SHARED_HI = 0.34
ACCENT_FRAC_EXCLUSIVE_LO = 0.18
ACCENT_FRAC_EXCLUSIVE_HI = 0.40

NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

SPRITE_RETRIES = 220
GEN_RETRIES = 900
FLIP_PROB = 0.5


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
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


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = seed
    if not (0 <= sr < h and 0 <= sc < w and mask[sr, sc]):
        return []
    seen = np.zeros_like(mask, dtype=bool)
    seen[sr, sc] = True
    stack = [(int(sr), int(sc))]
    comp: list[tuple[int, int]] = []
    while stack:
        rr, cc = stack.pop()
        comp.append((int(rr), int(cc)))
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((int(nr), int(nc)))
    return comp


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
        scores[int(k)] = int(score)
    return scores


def _best_axis(coords: list[tuple[int, int]], width: int) -> int:
    scores = _axis_scores(coords, width)
    best_score = -1
    best_k = 0
    for k, score in scores.items():
        if score > best_score:
            best_score = int(score)
            best_k = int(k)
    return best_k


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int) -> int:
    scores = _axis_scores(coords, width)
    return max(s for k, s in scores.items() if int(k) != int(true_k))


def _repair_component(g: np.ndarray, out: np.ndarray, cells: list[tuple[int, int]], bg: int) -> None:
    _, w = g.shape
    pts = {(int(r), int(c)) for r, c in cells}
    best_k = _best_axis(cells, w)
    for r, c in cells:
        mirror_c = int(best_k) - int(c)
        if not (0 <= mirror_c < w and (int(r), mirror_c) in pts and int(g[int(r), mirror_c]) == int(g[int(r), int(c)])):
            out[int(r), int(c)] = int(bg)


def _component_colors(g: np.ndarray, cells: list[tuple[int, int]]) -> set[int]:
    return {int(g[int(r), int(c)]) for r, c in cells}


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original per-color solver, used only to reject degenerate samples."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _, w = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        best_score = -1
        best_k = 0
        for k in range(0, 2 * (w - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k) - int(c)
                if 0 <= cc < w and int(g[int(r), cc]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

        for r, c in pts:
            cc = int(best_k) - int(c)
            if not (0 <= cc < w and int(g[int(r), cc]) == int(col)):
                out[int(r), int(c)] = int(bg)
    return out


def _shared_color_by_frequency(g: np.ndarray, bg: int) -> int:
    colors = colors_present(g, exclude={bg})
    counts = {int(col): int(np.sum(g == int(col))) for col in colors}
    return max(colors, key=lambda col: (counts[int(col)], -int(col)))


def _fusion_signature(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    shared = _shared_color_by_frequency(g, bg)
    comps = _components_of_mask(g != bg)
    shared_components = 0
    exclusive_components = 0
    for comp in comps:
        if shared in _component_colors(g, comp):
            shared_components += 1
        else:
            exclusive_components += 1
    counts = {int(col): int(np.sum(g == int(col))) for col in colors_present(g, exclude={bg})}
    sorted_counts = sorted(counts.values(), reverse=True)
    dominance_gap = sorted_counts[0] - sorted_counts[1] if len(sorted_counts) >= 2 else sorted_counts[0]
    return {
        "shared_color": int(shared),
        "shared_components": int(shared_components),
        "exclusive_components": int(exclusive_components),
        "shared_count": int(counts[int(shared)]),
        "dominance_gap": int(dominance_gap),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each two-color sprite by using one shared vertical axis, while delaying the shared-color cases.

    Input structure:
    - One background fills most cells. There are three separated noisy sprites, each using exactly two colors.
    - Two sprites reuse the same globally frequent color; the third sprite uses only colors that appear nowhere else.

    Rule:
    - Identify the globally most frequent non-background color as the shared color. First repair any full sprite that does not contain it.
    - Then, in the still-unclaimed area, use the remaining unique-color anchors to recover the shared-color sprites, and within each recovered sprite delete every pixel whose mirror across that sprite's best vertical axis is not the same color.

    Key insight:
    - The duplicated color cannot be cleaned as one global layer. Frequency only tells you which color to postpone; the actual symmetry axis must still be found on each whole two-color sprite, after spatially isolating that sprite from its own unique-color anchor.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    non_bg = g != bg
    assert bool(non_bg.any()), "grid must contain at least one non-background sprite"

    colors = colors_present(g, exclude={bg})
    counts = {int(col): int(np.sum(g == int(col))) for col in colors}
    shared_color = _shared_color_by_frequency(g, bg)
    claimed = np.zeros_like(non_bg, dtype=bool)
    pending_seeds: list[tuple[int, int]] = []

    ordered_colors = sorted([int(col) for col in colors if int(col) != int(shared_color)], key=lambda col: (counts[int(col)], int(col)))

    # Stage 1: colors with lower frequency reveal which sprite is fully exclusive.
    for col in ordered_colors:
        comps = find_components(
            g,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        for comp in comps:
            coords = [tuple(map(int, rc)) for rc in np.array(comp["coords"], dtype=int)]
            if not coords:
                continue
            seed = coords[0]
            if claimed[seed]:
                continue
            region = _component_from_seed(non_bg & (~claimed), seed)
            if not region:
                continue
            if int(shared_color) in _component_colors(g, region):
                pending_seeds.append(seed)
                continue
            _repair_component(g, out, region, bg)
            for r, c in region:
                claimed[int(r), int(c)] = True

    # Stage 2: repair the remaining sprites that contain the shared color.
    for seed in pending_seeds:
        if claimed[seed] or not non_bg[seed]:
            continue
        region = _component_from_seed(non_bg & (~claimed), seed)
        if not region:
            continue
        _repair_component(g, out, region, bg)
        for r, c in region:
            claimed[int(r), int(c)] = True

    # Fallback: if any legal region remains, clean it the same way.
    for region in _components_of_mask(non_bg & (~claimed)):
        _repair_component(g, out, region, bg)

    return out


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components_of_mask(mask)) == 1


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for h in range(HEIGHT_MIN, h_max + 1):
            for side_w in ODD_SIDE_W_RANGE:
                w = 2 * side_w + 1
                if w > max_w:
                    continue
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue
                for center_count in range(1, h + 1):
                    if (target_count - center_count) % 2 != 0:
                        continue
                    left_count = (target_count - center_count) // 2
                    if h <= left_count <= h * side_w:
                        options.append((h, w, target_count, center_count))
            if target_count % 2 != 0:
                continue
            for side_w in EVEN_SIDE_W_RANGE:
                w = 2 * side_w
                if w > max_w:
                    continue
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue
                left_count = target_count // 2
                if h <= left_count <= h * side_w:
                    options.append((h, w, target_count, 0))
    return options


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({tc for _, _, tc, _ in options})
    target_count = random.choice(counts)
    subset = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in subset]
    return random.choices(subset, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> tuple[set[tuple[int, int]], int]:
    max_extra = left_count - h
    min_backbone = max(0, side_w - 1 - max_extra)
    backbone_col = random.randint(min_backbone, side_w - 1)
    seam_row = random.randrange(h)

    cells: set[tuple[int, int]] = {(int(r), int(backbone_col)) for r in range(h)}
    for c in range(backbone_col, side_w):
        cells.add((int(seam_row), int(c)))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small for scaffold")

    while remaining > 0:
        frontier: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for r, c in cells:
            for rr, cc in _neighbors4(r, c):
                cand = (int(rr), int(cc))
                if 0 <= rr < h and 0 <= cc < side_w and cand not in cells and cand not in seen:
                    frontier.append(cand)
                    seen.add(cand)
        if not frontier:
            raise ValueError("no frontier while growing left half")
        cell = random.choice(frontier)
        cells.add(cell)
        remaining -= 1
    return cells, int(seam_row)


def _center_rows(h: int, center_count: int, bridge_row: int) -> np.ndarray:
    rows = np.zeros(h, dtype=bool)
    if center_count <= 0:
        return rows
    rows[int(bridge_row)] = True
    if center_count > 1:
        others = [r for r in range(h) if int(r) != int(bridge_row)]
        rows[random.sample(others, center_count - 1)] = True
    return rows


def _build_clean_mask(max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left, seam_row = _left_cells(h, side_w, left_count)
        centers = _center_rows(h, center_count, seam_row)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in left:
            mirror_c = w - 1 - c
            mask[int(r), int(c)] = True
            mask[int(r), int(mirror_c)] = True
        if center_count > 0:
            mask[centers, side_w] = True

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        true_k = w - 1
        if len(coords) != target_count:
            continue
        if not _is_connected(mask):
            continue
        if _second_best_score(coords, true_k, w) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return mask, int(true_k)
    raise ValueError("failed to build clean mask")


def _symmetry_groups(mask: np.ndarray, true_k: int) -> list[dict]:
    h, w = mask.shape
    pts = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    groups: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or (int(r), int(c)) in seen:
                continue
            mirror_c = int(true_k) - int(c)
            if mirror_c == c:
                cells = [(int(r), int(c))]
            else:
                cells = sorted([(int(r), int(c)), (int(r), int(mirror_c))])
            for cell in cells:
                seen.add(cell)
            boundary = False
            for rr, cc in cells:
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w and (int(nr), int(nc)) in pts):
                        boundary = True
                        break
                if boundary:
                    break
            dist = min(abs(2 * cc - int(true_k)) for _, cc in cells)
            groups.append(
                {
                    "cells": cells,
                    "size": len(cells),
                    "boundary": boundary,
                    "is_center": len(cells) == 1,
                    "dist": int(dist),
                    "rand": random.random(),
                }
            )
    return groups


def _single_color_component_count(sprite: np.ndarray, color: int) -> int:
    comps = find_components(
        sprite,
        bg=0,
        connectivity=4,
        monochromatic=True,
        colors=[int(color)],
        return_format="dict",
    )
    return len(comps)


def _paint_multicolor_sprite(
    mask: np.ndarray,
    colors: tuple[int, int],
    true_k: int,
    accent_frac_lo: float,
    accent_frac_hi: float,
) -> np.ndarray | None:
    base_color, accent_color = colors
    total = int(mask.sum())
    accent_min = max(2, math.ceil(accent_frac_lo * total))
    accent_max = min(total - 2, max(accent_min, math.floor(accent_frac_hi * total)))
    if accent_min > accent_max:
        return None

    groups = _symmetry_groups(mask, true_k)
    groups.sort(key=lambda g: (g["boundary"], g["dist"], g["is_center"], g["rand"]))
    target = random.randint(accent_min, accent_max)

    chosen: list[dict] = []
    accent_count = 0
    for group in groups:
        size = int(group["size"])
        if accent_count + size > accent_max:
            continue
        if accent_count >= accent_min and accent_count + size > target:
            continue
        chosen.append(group)
        accent_count += size

    if accent_count < accent_min:
        for group in groups:
            if group in chosen:
                continue
            size = int(group["size"])
            if accent_count + size > accent_max:
                continue
            chosen.append(group)
            accent_count += size
            if accent_count >= accent_min:
                break

    if not (accent_min <= accent_count <= accent_max):
        return None

    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = int(base_color)
    for group in chosen:
        for r, c in group["cells"]:
            sprite[int(r), int(c)] = int(accent_color)

    present = colors_present(sprite, exclude={0})
    if sorted(present) != sorted([int(base_color), int(accent_color)]):
        return None
    if _single_color_component_count(sprite, accent_color) != 1:
        return None
    return sprite


def _build_clean_sprite(
    colors: tuple[int, int],
    max_h: int,
    max_w: int,
    accent_frac_lo: float,
    accent_frac_hi: float,
) -> tuple[np.ndarray, int]:
    for _ in range(SPRITE_RETRIES):
        mask, true_k = _build_clean_mask(max_h, max_w)
        sprite = _paint_multicolor_sprite(mask, colors, true_k, accent_frac_lo, accent_frac_hi)
        if sprite is None:
            continue
        return sprite, int(true_k)
    raise ValueError("failed to build clean sprite")


def _slot_place_sprites(
    bg: int,
    sprite_specs: list[dict],
    slots: list[tuple[int, int, int, int]],
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
        clean_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        clean_mask[r0 : r0 + h, c0 : c0 + w] = sprite != 0
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
        spec["true_k"] = int(spec["axis_k_local"] + 2 * c0)
        spec["clean_mask"] = clean_mask
        placed.append(spec)
    return out, placed


def _noise_candidate_cells(clean_mask: np.ndarray, occupied_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    h, w = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~occupied_mask

    cols = np.arange(w)
    mirror_cols = int(true_k) - cols
    on_axis = mirror_cols == cols
    adj[:, on_axis] = False

    valid = (mirror_cols >= 0) & (mirror_cols < w) & ~on_axis
    for c in np.where(valid)[0]:
        adj[:, c] &= ~occupied_mask[:, mirror_cols[c]]

    rows, cols_arr = np.where(adj)
    return list(zip(rows.tolist(), cols_arr.tolist()))


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    base_color, accent_color = spec["colors"]
    true_k = int(spec["true_k"])
    clean_mask = np.array(spec["clean_mask"], dtype=bool)
    clean_coords = [tuple(map(int, rc)) for rc in np.argwhere(clean_mask)]
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    noise_target = random.randint(lo, hi)
    candidates = _noise_candidate_cells(clean_mask, inp != spec["bg"], true_k)
    if len(candidates) < noise_target:
        return False

    chosen: list[tuple[int, int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_coords = list(clean_coords)
    used_accent_noise = False

    for step in range(noise_target):
        options: list[tuple[int, int, float, tuple[int, int], int]] = []
        for cell in candidates:
            if cell in forbidden:
                continue
            r, c = cell
            mirror_c = true_k - c
            alt = _second_best_score(current_coords + [cell], true_k, GRID_SIZE)
            color_order = [accent_color, base_color] if (step == 0 or random.random() < 0.7) else [base_color, accent_color]
            for pref, color in enumerate(color_order):
                if 0 <= mirror_c < GRID_SIZE and int(out[r, mirror_c]) == int(color):
                    continue
                options.append((int(alt), int(pref), random.random(), cell, int(color)))

        if not options:
            return False

        options.sort(key=lambda x: (x[0], x[1], x[2]))
        best_alt = options[0][0]
        pool = [opt for opt in options if opt[0] <= best_alt + 1]
        min_pref = min(opt[1] for opt in pool)
        pool = [opt for opt in pool if opt[1] == min_pref]
        _, _, _, cell, color = random.choice(pool)
        r, c = cell
        chosen.append((int(r), int(c), int(color)))
        current_coords.append((int(r), int(c)))
        forbidden.add((int(r), int(c)))
        mirror = (int(r), int(true_k - c))
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden.add(mirror)
        if int(color) == int(accent_color):
            used_accent_noise = True

    if not used_accent_noise:
        return False
    if clean_count - _second_best_score(current_coords, true_k, GRID_SIZE) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c, color in chosen:
        inp[int(r), int(c)] = int(color)
    return True


def _spec_is_valid(clean_grid: np.ndarray) -> bool:
    bg = int(get_bg(clean_grid))
    sig = _fusion_signature(clean_grid)
    if sig["shared_components"] != 2 or sig["exclusive_components"] != 1:
        return False
    colors = colors_present(clean_grid, exclude={bg})
    counts = {int(col): int(np.sum(clean_grid == int(col))) for col in colors}
    shared = int(sig["shared_color"])
    sorted_pairs = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    if not sorted_pairs or int(sorted_pairs[0][0]) != shared:
        return False
    if len(sorted_pairs) >= 2 and int(sorted_pairs[0][1]) == int(sorted_pairs[1][1]):
        return False
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice(range(10))
    pool = [c for c in range(10) if c != bg]
    if len(pool) < 5:
        return None
    shared_color, unique_a, unique_b, exclusive_base, exclusive_accent = random.sample(pool, 5)

    sprite_specs = [
        {
            "role": "shared",
            "colors": (int(shared_color), int(unique_a)),
            "accent_lo": ACCENT_FRAC_SHARED_LO,
            "accent_hi": ACCENT_FRAC_SHARED_HI,
        },
        {
            "role": "shared",
            "colors": (int(shared_color), int(unique_b)),
            "accent_lo": ACCENT_FRAC_SHARED_LO,
            "accent_hi": ACCENT_FRAC_SHARED_HI,
        },
        {
            "role": "exclusive",
            "colors": (int(exclusive_base), int(exclusive_accent)),
            "accent_lo": ACCENT_FRAC_EXCLUSIVE_LO,
            "accent_hi": ACCENT_FRAC_EXCLUSIVE_HI,
        },
    ]
    random.shuffle(sprite_specs)
    slots = random.choice(LAYOUTS)

    try:
        for spec, slot in zip(sprite_specs, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(
                spec["colors"],
                max_h,
                max_w,
                spec["accent_lo"],
                spec["accent_hi"],
            )
            spec["sprite"] = sprite
            spec["axis_k_local"] = int(axis_k_local)
            spec["bg"] = int(bg)
    except ValueError:
        return None

    placed_result = _slot_place_sprites(bg, sprite_specs, slots)
    if placed_result is None:
        return None
    out, placed_specs = placed_result

    if len(_components_of_mask(out != bg)) != NUM_SPRITES:
        return None
    if not _spec_is_valid(out):
        return None

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out, spec):
            return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_legacy_main(inp), out):
        return None

    sig = _fusion_signature(out)
    if sig["shared_components"] != 2 or sig["exclusive_components"] != 1:
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
