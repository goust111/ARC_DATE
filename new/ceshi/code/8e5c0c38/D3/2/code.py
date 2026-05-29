# Variant of 8e5c0c38
# Dimension: D3 — 被变换的源对象类型 (src)
# Change: 3个独立颜色的近对称精灵，每个精灵使用单一颜色，精灵间互不重叠... → 多色精灵（每个精灵使用2种颜色）

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis
# a) 具体化与行为追踪
#    - 变体后的每个对象不再是“一个颜色 = 一个精灵”，而是“一个 4 连通精灵 = 两种颜色共同组成”。
#      典型外观是：外层轮廓用主色，内部或中轴附近的少量对称块用次色；两种颜色共享同一条垂直对称轴。
#    - 输入相对输出的差别仍是“单边噪声像素”，但噪声现在可以落在主色或次色上。
#    - 小例子追踪：
#      clean 对象的联合形状关于列 8.5 对称，主色轮廓很宽，次色只有中间两对像素。
#      原逻辑若按颜色层单独找轴，次色层加 1 个噪声后可能把最佳轴拉偏；随后它会围绕错误轴删点，导致保留噪声或误删真像素。
#      新逻辑先把这两种颜色看成同一精灵的联合形状来定轴，再逐像素检查“镜像位置是否为同色”，因此能用稳定的共享轴清掉两种颜色里的噪声。
# b) 复杂度校准
#    - 仍保持 22x22、3 个对象、输出先构造再反推输入，避免无关维度漂移。
#    - 每个对象固定只有 2 色，且 6 个对象色两两不同，降低跨对象同色混淆。
#    - 次色刻意做得更稀疏、更靠近中轴，既能肉眼看出“同一精灵里有两层颜色”，也能让“共享轴”真正成为必要条件。
# c) 隐式假设审查（对应原 main 的关键写法）
#    - `colors = colors_present(...)`：原本假设“每个非背景色就是一个对象”。变体中失效，因为一个对象有两色。
#    - `pts = np.argwhere(g == col)`：原本按颜色全局收点。变体中同一对象应按连通精灵分组，而不是按颜色分组。
#    - `if g[r, cc] == col` 用单色层得分：原本无害，因为轴就属于这个颜色层。变体中次色层可能过稀疏，单独定轴会错。
#    - 删除阶段直接围绕该单色轴删点：在变体中必须改成围绕“整个精灵的共享轴”删，并继续要求镜像是同色，避免两色互相充当镜像。
# d) 放置合法性
#    - 生成器继续使用互相分离的 3 个 slot；每个 slot 内只放一个对象，联合形状要求 4 连通。
#    - 噪声只放在对象联合形状 4 邻接的位置，且不同对象 slot 之间保留大间隔，因此不会桥接不同对象。
# e) 视觉可观察性
#    - 每个对象都明确含有两种颜色，而且两色都参与镜像结构；输入输出对比能直接看到“噪声可能落在任一颜色上，但修复时沿同一条轴恢复”。
#    - 生成器额外拒绝“旧版单色 solver 也能完全做对”的样本，避免退化回原题。
# f) 可视化整洁度
#    - 轮廓/内部双色分层比随机花纹更稳定，且对象之间通过固定 slot 分开，避免视觉拥挤。
#    - 联合形状强制 4 连通，确保解题者能自然地把两种颜色视为同一个精灵。
# g) 颜色变化可推断性
#    - 没有颜色映射或按颜色 id 排序；颜色只承担“同一精灵的两层材料”这一可见角色。
#    - 每个对象仅 2 色，且输出操作永远是删除不满足共享轴同色镜像的像素，人类可从少量样例稳定归纳。


# --------------- solver helpers ---------------

GRID_SIZE = 22


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
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


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    """Return {axis_k: symmetric_pair_count} for every candidate vertical axis."""
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


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int) -> int:
    scores = _axis_scores(coords, width)
    return max(s for k, s in scores.items() if k != true_k)


def _best_axis_for_component(coords: list[tuple[int, int]], width: int) -> int:
    scores = _axis_scores(coords, width)
    best_score = -1
    best_k = 0
    for k, score in scores.items():
        if score > best_score:
            best_score = int(score)
            best_k = int(k)
    return best_k


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original per-color solver, kept only to reject degenerate generator outputs."""
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

        k = int(best_k)
        for r, c in pts:
            rr = int(r)
            cc = int(k) - int(c)
            if not (0 <= cc < w and int(g[rr, cc]) == int(col)):
                out[rr, int(c)] = int(bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each multicolor sprite by enforcing one shared vertical symmetry axis.

    Input structure:
    - bg is the dominant background. Three separated non-bg sprites appear as connected components.
    - Each sprite uses exactly two colors, and a few extra pixels of either color may break the symmetry.

    Rule:
    - For each connected sprite, find the vertical axis that best matches the union of both colors.
    - Delete any pixel whose mirror across that shared axis is not occupied by the same color.

    Key insight: The symmetry axis belongs to the whole sprite, not to each color layer independently, so both colors must be cleaned against the same axis.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    out = g.copy()
    comps = _connected_components(g != bg)
    assert comps, "grid must contain at least one non-background sprite"

    for comp in comps:
        pt_set = {(int(r), int(c)) for r, c in comp}
        best_k = _best_axis_for_component(comp, w)

        for r, c in comp:
            cc = int(best_k) - int(c)
            if not (0 <= cc < w and (int(r), cc) in pt_set and int(g[int(r), cc]) == int(g[int(r), int(c)])):
                out[int(r), int(c)] = int(bg)

    return out


# --------------- generator ---------------

# ── CFG: canvas & layout ──
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
ODD_SIDE_W_RANGE = range(1, 6)
EVEN_SIDE_W_RANGE = range(2, 7)
DENSITY_LO = 0.35
DENSITY_HI = 0.89
ACCENT_FRAC_LO = 0.16
ACCENT_FRAC_HI = 0.38

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

# ── CFG: axis gap thresholds ──
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# ── CFG: retry limits ──
SPRITE_RETRIES = 180
GEN_RETRIES = 600

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _is_connected(mask: np.ndarray) -> bool:
    comps = _connected_components(mask)
    return len(comps) == 1


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate feasible (h, w, target_count, center_count) tuples."""
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
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    inner_anchor_row = random.randrange(h)
    for r in range(h):
        if r == 0:
            c = 0
        elif side_w > 1 and r == inner_anchor_row:
            c = side_w - 1
        else:
            c = random.randrange(side_w)
        cells.add((r, c))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small")

    pool = [(r, c) for r in range(h) for c in range(side_w) if (r, c) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _build_clean_mask(max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build a connected vertically symmetric mask and return (mask, local_axis_k)."""
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left = _left_cells(h, side_w, left_count)
        centers = _center_rows(h, center_count)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in left:
            mirror_c = w - 1 - c
            mask[r, c] = True
            mask[r, mirror_c] = True
        if center_count > 0:
            mask[centers, side_w] = True

        coords = [tuple(x) for x in np.argwhere(mask)]
        true_k = w - 1
        if len(coords) != target_count:
            continue
        if not _is_connected(mask):
            continue
        if _second_best_score(coords, true_k, w) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return mask, true_k
    raise ValueError("failed to build clean sprite")


def _symmetry_groups(mask: np.ndarray, true_k: int) -> list[dict]:
    """Return symmetry-linked cell groups for a symmetric local mask."""
    h, w = mask.shape
    pts = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    groups: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or (r, c) in seen:
                continue
            mirror_c = true_k - c
            if mirror_c == c:
                cells = [(r, c)]
            else:
                cells = sorted([(r, c), (r, mirror_c)])
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
            dist = min(abs(2 * cc - true_k) for _, cc in cells)
            groups.append(
                {
                    "cells": cells,
                    "boundary": boundary,
                    "is_center": len(cells) == 1,
                    "dist": dist,
                    "rand": random.random(),
                }
            )
    return groups


def _paint_multicolor_sprite(mask: np.ndarray, colors: tuple[int, int], true_k: int) -> np.ndarray | None:
    """Color a symmetric mask with exactly two symmetric color layers."""
    base_color, accent_color = colors
    total = int(mask.sum())
    accent_min = max(2, math.ceil(ACCENT_FRAC_LO * total))
    accent_max = min(total - 2, max(accent_min, math.floor(ACCENT_FRAC_HI * total)))
    if accent_min > accent_max:
        return None

    groups = _symmetry_groups(mask, true_k)
    groups.sort(key=lambda g: (g["boundary"], g["is_center"], g["dist"], g["rand"]))
    target = random.randint(accent_min, accent_max)

    chosen: list[dict] = []
    accent_count = 0
    for group in groups:
        size = len(group["cells"])
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
            size = len(group["cells"])
            if accent_count + size > accent_max:
                continue
            chosen.append(group)
            accent_count += size
            if accent_count >= accent_min:
                break

    if not (accent_min <= accent_count <= accent_max):
        return None

    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = base_color
    for group in chosen:
        for r, c in group["cells"]:
            sprite[r, c] = accent_color

    present = colors_present(sprite, exclude={0})
    if sorted(present) != sorted([base_color, accent_color]):
        return None
    return sprite


def _build_clean_sprite(colors: tuple[int, int], max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    for _ in range(SPRITE_RETRIES):
        mask, true_k = _build_clean_mask(max_h, max_w)
        sprite = _paint_multicolor_sprite(mask, colors, true_k)
        if sprite is None:
            continue
        return sprite, true_k
    raise ValueError("failed to color clean sprite")


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
        spec["r0"] = r0
        spec["c0"] = c0
        spec["true_k"] = spec["axis_k_local"] + 2 * c0
        placed.append(spec)
    return out, placed


def _noise_candidate_cells(clean_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    """4-adjacent background cells, excluding the symmetry axis itself."""
    h, w = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask

    cols = np.arange(w)
    mirror_cols = true_k - cols
    on_axis = mirror_cols == cols
    adj[:, on_axis] = False

    rows, cols_arr = np.where(adj)
    return list(zip(rows.tolist(), cols_arr.tolist()))


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Add noise pixels of either sprite color while preserving the shared-axis solution."""
    base_color, accent_color = spec["colors"]
    true_k = spec["true_k"]
    clean_mask = np.isin(out, [base_color, accent_color])
    clean_coords = [tuple(x) for x in np.argwhere(clean_mask)]
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    noise_target = random.randint(lo, hi)
    candidates = _noise_candidate_cells(clean_mask, true_k)
    if len(candidates) < noise_target:
        return False

    chosen: list[tuple[int, int, int]] = []
    forbidden_cells: set[tuple[int, int]] = set()
    current_coords = list(clean_coords)
    used_accent_noise = False

    for step in range(noise_target):
        options: list[tuple[int, int, float, tuple[int, int], int]] = []
        for cell in candidates:
            if cell in forbidden_cells:
                continue
            r, c = cell
            mirror_c = true_k - c
            alt = _second_best_score(current_coords + [cell], true_k, GRID_SIZE)
            color_order = [accent_color, base_color] if (step == 0 or random.random() < 0.7) else [base_color, accent_color]
            for pref, color in enumerate(color_order):
                if 0 <= mirror_c < GRID_SIZE and int(out[r, mirror_c]) == int(color):
                    continue
                options.append((alt, pref, random.random(), cell, color))

        if not options:
            return False

        options.sort(key=lambda x: (x[0], x[1], x[2]))
        best_alt = options[0][0]
        pool = [opt for opt in options if opt[0] <= best_alt + 1]
        alt_filtered = [opt for opt in pool if opt[1] == min(x[1] for x in pool)]
        _, _, _, cell, color = random.choice(alt_filtered)
        r, c = cell
        chosen.append((r, c, color))
        current_coords.append(cell)
        forbidden_cells.add(cell)
        mirror = (r, true_k - c)
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden_cells.add(mirror)
        if color == accent_color:
            used_accent_noise = True

    if not used_accent_noise and accent_color != base_color:
        return False
    if clean_count - _second_best_score(current_coords, true_k, GRID_SIZE) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c, color in chosen:
        inp[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build clean multicolor sprites, add noise, and validate."""
    bg = random.choice(range(10))
    object_pool = [c for c in range(1, 10) if c != bg]
    palette = random.sample(object_pool, 2 * NUM_SPRITES)
    color_pairs = [(palette[2 * i], palette[2 * i + 1]) for i in range(NUM_SPRITES)]
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    try:
        for colors, slot in zip(color_pairs, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(colors, max_h, max_w)
            sprite_specs.append({"colors": colors, "sprite": sprite, "axis_k_local": axis_k_local})
    except ValueError:
        return None

    result = _slot_place_sprites(bg, sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs = result

    if len(_connected_components(out != bg)) != NUM_SPRITES:
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
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
