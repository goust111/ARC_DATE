# Combo variant of 409aa875 (index 2)
# combo_spec: D2/0+D3/2
# Name: 双色精灵编码距离与方向
#
# Design analysis
# spec_variant:
#   core_change:
#     - 非背景对象仍然是 16x16 网格上的 3 像素、8 连通小三角组件，轮廓仍限制在原题的 L/V 家族里。
#     - 但每个组件改成 1 个 apex 色像素 + 2 个同色 arm 像素；因此求解时不能再按颜色分组件，必须先对所有非背景像素做整体 8 连通分量。
#     - apex 色同时编码两件事：射击方向属于对角还是轴向，以及投射距离是多少。
#     - 本实现固定两种 apex 码：2 -> 对角线方向且距离 3；8 -> 轴向方向且距离 6。
#   concrete_behavior:
#     - 先在每个 3 像素组件里找唯一颜色的那个像素，它就是 apex。
#     - 再把另外两枚 arm 像素相对 apex 的偏移相加，取其反向符号，得到精确发射方向。
#     - 如果 apex 色是 2，则这个方向必须是对角；如果 apex 色是 8，则这个方向必须是轴向。
#     - 落点不再固定为 apex + 5 * dir，而是 apex + distance(apex_color) * dir。
#     - 其余输出规则沿用原题：命中现有精灵则整块变 9；空背景格上若有多枚投射物重合则变 1，否则单枚落空变 9。
#   behavior_trace:
#     - 例 1：V 形轮廓 {(0,1),(1,0),(1,2)} 若唯一色 8 放在 (0,1)，arms 在下方两角，方向为向上，距离为 6。
#     - 例 2：同一轮廓若唯一色 2 放在 (1,0)，arms 在 (0,1),(1,2)，方向改为右下对角或左下对角中的符号反向，这里具体是 (1,-1)，距离改为 3。
#     - 原题对同一单色轮廓只会读出“天然 apex + 固定 5 格”；融合后必须先读颜色角色，再由 apex 色决定用哪一种距离和方向类型。
#   complexity_calibration:
#     - 只使用两种 apex 色和一种全局 arm 色，避免出现过多颜色映射。
#     - 同时强制两类 apex 色在每张图都出现，使“颜色决定距离”能在单图里直接观察到。
#     - 再强制至少一个 non-legacy apex（唯一色不在原题天然角点/顶点），保证“先找唯一色像素”不是可有可无的装饰。
#   implicit_assumptions_review:
#     - 原 solver 逐色找连通块，双色后会把一个精灵拆开；这里改成对 g != bg 做 8 连通分量。
#     - 原 solver 依赖几何度数直接找 apex；现在几何只负责验证轮廓属于 L/V 家族，真正的 apex 由 1+2 配色决定。
#     - 原 solver 把所有投射距离都写死为 5；这里距离改成查表，否则 D2 维度不会生效。
#     - 新 solver 额外假设每个有效组件恰有两色且频次为 1+2，并且 apex 色只来自 {2,8}；generator 会显式保证这一点。
#   placement_legality:
#     - 仍保持 16x16、背景 7、组件数 4..9、组件彼此 Chebyshev 距离至少 2、每个投射落点在界内。
#     - generator 继续走 output-first：先构造碰撞/命中事件，再把双色精灵画回输入，并用 main 反验。
#     - 不管是唯一落空、双投射同格，还是命中其他精灵，落点一旦预留为背景就不会被后续组件占据。
#   visual_observability:
#     - 人类看到输入时，可以先用“唯一色像素”定位 apex，再比较不同 apex 色对应的 3 格与 6 格落点差异。
#     - 由于每图都含对角 3 格类和轴向 6 格类，而且输出只新增 9/1 标记，颜色与空间关系的因果链比较清楚。
#   visual_cleanliness:
#     - 输入侧只出现背景、1 种 arm 色、2 种 apex 色，层次稳定。
#     - 输出侧仍只新增 9 与可能的 1，被击中的双色精灵整块统一染成 9，不会留下混色残影。
#   inference_guardrails:
#     - 不使用颜色编号大小做任何排序或算术；颜色只作为可见类别标签。
#     - 通过“必须同时出现两种 apex 色 + 至少一个 non-legacy apex + 至少一个命中或碰撞事件”避免样本退化成原题或退化成纯装饰换色。
#   solver_assumptions_guaranteed_by_generator:
#     - 每个非背景组件恰好 3 像素，且归一化轮廓属于允许的 L/V 朝向集合。
#     - 每个组件恰有一个 apex 色像素和两个同色 arm 像素。
#     - apex 色 2 只配对角方向与距离 3；apex 色 8 只配轴向方向与距离 6。
#     - 每张图至少含 1 个对角 3 格组件、1 个轴向 6 格组件、1 个 non-legacy apex 组件。

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1

DIAG_APEX_COLOR = 2
AXIS_APEX_COLOR = 8
PROJECTILE_DIST_BY_APEX = {
    DIAG_APEX_COLOR: 3,
    AXIS_APEX_COLOR: 6,
}
ARM_COLORS = [0, 3, 4, 5, 6]

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 500

BASE_SHAPES = [
    ((0, 0), (0, 1), (1, 0)),
    ((0, 1), (1, 0), (1, 2)),
]


def _normalize_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _all_orientations(shape: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    out: set[tuple[tuple[int, int], ...]] = set()
    pts = list(shape)
    for flip_r in (1, -1):
        for flip_c in (1, -1):
            cur = [(r * flip_r, c * flip_c) for r, c in pts]
            for _ in range(4):
                cur = [(c, -r) for r, c in cur]
                out.add(_normalize_cells(cur))
    return out


def _legacy_apex(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    pts = set(shape)
    deg4 = {}
    deg8 = {}
    for r, c in shape:
        deg4[(r, c)] = sum(1 for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if (rr, cc) in pts)
        deg8[(r, c)] = sum(
            1
            for rr in range(r - 1, r + 2)
            for cc in range(c - 1, c + 2)
            if (rr, cc) != (r, c) and (rr, cc) in pts
        )
    corners = [p for p, d in deg4.items() if d == 2]
    if corners:
        return corners[0]
    return [p for p, d in deg8.items() if d == 2][0]


def _shape_entries() -> list[dict]:
    entries: list[dict] = []
    seen: set[tuple[tuple[tuple[int, int], ...], tuple[int, int]]] = set()
    for base in BASE_SHAPES:
        for shape in sorted(_all_orientations(base)):
            legacy = _legacy_apex(shape)
            for apex in shape:
                others = [p for p in shape if p != apex]
                sr = sum(r - apex[0] for r, _ in others)
                sc = sum(c - apex[1] for _, c in others)
                dr = 0 if sr == 0 else (-1 if sr > 0 else 1)
                dc = 0 if sc == 0 else (-1 if sc > 0 else 1)
                dir_norm = abs(dr) + abs(dc)
                if dir_norm not in (1, 2):
                    continue
                key = (shape, apex)
                if key in seen:
                    continue
                seen.add(key)
                apex_color = AXIS_APEX_COLOR if dir_norm == 1 else DIAG_APEX_COLOR
                entries.append(
                    {
                        "shape": shape,
                        "apex_rel": apex,
                        "dir": (dr, dc),
                        "apex_color": apex_color,
                        "dist": PROJECTILE_DIST_BY_APEX[apex_color],
                        "legacy_apex": apex == legacy,
                    }
                )
    return entries


SHAPE_ENTRIES = _shape_entries()
ALLOWED_SHAPES = {entry["shape"] for entry in SHAPE_ENTRIES}


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _sprite_info(
    grid: np.ndarray, comp: list[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int], int, int] | None:
    if len(comp) != 3:
        return None
    if _normalize_cells(comp) not in ALLOWED_SHAPES:
        return None

    color_counts = Counter(int(grid[r, c]) for r, c in comp)
    if sorted(color_counts.values()) != [1, 2]:
        return None

    apex_color = next((col for col, cnt in color_counts.items() if cnt == 1), None)
    if apex_color not in PROJECTILE_DIST_BY_APEX:
        return None

    apex = next((rc for rc in comp if int(grid[rc]) == apex_color), None)
    if apex is None:
        return None

    ar, ac = apex
    others = [rc for rc in comp if rc != apex]
    sr = sum(r - ar for r, _ in others)
    sc = sum(c - ac for _, c in others)
    dr = 0 if sr == 0 else (-1 if sr > 0 else 1)
    dc = 0 if sc == 0 else (-1 if sc > 0 else 1)
    dir_norm = abs(dr) + abs(dc)
    if dir_norm not in (1, 2):
        return None
    if apex_color == DIAG_APEX_COLOR and dir_norm != 2:
        return None
    if apex_color == AXIS_APEX_COLOR and dir_norm != 1:
        return None
    return apex, (dr, dc), PROJECTILE_DIST_BY_APEX[apex_color], apex_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch projectiles from two-color triangle sprites whose apex color encodes both shot family and range.

    Input structure:
    - The background is the most common color. Each non-background 8-connected component has exactly 3 pixels and uses a 1+2 color split: one uniquely colored apex pixel plus two same-colored arm pixels.
    - The occupied silhouette is still one of the original tiny triangle outlines, but the singleton apex color is now essential: it identifies the firing cell and tells whether the shot is diagonal or axis-aligned, together with the travel distance.

    Rule:
    - For every valid sprite, treat the uniquely colored pixel as the apex. Sum the two arm offsets relative to that apex and fire in the opposite signed direction.
    - Apex color 2 means the direction must be diagonal and the projectile lands 3 cells away; apex color 8 means the direction must be axis-aligned and the projectile lands 6 cells away.
    - If the landing cell lies on another sprite, recolor that entire hit sprite to maroon (9). If multiple projectiles land on the same empty background cell, paint that cell blue (1); otherwise a lone empty-cell landing becomes maroon (9).

    Key insight: Geometry alone is no longer enough. You must first find the singleton-colored apex, then use that apex color as a lookup key for both the allowed direction family and the shooting distance.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps = _cc8(g != bg)
    cell_to_comp: dict[tuple[int, int], int] = {}
    valid_components: dict[int, list[tuple[int, int]]] = {}
    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)

    for cid, comp in enumerate(comps):
        info = _sprite_info(g, comp)
        if info is None:
            continue
        apex, direction, dist, _apex_color = info
        valid_components[cid] = comp
        for rc in comp:
            cell_to_comp[rc] = cid
        rr = apex[0] + dist * direction[0]
        cc = apex[1] + dist * direction[1]
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters[(rr, cc)].append(cid)

    hit_components = {cell_to_comp[dest] for dest in dest_to_shooters if dest in cell_to_comp}

    out = g.copy()
    for cid in hit_components:
        for r, c in valid_components[cid]:
            out[r, c] = HIT_COLOR

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

    return out


def _halo(cells: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for entry in SHAPE_ENTRIES:
        shape = entry["shape"]
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        for r0 in range(H - max_r):
            for c0 in range(W - max_c):
                cells = tuple((r0 + r, c0 + c) for r, c in shape)
                apex = (r0 + entry["apex_rel"][0], c0 + entry["apex_rel"][1])
                dr, dc = entry["dir"]
                dist = entry["dist"]
                dest = (apex[0] + dist * dr, apex[1] + dist * dc)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                placement = {
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "apex": apex,
                    "dir": (dr, dc),
                    "dest": dest,
                    "apex_color": entry["apex_color"],
                    "dist": dist,
                    "legacy_apex": entry["legacy_apex"],
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()
AXIS_PLACEMENTS = [p for p in PLACEMENTS if p["apex_color"] == AXIS_APEX_COLOR]
DIAG_PLACEMENTS = [p for p in PLACEMENTS if p["apex_color"] == DIAG_APEX_COLOR]
NONLEGACY_DIAG_PLACEMENTS = [
    p for p in PLACEMENTS if p["apex_color"] == DIAG_APEX_COLOR and not p["legacy_apex"]
]


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_input_sprite(grid: np.ndarray, placement: dict, arm_color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = arm_color
    ar, ac = placement["apex"]
    grid[ar, ac] = placement["apex_color"]


def _paint_hit_sprite(grid: np.ndarray, placement: dict) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = HIT_COLOR


def _choose_from_pool(
    rng: random.Random,
    pool: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = pool[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
            continue
        if _fits(placement, occupied, reserved_bg):
            return placement
    return None


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, reserved_bg)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ2:
                    continue
                return dest, [first, second]
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, reserved_bg)]
        if candidates:
            return rng.choice(candidates), cell
    return None


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 2) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, (n_sprites - 2) // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 2:
                continue
            if unique + e_coll < 1:
                continue
            if e_hit + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(rng, e_coll, sprites, occupied, reserved_bg) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            sprites.append({"placement": placement, "target": dest, "target_kind": "bg"})
            occupied.update(placement["cells"])
    return True


def _place_mandatory_uniques(rng, sprites, occupied, reserved_bg) -> bool:
    axis = _choose_from_pool(rng, AXIS_PLACEMENTS, occupied, reserved_bg)
    if axis is None:
        return False
    reserved_bg.add(axis["dest"])
    sprites.append({"placement": axis, "target": axis["dest"], "target_kind": "bg"})
    occupied.update(axis["cells"])

    off_natural = _choose_from_pool(rng, NONLEGACY_DIAG_PLACEMENTS, occupied, reserved_bg)
    if off_natural is None:
        return False
    reserved_bg.add(off_natural["dest"])
    sprites.append({"placement": off_natural, "target": off_natural["dest"], "target_kind": "bg"})
    occupied.update(off_natural["cells"])
    return True


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        sprites.append({"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx})
        occupied.update(placement["cells"])
        used_target_cells.add(cell)
    return True


def _place_unique_landings(rng, n, sprites, occupied, reserved_bg) -> bool:
    for _ in range(n):
        axis_count = sum(1 for s in sprites if s["placement"]["apex_color"] == AXIS_APEX_COLOR)
        diag_count = len(sprites) - axis_count
        preferred_pool = AXIS_PLACEMENTS if axis_count < diag_count else DIAG_PLACEMENTS
        placement = _choose_from_pool(rng, preferred_pool, occupied, reserved_bg)
        if placement is None:
            placement = _choose_from_pool(rng, PLACEMENTS, occupied, reserved_bg)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
    return True


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_hit_sprite(out, sprites[idx]["placement"])
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    comps = _cc8(grid != BG)
    metrics = {
        "valid": 0,
        "axis": 0,
        "diag": 0,
        "dist3": 0,
        "dist6": 0,
        "off_natural": 0,
    }
    for comp in comps:
        info = _sprite_info(grid, comp)
        if info is None:
            continue
        apex, _direction, dist, apex_color = info
        metrics["valid"] += 1
        if apex_color == AXIS_APEX_COLOR:
            metrics["axis"] += 1
            metrics["dist6"] += 1
        else:
            metrics["diag"] += 1
            metrics["dist3"] += 1
        legacy = _legacy_apex(_normalize_cells(comp))
        r0 = min(r for r, _ in comp)
        c0 = min(c for _, c in comp)
        if apex != (r0 + legacy[0], c0 + legacy[1]):
            metrics["off_natural"] += 1
        if dist not in (3, 6):
            return {}
    return metrics


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    comps = _cc8(grid != BG)
    if len(comps) != n_sprites:
        return False
    if not all(_sprite_info(grid, comp) is not None for comp in comps):
        return False
    metrics = _variant_metrics(grid)
    if not metrics or metrics["valid"] != n_sprites:
        return False
    return (
        metrics["axis"] >= 1
        and metrics["diag"] >= 1
        and metrics["dist3"] >= 1
        and metrics["dist6"] >= 1
        and metrics["off_natural"] >= 1
    )


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    arm_color = rng.choice(ARM_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, e_coll, _unique_total = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    if not _place_collisions(rng, e_coll, sprites, occupied, reserved_bg):
        return None
    if not _place_mandatory_uniques(rng, sprites, occupied, reserved_bg):
        return None
    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0:
        return None
    if not _place_unique_landings(rng, remaining, sprites, occupied, reserved_bg):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_input_sprite(inp, sprite["placement"], arm_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
