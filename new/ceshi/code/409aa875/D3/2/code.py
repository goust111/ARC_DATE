# Variant of 409aa875
# Dimension: D3 — 精灵形状与方向编码 (src)
# Change: 两种3像素三角形：L形（轴对齐直角，占2x2的3个角）和V形（顶点+两条对角臂），形状决定发射方向（L对角线/V轴方向）... → 双色精灵（顶点色与臂色不同）
#
# Design analysis
# spec_variant:
#   core_change:
#     - 非背景精灵仍然只占 3 个像素，外轮廓仍限制在原题的 L/V 两类三角轮廓内。
#     - 但每个精灵改成 2 色：1 个“顶点色”像素 + 2 个同色“臂”像素。
#     - 顶点色固定只有两类：2 表示该精灵必须发射对角线方向，8 表示必须发射轴向方向。
#   concrete_behavior:
#     - 先在一个 8 连通 3 像素组件内找唯一颜色的那个像素，它就是 apex。
#     - 再看另外两个臂像素相对 apex 的偏移；把两个偏移相加，投射方向就是这个和向量的反方向符号。
#     - 若 apex 色为 2，则该符号方向必须是对角；若 apex 色为 8，则必须是轴向。
#     - 投射物依旧落在 apex + 5 * dir；命中精灵整块变 9；双重落在空背景格则该格变 1；单次落空背景变 9。
#   behavior_trace:
#     - 原题在单色 L 形里，角点天然就是 apex，方向由形状直接给出。
#     - 本变体里，同一个 3 像素轮廓上可以把唯一色像素放在不同位置；例如 V 形若唯一色在中心则发轴向，若唯一色在某个腿端则发对角。
#     - 因此“哪个像素是发射点”不再能靠原题的单色几何规则直接决定，必须先读颜色角色。
#   complexity_calibration:
#     - 只保留两种 apex 色，且整张图只用一种臂色，避免颜色种类过多。
#     - 每张图强制同时出现轴向 apex 色与对角 apex 色，并至少出现一个“非原题天然 apex”的组件，保证变体肉眼可见。
#   implicit_assumptions_review:
#     - 原 solver 逐色做连通块；双色后会把一个精灵拆成 1 像素 + 2 像素两块，必须改成对所有非背景像素整体做 8 连通分量。
#     - 原 solver 用单色组件自身的度数找 apex；现在 apex 由“1+2 配色”决定，度数只用于验证轮廓是否仍属 L/V 家族。
#     - 原 generator 给整块刷同一色；现在必须分别刷 apex 色和臂色，否则新规则不可见。
#   placement_legality:
#     - 仍保持 16x16、背景 7、sprite 数 4..9、最小 Chebyshev 间隔 >= 2、目标点必须在界内。
#     - generator 继续 output-first：先构造命中/碰撞事件，再把 sprite 画回输入，并用 main 反验。
#   visual_cleanliness:
#     - 输入侧只有背景色、1 种臂色、2 种 apex 色，层次简单。
#     - 输出侧只新增 9 和可能的 1；被命中的双色精灵整块同染 9，不会残留混色噪声。
#   color_inference:
#     - 颜色不参与排序或数值运算；只承担“顶点角色 + 方向类型”这一可见语义。
#   solver_assumptions_guaranteed_by_generator:
#     - 每个非背景组件恰好 3 像素、恰好两种颜色且频次为 1+2。
#     - 组件轮廓只来自原题 L/V 模板。
#     - apex 色只会是 2 或 8，且与几何推出的 axis/diag 类型一致。

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color


H = W = 16
BG = 7
PROJECTILE_DIST = 5
HIT_COLOR = 9
COLLISION_COLOR = 1

DIAG_APEX_COLOR = 2
AXIS_APEX_COLOR = 8
ARM_COLORS = [0, 3, 4, 5, 6]

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 400

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
                entries.append(
                    {
                        "shape": shape,
                        "apex_rel": apex,
                        "dir": (dr, dc),
                        "apex_color": AXIS_APEX_COLOR if dir_norm == 1 else DIAG_APEX_COLOR,
                        "legacy_apex": apex == legacy,
                    }
                )
    return entries


SHAPE_ENTRIES = _shape_entries()
ALLOWED_SHAPES = {entry["shape"] for entry in SHAPE_ENTRIES}


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _sprite_apex_and_dir(grid: np.ndarray, comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(comp) != 3:
        return None
    if _normalize_cells(comp) not in ALLOWED_SHAPES:
        return None

    color_counts = Counter(int(grid[r, c]) for r, c in comp)
    if sorted(color_counts.values()) != [1, 2]:
        return None

    apex_color = next((col for col, cnt in color_counts.items() if cnt == 1), None)
    if apex_color not in (DIAG_APEX_COLOR, AXIS_APEX_COLOR):
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
    return apex, (dr, dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch projectiles from two-color 3-pixel triangle sprites and mark the landing effects.

    Input structure:
    - The background is the most common color. Each non-background 8-connected component has 3 pixels and uses one shared arm color plus one differently colored apex pixel.
    - The occupied silhouette is still one of the original tiny triangle patterns (L-like or V-like), but the singleton apex color identifies both the launcher cell and whether its shot must be diagonal or axis-aligned.

    Rule:
    - For every valid sprite, treat the uniquely colored pixel as the apex and the other two pixels as arms.
    - Add the two arm offsets relative to the apex; the projectile travels 5 cells in the opposite signed direction. Apex color 2 means that direction must be diagonal, while apex color 8 means it must be axis-aligned.
    - If a projectile lands on an existing sprite, recolor that entire sprite to maroon (9). If multiple projectiles land on the same empty background cell, paint it blue (1); otherwise a lone empty-cell landing becomes maroon (9).

    Key insight: The shape no longer tells you by itself which pixel is firing. You must first use the singleton apex color to find the tip, then aim away from the pair of arm pixels.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps = _cc8(g != bg)
    cell_to_comp: dict[tuple[int, int], int] = {}
    valid_components: dict[int, list[tuple[int, int]]] = {}
    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)

    for cid, comp in enumerate(comps):
        ad = _sprite_apex_and_dir(g, comp)
        if ad is None:
            continue
        valid_components[cid] = comp
        for rc in comp:
            cell_to_comp[rc] = cid
        (ar, ac), (dr, dc) = ad
        rr = ar + PROJECTILE_DIST * dr
        cc = ac + PROJECTILE_DIST * dc
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
                dest = (apex[0] + PROJECTILE_DIST * dr, apex[1] + PROJECTILE_DIST * dc)
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
                    "legacy_apex": entry["legacy_apex"],
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()
AXIS_PLACEMENTS = [p for p in PLACEMENTS if p["apex_color"] == AXIS_APEX_COLOR]
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
    axis = 0
    diag = 0
    off_natural = 0
    valid = 0
    for comp in comps:
        ad = _sprite_apex_and_dir(grid, comp)
        if ad is None:
            continue
        valid += 1
        apex, _ = ad
        apex_color = int(grid[apex])
        if apex_color == AXIS_APEX_COLOR:
            axis += 1
        else:
            diag += 1
        legacy = _legacy_apex(_normalize_cells(comp))
        r0 = min(r for r, _ in comp)
        c0 = min(c for _, c in comp)
        if apex != (r0 + legacy[0], c0 + legacy[1]):
            off_natural += 1
    return {"valid": valid, "axis": axis, "diag": diag, "off_natural": off_natural}


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    comps = _cc8(grid != BG)
    if len(comps) != n_sprites:
        return False
    if not all(_sprite_apex_and_dir(grid, comp) is not None for comp in comps):
        return False
    metrics = _variant_metrics(grid)
    return metrics["axis"] >= 1 and metrics["diag"] >= 1 and metrics["off_natural"] >= 1


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    arm_color = rng.choice(ARM_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, e_coll, unique_total = counts

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
    if unique_total < 2:
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
