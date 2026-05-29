# Variant of 64efde09
# Dimension: D17 — 精灵数量与多样性 (meta)
# Change: 3-4个精灵，从模板的去重二面体变体中选取，至少观察到3种不同朝向... → 包含干扰精灵（不参与射线发射）

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg

# Design analysis
# ----------------
# variant_spec:
#   core_change:
#     - 输入里除了真正参与射线传播的模板族外，还会出现 1-2 个稠密矩形干扰精灵。
#     - 干扰精灵保持“看起来像对象”的形式，但它们既不承接 marker，也不发射输出射线。
#   concrete_visualization:
#     - 真正的模板族仍是 3-4 个 2xN / Nx2 的多色细长矩形，彼此是同一模板的旋转/翻折。
#     - 干扰精灵改成 3xL / Lx3 的多色稠密矩形，尺寸与模板族明显不同，但仍是完整对象而非噪声。
#     - marker 仍是边界上的单像素，沿着真模板族活跃边的外向射线摆放；干扰精灵不会挡住这些射线。
#   behavior_trace:
#     - 原逻辑: 把所有非单像素稠密矩形都当作同一模板族成员，再拿第一个精灵当模板去匹配其余精灵。
#     - 变体逻辑: 先把所有稠密矩形按“二面体等价类”分组，只保留那个能共同解释全部 marker 的重复族。
#     - 差异点: 干扰精灵会让“所有矩形都属于同一族”这个旧前提失效，因此 solver 必须先做族筛选。
#   complexity_calibration:
#     - 干扰精灵数量限制为 1-2 个，避免画面过满。
#     - 干扰精灵尺寸统一落在 3x4..3x6 / 4..6x3，既能看出是完整对象，又不会与模板族混成同一几何档位。
#     - 真模板族仍保留 3-4 个成员和 3 个 marker，主任务信息量不变，变体只新增“筛掉无关对象”的步骤。
#   implicit_assumption_audit:
#     - 原 `variants = _dihedral_variants(sprites_raw[0]["sub"])` 假设第一个矩形就是真模板；现在不成立。
#     - 原 `for sprite in sprites_raw: ... assert len(matches) == 1` 假设所有矩形都能匹配同一模板；现在不成立。
#     - 原射线传播循环默认 `sprites` 已是纯净模板族；现在必须先筛出候选族，再对该族做原始映射与传播。
#   placement_legality:
#     - 先放模板族和 marker，再计算所有未来会被射线经过的背景格，把这些格加入保护区。
#     - 干扰精灵只能放在保护区外，且与现有非背景对象保持间隔，避免 8 邻接连成一个大组件。
#   observability:
#     - 人类在输入里能直接看到“多数细长矩形彼此是同构模板，而另有 1-2 个更厚的矩形不属于这一族”。
#     - 输出只给真模板族加射线，干扰精灵保持原样，因此“哪些对象会响应 marker”在 2-3 个样例里可见。
#   neatness:
#     - 干扰精灵不压住 marker，不穿过任何将来射线，避免输出层叠杂乱。
#   new_solver_assumptions:
#     - 恰好存在一个重复矩形族能解释全部 marker。
#     - 该族成员仍是非方形稠密矩形，且每个朝向的活跃边可由长边颜色多数唯一确定。
#     - generator 通过族数量、尺寸差异、marker 对齐和保护区放置保证这些前提成立。


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


def _unique_dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    seen: list[np.ndarray] = []
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, sub, ids in _dihedral_variants(template):
        if any(prev.shape == sub.shape and np.array_equal(prev, sub) for prev in seen):
            continue
        seen.append(sub.copy())
        result.append((name, sub, ids))
    return result


def _canonical_signature(sub: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    reps = []
    for _, variant, _ in _dihedral_variants(sub):
        reps.append((tuple(map(int, variant.shape)), tuple(int(x) for x in variant.flatten())))
    return min(reps)


def _active_side(sub: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"

    if h > w:
        sides = [
            ((0, -1), [(r, 0) for r in range(h)]),
            ((0, 1), [(r, w - 1) for r in range(h)]),
        ]
    else:
        sides = [
            ((-1, 0), [(0, c) for c in range(w)]),
            ((1, 0), [(h - 1, c) for c in range(w)]),
        ]

    scored = []
    for direction, coords in sides:
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((max(Counter(vals).values()), direction, set(coords)))

    scored.sort(key=lambda item: item[0], reverse=True)
    assert scored[0][0] > scored[1][0], "active side must be structurally unique"
    return scored[0][1], scored[0][2]


def _first_marker_on_ray(
    g: np.ndarray,
    bg: int,
    markers: dict[tuple[int, int], int],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> int | None:
    H, W = g.shape
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if int(g[r, c]) != bg:
            return markers.get((r, c))
        r += dr
        c += dc
    return None


def _parse_components(g: np.ndarray, bg: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False)
    sprites_raw: list[dict] = []
    markers: dict[tuple[int, int], int] = {}
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            markers[(r, c)] = int(g[r, c])
            continue
        assert area == h * w, "all non-marker components must be dense rectangles"
        sprites_raw.append({"bbox": (r0, c0, r1, c1), "sub": crop_bbox(g, (r0, c0, r1, c1))})
    return sprites_raw, markers


def _orient_family(sprites_raw: list[dict]) -> list[dict] | None:
    variants = _unique_dihedral_variants(sprites_raw[0]["sub"])
    sprites: list[dict] = []
    for sprite in sprites_raw:
        matches = [(ids, name) for name, sub, ids in variants if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])]
        if len(matches) != 1:
            return None
        ids, _name = matches[0]
        try:
            direction, active = _active_side(sprite["sub"])
        except AssertionError:
            return None
        sprites.append({**sprite, "ids": ids, "direction": direction, "active": active})
    return sprites


def _candidate_family(
    g: np.ndarray,
    bg: int,
    markers: dict[tuple[int, int], int],
    family_raw: list[dict],
) -> dict | None:
    sprites = _orient_family(family_raw)
    if sprites is None:
        return None

    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["active"]:
            marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if marker_color is None:
                continue
            template_id = int(sprite["ids"][lr, lc])
            if marker_color in color_to_template_id and color_to_template_id[marker_color] != template_id:
                return None
            if template_id in template_id_to_color and template_id_to_color[template_id] != marker_color:
                return None
            color_to_template_id[marker_color] = template_id
            template_id_to_color[template_id] = marker_color

    if set(color_to_template_id) != set(markers.values()):
        return None
    return {"sprites": sprites, "color_to_template_id": color_to_template_id}


def _select_true_family(g: np.ndarray, bg: int, sprites_raw: list[dict], markers: dict[tuple[int, int], int]) -> dict:
    families: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict]] = defaultdict(list)
    for sprite in sprites_raw:
        families[_canonical_signature(sprite["sub"])].append(sprite)

    candidates = []
    for family in families.values():
        if len(family) < 2:
            continue
        candidate = _candidate_family(g, bg, markers, family)
        if candidate is not None:
            candidates.append(candidate)

    assert candidates, "expected one repeated sprite family that explains all markers"
    best_size = max(len(candidate["sprites"]) for candidate in candidates)
    best = [candidate for candidate in candidates if len(candidate["sprites"]) == best_size]
    assert len(best) == 1, "template family must be uniquely identifiable"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate marker-labeled ports only across the repeated template family and leave unrelated rectangles untouched.

    Input structure:
    - bg is the most common color.
    - The grid contains one repeated family of dense non-square rectangles that are dihedral transforms of a shared template, plus 1-2 other dense rectangular distractors with different content.
    - Isolated single-pixel markers lie on outward rays from active-edge cells of the true template family; distractors have no such markers.

    Rule:
    - Partition the rectangles into dihedral-equivalence families and keep the unique repeated family whose members collectively account for every marker.
    - Within that family, the active edge is the long side with the stronger same-color majority, and the first marker seen along an outward ray assigns a color to that template port.
    - Emit each assigned color outward from the corresponding port of every member of the true family through bg cells until blocked or out of bounds; distractors remain unchanged.

    Key insight:
    - The extra rectangles are decoys. Marker alignment reveals which repeated shape family carries the transferable port identities, and only that family should broadcast rays under rotation/reflection.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    sprites_raw, markers = _parse_components(g, bg)

    assert len(sprites_raw) >= 3, "expected template sprites plus distractors"
    assert markers, "expected isolated control markers"
    assert len(set(markers.values())) == len(markers), "control marker colors must be unique"

    family = _select_true_family(g, bg, sprites_raw, markers)
    sprites = family["sprites"]
    color_to_template_id = family["color_to_template_id"]

    out = g.copy()
    H, W = g.shape
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for color, template_id in color_to_template_id.items():
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            assert (lr, lc) in sprite["active"], "mapped port must land on the sprite active edge"

            r = r0 + lr + dr
            c = c0 + lc + dc
            while 0 <= r < H and 0 <= c < W:
                if int(g[r, c]) != bg:
                    break
                assert int(out[r, c]) in (bg, color), "control rays must not conflict"
                out[r, c] = color
                r += dr
                c += dc

    return out


# --------------- generator (variant D17) ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_CLEARANCE = 2
_PLACEMENT_RETRIES = 100
_GEN_RETRIES = 400
_DISTRACTOR_COUNT_RANGE = (1, 2)
_DISTRACTOR_LONG_CHOICES = [4, 5, 6]
_DISTRACTOR_THICKNESS = 3


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    fill = others if others else colors
    for _ in range(n - majority):
        col.append(random.choice(fill))
    random.shuffle(col)
    return col


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)

    active_majority = random.randint(_ACTIVE_MAJORITY_RANGE[0], min(N, _ACTIVE_MAJORITY_RANGE[1]))
    left_col = _build_column(N, template_colors, active_majority, random.choice(template_colors))

    max_right_majority = active_majority - 1
    right_majority = random.randint(max(2, N // num_colors), max_right_majority)
    right_dominant = random.choice(template_colors)
    others = [c for c in template_colors if c != right_dominant]
    if not others:
        return None
    right_col = _build_column(N, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(int(x) for x in template.flat)) < num_colors:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _deduplicated_variants(template: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
    result = []
    for _, sub, ids_arr in _unique_dihedral_variants(template):
        try:
            direction, active_set = _active_side(sub)
        except AssertionError:
            continue
        result.append((sub, ids_arr, direction, list(active_set)))
    return result


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int]:
    pad = _PLACEMENT_CLEARANCE
    if dr == 0 and dc == -1:
        return random.randint(0, H - h), random.randint(pad, W - w)
    if dr == 0 and dc == 1:
        return random.randint(0, H - h), random.randint(0, W - w - pad - 1)
    if dr == -1 and dc == 0:
        return random.randint(pad, H - h), random.randint(0, W - w)
    return random.randint(0, H - h - pad - 1), random.randint(0, W - w)


def _occupied_set(placements: list[tuple]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _has_padded_conflict(
    H: int,
    W: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    pad: int,
) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H - 1, r0 + h - 1 + pad)
    cc1 = min(W - 1, c0 + w - 1 + pad)
    for rr in range(rr0, rr1 + 1):
        for cc in range(cc0, cc1 + 1):
            if (rr, cc) in occupied:
                return True
    return False


def _place_sprites(H: int, W: int, variants) -> list[tuple] | None:
    placements: list[tuple] = []
    for sub, ids_arr, direction, active_coords in variants:
        h, w = sub.shape
        dr, dc = direction
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w, dr, dc)
            occ = _occupied_set(placements)
            if _has_padded_conflict(H, W, r0, c0, h, w, occ, _PLACEMENT_PAD):
                continue
            placements.append((r0, c0, sub, ids_arr, direction, active_coords))
            placed = True
            break
        if not placed:
            return None
    return placements


def _boundary_position(r: int, c: int, dr: int, dc: int, H: int, W: int) -> tuple[int, int]:
    if dr == 0 and dc == -1:
        return r, 0
    if dr == 0 and dc == 1:
        return r, W - 1
    if dr == -1 and dc == 0:
        return 0, c
    return H - 1, c


def _is_isolated_marker_site(grid: np.ndarray, r: int, c: int) -> bool:
    H, W = grid.shape
    for rr in range(max(0, r - 1), min(H, r + 2)):
        for cc in range(max(0, c - 1), min(W, c + 2)):
            if (rr, cc) == (r, c):
                continue
            if int(grid[rr, cc]) != _BG:
                return False
    return True


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> dict[int, int] | None:
    H, W = grid.shape
    N = ids_base.shape[0]
    num_sprites = len(sprite_placements)

    active_template_ids = [int(ids_base[r, 0]) for r in range(N)]
    if len(active_template_ids) < _NUM_MARKERS:
        return None
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return None
    marker_colors = random.sample(available, _NUM_MARKERS)

    for color, tid in zip(marker_colors, labeled_ids):
        si = random.randint(0, num_sprites - 1)
        r0, c0, _, ids_arr, direction, active_coords = sprite_placements[si]
        dr, dc = direction

        locs = np.argwhere(ids_arr == tid)
        if locs.shape[0] != 1:
            return None
        lr, lc = map(int, locs[0])
        if (lr, lc) not in set(active_coords):
            return None

        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)
        cr, cc = r0 + lr + dr, c0 + lc + dc
        while (cr, cc) != (marker_r, marker_c) and 0 <= cr < H and 0 <= cc < W:
            if int(grid[cr, cc]) != _BG:
                return None
            cr += dr
            cc += dc

        if int(grid[marker_r, marker_c]) != _BG:
            return None
        if not _is_isolated_marker_site(grid, marker_r, marker_c):
            return None
        grid[marker_r, marker_c] = color

    return {color: tid for color, tid in zip(marker_colors, labeled_ids)}


def _ray_cells(grid: np.ndarray, sprite_placements: list[tuple], color_to_template_id: dict[int, int]) -> set[tuple[int, int]]:
    H, W = grid.shape
    cells: set[tuple[int, int]] = set()
    for r0, c0, _sub, ids_arr, direction, _active_coords in sprite_placements:
        dr, dc = direction
        for template_id in color_to_template_id.values():
            locs = np.argwhere(ids_arr == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            r = r0 + lr + dr
            c = c0 + lc + dc
            while 0 <= r < H and 0 <= c < W:
                if int(grid[r, c]) != _BG:
                    break
                cells.add((r, c))
                r += dr
                c += dc
    return cells


def _design_distractor(length: int, banned_signatures: set[tuple], banned_colors: set[int]) -> np.ndarray | None:
    for _ in range(_PLACEMENT_RETRIES):
        h, w = (_DISTRACTOR_THICKNESS, length) if random.random() < 0.5 else (length, _DISTRACTOR_THICKNESS)
        palette = [c for c in _NON_BG_COLORS if c not in banned_colors]
        if len(palette) < 2:
            palette = list(_NON_BG_COLORS)
        num_colors = random.randint(2, min(3, len(palette)))
        colors = random.sample(palette, num_colors)
        sub = np.array([[random.choice(colors) for _ in range(w)] for _ in range(h)], dtype=int)
        if len(set(int(x) for x in sub.flat)) < 2:
            continue
        signature = _canonical_signature(sub)
        if signature in banned_signatures:
            continue
        return sub
    return None


def _place_distractors(
    grid: np.ndarray,
    template: np.ndarray,
    marker_colors: set[int],
    protected_cells: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    distractor_count = random.randint(*_DISTRACTOR_COUNT_RANGE)
    available_lengths = random.sample(_DISTRACTOR_LONG_CHOICES, distractor_count)

    occupied = {(int(r), int(c)) for r, c in np.argwhere(grid != _BG)}
    banned_signatures = {_canonical_signature(template)}
    banned_colors = set(marker_colors)

    for length in available_lengths:
        sub = _design_distractor(length, banned_signatures, banned_colors)
        if sub is None:
            return False
        h, w = sub.shape
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0 = random.randint(0, H - h)
            c0 = random.randint(0, W - w)
            if _has_padded_conflict(H, W, r0, c0, h, w, occupied, _PLACEMENT_PAD):
                continue
            cells = {(r0 + r, c0 + c) for r in range(h) for c in range(w)}
            if cells & protected_cells:
                continue
            grid[r0:r0 + h, c0:c0 + w] = sub
            occupied |= cells
            banned_signatures.add(_canonical_signature(sub))
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors = design

    unique_variants = _deduplicated_variants(template)
    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    if len(unique_variants) < max(3, num_sprites):
        return None
    chosen = random.sample(unique_variants, num_sprites)

    placements = _place_sprites(H, W, chosen)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        grid[r0:r0 + h, c0:c0 + w] = sub

    color_to_template_id = _place_markers(grid, placements, ids_base, template_colors)
    if color_to_template_id is None:
        return None

    protected_cells = _ray_cells(grid, placements, color_to_template_id) | {(int(r), int(c)) for r, c in np.argwhere(grid != _BG)}
    if not _place_distractors(grid, template, set(color_to_template_id), protected_cells):
        return None

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
