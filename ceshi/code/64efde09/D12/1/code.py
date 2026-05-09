# Variant of 64efde09
# Dimension: D12 — 活跃边识别规则 (meta)
# Change: 活跃边是长边中最大同色计数（majority count）严格更高的那一边... → 活跃边由颜色种类数最少的边决定

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg

# Design analysis
# spec:
#   variant_core:
#     old_rule: 对每个 N×2 或 2×N 精灵，比较两条长边的最大同色计数，majority 更高的一边是 active。
#     new_rule: 改为比较两条长边上出现了多少种颜色，颜色种类更少的一边是 active。
#     generator_requirement: 不能只生成“新旧两种规则恰好选中同一边”的模板，否则视觉上退化回原题。
#   concrete_trace:
#     template_example:
#       left:  [A, A, A, A, B, B, B]
#       right: [A, A, A, A, A, B, C]
#     old_behavior:
#       left majority = 4, right majority = 5，所以旧规则会把右边当 active。
#     new_behavior:
#       left distinct = 2, right distinct = 3，所以新规则会把左边当 active。
#     visible_effect:
#       marker 只会贴在左边对应的边界射线上，输出也只会从所有同构精灵的左侧对应端口向外发射；
#       如果误用旧规则，marker 会被解释为“没贴在 active 边上”，或者输出会落到另一侧。
#   complexity_calibration:
#     kept:
#       - 精灵仍然是单个共享模板的二面体变换。
#       - marker 仍然是边界单像素，并给模板端口着色。
#       - 输出仍然是把这些端口颜色复制到所有同构精灵并沿外法线发射。
#     changed:
#       - 模板构造固定成 active 边颜色种类数 = 2，另一边颜色种类数 = 3。
#       - 同时强制 inactive 边的 majority 更高，让新旧规则稳定分歧。
#     reason:
#       这样差异肉眼可见，但搜索空间仍然小，不会把任务变成“同时猜多种判据”。
#   implicit_assumptions_audit:
#     solver_lines:
#       - 原 `_active_side()` 假设“majority 差值严格唯一”；这里替换成“distinct-count 差值严格唯一”。
#       - `main()` 里 marker 收集、模板 id 映射、射线绘制都只依赖 active 边坐标集合，不依赖 majority 本身，因此无需改动。
#       - `assert (lr, lc) in sprite["active"]` 变得更敏感：generator 必须保证 marker 标记的模板 id 真落在新 active 边上。
#     generator_guards:
#       - 模板必须满足 new_active != old_active，避免退化样本。
#       - 模板仍需避免二面体匹配歧义；最后继续用 `main(grid)` 做端到端验收。
#   placement_legality:
#     - 精灵仍然按 active 边朝向边界的方式摆放，保证 marker 到端口之间只有背景。
#     - 精灵之间保留 padding，最终再用 solver 验证不同颜色射线不冲突。
#   visual_observability:
#     - 排序依据只使用“每条边上能看见的颜色种类数”，不使用颜色 id 的大小。
#     - active 边始终更“单纯”，另一边颜色更杂，同时另一边 often 拥有更长同色段；这正好让人能从少量样例区分新旧规则。
#   cleanliness:
#     - 每个样例仍只有 3 个 marker、3-4 个精灵，避免边界和射线过度拥挤。
#   color_inference:
#     - marker 颜色只承担标签作用，仍与模板颜色集合分离；变体不引入新的颜色映射规则。


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


def _long_sides(sub: np.ndarray) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"
    if h > w:
        return [
            ((0, -1), [(r, 0) for r in range(h)]),
            ((0, 1), [(r, w - 1) for r in range(h)]),
        ]
    return [
        ((-1, 0), [(0, c) for c in range(w)]),
        ((1, 0), [(h - 1, c) for c in range(w)]),
    ]


def _active_side(sub: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    scored = []
    for direction, coords in _long_sides(sub):
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((len(set(vals)), direction, set(coords)))

    scored.sort(key=lambda item: item[0])
    assert scored[0][0] < scored[1][0], "active side must have uniquely lowest color diversity"
    return scored[0][1], scored[0][2]


def _majority_side(sub: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    scored = []
    for direction, coords in _long_sides(sub):
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((max(Counter(vals).values()), direction, set(coords)))

    scored.sort(key=lambda item: item[0], reverse=True)
    assert scored[0][0] > scored[1][0], "majority side must be structurally unique"
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer boundary marker labels from one low-diversity long edge to the matching ports of every transformed sprite.

    Input structure:
    - The background is the most frequent color.
    - Several solid N-by-2 or 2-by-N rectangles are dihedral transforms of one shared multicolor template.
    - A few isolated single-pixel markers lie on boundary rays extending outward from the long edge of some sprites whose edge colors are least diverse.

    Rule:
    - For each sprite, inspect its two long sides and choose the side that uses fewer distinct colors as the active edge.
    - If the first non-background cell on an outward ray from an active-edge cell is a marker, that marker color labels the corresponding template port.
    - For every sprite orientation, emit each learned port color outward from the matching active-edge cell until the ray hits a non-background cell or leaves the grid.

    Key insight:
    - Marker positions encode template-relative ports rather than fixed coordinates: once a color labels one template edge cell, the same cell is found on every rotated or reflected copy and the ray is redrawn there.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
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
        else:
            assert area == h * w, "sprite components must be dense rectangles"
            sprites_raw.append({"bbox": (r0, c0, r1, c1), "sub": crop_bbox(g, (r0, c0, r1, c1))})

    assert len(sprites_raw) >= 2, "expected multiple transformed rectangular sprites"
    assert markers, "expected isolated control markers"
    assert len(set(markers.values())) == len(markers), "control marker colors must be unique"

    variants = _dihedral_variants(sprites_raw[0]["sub"])
    sprites = []
    for sprite in sprites_raw:
        matches = [
            (ids, name)
            for name, sub, ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "each sprite must match the template in exactly one orientation"
        ids, _name = matches[0]
        direction, active = _active_side(sprite["sub"])
        sprites.append({**sprite, "ids": ids, "direction": direction, "active": active})

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
            if marker_color in color_to_template_id:
                assert color_to_template_id[marker_color] == template_id
            if template_id in template_id_to_color:
                assert template_id_to_color[template_id] == marker_color
            color_to_template_id[marker_color] = template_id
            template_id_to_color[template_id] = marker_color

    assert set(color_to_template_id) == set(markers.values()), "every control marker must label one active-edge cell"

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


# --------------- generator (D12) ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_CLEARANCE = 2
_PLACEMENT_RETRIES = 100
_GEN_RETRIES = 300


def _build_side(assignments: list[tuple[int, int]]) -> list[int]:
    vals: list[int] = []
    for color, count in assignments:
        vals.extend([color] * count)
    random.shuffle(vals)
    return vals


def _expected_direction_for_column(active_col_idx: int) -> tuple[int, int]:
    return (0, -1) if active_col_idx == 0 else (0, 1)


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int], int] | None:
    """Create an Nx2 canonical template where low diversity and high majority point to opposite long sides."""
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)
    active_on_left = random.choice([True, False])
    active_col_idx = 0 if active_on_left else 1

    active_colors = random.sample(template_colors, 2)
    remaining = [c for c in template_colors if c not in active_colors]

    if num_colors == 3:
        other_palette = active_colors + [remaining[0]]
    else:
        shared = random.choice(active_colors)
        other_palette = [shared] + random.sample(remaining, 2)
    assert len(set(other_palette)) == 3

    active_majority_low = (N + 1) // 2
    active_majority_high = N - 3
    if active_majority_low > active_majority_high:
        return None
    active_majority = random.randint(active_majority_low, active_majority_high)
    inactive_majority = random.randint(active_majority + 1, N - 2)

    active_dominant = random.choice(active_colors)
    active_other = next(c for c in active_colors if c != active_dominant)
    active_col = _build_side(
        [
            (active_dominant, active_majority),
            (active_other, N - active_majority),
        ]
    )

    inactive_dominant = random.choice(other_palette)
    inactive_others = [c for c in other_palette if c != inactive_dominant]
    rest = N - inactive_majority
    split = random.randint(1, rest - 1)
    inactive_col = _build_side(
        [
            (inactive_dominant, inactive_majority),
            (inactive_others[0], split),
            (inactive_others[1], rest - split),
        ]
    )

    if len(set(active_col)) != 2 or len(set(inactive_col)) != 3:
        return None

    columns = [active_col, inactive_col] if active_on_left else [inactive_col, active_col]
    template = np.column_stack(columns).astype(int)

    if len(set(map(int, template.flat))) != num_colors:
        return None

    new_direction, _ = _active_side(template)
    old_direction, _ = _majority_side(template)
    if new_direction != _expected_direction_for_column(active_col_idx):
        return None
    if old_direction == new_direction:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors, active_col_idx


def _deduplicated_variants(
    template: np.ndarray, ids_base: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
    """Return unique dihedral variants and their new-rule active edge."""
    raw = _dihedral_variants(template)
    raw_ids = _dihedral_variants(ids_base)

    seen: list[np.ndarray] = []
    result = []
    for (_, sub, _), (_, _, ids_arr) in zip(raw, raw_ids):
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            direction, active_set = _active_side(sub)
        except AssertionError:
            continue
        seen.append(sub)
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


def _place_sprites(H: int, W: int, variants) -> list[tuple] | None:
    placements: list[tuple] = []
    pad = _PLACEMENT_PAD
    for sub, ids_arr, direction, active_coords in variants:
        h, w = sub.shape
        dr, dc = direction
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w, dr, dc)
            occ = _occupied_set(placements)
            if any((r0 + r, c0 + c) in occ for r in range(-pad, h + pad) for c in range(-pad, w + pad)):
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


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
    active_col_idx: int,
) -> bool:
    H, W = grid.shape
    N = ids_base.shape[0]
    num_sprites = len(sprite_placements)

    active_template_ids = [int(ids_base[r, active_col_idx]) for r in range(N)]
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    for color, tid in zip(marker_colors, labeled_ids):
        si = random.randint(0, num_sprites - 1)
        r0, c0, _, ids_arr, direction, active_coords = sprite_placements[si]
        dr, dc = direction

        locs = np.argwhere(ids_arr == tid)
        if locs.shape[0] != 1:
            return False
        lr, lc = int(locs[0, 0]), int(locs[0, 1])
        if (lr, lc) not in active_coords:
            return False

        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)

        cr, cc = r0 + lr + dr, c0 + lc + dc
        while (cr, cc) != (marker_r, marker_c) and 0 <= cr < H and 0 <= cc < W:
            if grid[cr, cc] != _BG:
                return False
            cr += dr
            cc += dc

        if grid[marker_r, marker_c] != _BG:
            return False
        grid[marker_r, marker_c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors, active_col_idx = design

    unique_variants = _deduplicated_variants(template, ids_base)
    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    if len(unique_variants) < num_sprites:
        return None
    chosen = random.sample(unique_variants, num_sprites)

    placements = _place_sprites(H, W, chosen)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        grid[r0 : r0 + h, c0 : c0 + w] = sub

    if not _place_markers(grid, placements, ids_base, template_colors, active_col_idx):
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
