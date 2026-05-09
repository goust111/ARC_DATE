# Variant of 64efde09
# Dimension: D5 — 目标/停止条件：射线在何处终止 (dst)
# Change: 射线遇到任何非背景像素或到达网格边界时停止... → 射线停在距离边界N个像素处

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg

# Design analysis for the D5 variant
# ----------------------------------
# spec_variant:
#   preserved_structure:
#   - 背景色仍是全局多数色。
#   - 主体仍是同一个 Nx2 模板的若干二面体变体，marker 仍是单像素、唯一颜色。
#   - marker 仍然通过“从 active edge 向外看见的第一枚非背景像素”来给模板端口上色。
#   changed_core:
#   - 旧题中，无 marker 的复制射线会一直涂到边界；新题中，若途中没有障碍，射线只涂到“距所朝向边界恰好 N”的那一格。
#   - 这里 N 不是颜色或计数，而是模板长边长度；因为模板始终是 Nx2 或 2xN，所以 N 可从矩形长边直接读出。
#   concrete_example:
#   - 设某竖直 sprite 的 active edge 朝左，模板长边 N=7，active cell 在全局列 15。
#   - 旧逻辑会把列 14,13,...,1 全部涂满，遇到边界 marker/网格边界才停。
#   - 新逻辑只允许涂到列 7；列 6..0 保持背景，其中列 0 仍可放 marker。
#   - 因而输入里 marker 仍在边界，输出里却会留下一条长度为 N 的“未涂缓冲带”，这是人类可见的新规律。
#   complexity_calibration:
#   - 仍保留 3 个 marker、3-4 个 sprite，避免规则退化成单对象。
#   - N 保持在 7-9，缓冲带足够长，和原题“几乎涂到边界”的视觉效果明显不同。
#   implicit_assumptions_checked:
#   - `_first_marker_on_ray` 依赖 marker 是 outward ray 上第一枚非背景像素；此性质保留，所以只需保证生成器给被标记端口留出无遮挡视线。
#   - 原 `main()` 中的输出循环默认“边界就是合法终点”；变体后不成立，必须改为“距离边界至少 N 的格子才允许着色”。
#   - 原生成器只要求 sprite 面向边界且有少量空隙；变体后若空隙不超过 N，射线会退化成零长度，因此放置时必须强制 active edge 离边界超过 N。
#   placement_legality:
#   - 生成器对每个候选 sprite 强制预留至少 `_RAY_VISIBLE_MIN` 个可着色格，确保输出中每条有效射线都非空。
#   - marker 只从“该模板端口在所有 sprite 上到边界均无遮挡”的端口集合里选，避免有些复制射线提前撞上别的 sprite 而看不出新停止线。
#   observability_and_cleanliness:
#   - 所有被选端口在所有 sprite 上都保留清晰直廊，因此输出中的每条复制射线都会在同样的 inset 线停住。
#   - 颜色数量与原题同级，不引入基于颜色 id 的排序或隐藏状态。
#   new_solver_assumptions_and_generator_guarantees:
#   - 新 solver 假设所有 sprite 的长边长度相同，并把它当作 inset 偏移；生成器只放置同一模板的变体，因此满足。
#   - 新 solver 假设 marker 仍在 outward ray 上且唯一；生成器维持唯一 marker 颜色与无遮挡边界对齐。


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


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


def _facing_boundary_distance(r: int, c: int, dr: int, dc: int, H: int, W: int) -> int:
    if dr == 0 and dc == -1:
        return c
    if dr == 0 and dc == 1:
        return W - 1 - c
    if dr == -1 and dc == 0:
        return r
    return H - 1 - r


def _ray_cells(
    r: int,
    c: int,
    dr: int,
    dc: int,
    H: int,
    W: int,
    min_boundary_distance: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if _facing_boundary_distance(r, c, dr, dc, H, W) < min_boundary_distance:
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


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
    for rr, cc in _ray_cells(r, c, dr, dc, H, W, min_boundary_distance=0):
        if int(g[rr, cc]) != bg:
            return markers.get((rr, cc))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer boundary marker labels between transformed rectangular sprites, then emit inset rays from the matching ports.

    Input structure:
    - The background is the most frequent color.
    - Several solid Nx2 or 2xN rectangles are dihedral copies of one shared template.
    - Single-cell markers lie on outward sightlines from some sprites' dominant long edge, typically on the outer boundary.

    Rule:
    - For each sprite, locate the active long edge by the stronger same-color majority on that side.
    - The first non-background cell seen along an outward ray from an active-edge cell is a marker that labels the corresponding template port.
    - Copy each labeled port to every transformed sprite and draw its color outward through background cells, but only while the ray remains at least N cells away from the faced boundary, where N is the template long-side length.

    Key insight:
    - The markers identify shared template positions, and the stopping offset is not arbitrary: it equals the common rectangle length, so every copied ray halts on the same inset boundary line after rotation or reflection.
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

    stop_offset = max(map(int, sprites_raw[0]["sub"].shape))
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
        assert max(map(int, sprite["sub"].shape)) == stop_offset, "all sprites must share one long-side length"

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

            for rr, cc in _ray_cells(r0 + lr, c0 + lc, dr, dc, H, W, min_boundary_distance=stop_offset):
                if int(g[rr, cc]) != bg:
                    break
                assert int(out[rr, cc]) in (bg, color), "control rays must not conflict"
                out[rr, cc] = color

    return out


# --------------- generator (D5 variant) ---------------

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
_PLACEMENT_RETRIES = 100
_GEN_RETRIES = 300
_RAY_VISIBLE_MIN = 2


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
    if len([c for c in template_colors if c != right_dominant]) == 0:
        return None
    right_col = _build_column(N, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _deduplicated_variants(
    template: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], set[tuple[int, int]]]]:
    raw = _dihedral_variants(template)
    seen: list[np.ndarray] = []
    result = []
    for _name, sub, ids_arr in raw:
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            direction, active_set = _active_side(sub)
        except AssertionError:
            continue
        seen.append(sub)
        result.append((sub, ids_arr, direction, active_set))
    return result


def _random_sprite_position(
    H: int,
    W: int,
    h: int,
    w: int,
    dr: int,
    dc: int,
    stop_offset: int,
) -> tuple[int, int] | None:
    if dr == 0 and dc == -1:
        r_lo, r_hi = 0, H - h
        c_lo, c_hi = stop_offset + _RAY_VISIBLE_MIN, W - w
    elif dr == 0 and dc == 1:
        r_lo, r_hi = 0, H - h
        c_lo, c_hi = 0, W - w - stop_offset - _RAY_VISIBLE_MIN
    elif dr == -1 and dc == 0:
        r_lo, r_hi = stop_offset + _RAY_VISIBLE_MIN, H - h
        c_lo, c_hi = 0, W - w
    else:
        r_lo, r_hi = 0, H - h - stop_offset - _RAY_VISIBLE_MIN
        c_lo, c_hi = 0, W - w

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _occupied_set(placements: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for sprite in placements:
        r0, c0, r1, c1 = sprite["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cells.add((r, c))
    return cells


def _place_sprites(
    H: int,
    W: int,
    variants: list[tuple[np.ndarray, np.ndarray, tuple[int, int], set[tuple[int, int]]]],
    stop_offset: int,
) -> list[dict] | None:
    placements: list[dict] = []
    pad = _PLACEMENT_PAD
    for sub, ids_arr, direction, active_coords in variants:
        h, w = sub.shape
        dr, dc = direction
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            pos = _random_sprite_position(H, W, h, w, dr, dc, stop_offset)
            if pos is None:
                return None
            r0, c0 = pos
            occ = _occupied_set(placements)
            if any(
                (r0 + r, c0 + c) in occ
                for r in range(-pad, h + pad)
                for c in range(-pad, w + pad)
            ):
                continue
            placements.append(
                {
                    "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                    "sub": sub,
                    "ids": ids_arr,
                    "direction": direction,
                    "active": set(active_coords),
                }
            )
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


def _port_for_template_id(sprite: dict, template_id: int) -> tuple[int, int] | None:
    locs = np.argwhere(sprite["ids"] == template_id)
    if locs.shape[0] != 1:
        return None
    lr, lc = map(int, locs[0])
    if (lr, lc) not in sprite["active"]:
        return None
    return lr, lc


def _clear_corridor_specs(
    grid: np.ndarray,
    sprite_placements: list[dict],
    template_id: int,
    stop_offset: int,
) -> list[tuple[dict, int, int, tuple[int, int]]] | None:
    H, W = grid.shape
    specs: list[tuple[dict, int, int, tuple[int, int]]] = []
    for sprite in sprite_placements:
        port = _port_for_template_id(sprite, template_id)
        if port is None:
            return None
        lr, lc = port
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        full_ray = _ray_cells(r0 + lr, c0 + lc, dr, dc, H, W, min_boundary_distance=0)
        inset_ray = _ray_cells(r0 + lr, c0 + lc, dr, dc, H, W, min_boundary_distance=stop_offset)
        if len(inset_ray) < _RAY_VISIBLE_MIN:
            return None
        if not full_ray or any(int(grid[rr, cc]) != _BG for rr, cc in full_ray):
            return None
        specs.append((sprite, lr, lc, full_ray[-1]))
    return specs


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[dict],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    stop_offset = int(ids_base.shape[0])

    active_template_ids = [int(ids_base[r, 0]) for r in range(stop_offset)]
    available_ids: list[int] = []
    clear_specs: dict[int, list[tuple[dict, int, int, tuple[int, int]]]] = {}
    for tid in active_template_ids:
        specs = _clear_corridor_specs(grid, sprite_placements, tid, stop_offset)
        if specs is None:
            continue
        available_ids.append(tid)
        clear_specs[tid] = specs

    if len(available_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(available_ids, _NUM_MARKERS)

    available_colors = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available_colors) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available_colors, _NUM_MARKERS)

    used_boundary_cells: set[tuple[int, int]] = set()
    assignments: list[tuple[int, tuple[int, int]]] = []
    for tid in labeled_ids:
        options = clear_specs[tid][:]
        random.shuffle(options)
        chosen_boundary: tuple[int, int] | None = None
        for sprite, lr, lc, _boundary_cell in options:
            r0, c0, _r1, _c1 = sprite["bbox"]
            dr, dc = sprite["direction"]
            marker_cell = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)
            if marker_cell in used_boundary_cells:
                continue
            chosen_boundary = marker_cell
            break
        if chosen_boundary is None:
            return False
        used_boundary_cells.add(chosen_boundary)
        assignments.append((tid, chosen_boundary))

    for color, (_tid, (marker_r, marker_c)) in zip(marker_colors, assignments):
        if int(grid[marker_r, marker_c]) != _BG:
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
    template, ids_base, template_colors = design

    unique_variants = _deduplicated_variants(template)
    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    if len(unique_variants) < num_sprites:
        return None
    chosen = random.sample(unique_variants, num_sprites)

    placements = _place_sprites(H, W, chosen, stop_offset=N)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for sprite in placements:
        r0, c0, r1, c1 = sprite["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = sprite["sub"]

    if not _place_markers(grid, placements, ids_base, template_colors):
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
