# Variant of 64efde09
# Dimension: D1 — 匹配条件：精灵间的对应关系 (match_cond)
# Change: stripe_signature_match — 精灵通过二面体变换（旋转/翻转）与同一模板匹配，活跃边由最大同色计数严... → 活跃边由最小同色计数决定（取反）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg

# Design analysis:
# spec_delta:
#   preserved:
#     - bg 仍为最常见颜色，sprite 仍是同一 Nx2 模板的二面体变换，marker 仍是边界上的单像素。
#     - 输出仍是“marker 先给模板端口着色，再把该颜色从所有同构 sprite 的对应端口向外发射”。
#   changed_core:
#     - 原题按长边上“最大同色计数最高”的一侧选 active edge。
#     - 本变体改为按“最大同色计数最低”的一侧选 active edge，也就是更杂乱、更分散的长边。
# concrete_trace:
#   - 例: N=7 时，左列颜色频数峰值为 2，右列峰值为 4。
#   - 原题会把右列当 active，所以 marker 必须放在右列朝向的边界射线上。
#   - 变体必须把左列当 active；同一组 sprite 若 marker 仍按原题放在右侧，solver 会给错误模板端口贴标。
#   - 因此需要同时翻转 solver 的判定逻辑，以及 generator 的模板设计、朝向放置与 marker 对齐逻辑。
# complexity_calibration:
#   - 为了让“更杂乱的一侧”肉眼可见，active 边的频数峰值限制在 2-3，另一侧限制为至少高 2。
#   - active 边强制使用全部模板颜色，避免退化成看不出差别的近纯色边。
#   - 模板拒绝二面体对称，避免同一个 sprite 匹配多个 orientation，保持 shared-template 对应关系清晰。
# implicit_assumptions_review:
#   - _active_side 以前依赖“更整齐的一边唯一”，现在改成“更杂乱的一边唯一”；排序方向与断言都必须反转。
#   - main() 中读取 marker 的循环假设: 真正的模板端口一定落在 active edge 上。generator 通过把 canonical 左列设计成低峰值边来保证此事。
#   - _random_sprite_position / _boundary_position 并不关心颜色统计，只依赖 direction；direction 由新 _active_side 产出后仍然成立。
#   - 匹配阶段仍断言每个 sprite 恰好匹配一个 orientation，所以 generator 额外保证 8 个二面体结果全异。
# placement_legality_and_cleanliness:
#   - sprite 之间仍保留 padding 与朝边界的 clearance，保证 marker 视线干净，射线不穿过别的对象。
#   - marker 仍只放在边界，颜色与模板颜色分离，减少视觉重叠。
# variant_metric_plan:
#   - 量化指标使用 (inactive_peak - active_peak)；生成器强制其 >= 2。
#   - 验证时额外比较“旧规则 solver”与“新规则 solver”在同一输入上的行为，确保多数样本不会退化成原题等价实例。


def _dihedral_variants_with_ids(
    template: np.ndarray, ids: np.ndarray
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    return _dihedral_variants_with_ids(template, ids)


def _side_peak(vals: list[int] | np.ndarray) -> int:
    return max(Counter(map(int, vals)).values())


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
        scored.append((_side_peak(vals), direction, set(coords)))

    scored.sort(key=lambda item: item[0])
    assert scored[0][0] < scored[1][0], "active side must be structurally unique"
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
    Transfer boundary marker colors from the most color-dispersed long edge of one sprite orientation to all matching transforms.

    Input structure:
    - The background is the most frequent color.
    - Several dense N-by-2 or 2-by-N rectangles are dihedral transforms of one shared non-square template.
    - Isolated single-pixel markers lie on boundary-facing rays that extend outward from the more mixed long edge of some sprites.

    Rule:
    - For each sprite, compare its two long sides and choose the one whose maximum same-color count is strictly smaller.
    - If a marker is the first non-background cell seen on an outward ray from one of those active-edge cells, that marker color labels the corresponding template port.
    - For every transformed sprite, emit that label color outward from the matching active-edge port until a non-background blocker or the grid boundary stops the ray.

    Key insight:
    - The markers do not name absolute coordinates; they identify template edge cells on the least uniform long side, and those template ports move with each rotation or reflection.
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


# --------------- generator (att1) ---------------

# --- CFG: tuneable generation parameters ---
_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_PEAK_RANGE = (2, 3)
_INACTIVE_PEAK_RANGE = (4, 7)
_MIN_PEAK_GAP = 2
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_CLEARANCE = 2
_PLACEMENT_RETRIES = 100
_GEN_RETRIES = 500


def _build_column_with_peak(
    n: int,
    colors: list[int],
    peak: int,
    dominant: int,
    required_colors: set[int],
) -> list[int] | None:
    counts = {color: 0 for color in colors}
    counts[dominant] = peak
    remaining = n - peak

    for color in required_colors:
        if color == dominant:
            continue
        if remaining <= 0:
            return None
        counts[color] += 1
        remaining -= 1

    choices = list(colors)
    while remaining > 0:
        options = [color for color in choices if counts[color] < peak]
        if not options:
            return None
        pick = random.choice(options)
        counts[pick] += 1
        remaining -= 1

    if max(counts.values()) != peak or counts[dominant] != peak:
        return None

    col: list[int] = []
    for color, count in counts.items():
        col.extend([color] * count)
    random.shuffle(col)
    return col


def _all_dihedral_unique(template: np.ndarray) -> bool:
    seen: list[np.ndarray] = []
    for _name, sub, _ids in _dihedral_variants(template):
        if any(prev.shape == sub.shape and np.array_equal(prev, sub) for prev in seen):
            return False
        seen.append(sub.copy())
    return True


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    """Create an Nx2 canonical template whose left edge is uniquely the most mixed long side."""
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)

    min_active_peak = max(_ACTIVE_PEAK_RANGE[0], (N + num_colors - 1) // num_colors)
    max_active_peak = min(_ACTIVE_PEAK_RANGE[1], N)
    if min_active_peak > max_active_peak:
        return None
    active_peak = random.randint(min_active_peak, max_active_peak)

    min_inactive_peak = max(active_peak + _MIN_PEAK_GAP, _INACTIVE_PEAK_RANGE[0])
    max_inactive_peak = min(N, _INACTIVE_PEAK_RANGE[1])
    if min_inactive_peak > max_inactive_peak:
        return None
    inactive_peak = random.randint(min_inactive_peak, max_inactive_peak)

    left_dom = random.choice(template_colors)
    right_dom = random.choice(template_colors)
    left_col = _build_column_with_peak(
        N,
        template_colors,
        active_peak,
        left_dom,
        required_colors=set(template_colors),
    )
    right_col = _build_column_with_peak(
        N,
        template_colors,
        inactive_peak,
        right_dom,
        required_colors=set(random.sample(template_colors, k=min(2, num_colors))),
    )
    if left_col is None or right_col is None:
        return None

    if _side_peak(left_col) >= _side_peak(right_col):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None
    if not _all_dihedral_unique(template):
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _deduplicated_variants(
    template: np.ndarray, ids_base: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
    """Return unique dihedral variants with their active-edge direction."""
    raw = _dihedral_variants_with_ids(template, ids_base)

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
        result.append((sub, ids_arr, direction, list(active_set)))
    return result


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int]:
    """Pick a random top-left corner so the active edge faces its boundary with clearance."""
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
    """Place variants without overlap, keeping their active edges boundary-facing."""
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
) -> bool:
    """Choose and place 3 markers on boundary cells. Mutates *grid* in place."""
    H, W = grid.shape
    N = ids_base.shape[0]
    num_sprites = len(sprite_placements)

    active_template_ids = [int(ids_base[r, 0]) for r in range(N)]
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    for color, tid in zip(marker_colors, labeled_ids):
        si = random.randint(0, num_sprites - 1)
        r0, c0, _sub, ids_arr, direction, active_coords = sprite_placements[si]
        dr, dc = direction

        locs = np.argwhere(ids_arr == tid)
        if locs.shape[0] != 1:
            return False
        lr, lc = int(locs[0, 0]), int(locs[0, 1])
        if (lr, lc) not in [(r, c) for r, c in active_coords]:
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
    template, ids_base, template_colors = design

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
