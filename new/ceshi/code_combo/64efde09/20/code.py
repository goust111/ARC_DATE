# Combo variant of 64efde09 (index 20)
# combo_spec: D14/1+D16/1
# Name: 角落偏移映射+按颜色顺序处理

from __future__ import annotations

from collections import Counter
import itertools

import numpy as np
import random

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# source_task:
#   - Base task 64efde09 copies boundary-marker labels from one rectangular sprite
#     template onto every dihedral copy, then emits outward rays from the matched ports.
# combo concretization:
#   objects:
#     - Objects remain dense Nx2 / 2xN rectangles with one structurally unique active
#       long edge, plus three isolated boundary markers.
#     - The marker colors are three distinct non-template colors. Their numeric order
#       is the processing queue, and the generator varies the chosen trio so that this
#       ordering rule is part of the observable task rather than a hardcoded palette.
#   fused changes:
#     - D14/1: a marker labels offset k along the active edge, counted from the same
#       visible corner on every sprite. Vertical active edges count top->bottom;
#       horizontal ones count left->right.
#     - D16/1: marker colors are processed from smaller value to larger value. All rays
#       of one color are emitted simultaneously; later colors treat earlier colors as
#       blockers.
# behavior trace:
#   - Small example: suppose marker color 2 labels offset 1 on a forward vertical host
#     sprite, and marker color 6 labels offset 5.
#   - Old rule: on a vertically reversed copy those labels would land on template IDs
#     mirrored to offsets N-2 and N-6. With the fused rule they stay at visible offsets
#     1 and 5 instead.
#   - When the forward horizontal copy emits the same 2/6 columns, the color-2 row
#     and column are drawn first and occupy their full free corridors. Color 6 then
#     starts from its own row/column but stops as soon as it reaches the already-drawn
#     color-2 cross. A still larger third color is truncated by whichever earlier color
#     it meets first.
#   - Therefore the output depends jointly on both the offset mapping and the color
#     processing queue.
# complexity calibration:
#   - Pure random placement often degenerates: either no reversed sprite makes D14
#     observable, or no orthogonal crossings make D16 observable. The generator instead
#     enforces four roles: forward vertical host, forward horizontal crosser, reversed
#     vertical receiver, reversed horizontal receiver.
#   - Marker offsets avoid the self-mirroring middle cell when N is odd, and the
#     template is made asymmetric enough that forward/reversed variants stay distinct.
# implicit assumption review:
#   old-main dependencies broken by the combo:
#     - Exact per-pixel template IDs are no longer the public port identity. Offset k on
#       the ordered active edge replaces transformed ID equality.
#     - "All rays can be drawn against the original grid only" no longer holds. Earlier
#       marker colors must become blockers for later colors.
#     - The original no-conflict assertion between different colors is invalid because
#       orthogonal crossings are now intentional and resolved by the processing order.
#   new assumptions introduced here:
#     - Every sprite is still a dihedral transform of one asymmetric template; the
#       generator rejects symmetric templates so orientation matching stays unique.
#     - Exactly three distinct marker colors appear, and sorting those values defines
#       the global processing queue.
#     - At least one reversed sprite exists, at least one orthogonal crossing exists,
#       and the host marker lines to the boundary are clear in the input.
# placement legality:
#   - Sprites are placed in four separated zones: left-middle, top-middle, right-top,
#     bottom-left. Their bodies never overlap and stay in-bounds.
#   - Markers sit only on boundary cells reached by unobstructed host rays.
#   - Structured placement guarantees a large central crossing region while keeping the
#     marker host corridors clear.
# visual observability:
#   - D14 is visible because the reversed vertical and reversed horizontal receivers put
#     the same marker colors on the same visible offsets, not on mirrored template IDs.
#   - D16 is visible because every sample creates the same type of crossing contest:
#     the smallest marker color survives farthest, the middle color is partially
#     truncated by it, and the largest color is truncated by one or both earlier colors.
#   - The generator rejects samples where the combo output matches the sequential
#     ID-based baseline or where no truncation from color priority happens.
# cleanliness and inference metrics:
#   - Metric 1: `sequential_loss` measures how many potential ray cells are removed by
#     earlier colors relative to the empty-grid corridors. It must be positive.
#   - Metric 2: `offset_id_diff` counts output cells that differ between the combo rule
#     and the same ordered process with old ID-based port correspondence. It must be
#     positive.


_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_TEMPLATE_COLORS = list(_NON_BG_COLORS)

_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_NUM_MARKERS = 3
_GEN_RETRIES = 300


def _dihedral_variants(template: np.ndarray, ids: np.ndarray | None = None) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    variants: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = None if ids is None else np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        flipped = np.fliplr(rotated)
        flipped_ids = None if rotated_ids is None else np.fliplr(rotated_ids)
        variants.append((f"rot{k}_fliplr", flipped, flipped_ids))
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


def _ordered_active_edge(active: set[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction[0] == 0:
        return sorted(active, key=lambda rc: (rc[0], rc[1]))
    return sorted(active, key=lambda rc: (rc[1], rc[0]))


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


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict], dict[tuple[int, int], int]]:
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
    assert len(markers) == _NUM_MARKERS, "expected exactly three control markers"
    assert len(set(markers.values())) == _NUM_MARKERS, "control marker colors must be distinct"
    return g, bg, sprites_raw, markers


def _marker_priority(markers: dict[tuple[int, int], int]) -> list[int]:
    priority = sorted(set(markers.values()))
    assert len(priority) == _NUM_MARKERS, "expected exactly three distinct marker colors"
    return priority


def _build_offset_sprites(sprites_raw: list[dict]) -> list[dict]:
    variants = _dihedral_variants(sprites_raw[0]["sub"])
    sprites: list[dict] = []
    active_len = None
    for sprite in sprites_raw:
        matches = [
            name
            for name, sub, _ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "each sprite must match the shared template in one orientation"
        direction, active = _active_side(sprite["sub"])
        ordered_active = _ordered_active_edge(active, direction)
        if active_len is None:
            active_len = len(ordered_active)
        assert len(ordered_active) == active_len, "active-edge lengths must agree"
        sprites.append({**sprite, "direction": direction, "ordered_active": ordered_active})
    return sprites


def _build_id_sprites(sprites_raw: list[dict]) -> list[dict]:
    variants = _dihedral_variants(sprites_raw[0]["sub"], np.arange(sprites_raw[0]["sub"].size, dtype=int).reshape(sprites_raw[0]["sub"].shape))
    sprites: list[dict] = []
    active_len = None
    for sprite in sprites_raw:
        matches = [
            (name, ids)
            for name, sub, ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "ID-based reference needs unique orientation matching"
        _name, ids = matches[0]
        assert ids is not None
        direction, active = _active_side(sprite["sub"])
        ordered_active = _ordered_active_edge(active, direction)
        if active_len is None:
            active_len = len(ordered_active)
        assert len(ordered_active) == active_len, "active-edge lengths must agree"
        sprites.append({**sprite, "direction": direction, "ordered_active": ordered_active, "ids": ids})
    return sprites


def _color_to_offset(g: np.ndarray, bg: int, sprites: list[dict], markers: dict[tuple[int, int], int]) -> dict[int, int]:
    color_to_offset: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for offset, (lr, lc) in enumerate(sprite["ordered_active"]):
            marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if marker_color is None:
                continue
            if marker_color in color_to_offset:
                assert color_to_offset[marker_color] == offset
            color_to_offset[marker_color] = offset
    assert set(color_to_offset) == set(markers.values()), "every marker must label one active-edge offset"
    return color_to_offset


def _color_to_template_id(g: np.ndarray, bg: int, sprites: list[dict], markers: dict[tuple[int, int], int]) -> dict[int, int]:
    color_to_template_id: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["ordered_active"]:
            marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if marker_color is None:
                continue
            template_id = int(sprite["ids"][lr, lc])
            if marker_color in color_to_template_id:
                assert color_to_template_id[marker_color] == template_id
            color_to_template_id[marker_color] = template_id
    assert set(color_to_template_id) == set(markers.values()), "every marker must label one template ID"
    return color_to_template_id


def _emit_from_offsets(
    g: np.ndarray,
    bg: int,
    sprites: list[dict],
    color_to_offset: dict[int, int],
    priority: list[int],
    *,
    sequential: bool,
) -> np.ndarray:
    out = g.copy()
    H, W = g.shape
    base_blocked = g.copy()
    for color in priority:
        offset = color_to_offset[color]
        blocked = out.copy() if sequential else base_blocked
        for sprite in sprites:
            assert 0 <= offset < len(sprite["ordered_active"])
            lr, lc = sprite["ordered_active"][offset]
            r0, c0, _r1, _c1 = sprite["bbox"]
            dr, dc = sprite["direction"]
            r = r0 + lr + dr
            c = c0 + lc + dc
            while 0 <= r < H and 0 <= c < W:
                if int(blocked[r, c]) != bg:
                    break
                out[r, c] = color
                r += dr
                c += dc
    return out


def _emit_from_ids(
    g: np.ndarray,
    bg: int,
    sprites: list[dict],
    color_to_template_id: dict[int, int],
    priority: list[int],
    *,
    sequential: bool,
) -> np.ndarray:
    out = g.copy()
    H, W = g.shape
    base_blocked = g.copy()
    for color in priority:
        template_id = color_to_template_id[color]
        blocked = out.copy() if sequential else base_blocked
        for sprite in sprites:
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            assert (lr, lc) in sprite["ordered_active"], "mapped ID must stay on the active edge"
            r0, c0, _r1, _c1 = sprite["bbox"]
            dr, dc = sprite["direction"]
            r = r0 + lr + dr
            c = c0 + lc + dc
            while 0 <= r < H and 0 <= c < W:
                if int(blocked[r, c]) != bg:
                    break
                out[r, c] = color
                r += dr
                c += dc
    return out


def _solve_offset(grid: np.ndarray, *, sequential: bool) -> np.ndarray:
    g, bg, sprites_raw, markers = _parse_scene(grid)
    sprites = _build_offset_sprites(sprites_raw)
    color_to_offset = _color_to_offset(g, bg, sprites, markers)
    return _emit_from_offsets(g, bg, sprites, color_to_offset, _marker_priority(markers), sequential=sequential)


def _solve_id(grid: np.ndarray, *, sequential: bool) -> np.ndarray:
    g, bg, sprites_raw, markers = _parse_scene(grid)
    sprites = _build_id_sprites(sprites_raw)
    color_to_template_id = _color_to_template_id(g, bg, sprites, markers)
    return _emit_from_ids(g, bg, sprites, color_to_template_id, _marker_priority(markers), sequential=sequential)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy boundary-labeled ports across transformed sprites by active-edge offset, then emit the marker colors in ascending color order so earlier colors block later ones.

    Input structure:
    - The background is the most common color.
    - Several dense 2xN or Nx2 sprites are dihedral transforms of one asymmetric template.
    - Exactly three isolated boundary markers use distinct non-template colors, and each sits on an outward ray from one active-edge cell.

    Rule:
    - For every sprite, find the active long edge: the long side with the stronger same-color majority.
    - A marker labels the offset of that active-edge cell counted from the same visible corner on every sprite.
    - Process the three marker colors from smallest value to largest value. In each color round, emit that color from the corresponding offset on every sprite, stopping at any non-background cell already present from the input or from earlier color rounds.

    Key insight:
    - The markers identify "the kth port from the reference corner" rather than a transformed template pixel, and the sorted marker colors also define a global emission queue. Reversed sprites preserve visible offsets, while earlier colors carve blockers that truncate later orthogonal rays.
    """
    return _solve_offset(grid, sequential=True)


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    for _ in range(n - majority):
        col.append(random.choice(others))
    random.shuffle(col)
    return col


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_TEMPLATE_COLORS, num_colors)

    active_majority = random.randint(_ACTIVE_MAJORITY_RANGE[0], min(N, _ACTIVE_MAJORITY_RANGE[1]))
    left_col = _build_column(N, template_colors, active_majority, random.choice(template_colors))

    max_right_majority = active_majority - 1
    min_right_majority = max(2, N // num_colors)
    if min_right_majority > max_right_majority:
        return None
    right_majority = random.randint(min_right_majority, max_right_majority)
    right_dominant = random.choice(template_colors)
    others = [c for c in template_colors if c != right_dominant]
    if not others:
        return None
    right_col = _build_column(N, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None

    row_pairs = [tuple(map(int, row)) for row in template.tolist()]
    if row_pairs == list(reversed(row_pairs)):
        return None
    if row_pairs[0] == row_pairs[-1]:
        return None
    if len(set(row_pairs)) < min(5, N):
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _deduplicated_variants(template: np.ndarray, ids_base: np.ndarray) -> list[dict]:
    raw = _dihedral_variants(template, ids_base)
    id_to_offset = {int(ids_base[r, 0]): r for r in range(ids_base.shape[0])}
    forward_seq = list(range(ids_base.shape[0]))
    reverse_seq = list(reversed(forward_seq))

    seen: list[np.ndarray] = []
    result: list[dict] = []
    for name, sub, ids_arr in raw:
        assert ids_arr is not None
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            direction, active_set = _active_side(sub)
        except AssertionError:
            continue

        ordered_active = _ordered_active_edge(active_set, direction)
        try:
            offset_sequence = [id_to_offset[int(ids_arr[r, c])] for r, c in ordered_active]
        except KeyError:
            continue

        if offset_sequence == forward_seq:
            reversed_order = False
        elif offset_sequence == reverse_seq:
            reversed_order = True
        else:
            continue

        seen.append(sub)
        result.append(
            {
                "name": name,
                "sub": sub,
                "direction": direction,
                "ordered_active": ordered_active,
                "reversed_order": reversed_order,
            }
        )
    return result


def _select_variants(unique_variants: list[dict]) -> dict[str, dict] | None:
    roles = {
        "host_v": ((0, 1), False),
        "host_h": ((1, 0), False),
        "rev_v": ((0, -1), True),
        "rev_h": ((-1, 0), True),
    }
    selected: dict[str, dict] = {}
    used_names: set[str] = set()
    for role, (direction, reversed_order) in roles.items():
        candidates = [
            v
            for v in unique_variants
            if v["direction"] == direction and v["reversed_order"] == reversed_order and v["name"] not in used_names
        ]
        if not candidates:
            return None
        choice = random.choice(candidates)
        selected[role] = choice
        used_names.add(choice["name"])
    return selected


def _place_structured_sprites(H: int, W: int, selected: dict[str, dict], N: int) -> list[dict] | None:
    placements = [
        {**selected["host_v"], "r0": H // 2 - N // 2, "c0": 2, "role": "host_v"},
        {**selected["host_h"], "r0": 2, "c0": W // 2 - N // 2, "role": "host_h"},
        {**selected["rev_v"], "r0": max(0, H // 2 - N // 2 - (N - 2)), "c0": W - 4, "role": "rev_v"},
        {**selected["rev_h"], "r0": H - 4, "c0": max(0, W // 2 - N // 2 - (N - 2)), "role": "rev_h"},
    ]

    occ: set[tuple[int, int]] = set()
    for sprite in placements:
        r0 = int(sprite["r0"])
        c0 = int(sprite["c0"])
        sub = sprite["sub"]
        h, w = sub.shape
        if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
            return None
        for r in range(h):
            for c in range(w):
                cell = (r0 + r, c0 + c)
                if cell in occ:
                    return None
                occ.add(cell)
    return placements


def _boundary_position(r: int, c: int, dr: int, dc: int, H: int, W: int) -> tuple[int, int]:
    if dr == 0 and dc == -1:
        return r, 0
    if dr == 0 and dc == 1:
        return r, W - 1
    if dr == -1 and dc == 0:
        return 0, c
    return H - 1, c


def _clear_path_to_boundary(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> bool:
    H, W = grid.shape
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if grid[r, c] != _BG:
            return False
        r += dr
        c += dc
    return True


def _candidate_offsets_for_host(grid: np.ndarray, host_sprite: dict, N: int) -> list[int]:
    candidates: list[int] = []
    r0, c0 = int(host_sprite["r0"]), int(host_sprite["c0"])
    dr, dc = host_sprite["direction"]
    for offset, (lr, lc) in enumerate(host_sprite["ordered_active"]):
        if 2 * offset == N - 1:
            continue
        if _clear_path_to_boundary(grid, r0 + lr, c0 + lc, dr, dc):
            candidates.append(offset)
    return candidates


def _choose_labeled_offsets(candidates: list[int], N: int) -> list[int] | None:
    if len(candidates) < _NUM_MARKERS:
        return None
    good = []
    for trio in itertools.combinations(candidates, _NUM_MARKERS):
        if max(trio) - min(trio) < 3:
            continue
        diffs = sorted(abs(a - b) for i, a in enumerate(trio) for b in trio[i + 1 :])
        if diffs[0] < 2:
            continue
        good.append(list(trio))
    if not good:
        return None
    return random.choice(good)


def _place_markers(
    grid: np.ndarray,
    placements: list[dict],
    labeled_offsets: list[int],
    marker_colors: list[int],
) -> bool:
    H, W = grid.shape
    host = next(sprite for sprite in placements if sprite["role"] == "host_v")
    r0, c0 = int(host["r0"]), int(host["c0"])
    dr, dc = host["direction"]
    for color, offset in zip(marker_colors, labeled_offsets):
        lr, lc = host["ordered_active"][offset]
        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)
        if grid[marker_r, marker_c] != _BG:
            return False
        if not _clear_path_to_boundary(grid, r0 + lr, c0 + lc, dr, dc):
            return False
        grid[marker_r, marker_c] = color
    return True


def _build_grid(H: int, W: int, placements: list[dict]) -> np.ndarray:
    grid = np.full((H, W), _BG, dtype=int)
    for sprite in placements:
        r0 = int(sprite["r0"])
        c0 = int(sprite["c0"])
        sub = sprite["sub"]
        h, w = sub.shape
        grid[r0 : r0 + h, c0 : c0 + w] = sub
    return grid


def _ray_potential_length(g: np.ndarray, bg: int, start: tuple[int, int], direction: tuple[int, int]) -> int:
    H, W = g.shape
    r, c = start
    dr, dc = direction
    length = 0
    while 0 <= r < H and 0 <= c < W and int(g[r, c]) == bg:
        length += 1
        r += dr
        c += dc
    return length


def _ray_realized_length(out: np.ndarray, color: int, start: tuple[int, int], direction: tuple[int, int]) -> int:
    H, W = out.shape
    r, c = start
    dr, dc = direction
    length = 0
    while 0 <= r < H and 0 <= c < W and int(out[r, c]) == color:
        length += 1
        r += dr
        c += dc
    return length


def _combo_metrics(inp: np.ndarray) -> dict[str, int]:
    combo = _solve_offset(inp, sequential=True)
    id_ordered = _solve_id(inp, sequential=True)

    g, bg, sprites_raw, markers = _parse_scene(inp)
    priority = _marker_priority(markers)
    sprites = _build_offset_sprites(sprites_raw)
    color_to_offset = _color_to_offset(g, bg, sprites, markers)

    sequential_loss = 0
    for color in priority[1:]:
        offset = color_to_offset[color]
        for sprite in sprites:
            lr, lc = sprite["ordered_active"][offset]
            r0, c0, _r1, _c1 = sprite["bbox"]
            dr, dc = sprite["direction"]
            start = (r0 + lr + dr, c0 + lc + dc)
            sequential_loss += _ray_potential_length(g, bg, start, (dr, dc)) - _ray_realized_length(combo, color, start, (dr, dc))

    offset_id_diff = int(np.count_nonzero(combo != id_ordered))
    return {"sequential_loss": int(sequential_loss), "offset_id_diff": offset_id_diff}


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, _template_colors = design

    unique_variants = _deduplicated_variants(template, ids_base)
    selected = _select_variants(unique_variants)
    if selected is None:
        return None

    placements = _place_structured_sprites(H, W, selected, N)
    if placements is None:
        return None

    grid = _build_grid(H, W, placements)
    host = next(sprite for sprite in placements if sprite["role"] == "host_v")
    candidates = _candidate_offsets_for_host(grid, host, N)
    labeled_offsets = _choose_labeled_offsets(candidates, N)
    if labeled_offsets is None:
        return None
    random.shuffle(labeled_offsets)
    available_markers = sorted(c for c in _NON_BG_COLORS if c not in set(template.flat))
    if len(available_markers) < _NUM_MARKERS:
        return None
    marker_colors = sorted(random.sample(available_markers, _NUM_MARKERS))

    if not _place_markers(grid, placements, labeled_offsets, marker_colors):
        return None

    try:
        combo = main(grid)
        metrics = _combo_metrics(grid)
    except Exception:
        return None

    if metrics["sequential_loss"] <= 0:
        return None
    if metrics["offset_id_diff"] <= 0:
        return None
    if np.array_equal(combo, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
