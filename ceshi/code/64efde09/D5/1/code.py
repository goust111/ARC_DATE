# Variant of 64efde09
# Dimension: D5 — 目标/停止条件：射线在何处终止 (dst)
# Change: 射线遇到任何非背景像素或到达网格边界时停止... → 射线停在固定距离处

from __future__ import annotations

# Design analysis for the D5 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The input still contains several dense dihedral copies of one thin N-by-2 template.
#      Single-pixel markers still sit on boundary-aligned outward rays and still label template
#      ports by "first non-background seen on that ray".
#    - The only semantic change is in the emitted output rays: every labeled port now paints
#      exactly N cells outward, where N is the template's long dimension. The painted segment
#      ignores intermediate obstacles conceptually, so the generator must keep those N cells empty
#      to avoid clutter. Markers remain farther away on the boundary, leaving a visible gap.
#    - Small trace: if N=8 and a labeled left-facing port sits at column 12, the old task would
#      paint left through all bg cells until the boundary marker/border; this might produce 11 or
#      12 colored cells. The new task paints only columns 11..4 (8 cells) and stops, even though
#      the marker is still visible at column 0. The difference is therefore local and countable.
#
# b) Complexity calibration
#    - Keep the original template/port-matching mechanism, 3 markers, 3-4 sprites, and dihedral
#      matching intact so the task is not simplified into "just draw N".
#    - Vary N in 7..9 so the fixed-length rule is inferable from examples rather than collapsing
#      to a hard-coded constant. Marker colors remain unique and disjoint from template colors.
#
# c) Implicit assumptions audit for main()
#    - Component parsing, template matching, active-side detection, and marker-to-template-ID
#      inference do not depend on the old stop condition and stay unchanged.
#    - The original drawing loop assumed "first obstacle/boundary" and therefore safely wrote only
#      through background. That assumption is false here. The new loop instead iterates for exactly
#      N steps and allows overwriting original input pixels if a malformed input places obstacles
#      there; generator-side construction prevents that on valid samples.
#    - Because ray length is now inferred, main() assumes every sprite shares one long dimension N.
#      The generator preserves that by deriving every sprite from one template.
#
# d) Placement legality
#    - Each sprite is still a dense rectangle, non-overlapping with padding.
#    - To make fixed-length rays valid, every sprite's active side is placed at least N+2 cells from
#      the boundary it faces. This guarantees N in-bounds output cells plus at least one visible bg
#      gap before the boundary marker.
#    - Marker hosts are chosen only from ports whose entire boundary corridor is clear, and each
#      selected template ID is accepted only if all of its emitted N-cell rays across all sprites
#      land on currently empty cells without pairwise overlap.
#
# e) Visual observability
#    - The marker still sits on the boundary, but the output ray now ends short of it. Across
#      examples with different N, a solver can count that the colored segment length equals the
#      sprite's long side rather than the remaining distance to the wall.
#    - No invisible metadata such as raw color-id ordering is used for matching or selection.
#
# f) Visual cleanliness
#    - The generator rejects candidate labeled ports whose emitted segments would hit sprites or
#      collide with one another. This avoids outputs where the new "ignore obstacles" semantics
#      would scribble through objects and hide the intended fixed-distance pattern.
#
# g) Color inferability
#    - Colors do not transform. Marker color = ray color, exactly as in the original task, so the
#      only new inference burden is the stopping distance. Keeping that mapping unchanged isolates
#      D5 cleanly.

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


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


def _template_length(sub: np.ndarray) -> int:
    return int(max(sub.shape))


def _boundary_position(r: int, c: int, dr: int, dc: int, H: int, W: int) -> tuple[int, int]:
    if dr == 0 and dc == -1:
        return r, 0
    if dr == 0 and dc == 1:
        return r, W - 1
    if dr == -1 and dc == 0:
        return 0, c
    return H - 1, c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer port labels from one rectangular sprite family to every transformed copy, then emit fixed-length rays.

    Input structure:
    - The dominant color is background.
    - Several dense N-by-2 or 2-by-N rectangles are dihedral variants of one shared template.
    - Single-pixel markers lie somewhere farther out along boundary-facing rays from active-edge cells.

    Rule:
    - On each sprite, the active edge is the long side with the stronger same-color majority.
    - The first marker encountered outward from an active-edge cell assigns its color to that template port.
    - For every sprite, emit that color from the corresponding port for exactly N cells, where N is the template's long dimension.

    Key insight:
    - Marker positions identify template ports as before, but ray extent is no longer "until blocked";
    - it is encoded by the shared template size and must be counted from the sprite itself.
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

    ray_len = _template_length(sprites_raw[0]["sub"])
    assert all(_template_length(sprite["sub"]) == ray_len for sprite in sprites_raw)

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

            for step in range(1, ray_len + 1):
                r = r0 + lr + dr * step
                c = c0 + lc + dc * step
                if not (0 <= r < H and 0 <= c < W):
                    break
                assert int(out[r, c]) in {bg, int(g[r, c]), color}, "fixed-length rays must not conflict"
                out[r, c] = color

    return out


# --------------- generator ---------------

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
_POST_RAY_GAP = 1
_PLACEMENT_RETRIES = 120
_GEN_RETRIES = 400


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
    if len(set(template.flat)) < num_colors:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _deduplicated_variants(
    template: np.ndarray, ids_base: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
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
        result.append((sub, ids_arr, direction, sorted(active_set)))
    return result


def _min_boundary_distance(ray_len: int) -> int:
    return int(ray_len + 1 + _POST_RAY_GAP)


def _random_sprite_position(
    H: int, W: int, h: int, w: int, dr: int, dc: int, ray_len: int
) -> tuple[int, int] | None:
    min_dist = _min_boundary_distance(ray_len)
    if dr == 0 and dc == -1:
        r_lo, r_hi = 0, H - h
        c_lo, c_hi = min_dist, W - w
    elif dr == 0 and dc == 1:
        r_lo, r_hi = 0, H - h
        c_lo, c_hi = 0, W - w - min_dist
    elif dr == -1 and dc == 0:
        r_lo, r_hi = min_dist, H - h
        c_lo, c_hi = 0, W - w
    else:
        r_lo, r_hi = 0, H - h - min_dist
        c_lo, c_hi = 0, W - w

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _occupied_set(placements: list[tuple]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _bbox_with_pad_cells(r0: int, c0: int, h: int, w: int, pad: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(r0 - pad, r0 + h + pad):
        for c in range(c0 - pad, c0 + w + pad):
            cells.add((r, c))
    return cells


def _place_sprites(H: int, W: int, variants, ray_len: int) -> list[tuple] | None:
    placements: list[tuple] = []
    for sub, ids_arr, direction, active_coords in variants:
        h, w = sub.shape
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            pos = _random_sprite_position(H, W, h, w, direction[0], direction[1], ray_len)
            if pos is None:
                return None
            r0, c0 = pos
            occ = _occupied_set(placements)
            if any((r, c) in occ for r, c in _bbox_with_pad_cells(r0, c0, h, w, _PLACEMENT_PAD)):
                continue
            placements.append((r0, c0, sub, ids_arr, direction, active_coords))
            placed = True
            break
        if not placed:
            return None
    return placements


def _ray_cells_from_port(r: int, c: int, dr: int, dc: int, steps: int) -> list[tuple[int, int]]:
    return [(r + dr * k, c + dc * k) for k in range(1, steps + 1)]


def _distance_to_boundary(r: int, c: int, dr: int, dc: int, H: int, W: int) -> int:
    br, bc = _boundary_position(r, c, dr, dc, H, W)
    return abs(br - r) + abs(bc - c)


def _analyze_template_id(
    grid: np.ndarray,
    placements: list[tuple],
    tid: int,
    ray_len: int,
) -> dict | None:
    H, W = grid.shape
    emit_cells: set[tuple[int, int]] = set()
    hosts = []
    ports = []

    for idx, (r0, c0, _sub, ids_arr, direction, active_coords) in enumerate(placements):
        dr, dc = direction
        locs = np.argwhere(ids_arr == tid)
        if locs.shape[0] != 1:
            return None
        lr, lc = map(int, locs[0])
        if (lr, lc) not in set(active_coords):
            return None

        port_r = r0 + lr
        port_c = c0 + lc
        ports.append((port_r, port_c, dr, dc))
        ray_cells = _ray_cells_from_port(port_r, port_c, dr, dc, ray_len)
        for rr, cc in ray_cells:
            if not (0 <= rr < H and 0 <= cc < W):
                return None
            if int(grid[rr, cc]) != _BG:
                return None
            if (rr, cc) in emit_cells:
                return None
            emit_cells.add((rr, cc))

        dist = _distance_to_boundary(port_r, port_c, dr, dc, H, W)
        if dist < _min_boundary_distance(ray_len):
            continue
        marker_r, marker_c = _boundary_position(port_r, port_c, dr, dc, H, W)
        clear = True
        for step in range(1, dist):
            rr = port_r + dr * step
            cc = port_c + dc * step
            if int(grid[rr, cc]) != _BG:
                clear = False
                break
        if clear and int(grid[marker_r, marker_c]) == _BG:
            hosts.append({"sprite_index": idx, "boundary": (marker_r, marker_c)})

    if not hosts:
        return None
    return {"tid": tid, "emit_cells": emit_cells, "hosts": hosts, "ports": ports}


def _legacy_is_strictly_longer(
    infos: list[dict], marker_cells: set[tuple[int, int]], ray_len: int, grid: np.ndarray
) -> bool:
    H, W = grid.shape
    for info in infos:
        for port_r, port_c, dr, dc in info["ports"]:
            legacy_len = 0
            rr = port_r + dr
            cc = port_c + dc
            while 0 <= rr < H and 0 <= cc < W:
                if (rr, cc) in marker_cells or int(grid[rr, cc]) != _BG:
                    break
                legacy_len += 1
                rr += dr
                cc += dc
            if legacy_len <= ray_len:
                return False
    return True


def _choose_marker_assignments(
    grid: np.ndarray,
    placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> list[tuple[int, int, tuple[int, int]]] | None:
    N = ids_base.shape[0]
    ray_len = int(N)
    active_template_ids = [int(ids_base[r, 0]) for r in range(N)]
    candidates = []
    for tid in active_template_ids:
        info = _analyze_template_id(grid, placements, tid, ray_len)
        if info is not None:
            candidates.append(info)

    if len(candidates) < _NUM_MARKERS:
        return None

    available_colors = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available_colors) < _NUM_MARKERS:
        return None

    for _ in range(80):
        chosen = random.sample(candidates, _NUM_MARKERS)
        all_emit_cells = set().union(*(info["emit_cells"] for info in chosen))
        used_boundary: set[tuple[int, int]] = set()
        assignments: list[tuple[int, int, tuple[int, int]]] = []
        ok = True
        for info in chosen:
            host_options = info["hosts"][:]
            random.shuffle(host_options)
            picked = None
            for host in host_options:
                if host["boundary"] in all_emit_cells:
                    continue
                if host["boundary"] not in used_boundary:
                    picked = host
                    break
            if picked is None:
                ok = False
                break
            used_boundary.add(picked["boundary"])
            assignments.append((int(info["tid"]), int(picked["sprite_index"]), picked["boundary"]))
        if not ok:
            continue
        marker_cells = {boundary for _tid, _si, boundary in assignments}
        if not _legacy_is_strictly_longer(chosen, marker_cells, ray_len, grid):
            continue
        colors = random.sample(available_colors, _NUM_MARKERS)
        random.shuffle(assignments)
        return [(color, tid, boundary) for color, (tid, _si, boundary) in zip(colors, assignments)]

    return None


def _place_markers(
    grid: np.ndarray,
    placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    assignments = _choose_marker_assignments(grid, placements, ids_base, template_colors)
    if assignments is None:
        return False
    for color, _tid, (r, c) in assignments:
        if int(grid[r, c]) != _BG:
            return False
        grid[r, c] = int(color)
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

    placements = _place_sprites(H, W, chosen, N)
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
