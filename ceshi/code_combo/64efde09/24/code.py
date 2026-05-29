# Combo variant of 64efde09 (index 24)
# combo_spec: D3/0+D5/1+D10/0
# Name: 间歇涂色+固定长度+动态标记数

from __future__ import annotations

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Keep the original object family from the spec: 3-4 dense dihedral copies of one thin N-by-2
#      template, where N is 7-9, plus isolated boundary markers whose colors label active-edge
#      template ports.
#    - Fuse the three changes into one trajectory rule. Example: let N=8 and suppose a left-facing
#      sprite has a labeled port at column 14 while the boundary marker for that port sits at
#      column 0. Original behavior would fill every background cell 13..1 continuously. In this
#      combo, the same port paints only a fixed segment of length N and only on odd steps:
#      columns 13,11,9,7 are colored; columns 12,10,8,6 are left as background; columns 5..1 stay
#      background because the ray stops after 8 steps even though the marker is still visible
#      farther away on the boundary.
#    - The dynamic-meta change means the number of labeled ports is no longer fixed at 3. If a
#      sample contains 4 sprites, it also contains 4 distinct marker colors and therefore 4 dashed
#      fixed-length ray families in the output. More sprites literally create a denser repeated dot
#      lattice, which makes the count relation visible.
# b) Complexity calibration:
#    - Using only dashed painting would be weak if the fixed-length segment coincided with the wall.
#      The generator therefore places every sprite at least N+3 cells from the boundary it faces.
#      That guarantees two things at once: the dashed pattern has several visible colored/blank
#      alternations, and the fixed-length stop happens strictly before the boundary marker.
#    - Dynamic marker count could explode the search space if we allowed many possibilities. We keep
#      sprite count in the original 3-4 range, then deterministically set marker count equal to the
#      chosen sprite count. This preserves repetition without making the puzzle noisy.
# c) Hidden-assumption audit of main():
#    - Component parsing, active-side detection, and dihedral template matching still rely on the
#      original rectangular-sprite assumptions and remain valid.
#    - The old drawing loop assumed "scan until blocker" and overwrite every traversed background
#      cell. That breaks twice here: stopping is no longer obstacle-driven, and only odd steps are
#      painted. The new loop must iterate exactly N steps from each mapped port and gate writes on
#      step parity.
#    - Once rays ignore blockers conceptually, main() implicitly assumes the first N cells from each
#      emitted port are safe to paint. The generator now guarantees those cells are background and
#      that odd-step painted cells from different colors never overlap.
#    - The new meta rule adds an assumption that every marker color corresponds to a unique template
#      port and that marker count equals sprite count. The generator enforces both.
# d) Placement legality:
#    - Sprites are still placed as non-overlapping dense rectangles with padding.
#    - Every sprite faces a boundary and is far enough from that boundary to host a marker beyond
#      the fixed-length dashed segment. The first N cells on every selected ray stay in-bounds.
#    - Candidate labeled template IDs are accepted only if their odd-step painted cells across all
#      sprites are pairwise disjoint; this avoids color conflicts while still allowing untouched
#      even-step cells to remain shared background when geometry happens to line up.
# e) Visual observability:
#    - A human can see three coupled regularities from a few examples: markers sit on boundary
#      sightlines, output rays are dotted rather than solid, and those dotted rays stop after a
#      count equal to the template's long side instead of reaching the marker.
#    - The dynamic count is also visible because 3-sprite examples show 3 marker colors while
#      4-sprite examples show 4. No hidden sorting by numeric color IDs is used.
# f) Visual cleanliness:
#    - Fixed-length segments end before the boundary, leaving a clean blank gap between the last
#      painted dash and the boundary marker. This prevents the marker from visually merging into the
#      emitted ray.
#    - Rejecting odd-step overlaps keeps the point-lattice output readable even when several sprite
#      orientations coexist in one grid.
# g) Change inferability:
#    - Colors remain structurally simple: marker color equals ray color, exactly as in the base
#      task. The new inference burden is therefore limited to three visible structural cues:
#      alternating occupancy, length = N, and marker count = sprite count.

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


def _ray_cells_from_port(r: int, c: int, dr: int, dc: int, steps: int) -> list[tuple[int, int, int]]:
    return [(r + dr * step, c + dc * step, step) for step in range(1, steps + 1)]


def _odd_step_cells(cells: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    return {(r, c) for r, c, step in cells if step % 2 == 1}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer boundary marker labels between all template-equivalent sprites, then emit fixed-length dashed rays.

    Input structure:
    - The most frequent color is the background.
    - Several dense 2-by-N or N-by-2 rectangles are dihedral transforms of one shared template, with N given by the template's long side.
    - Single-pixel markers sit on boundary cells along outward sightlines from active-edge ports, and the number of markers matches the number of sprites.

    Rule:
    - For each sprite, identify the active long edge as the side whose cells have the strongest same-color majority.
    - If the first non-background cell seen outward from an active-edge cell is a marker, that marker color labels the corresponding template port.
    - For every sprite and every labeled template port, emit the marker color for exactly N outward steps, but only paint odd-numbered steps (1, 3, 5, ...); even-numbered steps remain background.

    Key insight:
    - The markers still label shared template ports rather than absolute coordinates, but now three cues must be combined: the number of labels scales with sprite count, each label propagates to every transformed sprite, and each propagated ray is both dashed and capped by the common template length instead of by the wall.
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
    assert len(markers) == len(sprites_raw), "marker count must equal sprite count"

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
    assert len(color_to_template_id) == len(sprites), "the number of labeled ports must equal the sprite count"

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

            for r, c, step in _ray_cells_from_port(r0 + lr, c0 + lc, dr, dc, ray_len):
                assert 0 <= r < H and 0 <= c < W, "fixed-length ray must stay in bounds"
                assert int(g[r, c]) == bg, "fixed-length corridor must be empty in valid inputs"
                if step % 2 == 0:
                    continue
                assert int(out[r, c]) in (bg, color), "dashed fixed-length rays must not conflict"
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
_PLACEMENT_PAD = 2
_POST_RAY_GAP = 2
_PLACEMENT_RETRIES = 160
_GEN_RETRIES = 500
_ASSIGNMENT_RETRIES = 120
_SPRITE_COUNT_SEQUENCE = tuple(range(_SPRITE_COUNT_RANGE[0], _SPRITE_COUNT_RANGE[1] + 1))
_SPRITE_COUNT_CURSOR = 0


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
    if not [c for c in template_colors if c != right_dominant]:
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
    template: np.ndarray,
    ids_base: np.ndarray,
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
    H: int,
    W: int,
    h: int,
    w: int,
    dr: int,
    dc: int,
    ray_len: int,
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
    painted_cells: set[tuple[int, int]] = set()
    all_cells: set[tuple[int, int]] = set()
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
        for rr, cc, _step in ray_cells:
            if not (0 <= rr < H and 0 <= cc < W):
                return None
            if int(grid[rr, cc]) != _BG:
                return None
            if (rr, cc) in all_cells:
                return None
            all_cells.add((rr, cc))

        painted = _odd_step_cells(ray_cells)
        if painted & painted_cells:
            return None
        painted_cells |= painted

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
    return {"tid": tid, "painted_cells": painted_cells, "all_cells": all_cells, "hosts": hosts, "ports": ports}


def _find_host_assignment(chosen_infos: list[dict]) -> list[tuple[int, tuple[int, int]]] | None:
    ordered = sorted(chosen_infos, key=lambda info: len(info["hosts"]))
    used_sprites: set[int] = set()
    used_boundary: set[tuple[int, int]] = set()
    assignment: dict[int, tuple[int, int]] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(ordered):
            return True
        info = ordered[pos]
        host_options = info["hosts"][:]
        random.shuffle(host_options)
        for host in host_options:
            sprite_index = int(host["sprite_index"])
            boundary = tuple(host["boundary"])
            if sprite_index in used_sprites or boundary in used_boundary:
                continue
            used_sprites.add(sprite_index)
            used_boundary.add(boundary)
            assignment[int(info["tid"])] = boundary
            if backtrack(pos + 1):
                return True
            used_sprites.remove(sprite_index)
            used_boundary.remove(boundary)
            del assignment[int(info["tid"])]
        return False

    if not backtrack(0):
        return None
    return [(int(info["tid"]), assignment[int(info["tid"])]) for info in chosen_infos]


def _choose_marker_assignments(
    grid: np.ndarray,
    placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> list[tuple[int, int, tuple[int, int]]] | None:
    N = int(ids_base.shape[0])
    num_markers = len(placements)
    active_template_ids = [int(ids_base[r, 0]) for r in range(N)]
    candidates = []
    for tid in active_template_ids:
        info = _analyze_template_id(grid, placements, tid, N)
        if info is not None:
            candidates.append(info)

    if len(candidates) < num_markers:
        return None

    available_colors = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available_colors) < num_markers:
        return None

    for _ in range(_ASSIGNMENT_RETRIES):
        chosen_infos = random.sample(candidates, num_markers)
        corridor_union: set[tuple[int, int]] = set()
        ok = True
        for info in chosen_infos:
            overlap = corridor_union & info["all_cells"]
            if overlap:
                ok = False
                break
            corridor_union |= info["all_cells"]
        if not ok:
            continue

        host_assignment = _find_host_assignment(chosen_infos)
        if host_assignment is None:
            continue

        colors = random.sample(available_colors, num_markers)
        random.shuffle(host_assignment)
        return [
            (int(color), int(tid), boundary)
            for color, (tid, boundary) in zip(colors, host_assignment)
        ]
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
    if len(assignments) != len(placements):
        return False
    for color, _tid, (r, c) in assignments:
        if int(grid[r, c]) != _BG:
            return False
        grid[r, c] = int(color)
    return True


def _gen_attempt(num_sprites: int) -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors = design

    unique_variants = _deduplicated_variants(template, ids_base)
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
    global _SPRITE_COUNT_CURSOR

    preferred = _SPRITE_COUNT_SEQUENCE[_SPRITE_COUNT_CURSOR % len(_SPRITE_COUNT_SEQUENCE)]
    ordered_counts = [preferred] + [count for count in _SPRITE_COUNT_SEQUENCE if count != preferred]
    budgets = [_GEN_RETRIES] + [max(1, _GEN_RETRIES // 4)] * (len(ordered_counts) - 1)

    for count, budget in zip(ordered_counts, budgets):
        for _ in range(budget):
            result = _gen_attempt(count)
            if result is not None:
                _SPRITE_COUNT_CURSOR += 1
                return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
