# Combo variant of 64efde09 (index 5)
# combo_spec: D4/1+D12/0
# Name: 内部标记+特定颜色活跃边

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The preserved backbone is still "one N-by-2 template -> 3-4 dihedral sprite copies ->
#    3 marker colors label 3 template ports -> emit those colors from every homologous port".
#    The fused change affects two consecutive inference steps:
#    - Original local story on a canonical vertical sprite:
#        choose the long side with the stronger same-color majority;
#        follow that side's outward ray until the first non-bg boundary pixel marker.
#    - Combo local story:
#        first find the unique rarest sprite color ("signal color");
#        the active long side is the only long side containing that signal color;
#        then look outward from that side and find an isolated interior marker on the ray.
#      Example row pattern for a left-facing sprite becomes:
#        boundary . m . active_edge
#      rather than
#        boundary_marker . . majority_edge
#      so the solver must identify the signal-bearing edge before the floating marker makes sense.
# b) Complexity calibration:
#    The generator intentionally makes the old majority heuristic disagree with the new signal rule:
#    the signal color lives only on the canonical left side, while the opposite side has the larger
#    same-color repeat count. This prevents silent collapse back to the original task.
#    Marker colors remain unique and outside the template palette so the added difficulty comes from
#    the two visible structural cues, not from hidden color remapping.
# c) Hidden assumptions audited against main():
#    - Components of area 1 are treated as markers. Therefore markers must stay 8-disconnected from
#      sprites and from each other; generator enforces isolated interior placement.
#    - `_signal_color` assumes each sprite has one unique rarest non-bg color; generator builds the
#      template so exactly one color has the minimum count.
#    - `_active_side` assumes exactly one long side contains that rarest color; generator keeps the
#      signal color entirely off the opposite long side.
#    - `_first_marker_on_ray` assumes the first non-bg cell encountered on a labeled active ray is
#      the marker, not a different sprite. Generator places markers only on clear active rays and
#      rejects candidate cells shared by multiple active-ray owners.
#    - The label-propagation loop assumes labeled template ids lie on the active side in every
#      orientation. Generator samples labels only from the canonical signal-bearing side.
# d) Placement legality:
#    Sprites stay dense rectangles, remain inside bounds, and are spaced with padding. Each sprite is
#    placed so its active side faces a boundary with enough clearance to fit at least one background
#    cell, then a floating marker, then more background before the boundary. Marker candidates must
#    be background cells on exactly one active ray, strictly interior, and 8-disconnected from all
#    non-bg neighbors.
# e) Visual observability:
#    Across examples, humans can see two repeated cues: the emitting side is the one containing the
#    uniquely rare template color, and the label dot is not on the frame but floating in the open
#    corridor beyond that side. Because the majority-heavy side is deliberately the wrong side, the
#    D12 change is visible rather than cosmetic.
# f) Visual cleanliness:
#    The grid contains only rectangles, isolated dots, and straight rays. We forbid marker adjacency
#    and ambiguous shared-ray cells, so cues do not pile up into unreadable clusters.
# g) Change inferability:
#    No ordering by numeric color id is used. The signal color is chosen by visible multiplicity
#    inside the rectangle, and marker colors stay unique but low-count. From 2-4 examples a solver
#    can infer "rarest-color side emits" plus "interior dot on that ray chooses the port".
# 9) New assumptions checked against the generator:
#    - Every generated marker is interior and lies on the outward ray of exactly one active-edge
#      port belonging to the signal-based side.
#    - The signal-based active direction differs from the original majority-based direction in the
#      canonical template, so the combo does not degenerate to the seed task.
#    - All labeled ids are canonical left-column ids, which remain the active-side ids under every
#      dihedral transform because `_active_side` is recomputed after each transform.


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


def _signal_color(sub: np.ndarray) -> int:
    counts = Counter(map(int, sub.flat))
    rare_count = min(counts.values())
    rare_colors = [color for color, count in counts.items() if count == rare_count]
    assert len(rare_colors) == 1, "template must have a unique rarest color"
    return int(rare_colors[0])


def _active_side(sub: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    signal = _signal_color(sub)
    hits = []
    for direction, coords in _long_sides(sub):
        contains_signal = any(int(sub[r, c]) == signal for r, c in coords)
        hits.append((contains_signal, direction, set(coords)))
    active = [item for item in hits if item[0]]
    assert len(active) == 1, "exactly one long side must contain the signal color"
    _contains_signal, direction, coords = active[0]
    return direction, coords


def _majority_side_direction(sub: np.ndarray) -> tuple[int, int] | None:
    scored = []
    for direction, coords in _long_sides(sub):
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((max(Counter(vals).values()), direction))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


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


def _is_interior_cell(r: int, c: int, H: int, W: int) -> bool:
    return 0 < r < H - 1 and 0 < c < W - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer floating marker labels from the signal-bearing long edge of one rectangular sprite template to all transforms.

    Input structure:
    - The background is the most frequent color.
    - Several dense 2-by-N or N-by-2 rectangles are dihedral copies of one shared template.
    - Three isolated single-pixel markers sit strictly inside the grid on outward rays from some sprites, and each marker color is outside the template palette.

    Rule:
    - For each sprite, find the unique rarest color inside that rectangle and take the only long side containing that color as the active edge.
    - Scan outward from each active-edge cell; if the first non-bg cell on that ray is one of the interior markers, that marker color labels the corresponding template port.
    - For every labeled port, draw the marker color outward from the homologous active-edge cell of every transformed sprite through background cells until blocked or off-grid.

    Key insight:
    - The floating dots do not mark absolute destinations. They name template ports, but only after you identify the correct long side via the rectangle's rare signal color; those port identities then follow the whole dihedral sprite family.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
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
            assert _is_interior_cell(r, c, H, W), "control markers must be interior in this combo"
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


# --------------- generator (combo att1) ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_SIGNAL_COUNT_RANGE = (1, 2)
_ACTIVE_SIDE_MAX_REPEAT = 3
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_CLEARANCE = 4
_MARKER_MIN_STEPS_FROM_EDGE = 2
_MARKER_MIN_STEPS_TO_BOUNDARY = 2
_PLACEMENT_RETRIES = 100
_GEN_RETRIES = 400


def _build_balanced_column(length: int, colors: list[int], max_repeat: int) -> list[int] | None:
    if length < len(colors):
        return None
    col = list(colors)
    counts = Counter(col)
    while len(col) < length:
        allowed = [c for c in colors if counts[c] < max_repeat]
        if not allowed:
            return None
        min_count = min(counts[c] for c in allowed)
        choices = [c for c in allowed if counts[c] == min_count]
        choice = random.choice(choices)
        counts[choice] += 1
        col.append(choice)
    random.shuffle(col)
    return col


def _build_dominant_column(length: int, colors: list[int], dominant: int, majority: int) -> list[int] | None:
    others = [c for c in colors if c != dominant]
    if majority >= length or length - majority < len(others):
        return None
    col = [dominant] * majority + list(others)
    while len(col) < length:
        col.append(random.choice(others))
    random.shuffle(col)
    return col


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int], int] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)
    signal_color = random.choice(template_colors)
    support_colors = [c for c in template_colors if c != signal_color]
    if not support_colors:
        return None

    signal_count = random.randint(*_SIGNAL_COUNT_RANGE)
    left_fill = _build_balanced_column(N - signal_count, support_colors, _ACTIVE_SIDE_MAX_REPEAT)
    if left_fill is None:
        return None
    left_col = left_fill + [signal_color] * signal_count
    random.shuffle(left_col)
    left_max = max(Counter(left_col).values())

    dominant = random.choice(support_colors)
    max_majority = N - (len(support_colors) - 1)
    min_majority = max(left_max + 1, 4)
    if min_majority > max_majority:
        return None
    right_majority = random.randint(min_majority, max_majority)
    right_col = _build_dominant_column(N, support_colors, dominant, right_majority)
    if right_col is None:
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    counts = Counter(map(int, template.flat))
    rare_count = counts[signal_color]
    if sum(1 for count in counts.values() if count == rare_count) != 1:
        return None
    if any(int(template[r, 1]) == signal_color for r in range(N)):
        return None
    if not any(int(template[r, 0]) == signal_color for r in range(N)):
        return None

    signal_direction, _active = _active_side(template)
    majority_direction = _majority_side_direction(template)
    if signal_direction != (0, -1):
        return None
    if majority_direction != (0, 1):
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors, signal_color


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
        result.append((sub, ids_arr, direction, list(active_set)))
    return result


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int]:
    pad = _PLACEMENT_CLEARANCE
    if dr == 0 and dc == -1:
        return random.randint(0, H - h), random.randint(pad, W - w)
    if dr == 0 and dc == 1:
        return random.randint(0, H - h), random.randint(0, W - w - pad)
    if dr == -1 and dc == 0:
        return random.randint(pad, H - h), random.randint(0, W - w)
    return random.randint(0, H - h - pad), random.randint(0, W - w)


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
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w, *direction)
            occ = _occupied_set(placements)
            if any(
                (r0 + r, c0 + c) in occ
                for r in range(-pad, h + pad)
                for c in range(-pad, w + pad)
            ):
                continue
            placements.append((r0, c0, sub, ids_arr, direction, active_coords))
            placed = True
            break
        if not placed:
            return None
    return placements


def _ray_cells(r: int, c: int, dr: int, dc: int, H: int, W: int) -> list[tuple[int, int]]:
    cells = []
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _clear_segment_to_marker(
    grid: np.ndarray, r: int, c: int, dr: int, dc: int, marker: tuple[int, int]
) -> bool:
    mr, mc = marker
    r += dr
    c += dc
    while (r, c) != (mr, mc):
        if int(grid[r, c]) != _BG:
            return False
        r += dr
        c += dc
    return int(grid[mr, mc]) == _BG


def _marker_is_isolated(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    markers: set[tuple[int, int]],
    H: int,
    W: int,
) -> bool:
    r, c = cell
    if not _is_interior_cell(r, c, H, W):
        return False
    if cell in occupied or cell in markers:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) in occupied or (rr, cc) in markers:
                return False
    return True


def _port_records(sprite_placements: list[tuple], H: int, W: int) -> list[dict]:
    ports: list[dict] = []
    for si, placement in enumerate(sprite_placements):
        r0, c0, _sub, ids_arr, direction, active_coords = placement
        dr, dc = direction
        for lr, lc in set(active_coords):
            rr = r0 + lr
            cc = c0 + lc
            ports.append(
                {
                    "sprite_index": si,
                    "tid": int(ids_arr[lr, lc]),
                    "origin": (rr, cc),
                    "direction": direction,
                    "ray": _ray_cells(rr, cc, dr, dc, H, W),
                }
            )
    return ports


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    N = ids_base.shape[0]

    active_template_ids = [int(ids_base[r, 0]) for r in range(N)]
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    occupied = _occupied_set(sprite_placements)
    ports = _port_records(sprite_placements, H, W)
    ray_owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for port in ports:
        owner = (port["sprite_index"], port["tid"])
        for cell in port["ray"]:
            ray_owners.setdefault(cell, []).append(owner)

    placed_markers: set[tuple[int, int]] = set()
    for color, tid in zip(marker_colors, labeled_ids):
        candidates: list[tuple[int, int]] = []
        for port in ports:
            if port["tid"] != tid:
                continue
            si = port["sprite_index"]
            rr, cc = port["origin"]
            dr, dc = port["direction"]
            ray = port["ray"]
            for step, cell in enumerate(ray, start=1):
                remaining = len(ray) - step
                if step < _MARKER_MIN_STEPS_FROM_EDGE:
                    continue
                if remaining < _MARKER_MIN_STEPS_TO_BOUNDARY:
                    continue
                if ray_owners.get(cell) != [(si, tid)]:
                    continue
                if not _marker_is_isolated(cell, occupied, placed_markers, H, W):
                    continue
                if not _clear_segment_to_marker(grid, rr, cc, dr, dc, cell):
                    continue
                candidates.append(cell)
        if not candidates:
            return False

        marker_r, marker_c = random.choice(candidates)
        grid[marker_r, marker_c] = color
        placed_markers.add((marker_r, marker_c))
    return True


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors, _signal_color_value = design

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
