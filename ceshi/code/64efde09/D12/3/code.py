# Variant of 64efde09
# Dimension: D12 — 活跃边识别规则 (meta)
# Change: 活跃边是长边中最大同色计数（majority count）严格更高的那一边... → 两条长边都是活跃边

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the D12 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     old: only one long edge of the 2xN / Nx2 sprite was active, chosen by a strict majority-gap rule
#     new: both long edges are active; ports may be labeled on either side and rays fire outward from both directions
#   concretization:
#     - The shared sprite template is still an Nx2 dense rectangle with 3-4 visible colors.
#     - The two long edges are simply the two columns (or, after rotation, the two rows).
#     - A boundary marker now identifies a specific port by sitting on the boundary reached by that port's outward normal.
#       Because the labeled port can belong to either template column, the boundary side itself reveals whether the marker
#       refers to the "left-column family" or the "right-column family" under the current sprite transform.
#     - Output rays therefore appear on opposite sides of the same sprite once markers cover both template columns.
#   behavior_trace:
#     input_example:
#       - Suppose the canonical template is 8x2.
#       - Marker color A labels id (row 2, col 0) through a left-facing sprite edge.
#       - Marker color B labels id (row 5, col 1) through a right-facing sprite edge.
#     old_flow:
#       - Only the majority-winning long edge was scanned.
#       - A marker on the other long edge would be ignored or make generation impossible.
#     new_flow:
#       - Both long edges are scanned.
#       - A discovers one canonical-side port, B discovers the opposite canonical-side port.
#       - Every sprite emits A from whichever physical side contains id(row 2, col 0), and B from the opposite physical side.
#     visible_difference:
#       - A single sprite can now have colored rays leaving from both opposing long sides in the same output.
#   complexity_calibration:
#     - Marker count stays at 3 to preserve the original puzzle scale, but generation forces coverage of both canonical sides.
#     - Template symmetries are rejected so each placed sprite still matches exactly one dihedral orientation.
#     - Colors are not used for sorting or tie-breaking; only visible geometry and alignment matter.
#   old_assumptions_removed_from_main:
#     - "_active_side" no longer chooses one winner or asserts a strict majority gap.
#     - Marker-to-port mapping is no longer restricted to one active-edge set per sprite.
#     - Output drawing no longer assumes every labeled port lies on the same outward-facing side in each orientation.
#   placement_legality:
#     - Each sprite is placed with one chosen long side guaranteed to have boundary clearance for marker placement.
#     - Different sprites keep padding, markers remain isolated single pixels, and solver validation rejects any accidental
#       ambiguity or ray-color conflict.
#   observability_and_cleanliness:
#     - Every generated sample labels at least one port from each canonical long side, so opposite-direction propagation is
#       present in the outputs rather than only being theoretically possible.
#     - Marker colors are distinct from template colors, keeping control pixels visually separable from sprite pixels.
#   new_solver_assumptions_checked_against_generator:
#     - Every non-singleton component is a dense non-square rectangle: enforced by template construction and placement.
#     - Every sprite orientation is unique under dihedral transforms: enforced by rejecting symmetric templates.
#     - Every marker labels exactly one long-edge template id: enforced by boundary-line clearance checks and final validation.


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


def _long_sides(sub: np.ndarray) -> list[dict]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"

    if h > w:
        raw = [
            ((0, -1), [(r, 0) for r in range(h)]),
            ((0, 1), [(r, w - 1) for r in range(h)]),
        ]
    else:
        raw = [
            ((-1, 0), [(0, c) for c in range(w)]),
            ((1, 0), [(h - 1, c) for c in range(w)]),
        ]

    return [
        {"direction": direction, "coords": coords, "coord_set": set(coords)}
        for direction, coords in raw
    ]


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
    Transfer boundary marker labels from either long edge of one rectangular sprite template to all transformed copies.

    Input structure:
    - bg is the most common color.
    - Several dense 2-by-N or N-by-2 rectangular sprites are dihedral transforms of one shared template.
    - Isolated single-pixel markers lie on boundary-aligned rays extending outward from ports on either of the two long edges.

    Rule:
    - Treat both long edges of every sprite as active port sets.
    - If the first non-bg cell seen from a long-edge port along its outward normal is a marker, that marker color labels the shared template port at that location.
    - For every labeled template port, emit its color outward from the corresponding port of every transformed sprite, following that port's own outward normal until blocked or out of bounds.

    Key insight:
    - A marker does not describe an absolute position; it tags one template port on one of the two long-edge families, and that tagged port follows the template through rotation and reflection.
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
        sprites.append({**sprite, "ids": ids, "sides": _long_sides(sprite["sub"])})

    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        for side in sprite["sides"]:
            dr, dc = side["direction"]
            for lr, lc in side["coords"]:
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

    assert set(color_to_template_id) == set(markers.values()), "every control marker must label one long-edge port"

    out = g.copy()
    H, W = g.shape
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        for color, template_id in color_to_template_id.items():
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            matching_sides = [side for side in sprite["sides"] if (lr, lc) in side["coord_set"]]
            assert len(matching_sides) == 1, "mapped port must land on exactly one long edge"
            dr, dc = matching_sides[0]["direction"]

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


# --------------- generator ---------------

# --- CFG: tuneable generation parameters ---
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
_PLACEMENT_RETRIES = 120
_GEN_RETRIES = 500


def _random_column(n: int, colors: list[int]) -> list[int]:
    return [random.choice(colors) for _ in range(n)]


def _unique_variants_with_sides(
    template: np.ndarray,
    ids_base: np.ndarray,
) -> list[dict]:
    left_ids = {int(ids_base[r, 0]) for r in range(ids_base.shape[0])}
    right_ids = {int(ids_base[r, 1]) for r in range(ids_base.shape[0])}

    raw = _dihedral_variants(template)
    raw_ids = _dihedral_variants(ids_base)

    seen: list[np.ndarray] = []
    result: list[dict] = []
    for (_, sub, _), (_, _, ids_arr) in zip(raw, raw_ids):
        if any(prev.shape == sub.shape and np.array_equal(prev, sub) for prev in seen):
            continue

        sides = []
        for side in _long_sides(sub):
            template_ids = [int(ids_arr[r, c]) for r, c in side["coords"]]
            tid_set = set(template_ids)
            if tid_set == left_ids:
                canonical_side = 0
            elif tid_set == right_ids:
                canonical_side = 1
            else:
                raise AssertionError("long edge must correspond to one canonical template column")

            sides.append(
                {
                    **side,
                    "template_ids": template_ids,
                    "canonical_side": canonical_side,
                }
            )

        seen.append(sub.copy())
        result.append({"sub": sub, "ids": ids_arr, "sides": sides})

    return result


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int], list[dict]] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)

    left_col = _random_column(N, template_colors)
    right_col = _random_column(N, template_colors)

    if len(set(left_col)) < 2 or len(set(right_col)) < 2:
        return None
    if left_col == right_col or left_col == list(reversed(right_col)):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None
    if sum(int(left_col[r] != right_col[r]) for r in range(N)) < 2:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    variants = _unique_variants_with_sides(template, ids_base)
    if len(variants) < 8:
        return None

    return template, ids_base, template_colors, variants


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int]:
    pad = _PLACEMENT_CLEARANCE
    if dr == 0 and dc == -1:
        return random.randint(0, H - h), random.randint(pad, W - w)
    if dr == 0 and dc == 1:
        return random.randint(0, H - h), random.randint(0, W - w - pad - 1)
    if dr == -1 and dc == 0:
        return random.randint(pad, H - h), random.randint(0, W - w)
    return random.randint(0, H - h - pad - 1), random.randint(0, W - w)


def _occupied_set(placements: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for placement in placements:
        r0, c0 = placement["r0"], placement["c0"]
        h, w = placement["sub"].shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _assign_label_sides(chosen: list[dict]) -> list[dict]:
    targets = [0, 1] + [random.choice([0, 1]) for _ in range(len(chosen) - 2)]
    random.shuffle(targets)

    assigned = []
    for variant, canonical_side in zip(chosen, targets):
        label_side = next(side for side in variant["sides"] if side["canonical_side"] == canonical_side)
        assigned.append({**variant, "label_side": label_side})
    return assigned


def _place_sprites(H: int, W: int, variants: list[dict]) -> list[dict] | None:
    placements: list[dict] = []
    pad = _PLACEMENT_PAD
    for variant in variants:
        sub = variant["sub"]
        h, w = sub.shape
        dr, dc = variant["label_side"]["direction"]
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w, dr, dc)
            occ = _occupied_set(placements)
            if any((r0 + r, c0 + c) in occ for r in range(-pad, h + pad) for c in range(-pad, w + pad)):
                continue
            placements.append({**variant, "r0": r0, "c0": c0})
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


def _clear_to_boundary(
    grid: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    marker_r: int,
    marker_c: int,
) -> bool:
    cr, cc = r + dr, c + dc
    while (cr, cc) != (marker_r, marker_c):
        if not (0 <= cr < grid.shape[0] and 0 <= cc < grid.shape[1]):
            return False
        if int(grid[cr, cc]) != _BG:
            return False
        cr += dr
        cc += dc
    return int(grid[marker_r, marker_c]) == _BG


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[dict],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    N = ids_base.shape[0]

    marker_sides = [0, 1] + [random.choice([0, 1]) for _ in range(_NUM_MARKERS - 2)]
    random.shuffle(marker_sides)

    ids_by_side = {
        0: [int(ids_base[r, 0]) for r in range(N)],
        1: [int(ids_base[r, 1]) for r in range(N)],
    }

    labeled_ids: list[tuple[int, int]] = []
    used_tids: set[int] = set()
    for side in marker_sides:
        candidates = [tid for tid in ids_by_side[side] if tid not in used_tids]
        if not candidates:
            return False
        tid = random.choice(candidates)
        labeled_ids.append((side, tid))
        used_tids.add(tid)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    for color, (canonical_side, tid) in zip(marker_colors, labeled_ids):
        hosts = [p for p in sprite_placements if p["label_side"]["canonical_side"] == canonical_side]
        random.shuffle(hosts)

        placed = False
        for sprite in hosts:
            r0, c0 = sprite["r0"], sprite["c0"]
            locs = np.argwhere(sprite["ids"] == tid)
            if locs.shape[0] != 1:
                return False
            lr, lc = map(int, locs[0])
            matching_sides = [side for side in sprite["sides"] if (lr, lc) in side["coord_set"]]
            if len(matching_sides) != 1:
                return False
            side = matching_sides[0]
            if side["canonical_side"] != canonical_side:
                return False

            dr, dc = side["direction"]
            marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)
            if not _clear_to_boundary(grid, r0 + lr, c0 + lc, dr, dc, marker_r, marker_c):
                continue

            grid[marker_r, marker_c] = color
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
    template, ids_base, template_colors, variants = design

    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    if len(variants) < num_sprites:
        return None

    chosen = random.sample(variants, num_sprites)
    chosen = _assign_label_sides(chosen)

    placements = _place_sprites(H, W, chosen)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for placement in placements:
        r0, c0 = placement["r0"], placement["c0"]
        sub = placement["sub"]
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
