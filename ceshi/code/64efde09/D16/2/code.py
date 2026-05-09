# Variant of 64efde09
# Dimension: D16 — 多步处理序列 (multi_step)
# Change: 单步处理——识别标记、映射端口、发射所有射线在同一步完成... → 先移动标记再发射射线

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg

# Design analysis for the D16 variant
# -----------------------------------
# concrete_behavior:
#   input_objects:
#     - The input still contains 3-4 dense 2xN / Nx2 rectangular sprites that are
#       dihedral transforms of one shared template.
#     - There are still 3 isolated marker pixels on grid boundaries, and each one
#       is aligned with one active-edge cell of some sprite.
#   output_objects:
#     - Stage 1 is now explicit in the final state: each boundary marker is moved
#       inward along its line of sight until it lands on the nearest sprite cell,
#       which must be an active-edge port. That sprite cell is recolored with the
#       marker color.
#     - Stage 2 uses the stage-1 attachment as the port label and draws the same
#       color outward from every transformed copy of that template port.
#   traced_example:
#     - Old behavior on a left-facing sprite row: boundary marker at (5, 0) labels
#       the active-edge cell at (5, 4), but only cells (5, 1..3) get colored.
#     - New behavior: the marker first moves onto (5, 4), recoloring that sprite
#       cell, then the emission step draws from the corresponding ports of every
#       sprite, so the source lane reaches the boundary and the colored cell is now
#       visibly attached to the sprite.
# complexity_calibration:
#   - Keep the same thin non-square rectangular family, unique active edge, and
#     three unique marker colors, so port matching stays constrained and repeated.
#   - Preserve multiple sprite orientations; this keeps the mapping problem from
#     collapsing into a single absolute-position rule.
# implicit_assumption_review:
#   - The original `_first_marker_on_ray` assumes the marker is already the final
#     boundary-side blocker. That is no longer the right semantic primitive.
#   - Emission can no longer use the raw input as the blocker grid, because the
#     stage-1 move removes boundary markers before stage 2 starts.
#   - Boundary markers on corners would make the inward direction ambiguous, so the
#     generator must forbid that geometry instead of asking the solver to guess.
# placement_legality:
#   - Sprites still need clear outward corridors to the boundary for their chosen
#     labeling markers.
#   - The generator also validates the full grid with `main()` so any residual ray
#     conflict or malformed attachment is rejected.
# visual_observability:
#   - The changed behavior is visible without hidden numeric ordering:
#       1. a marker-colored cell appears on a sprite edge;
#       2. the source boundary lane becomes continuous to the edge;
#       3. homologous rays still appear on all transformed sprites.
#   - Marker colors stay disjoint from template colors so the attachment step is
#     easy to see.
# main_new_assumptions_and_generator_guarantees:
#   - Every marker lies on exactly one boundary side, never on a corner.
#   - Scanning inward from a marker first hits an active-edge cell, not a sprite
#     interior cell or another object.
#   - Each marker color still labels exactly one template id, and the generator
#     keeps those color/template assignments bijective.


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


def _marker_inward_direction(r: int, c: int, H: int, W: int) -> tuple[int, int]:
    sides = []
    if r == 0:
        sides.append((1, 0))
    if r == H - 1:
        sides.append((-1, 0))
    if c == 0:
        sides.append((0, 1))
    if c == W - 1:
        sides.append((0, -1))
    assert len(sides) == 1, "markers must lie on exactly one boundary side"
    return sides[0]


def _scan_to_first_non_bg(g: np.ndarray, bg: int, r: int, c: int, dr: int, dc: int) -> tuple[int, int] | None:
    H, W = g.shape
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if int(g[r, c]) != bg:
            return r, c
        r += dr
        c += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move boundary markers onto sprite ports, then copy those port colors to rays on every transformed sprite.

    Input structure:
    - bg is the most common color.
    - Several dense 2-by-N or N-by-2 rectangular sprites are dihedral transforms of one template.
    - Isolated single-pixel markers sit on boundary cells, each aligned with an outward corridor from some sprite's dominant long edge.

    Rule:
    - For each sprite, the active edge is the long side whose cells have the strongest same-color majority.
    - First move each boundary marker inward along its corridor until it lands on the nearest active-edge cell; that attached sprite cell now carries the marker color.
    - Then use that attached port-template mapping to draw rays outward from the corresponding active-edge cell of every transformed sprite, through bg cells until blocked or out of bounds.

    Key insight:
    - The boundary pixels are only temporary controls. Their real role is to identify template ports after a visible attachment step, and those template ports are then replayed across the whole transform family.
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

    port_lookup: dict[tuple[int, int], tuple[int, int, int]] = {}
    for idx, sprite in enumerate(sprites):
        r0, c0, _r1, _c1 = sprite["bbox"]
        for lr, lc in sprite["active"]:
            port_lookup[(r0 + lr, c0 + lc)] = (idx, lr, lc)

    stage1 = g.copy()
    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    attached_targets: dict[tuple[int, int], int] = {}
    H, W = g.shape

    for (mr, mc), marker_color in markers.items():
        inward_dr, inward_dc = _marker_inward_direction(mr, mc, H, W)
        hit = _scan_to_first_non_bg(g, bg, mr, mc, inward_dr, inward_dc)
        assert hit is not None, "every marker must see a sprite inward"
        assert hit in port_lookup, "marker must attach to an active-edge cell"

        sprite_idx, lr, lc = port_lookup[hit]
        sprite = sprites[sprite_idx]
        dr, dc = sprite["direction"]
        assert (dr, dc) == (-inward_dr, -inward_dc), "marker must face its sprite's active side"

        template_id = int(sprite["ids"][lr, lc])
        if marker_color in color_to_template_id:
            assert color_to_template_id[marker_color] == template_id
        if template_id in template_id_to_color:
            assert template_id_to_color[template_id] == marker_color
        color_to_template_id[marker_color] = template_id
        template_id_to_color[template_id] = marker_color

        if hit in attached_targets:
            assert attached_targets[hit] == marker_color
        attached_targets[hit] = marker_color
        stage1[mr, mc] = bg
        stage1[hit] = marker_color

    assert set(color_to_template_id) == set(markers.values()), "every control marker must label one active-edge cell"

    out = stage1.copy()
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
                if int(stage1[r, c]) != bg:
                    break
                assert int(out[r, c]) in (bg, color), "control rays must not conflict"
                out[r, c] = color
                r += dr
                c += dc

    return out


# --------------- generator (D16 variant) ---------------

# --- CFG: tuneable generation parameters ---
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
_GEN_RETRIES = 300


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
        result.append((sub, ids_arr, direction, list(active_set)))
    return result


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int]:
    pad = _PLACEMENT_CLEARANCE
    if dr == 0 and dc == -1:
        return random.randint(1, H - h - 1), random.randint(pad, W - w)
    if dr == 0 and dc == 1:
        return random.randint(1, H - h - 1), random.randint(0, W - w - pad - 1)
    if dr == -1 and dc == 0:
        return random.randint(pad, H - h), random.randint(1, W - w - 1)
    return random.randint(0, H - h - pad - 1), random.randint(1, W - w - 1)


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
    H, W = grid.shape
    N = ids_base.shape[0]
    num_sprites = len(sprite_placements)

    active_template_ids = [ids_base[r, 0] for r in range(N)]
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
        if (lr, lc) not in active_coords:
            return False

        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)
        if marker_r in (0, H - 1) and marker_c in (0, W - 1):
            return False

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
