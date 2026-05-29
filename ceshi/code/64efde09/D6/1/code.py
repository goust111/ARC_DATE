# Variant of 64efde09
# Dimension: D6 — 变换：输出中对对象的处理 (transform)
# Change: keep_source — 原始精灵和标记保持不变，仅添加射线... → 射线涂色后将精灵重新着色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg

# Design analysis for the D6 variant
# ---
# variant_spec:
#   core_change:
#     before: marker colors only propagate outward as rays; sprite bodies stay unchanged
#     after: the same labeled ports still emit rays, then every sprite recolors those
#       active-edge port cells to the corresponding marker colors
#   concrete_visual:
#     input: identical to the base task. Dense 2xN / Nx2 sprites are dihedral transforms
#       of one shared thin template, and three single-pixel boundary markers label three
#       template ids on the structurally dominant long edge.
#     output: every labeled template id still creates a straight outward ray on every
#       matching sprite, but now the sprite's own port cell also turns into that marker
#       color. Each labeled color therefore appears both on the boundary marker, along
#       each propagated ray, and on the aligned edge cells of every sprite.
#   behavior_trace:
#     example:
#       old_logic:
#         - marker color 4 labels template id T on one sprite edge cell.
#         - every transformed copy emits a color-4 ray from its local copy of T.
#         - the emitting sprite cells keep their original template colors.
#       new_logic:
#         - the same ray propagation happens first.
#         - then each local copy of T inside every sprite is recolored to 4.
#       visible_difference:
#         - old outputs show color 4 only on the marker and the external ray.
#         - new outputs additionally show color 4 embedded in each sprite body exactly
#           at the active-edge port positions.
#   complexity_calibration:
#     - Keep exactly three markers, so the recolor rule repeats enough to be learnable.
#     - Marker colors remain disjoint from template colors, guaranteeing that each
#       recolored port is visibly new rather than an accidental no-op.
#     - Template ids are still tied to dihedral correspondence, so the solver must track
#       both orientation transfer and the extra sprite recolor step.
#   implicit_assumption_audit:
#     - Component parsing still assumes markers are area-1 components and sprites are
#       dense rectangles; the variant does not change the input, so this remains valid.
#     - Active-edge detection still relies on one long side having a strictly larger
#       same-color majority. Generator must keep that uniqueness.
#     - Ray drawing must continue to test obstacles against the original input grid `g`,
#       not the progressively recolored output, otherwise an early recolored port could
#       incorrectly block later rays. The solver below preserves the old `g`-based test.
#     - Recoloring must happen after port discovery, because matching a sprite against
#       the template depends on the unchanged input sprite colors.
#   placement_legality:
#     - Sprites are still placed with padding and boundary-facing active sides.
#     - Boundary markers still require clear line-of-sight through background cells.
#     - Because recoloring only changes cells already occupied by sprites, it cannot
#       create new geometric conflicts; only ray conflicts need validation, as before.
#   visual_observability:
#     - Human solvers can see a three-part chain: boundary marker -> same-colored rays ->
#       same-colored port cells embedded on every sprite edge.
#     - The recolor rule does not depend on hidden ordering or color-id arithmetic; it is
#       a direct visible copy from marker color to corresponding ports.
#   cleanliness:
#     - No extra object classes are introduced, so the grid remains readable.
#     - The added signal stays localized to active-edge cells, avoiding clutter inside
#       the thin rectangles.
#   generator_guarantees:
#     - Validation checks that the variant changes exactly sprite_count * 3 sprite cells,
#       so generated samples cannot silently collapse back to the keep-source behavior.


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


def _solve(grid: np.ndarray, *, recolor_ports: bool) -> np.ndarray:
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
        recolors: list[tuple[int, int, int]] = []
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

            recolors.append((r0 + lr, c0 + lc, color))

        if recolor_ports:
            for rr, cc, color in recolors:
                out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast boundary marker colors from template ports onto rays and back onto every transformed sprite.

    Input structure:
    - The background is the most frequent color.
    - Several dense 2-by-N or N-by-2 rectangles are dihedral copies of one thin template.
    - Single-pixel markers sit on boundary-aligned sightlines extending outward from some active-edge cells.

    Rule:
    - For each sprite, find the active long edge: the side whose cells have the stronger repeated-color majority.
    - The first marker seen along an outward sightline from an active-edge cell assigns its color to that shared template port.
    - For every transformed sprite, emit that color outward from the corresponding port through background cells, then recolor the emitting port cell itself to the same marker color.

    Key insight:
    - Markers identify template-edge positions rather than fixed coordinates, so each learned color follows the template through rotations/reflections and affects both the external ray and the matching sprite port.
    """
    return _solve(grid, recolor_ports=True)


def _keep_source_output(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, recolor_ports=False)


# --------------- generator (D6 recolor variant) ---------------

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
        r0, c0, _, ids_arr, direction, active_coords = sprite_placements[si]
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


def _sprite_mask(shape: tuple[int, int], placements: list[tuple]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


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
        out = main(grid)
        legacy_out = _keep_source_output(grid)
    except Exception:
        return None

    sprite_mask = _sprite_mask(grid.shape, placements)
    recolored_inside = int(np.count_nonzero((out != grid) & sprite_mask))
    if recolored_inside != len(placements) * _NUM_MARKERS:
        return None
    if np.array_equal(out, legacy_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
