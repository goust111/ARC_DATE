# Variant of 64efde09
# Dimension: D5 — 目标/停止条件：射线在何处终止 (dst)
# Change: 射线遇到任何非背景像素或到达网格边界时停止... → 射线穿透其他精灵继续传播

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the D5 variant
# ----------------------------------
# a) Concrete behavior:
# - Objects are still dense N x 2 / 2 x N sprites that share one rectangular template.
# - A single-pixel marker still labels one active-edge template port, but the sight-line from
#   that port to the boundary marker may now pass through another sprite.
# - Output rays still move in straight cardinal directions and only color background cells.
#   The D5 change is only the stop condition: sprite pixels no longer terminate the ray;
#   they are skipped and the color continues to the boundary.
# - Small trace: if a left-facing sprite at col 23 has a matching left-facing sprite at col 6
#   on the same rows, then the old rule stopped at col 7 when the ray hit the blocker sprite.
#   The new rule colors bg cells from col 22 down to col 0, leaving the blocker sprite intact
#   at cols 6..7 and resuming on bg cells at cols 5..0. The same transparency also lets a
#   boundary marker at col 0 label the right-hand sprite even though another sprite lies in between.
#
# b) Complexity calibration:
# - The template logic, color uniqueness, and template-port matching stay unchanged.
# - The generator keeps 3-4 sprites and 3 distinct markers, so the puzzle still has repeated
#   evidence for port correspondence rather than a single special case.
# - To avoid orthogonal long-ray intersections introduced by the new stop rule, the generator
#   uses vertically oriented sprites only. It pairs opposite-facing vertical sprites so one sprite
#   can block the line-of-sight and the output path without also reproducing the same boundary-side
#   color segment under the old rule. This is an adaptation for D5 visibility, not a rule change:
#   the solver still accepts any orientation.
#
# c) Old assumptions that break under D5:
# - `_first_marker_on_ray(...)` previously returned at the first non-background cell. That only
#   worked because markers had guaranteed clear line-of-sight. Under D5 this must ignore sprite
#   pixels and keep scanning until a marker or the boundary is reached.
# - The drawing loop previously `break`ed on any non-background cell. Under D5 that would erase
#   the new behavior entirely. The correct logic is: paint bg cells, preserve non-bg cells, and
#   continue stepping until out of bounds.
# - The old generator's clear-line-of-sight check for markers is no longer valid. It would reject
#   exactly the bridged cases that this variant needs.
#
# d) Placement legality:
# - Sprites remain dense rectangles, non-overlapping, fully in-bounds, and separated into an
#   upper band and a lower band.
# - The upper band always contains a right-facing blocker on the left and a left-facing emitter on
#   the right, aligned on the same rows. Every chosen marker sits on the emitter's left-boundary
#   ray behind that blocker, so the input itself depends on transparency.
# - Optional lower-band right-facing sprites provide more matched copies without creating
#   orthogonal ray conflicts.
#
# e) Visual observability:
# - The changed rule is directly visible: colored lines appear on both sides of an intervening
#   sprite, and the boundary markers for the upper-right sprite sit behind another sprite.
# - This is inferable from 2-3 examples because the same transparency pattern repeats for three
#   marker colors and across multiple sprite copies.
#
# f) Visual cleanliness:
# - All rays are horizontal in generated examples, so there is no clutter from dense crossing
#   vertical/horizontal beams.
# - Upper and lower sprite bands are separated, making the copied colors easy to track.
#
# g) Color inference:
# - Marker colors are still unique and disjoint from template colors.
# - Colors are never sorted or transformed by numeric id; they are only used as visible labels
#   attached to repeated template ports.
#
# Post-change solver assumptions checked against the generator:
# - Every sprite is still a dense non-square rectangle.
# - Every marker color labels exactly one active-edge template id.
# - Rays of different colors never need to share the same background cell in generated samples.
# - At least one chosen marker is placed behind another sprite, and at least one output segment on
#   the marker side of that blocker is reachable only under the new stop rule.


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
        marker_color = markers.get((r, c))
        if marker_color is not None:
            return marker_color
        r += dr
        c += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy boundary-labeled template ports across matching sprites, with rays that tunnel through other sprites.

    Input structure:
    - The background is the grid's majority color.
    - Several dense non-square rectangular sprites are dihedral transforms of one shared template.
    - Single-pixel markers sit on boundary cells along outward active-edge rays, and those marker sight-lines may cross other sprites.

    Rule:
    - For each sprite, identify the active long edge by the stronger repeated-color majority.
    - Following each active-edge direction outward, ignore intervening non-background pixels until a boundary marker is found; that marker color labels the corresponding template port.
    - From every matching port on every sprite, send that color outward to the grid boundary, painting only background cells and passing through sprite/marker cells without stopping.

    Key insight:
    - The markers assign colors to shared template ports rather than to absolute positions, and in this variant sprites are transparent to the ray logic.
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
            marker_color = _first_marker_on_ray(g, markers, r0 + lr, c0 + lc, dr, dc)
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
                if int(g[r, c]) == bg:
                    assert int(out[r, c]) in (bg, color), "control rays must not conflict on background cells"
                    out[r, c] = color
                else:
                    assert int(out[r, c]) == int(g[r, c]), "rays may cross objects but may not rewrite them"
                r += dr
                c += dc

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


def _directional_variants(
    template: np.ndarray,
    ids_base: np.ndarray,
) -> dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]]:
    raw = _dihedral_variants(template)
    raw_ids = _dihedral_variants(ids_base)

    seen: list[np.ndarray] = []
    by_direction: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]] = {}
    for (_, sub, _), (_, _, ids_arr) in zip(raw, raw_ids):
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        direction, active_set = _active_side(sub)
        seen.append(sub.copy())
        by_direction.setdefault(direction, []).append((sub, ids_arr, direction, sorted(active_set)))
    return by_direction


def _active_id_order(ids_arr: np.ndarray, active_coords: list[tuple[int, int]]) -> list[int]:
    return [int(ids_arr[r, c]) for r, c in sorted(active_coords)]


def _make_sprite(
    r0: int,
    c0: int,
    sub: np.ndarray,
    ids_arr: np.ndarray,
    direction: tuple[int, int],
    active_coords: list[tuple[int, int]],
) -> dict:
    h, w = sub.shape
    return {
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        "sub": sub,
        "ids": ids_arr,
        "direction": direction,
        "active": set(active_coords),
    }


def _place_structured_sprites(
    H: int,
    W: int,
    N: int,
    variants_by_direction: dict,
    num_sprites: int,
) -> list[dict] | None:
    left_choices = variants_by_direction.get((0, -1), [])
    right_choices = variants_by_direction.get((0, 1), [])
    if not left_choices or not right_choices:
        return None

    compatible_pairs = []
    for left_variant in left_choices:
        left_order = _active_id_order(left_variant[1], left_variant[3])
        for right_variant in right_choices:
            if _active_id_order(right_variant[1], right_variant[3]) == left_order:
                compatible_pairs.append((left_variant, right_variant))
    if not compatible_pairs:
        return None
    left_variant, right_variant = random.choice(compatible_pairs)

    upper_top = 1
    lower_top = H - N - 1
    if lower_top <= upper_top + N:
        return None

    left_blocker_c = random.randint(5, 7)
    left_emitter_c = W - 4
    right_emitter_c = 2
    right_blocker_c = random.randint(W - 8, W - 6)

    placements = [
        _make_sprite(upper_top, left_blocker_c, *right_variant),
        _make_sprite(upper_top, left_emitter_c, *left_variant),
        _make_sprite(lower_top, right_emitter_c, *right_variant),
    ]
    if num_sprites == 4:
        placements.append(_make_sprite(lower_top, right_blocker_c, *left_variant))
    return placements


def _paint_sprites(H: int, W: int, placements: list[dict]) -> np.ndarray | None:
    grid = np.full((H, W), _BG, dtype=int)
    for sprite in placements:
        r0, c0, r1, c1 = sprite["bbox"]
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            return None
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != _BG):
            return None
        grid[r0 : r1 + 1, c0 : c1 + 1] = sprite["sub"]
    return grid


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
    placements: list[dict],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    active_template_ids = [int(ids_base[r, 0]) for r in range(ids_base.shape[0])]
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    emitter = placements[1]  # upper-right left-facing sprite; all its markers lie behind the blocker.
    used_cells: set[tuple[int, int]] = set()
    for color, tid in zip(marker_colors, labeled_ids):
        locs = np.argwhere(emitter["ids"] == tid)
        if locs.shape[0] != 1:
            return False
        lr, lc = map(int, locs[0])
        if (lr, lc) not in emitter["active"]:
            return False
        r0, c0, _r1, _c1 = emitter["bbox"]
        dr, dc = emitter["direction"]
        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)
        if (marker_r, marker_c) in used_cells or int(grid[marker_r, marker_c]) != _BG:
            return False
        grid[marker_r, marker_c] = color
        used_cells.add((marker_r, marker_c))
    return True


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)
    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors = design

    variants_by_direction = _directional_variants(template, ids_base)
    placements = _place_structured_sprites(H, W, N, variants_by_direction, num_sprites)
    if placements is None:
        return None

    grid = _paint_sprites(H, W, placements)
    if grid is None:
        return None

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
