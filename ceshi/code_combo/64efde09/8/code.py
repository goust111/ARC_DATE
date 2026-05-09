# Combo variant of 64efde09 (index 8)
# combo_spec: D13/2+D15/0
# Name: 面对面射线交叉混合

from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg

# Design analysis for the fused variant
# a) Concrete behavior:
#    - Original task: a marker on a boundary ray labels one active-edge template cell, then every
#      transformed sprite emits that color straight toward the boundary, and different colors never meet.
#    - New task: four sprites are arranged as left/right/up/down pairs around one empty N-by-N box.
#      Each chosen template port now has one marker placed at the crossing of its horizontal and vertical
#      partner rays inside that box. In the output, left+right emit the same-color horizontal band and
#      top+bottom emit the same-color vertical band. Off-diagonal crossings become a dedicated mix color.
#    - Small example with 3 labels A/B/C:
#      horizontal rows = A,B,C and vertical cols = A,B,C. Cell (row A, col A) stays A because every ray
#      reaching it has color A; cell (row A, col B) becomes MIX because A-horizontal meets B-vertical.
# b) Complexity calibration:
#    - Using exactly four sprites keeps the geometry readable but still forces the learner to combine
#      template matching, opposite-facing symmetry, and collision coloring.
#    - Using exactly three labels gives repeated evidence: 3 horizontal lines, 3 vertical lines, and
#      6 mixed crossings. That is visibly different from the original no-conflict behavior.
# c) Broken original assumptions:
#    - Original `_random_sprite_position` assumed active sides face an outer boundary; that is replaced
#      by a fixed inward-facing cross layout.
#    - Original drawing stopped at any non-bg cell and asserted no color conflicts. That is no longer
#      valid because markers sit on the ray corridor and orthogonal rays must cross. The solver therefore
#      treats markers as transparent control cells when tracing, then resolves multi-color visits after
#      all rays are collected.
#    - Original generator only had to avoid collisions; this generator must instead guarantee intentional
#      collisions while keeping markers isolated components in the input.
# d) Placement legality:
#    - The generator derives a central empty box and places the four dense rectangles flush against its
#      four sides, so sprites never overlap and always stay in bounds.
#    - Marker cells are chosen inside the box at pairwise non-8-adjacent coordinates, so `find_components`
#      still sees them as isolated single-pixel components.
# e) Visual observability:
#    - Inputs show four matched sprites staring into one empty square with a few interior markers.
#      Outputs turn that square into a grid of colored bars whose non-matching crossings all share one
#      dedicated mix color. The collision rule is therefore visible without relying on color-number order.
# f) Visual cleanliness:
#    - All interactions are confined to the central box. Sprites remain untouched, markers are sparse,
#      and the only overlaps are the intended orthogonal crossings.
# g) Inferrable variation:
#    - The only new color transform is "different ray colors at one cell -> fixed mix color". Because the
#      same mix color appears repeatedly at structurally identical off-diagonal crossings, a human can
#      infer the rule from a few training pairs.
# Additional solver assumptions introduced here and guaranteed by the generator:
#    - There are exactly four sprites, one per cardinal active direction.
#    - Each marker lies on one selected horizontal ray and one selected vertical ray for the same
#      template port, so every marker-to-template mapping is self-consistent from all four sprites.
#    - Marker colors are unique and disjoint from sprite colors and the mix color.

_BG = 8
_MIX_COLOR = 9
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7]

_GRID_H_RANGE = (22, 29)
_GRID_W_RANGE = (22, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_NUM_MARKERS = 3
_GEN_RETRIES = 300


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


def _parse_scene(g: np.ndarray) -> tuple[int, list[dict], dict[tuple[int, int], int]]:
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

    assert len(sprites_raw) == 4, "expected four inward-facing sprites"
    assert len(markers) == _NUM_MARKERS, "expected exactly three isolated control markers"
    assert len(set(markers.values())) == len(markers), "control marker colors must be unique"
    assert _MIX_COLOR not in markers.values(), "mix color is reserved for output only"
    return bg, sprites_raw, markers


def _identify_sprites(sprites_raw: list[dict]) -> list[dict]:
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

    dirs = {sprite["direction"] for sprite in sprites}
    assert dirs == {(0, -1), (0, 1), (-1, 0), (1, 0)}, "need one sprite per cardinal direction"
    return sprites


def _label_mapping(
    g: np.ndarray,
    bg: int,
    sprites: list[dict],
    markers: dict[tuple[int, int], int],
) -> dict[int, int]:
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

    assert set(color_to_template_id) == set(markers.values()), "every marker must label a template port"
    return color_to_template_id


def _collect_ray_visits(
    g: np.ndarray,
    bg: int,
    sprites: list[dict],
    markers: dict[tuple[int, int], int],
    color_to_template_id: dict[int, int],
) -> dict[tuple[int, int], set[int]]:
    visits: dict[tuple[int, int], set[int]] = {}
    H, W = g.shape
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for color, template_id in color_to_template_id.items():
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            assert (lr, lc) in sprite["active"], "mapped port must land on the active edge"

            r = r0 + lr + dr
            c = c0 + lc + dc
            while 0 <= r < H and 0 <= c < W:
                if int(g[r, c]) != bg and (r, c) not in markers:
                    break
                visits.setdefault((r, c), set()).add(color)
                r += dr
                c += dc
    return visits


def _render_from_visits(g: np.ndarray, visits: dict[tuple[int, int], set[int]]) -> np.ndarray:
    out = np.array(g, dtype=int, copy=True)
    for (r, c), colors in visits.items():
        uniq = set(colors)
        if len(uniq) == 1:
            out[r, c] = next(iter(uniq))
        else:
            out[r, c] = _MIX_COLOR
    return out


def _render_without_mix(g: np.ndarray, visits: dict[tuple[int, int], set[int]]) -> np.ndarray:
    out = np.array(g, dtype=int, copy=True)
    for (r, c), colors in visits.items():
        out[r, c] = sorted(colors)[0]
    return out


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg, sprites_raw, markers = _parse_scene(g)
    sprites = _identify_sprites(sprites_raw)
    color_to_template_id = _label_mapping(g, bg, sprites, markers)
    visits = _collect_ray_visits(g, bg, sprites, markers, color_to_template_id)
    out = _render_from_visits(g, visits)
    metrics = {
        "collision_cells": {cell for cell, cols in visits.items() if len(cols) >= 2},
        "mixed_cells": {cell for cell, cols in visits.items() if len(cols) >= 2},
        "visits": visits,
    }
    return out, metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reuse marker-labeled template ports across four inward-facing sprite copies, then mix orthogonal ray crossings.

    Input structure:
    - bg is the most common color.
    - Exactly four dense rectangular sprites are dihedral transforms of one non-square template, and their
      active long edges face inward from left, right, top, and bottom around one empty central box.
    - Three isolated single-pixel markers sit inside that box. Each marker lies simultaneously on one
      horizontal pair ray and one vertical pair ray for the same template port.

    Rule:
    - For each sprite, the active edge is the long side whose cells have the strongest same-color majority.
    - A marker first seen along an outward active-edge ray labels that template port with the marker color.
    - Every sprite emits that color from the corresponding active-edge cell. Markers are transparent control
      cells while tracing, so opposite rays pass through them. If only one color visits a cell, paint it
      with that color; if different colors visit the same cell, paint that crossing with the dedicated mix color.

    Key insight:
    - The markers do not encode absolute destinations. They identify shared template ports, and the four
      symmetric inward-facing sprites turn those labels into a horizontal/vertical crossing lattice whose
      off-diagonal intersections reveal the mixing rule.
    """
    out, _metrics = _solve_core(grid)
    return out


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
    right_col = _build_column(N, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _layout_variants(
    template: np.ndarray, ids_base: np.ndarray
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]]:
    variants = {
        (0, -1): (np.array(template, copy=True), np.array(ids_base, copy=True)),
        (0, 1): (np.fliplr(template), np.fliplr(ids_base)),
        (1, 0): (np.rot90(template, 1), np.rot90(ids_base, 1)),
        (-1, 0): (np.fliplr(np.rot90(template, 3)), np.fliplr(np.rot90(ids_base, 3))),
    }
    result: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]] = {}
    for expected_dir, (sub, ids_arr) in variants.items():
        direction, active_set = _active_side(sub)
        assert direction == expected_dir, "layout variant must realize the intended inward direction"
        result[direction] = (sub, ids_arr, list(active_set))
    return result


def _pick_labeled_ids(
    ids_base: np.ndarray,
    right_variant: tuple[np.ndarray, np.ndarray, list[tuple[int, int]]],
    down_variant: tuple[np.ndarray, np.ndarray, list[tuple[int, int]]],
) -> list[int] | None:
    candidate_ids = [int(ids_base[r, 0]) for r in range(ids_base.shape[0])]
    _sub_r, ids_right, active_right = right_variant
    _sub_d, ids_down, active_down = down_variant
    right_active = set(active_right)
    down_active = set(active_down)

    candidate_positions = []
    for tid in candidate_ids:
        loc_r = np.argwhere(ids_right == tid)
        loc_d = np.argwhere(ids_down == tid)
        if loc_r.shape[0] != 1 or loc_d.shape[0] != 1:
            continue
        rr, rc = map(int, loc_r[0])
        dr, dc = map(int, loc_d[0])
        if (rr, rc) not in right_active or (dr, dc) not in down_active:
            continue
        if rr in (0, ids_base.shape[0] - 1) or dc in (0, ids_base.shape[0] - 1):
            continue
        candidate_positions.append((tid, rr, dc))

    all_picks = list(combinations(candidate_positions, _NUM_MARKERS))
    random.shuffle(all_picks)
    for picked in all_picks:
        marker_cells = [(row, col) for _tid, row, col in picked]
        if all(max(abs(r1 - r2), abs(c1 - c2)) > 1 for (r1, c1), (r2, c2) in combinations(marker_cells, 2)):
            return [tid for tid, _row, _col in picked]
    return None


def _place_cross_layout(
    grid: np.ndarray,
    variants: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]],
    N: int,
) -> tuple[list[tuple], tuple[int, int]] | None:
    H, W = grid.shape
    if H < N + 7 or W < N + 7:
        return None

    core_r = random.randint(3, H - N - 4)
    core_c = random.randint(3, W - N - 4)

    placements = [
        (core_r, core_c - 3, *variants[(0, 1)], (0, 1)),        # left sprite, active edge faces right
        (core_r, core_c + N + 2, *variants[(0, -1)], (0, -1)),  # right sprite, active edge faces left
        (core_r - 3, core_c, *variants[(1, 0)], (1, 0)),        # top sprite, active edge faces down
        (core_r + N + 2, core_c, *variants[(-1, 0)], (-1, 0)),  # bottom sprite, active edge faces up
    ]

    for r0, c0, sub, _ids_arr, _active, _direction in placements:
        h, w = sub.shape
        if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
            return None
        if np.any(grid[r0 : r0 + h, c0 : c0 + w] != _BG):
            return None
        grid[r0 : r0 + h, c0 : c0 + w] = sub
    return placements, (core_r, core_c)


def _place_markers(
    grid: np.ndarray,
    box_origin: tuple[int, int],
    labeled_ids: list[int],
    ids_right: np.ndarray,
    ids_down: np.ndarray,
    template_colors: list[int],
) -> bool:
    box_r, box_c = box_origin
    available = [c for c in _NON_BG_COLORS if c not in template_colors]
    if len(available) < len(labeled_ids):
        return False
    marker_colors = random.sample(available, len(labeled_ids))

    seen_cells: set[tuple[int, int]] = set()
    for color, tid in zip(marker_colors, labeled_ids):
        loc_r = np.argwhere(ids_right == tid)
        loc_d = np.argwhere(ids_down == tid)
        if loc_r.shape[0] != 1 or loc_d.shape[0] != 1:
            return False
        row = int(loc_r[0, 0])
        col = int(loc_d[0, 1])
        cell = (box_r + row, box_c + col)
        if cell in seen_cells or grid[cell] != _BG:
            return False
        seen_cells.add(cell)
        grid[cell] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors = design

    variants = _layout_variants(template, ids_base)
    need_dirs = {(0, -1), (0, 1), (-1, 0), (1, 0)}
    if set(variants) != need_dirs:
        return None

    labeled_ids = _pick_labeled_ids(ids_base, variants[(0, 1)], variants[(1, 0)])
    if labeled_ids is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    placed = _place_cross_layout(grid, variants, N)
    if placed is None:
        return None
    placements, box_origin = placed
    _ = placements

    _sub_r, ids_right, _active_r = variants[(0, 1)]
    _sub_d, ids_down, _active_d = variants[(1, 0)]
    if not _place_markers(grid, box_origin, labeled_ids, ids_right, ids_down, template_colors):
        return None

    try:
        out, metrics = _solve_core(grid)
    except Exception:
        return None

    no_mix = _render_without_mix(grid, metrics["visits"])
    if len(metrics["mixed_cells"]) < 3:
        return None
    if np.array_equal(out, no_mix):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
