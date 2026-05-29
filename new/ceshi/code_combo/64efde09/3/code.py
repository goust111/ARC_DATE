# Combo variant of 64efde09 (index 3)
# combo_spec: D3/1+D10/2
# Name: 密集颜色循环射线

# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior trace
#    Original task:
#    - Only a few active-edge ports are labeled.
#    - Each labeled port emits a solid-color ray on every matching sprite.
#    Fused task:
#    - Every active-edge port on one anchor sprite is labeled, so there are N markers for an N-cell edge.
#    - Read those N marker colors in visible edge order (top->bottom for vertical edges, left->right for horizontal).
#    - A ray no longer stays solid. If the ordered marker strip is [A,B,C,D], then the port tagged by A emits
#      A,B,C,D,A,B,...; the port tagged by C emits C,D,A,B,C,D,... .
#    Example with a 4x2 vertical template and anchor markers [1,5,7,2] from top to bottom:
#    - Original output from the 2nd port would be 5,5,5,5,...
#    - New output from the 2nd port is 5,7,2,1,5,7,... .
#    The difference is both denser (all ports active) and structurally linked across ports (phase-shifted cycle).
#
# b) Complexity calibration
#    - The cycle order is geometric, not hidden: it is read directly from the one fully labeled sprite edge.
#    - Marker colors are unique within the cycle, so the repeating order is easy to track by eye.
#    - Rays are long enough to show at least one wraparound, so the output is visibly different from "solid rays".
#
# c) Implicit-assumption audit
#    - Original `color -> template_id` uniqueness no longer scales well if every port is labeled; the solver now
#      uses `template_id -> cycle index` from the anchor sprite instead.
#    - Original generation put markers on the boundary. With N consecutive ports this would merge into one 8-connected
#      component, breaking the "isolated single pixels" assumption. The generator therefore keeps markers isolated by
#      staggering them along the outward rays while preserving one marker per port and clear line-of-sight.
#    - Output painting can no longer write one constant color per ray; it must paint cell-by-cell using the inferred cycle.
#
# d) Placement legality
#    - Each scene uses one orientation family at a time: either tall Nx2 variants with horizontal rays or wide 2xN
#      variants with vertical rays.
#    - Sprite bands are disjoint along the axis orthogonal to the rays, which prevents inter-sprite ray crossings.
#    - Anchor markers sit on alternating gutter cells, keeping them isolated under 8-connectivity.
#
# e) / f) Observability and cleanliness
#    - One sprite carries the full marker strip, making the cycle order explicit in a single place.
#    - The other sprites are unlabeled copies, so the solver must transfer that cycle through rotation/reflection.
#    - Dense rainbow rows are visually distinct from the original sparse solid rays and remain easy to segment because
#      row bands do not overlap.
#
# g) New solver assumptions guaranteed by `generate_input()`
#    - Exactly one sprite has all N ports labeled.
#    - That fully labeled sprite defines the cycle order.
#    - Every sprite is a unique dihedral variant of the same template within the scene's chosen orientation family.
#    - Every ray has enough empty runway to show a repeated cycle and no two ray systems overlap.

from __future__ import annotations

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


def _ordered_edge_coords(active: set[tuple[int, int]]) -> list[tuple[int, int]]:
    coords = sorted(active)
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1:
        return sorted(coords, key=lambda rc: rc[1])
    assert len(cols) == 1
    return sorted(coords, key=lambda rc: rc[0])


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


def _parse_scene(
    grid: np.ndarray,
) -> tuple[int, list[dict], list[int], dict[int, int]]:
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

    variants = _dihedral_variants(sprites_raw[0]["sub"])
    sprites: list[dict] = []
    for sprite in sprites_raw:
        matches = [
            (ids, name)
            for name, sub, ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "each sprite must match the template in exactly one orientation"
        ids, _name = matches[0]
        direction, active = _active_side(sprite["sub"])
        ordered_active = _ordered_edge_coords(active)
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = direction
        labels = [
            _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            for lr, lc in ordered_active
        ]
        sprites.append(
            {
                **sprite,
                "ids": ids,
                "direction": direction,
                "active": active,
                "ordered_active": ordered_active,
                "labels": labels,
            }
        )

    anchor_candidates = [sprite for sprite in sprites if all(color is not None for color in sprite["labels"])]
    assert len(anchor_candidates) == 1, "exactly one sprite must carry the full marker strip"
    anchor = anchor_candidates[0]
    assert len(markers) == len(anchor["ordered_active"]), "marker count must equal active-edge length"

    cycle_colors: list[int] = []
    template_id_to_index: dict[int, int] = {}
    for idx, ((lr, lc), color) in enumerate(zip(anchor["ordered_active"], anchor["labels"])):
        assert color is not None
        template_id = int(anchor["ids"][lr, lc])
        assert template_id not in template_id_to_index
        template_id_to_index[template_id] = idx
        cycle_colors.append(int(color))

    assert len(cycle_colors) >= 2
    for sprite in sprites:
        for (lr, lc), color in zip(sprite["ordered_active"], sprite["labels"]):
            if color is None:
                continue
            template_id = int(sprite["ids"][lr, lc])
            assert template_id in template_id_to_index
            assert cycle_colors[template_id_to_index[template_id]] == int(color)

    return bg, sprites, cycle_colors, template_id_to_index


def _paint_output(
    grid: np.ndarray,
    *,
    cycle_colors: list[int],
    template_id_to_index: dict[int, int],
    solid: bool,
) -> np.ndarray:
    bg, sprites, _cycle_colors_check, _template_index_check = _parse_scene(grid)
    assert cycle_colors == _cycle_colors_check
    assert template_id_to_index == _template_index_check

    g = np.asarray(grid, dtype=int)
    out = g.copy()
    H, W = g.shape
    cycle_len = len(cycle_colors)

    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["ordered_active"]:
            template_id = int(sprite["ids"][lr, lc])
            start_idx = int(template_id_to_index[template_id])
            r = r0 + lr + dr
            c = c0 + lc + dc
            step = 0
            while 0 <= r < H and 0 <= c < W:
                if int(g[r, c]) != bg:
                    break
                color = int(cycle_colors[start_idx if solid else (start_idx + step) % cycle_len])
                assert int(out[r, c]) in (bg, color), "control rays must not conflict"
                out[r, c] = color
                r += dr
                c += dc
                step += 1

    return out


def _solid_baseline_output(grid: np.ndarray) -> np.ndarray:
    bg, sprites, cycle_colors, template_id_to_index = _parse_scene(grid)
    _ = bg, sprites
    return _paint_output(
        grid,
        cycle_colors=cycle_colors,
        template_id_to_index=template_id_to_index,
        solid=True,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer a fully labeled edge cycle across matching rectangular sprites and emit dense phase-shifted rainbow rays.

    Input structure:
    - The background is the most frequent color.
    - Several dense Nx2 rectangles are dihedral transforms of one shared template.
    - Exactly one sprite has every active-edge port labeled by an isolated single-pixel marker placed outward on its ray.

    Rule:
    - For each sprite, find the active long edge as the side with the stronger same-color majority.
    - On the fully labeled sprite, read the marker colors in visible edge order to obtain the repeating cycle.
    - Each active-edge port on every sprite emits an outward ray through background cells, but the ray now advances through
      the whole cycle cell-by-cell, starting at the phase assigned to that corresponding template port.

    Key insight:
    - The anchor sprite does not merely assign one color per port. Its complete ordered marker strip defines a cyclic code,
      and every transformed copy reuses that code with the correct port-specific phase shift.
    """
    bg, sprites, cycle_colors, template_id_to_index = _parse_scene(grid)
    _ = bg, sprites
    return _paint_output(
        grid,
        cycle_colors=cycle_colors,
        template_id_to_index=template_id_to_index,
        solid=False,
    )


# --------------- generator (dense_cycle_anchor) ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_SPRITE_COUNT = 3
_BAND_GAP = 1
_GEN_RETRIES = 500
_NEXT_SCENE_MODE = 0


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    for _ in range(n - majority):
        col.append(random.choice(others))
    random.shuffle(col)
    return col


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)

    active_majority = random.randint(_ACTIVE_MAJORITY_RANGE[0], min(N, _ACTIVE_MAJORITY_RANGE[1]))
    left_col = _build_column(N, template_colors, active_majority, random.choice(template_colors))

    max_right_majority = active_majority - 1
    if max_right_majority < 2:
        return None
    min_right_majority = max(2, N // num_colors)
    if min_right_majority > max_right_majority:
        return None
    right_majority = random.randint(min_right_majority, max_right_majority)
    right_col = _build_column(N, template_colors, right_majority, random.choice(template_colors))

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base


def _scene_variants(
    template: np.ndarray,
    mode: str,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
    seen: list[np.ndarray] = []
    result = []
    for _name, sub, ids_arr in _dihedral_variants(template):
        if mode == "vertical":
            if sub.shape[0] <= sub.shape[1]:
                continue
        else:
            if sub.shape[1] <= sub.shape[0]:
                continue
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        direction, active_set = _active_side(sub)
        if mode == "vertical" and direction[0] != 0:
            continue
        if mode == "horizontal" and direction[1] != 0:
            continue
        seen.append(sub)
        result.append((sub, ids_arr, direction, _ordered_edge_coords(active_set)))
    return result


def _band_starts(total_span: int, item_span: int) -> list[int]:
    base_used = _SPRITE_COUNT * item_span + (_SPRITE_COUNT - 1) * _BAND_GAP
    slack = total_span - base_used
    pads = [0] * (_SPRITE_COUNT + 1)
    for _ in range(slack):
        pads[random.randrange(len(pads))] += 1

    starts: list[int] = []
    r = pads[0]
    for idx in range(_SPRITE_COUNT):
        starts.append(r)
        r += item_span
        if idx < _SPRITE_COUNT - 1:
            r += _BAND_GAP + pads[idx + 1]
    return starts


def _sample_col(W: int, N: int, direction: tuple[int, int], is_anchor: bool) -> int | None:
    min_len = N + 2
    if direction == (0, -1):
        min_c0 = (2 if is_anchor else 0) + min_len + 1
        max_c0 = W - 2
        if min_c0 > max_c0:
            return None
        return random.randint(min_c0, max_c0)
    if direction == (0, 1):
        max_c0 = (W - 3) - min_len - 2 if is_anchor else W - min_len - 2
        if max_c0 < 0:
            return None
        return random.randint(0, max_c0)
    return None


def _sample_row(H: int, N: int, direction: tuple[int, int], is_anchor: bool) -> int | None:
    min_len = N + 2
    if direction == (-1, 0):
        min_r0 = (2 if is_anchor else 0) + min_len + 1
        max_r0 = H - 2
        if min_r0 > max_r0:
            return None
        return random.randint(min_r0, max_r0)
    if direction == (1, 0):
        max_r0 = (H - 3) - min_len - 2 if is_anchor else H - min_len - 2
        if max_r0 < 0:
            return None
        return random.randint(0, max_r0)
    return None


def _marker_position(
    abs_r: int,
    abs_c: int,
    direction: tuple[int, int],
    idx: int,
    H: int,
    W: int,
) -> tuple[int, int]:
    dr, dc = direction
    if (dr, dc) == (0, -1):
        return abs_r, 0 if idx % 2 == 0 else 2
    if (dr, dc) == (0, 1):
        return abs_r, W - 1 if idx % 2 == 0 else W - 3
    if (dr, dc) == (-1, 0):
        return 0 if idx % 2 == 0 else 2, abs_c
    return H - 1 if idx % 2 == 0 else H - 3, abs_c


def _place_markers(
    grid: np.ndarray,
    anchor: tuple[int, int, np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]],
) -> bool:
    H, W = grid.shape
    r0, c0, _sub, _ids_arr, direction, active_coords = anchor
    marker_colors = random.sample(_NON_BG_COLORS, len(active_coords))
    used_positions: set[tuple[int, int]] = set()

    dr, dc = direction
    for idx, ((lr, lc), color) in enumerate(zip(active_coords, marker_colors)):
        marker_r, marker_c = _marker_position(r0 + lr, c0 + lc, direction, idx, H, W)
        if (marker_r, marker_c) in used_positions:
            return False
        if int(grid[marker_r, marker_c]) != _BG:
            return False

        cr, cc = r0 + lr + dr, c0 + lc + dc
        while (cr, cc) != (marker_r, marker_c):
            if not (0 <= cr < H and 0 <= cc < W):
                return False
            if int(grid[cr, cc]) != _BG:
                return False
            cr += dr
            cc += dc

        grid[marker_r, marker_c] = int(color)
        used_positions.add((marker_r, marker_c))
    return True


def _gen_attempt(mode: str) -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    min_band_total = _SPRITE_COUNT * N + (_SPRITE_COUNT - 1) * _BAND_GAP
    if mode == "vertical":
        min_H = max(_GRID_H_RANGE[0], min_band_total)
        if min_H > _GRID_H_RANGE[1]:
            return None
        H = random.randint(min_H, _GRID_H_RANGE[1])
        W = random.randint(*_GRID_W_RANGE)
    else:
        min_W = max(_GRID_W_RANGE[0], min_band_total)
        if min_W > _GRID_W_RANGE[1]:
            return None
        H = random.randint(*_GRID_H_RANGE)
        W = random.randint(min_W, _GRID_W_RANGE[1])

    design = _design_template(N)
    if design is None:
        return None
    template, _ids_base = design

    variants = _scene_variants(template, mode)
    if len(variants) < _SPRITE_COUNT:
        return None

    chosen = random.sample(variants, _SPRITE_COUNT)
    starts = _band_starts(H if mode == "vertical" else W, N)
    anchor_idx = random.randrange(_SPRITE_COUNT)

    placements: list[tuple[int, int, np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]] = []
    for idx, ((sub, ids_arr, direction, active_coords), start) in enumerate(zip(chosen, starts)):
        if mode == "vertical":
            r0 = start
            c0 = _sample_col(W, N, direction, is_anchor=(idx == anchor_idx))
            if c0 is None:
                return None
        else:
            c0 = start
            r0 = _sample_row(H, N, direction, is_anchor=(idx == anchor_idx))
            if r0 is None:
                return None
        placements.append((r0, c0, sub, ids_arr, direction, active_coords))

    grid = np.full((H, W), _BG, dtype=int)
    for r0, c0, sub, *_rest in placements:
        h, w = sub.shape
        if np.any(grid[r0 : r0 + h, c0 : c0 + w] != _BG):
            return None
        grid[r0 : r0 + h, c0 : c0 + w] = sub

    if not _place_markers(grid, placements[anchor_idx]):
        return None

    try:
        out = main(grid)
        solid = _solid_baseline_output(grid)
    except Exception:
        return None

    if np.array_equal(out, solid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _NEXT_SCENE_MODE
    preferred_mode = "vertical" if _NEXT_SCENE_MODE % 2 == 0 else "horizontal"
    fallback_mode = "horizontal" if preferred_mode == "vertical" else "vertical"
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt(preferred_mode)
        if result is None:
            result = _gen_attempt(fallback_mode)
        if result is not None:
            _NEXT_SCENE_MODE += 1
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
