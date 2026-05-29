# Combo variant of 64efde09 (index 10)
# combo_spec: D2/1+D13/2
# Name: 反射乒乓射线

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the fused variant:
# a) Original markers lived on boundary-facing rays; here the active edges face another sprite,
#    so markers must move into the corridor between paired sprites. A ray now travels from one
#    active port into that corridor, can hit the opposite sprite, reflect, and ping-pong back.
# b) Pure axis-aligned reflection would otherwise look too similar to a straight stop-at-blocker
#    fill, so the reflected contact on the sprite face is also painted. That makes reflection
#    observable in output and prevents degeneration back to the original behavior.
# c) The original solver assumed "first non-bg on the ray is a boundary marker" and "markers are
#    blockers during drawing". Both break here. We instead read the first marker before the next
#    sprite, then treat markers as transparent color seeds during propagation.
# d) Generator legality is enforced by using two separated face-to-face pairs on a single axis,
#    wide corridors, isolated markers, and lane-separated labels so nothing overlaps or merges.
# e) Visual observability comes from repeated pair structure: two corridors, three marker colors,
#    and the same template ports lighting up on all four sprites across horizontal or vertical layouts.
# f) Cleanliness is preserved by keeping all rays axis-aligned and putting the two pairs on disjoint
#    bands, so different colors never cross.
# g) Color inference stays small: three unique marker colors each bind one template port, then the
#    same port repeats across every isomorphic sprite and lights both the corridor and bounce port.


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


def _first_marker_on_lane(
    g: np.ndarray,
    markers: dict[tuple[int, int], int],
    sprite_cells: set[tuple[int, int]],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> int | None:
    H, W = g.shape
    seen: int | None = None
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if (r, c) in sprite_cells:
            return seen
        marker_color = markers.get((r, c))
        if marker_color is not None:
            if seen is None:
                seen = int(marker_color)
            else:
                assert seen == int(marker_color), "a lane cannot contain multiple marker colors"
        r += dr
        c += dc
    return seen


def _paint_path_cell(
    out: np.ndarray,
    original: np.ndarray,
    markers: dict[tuple[int, int], int],
    color: int,
    r: int,
    c: int,
) -> None:
    current = int(out[r, c])
    orig = int(original[r, c])
    if (r, c) in markers:
        assert orig == color, "rays may only pass through their own marker"
        assert current in (orig, color), "marker cells cannot be recolored by another ray"
    elif current not in (orig, color):
        raise AssertionError("reflective rays must not conflict")
    out[r, c] = int(color)


def _trace_reflective_ray(
    out: np.ndarray,
    original: np.ndarray,
    markers: dict[tuple[int, int], int],
    sprite_cells: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    max_reflections: int,
) -> None:
    H, W = original.shape
    cur_r, cur_c = map(int, start)
    dr, dc = map(int, direction)
    reflections = 0

    while reflections <= max_reflections:
        nr = cur_r + dr
        nc = cur_c + dc

        if not (0 <= nr < H and 0 <= nc < W):
            reflections += 1
            dr, dc = -dr, -dc
            continue

        if (nr, nc) in sprite_cells:
            _paint_path_cell(out, original, markers, color, nr, nc)
            reflections += 1
            dr, dc = -dr, -dc
            continue

        _paint_path_cell(out, original, markers, color, nr, nc)
        cur_r, cur_c = nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read corridor markers as template-port labels, then fire finite reflective ping-pong rays between facing sprites.

    Input structure:
    - bg is the most common color.
    - Several dense 2-by-N or N-by-2 rectangular sprites are dihedral transforms of one template.
    - Sprites are arranged in face-to-face pairs, and isolated single-pixel markers sit inside the corridor in front of some active-edge ports.

    Rule:
    - For each sprite, the active edge is the long side whose cells have the strongest same-color majority.
    - The first marker seen from an active-edge cell before the next sprite labels that shared template port.
    - From every sprite's corresponding labeled port, launch a ray along the active-edge normal; marker cells are transparent, and when the ray hits a sprite face or the grid boundary it reflects and continues for a small fixed number of bounces, coloring every visited cell including reflected sprite-contact ports.

    Key insight:
    - The marker does not name an absolute location. It identifies one template port inside a face-to-face corridor, and that port must be replayed on every transformed copy while respecting the paired reflective geometry.
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
    sprite_cells: set[tuple[int, int]] = set()
    for sprite in sprites_raw:
        matches = [
            (ids, name)
            for name, sub, ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "each sprite must match the template in exactly one orientation"
        ids, _name = matches[0]
        direction, active = _active_side(sprite["sub"])
        r0, c0, r1, c1 = sprite["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                sprite_cells.add((r, c))
        sprites.append({**sprite, "ids": ids, "direction": direction, "active": active})

    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["active"]:
            marker_color = _first_marker_on_lane(g, markers, sprite_cells, r0 + lr, c0 + lc, dr, dc)
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
        direction = sprite["direction"]
        for color, template_id in color_to_template_id.items():
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            assert (lr, lc) in sprite["active"], "mapped port must land on the sprite active edge"
            _trace_reflective_ray(
                out,
                g,
                markers,
                sprite_cells,
                (r0 + lr, c0 + lc),
                direction,
                color,
                max_reflections=4,
            )

    return out


_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (22, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_NUM_MARKERS = 3
_PAIR_GAP_RANGE = (5, 8)
_PAIR_BAND_GAP_MIN = 3
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
    if max_right_majority < 2:
        return None
    right_majority = random.randint(max(2, N // num_colors), max_right_majority)
    right_col = _build_column(N, template_colors, right_majority, random.choice(template_colors))

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _facing_variants(
    template: np.ndarray,
    ids_base: np.ndarray,
    axis: str,
) -> tuple[tuple[np.ndarray, np.ndarray, tuple[int, int], set[tuple[int, int]]], ...] | None:
    raw = _dihedral_variants(template)
    raw_ids = _dihedral_variants(ids_base)
    by_dir: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, tuple[int, int], set[tuple[int, int]]]]] = {}

    for (_, sub, _), (_, _, ids_arr) in zip(raw, raw_ids):
        try:
            direction, active = _active_side(sub)
        except AssertionError:
            continue
        by_dir.setdefault(direction, []).append((sub.copy(), ids_arr.copy(), direction, set(active)))

    needed = ((0, 1), (0, -1)) if axis == "horizontal" else ((1, 0), (-1, 0))
    if needed[0] not in by_dir or needed[1] not in by_dir:
        return None

    def lane_map(
        variant: tuple[np.ndarray, np.ndarray, tuple[int, int], set[tuple[int, int]]],
    ) -> dict[int, int]:
        _sub, ids_arr, direction, active = variant
        return {
            int(ids_arr[lr, lc]): int(lr if direction[0] == 0 else lc)
            for lr, lc in active
        }

    for forward in by_dir[needed[0]]:
        forward_map = lane_map(forward)
        for backward in by_dir[needed[1]]:
            if forward_map == lane_map(backward):
                return forward, backward
    return None


def _two_band_starts(span: int, total: int) -> tuple[int, int] | None:
    max_between = total - (2 * span)
    if max_between < _PAIR_BAND_GAP_MIN:
        return None
    between = random.randint(_PAIR_BAND_GAP_MIN, max_between)
    slack = total - (2 * span + between)
    if slack < 0:
        return None
    first = random.randint(0, slack)
    return first, first + span + between


def _choose_marker_lanes(length: int) -> tuple[list[int], list[int]] | None:
    pair_assignments = [0, 1, random.randint(0, 1)]
    random.shuffle(pair_assignments)
    rows_by_pair: dict[int, list[int]] = {0: [], 1: []}
    chosen_rows: list[int] = []

    for pair_idx in pair_assignments:
        candidates = [
            r for r in range(length)
            if r not in chosen_rows and all(abs(r - prev) >= 2 for prev in rows_by_pair[pair_idx])
        ]
        if not candidates:
            return None
        row = random.choice(candidates)
        rows_by_pair[pair_idx].append(row)
        chosen_rows.append(row)

    return chosen_rows, pair_assignments


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)
    axis = random.choice(("horizontal", "vertical"))

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors = design

    facing_variants = _facing_variants(template, ids_base, axis)
    if facing_variants is None:
        return None
    forward_variant, backward_variant = facing_variants
    forward_sub, forward_ids, forward_dir, forward_active = forward_variant
    backward_sub, backward_ids, backward_dir, backward_active = backward_variant

    if axis == "horizontal":
        band_starts = _two_band_starts(forward_sub.shape[0], H)
        if band_starts is None:
            return None
        pair0_r, pair1_r = band_starts

        gap = random.randint(*_PAIR_GAP_RANGE)
        total_w = forward_sub.shape[1] + gap + backward_sub.shape[1]
        if total_w > W:
            return None
        left_c0 = random.randint(0, W - total_w)
        right_c0 = left_c0 + forward_sub.shape[1] + gap

        placements = [
            {
                "pair": 0,
                "side": "forward",
                "r0": pair0_r,
                "c0": left_c0,
                "sub": forward_sub,
                "ids": forward_ids,
                "direction": forward_dir,
                "active": forward_active,
            },
            {
                "pair": 0,
                "side": "backward",
                "r0": pair0_r,
                "c0": right_c0,
                "sub": backward_sub,
                "ids": backward_ids,
                "direction": backward_dir,
                "active": backward_active,
            },
            {
                "pair": 1,
                "side": "forward",
                "r0": pair1_r,
                "c0": left_c0,
                "sub": forward_sub,
                "ids": forward_ids,
                "direction": forward_dir,
                "active": forward_active,
            },
            {
                "pair": 1,
                "side": "backward",
                "r0": pair1_r,
                "c0": right_c0,
                "sub": backward_sub,
                "ids": backward_ids,
                "direction": backward_dir,
                "active": backward_active,
            },
        ]
    else:
        band_starts = _two_band_starts(forward_sub.shape[1], W)
        if band_starts is None:
            return None
        pair0_c, pair1_c = band_starts

        gap = random.randint(*_PAIR_GAP_RANGE)
        total_h = forward_sub.shape[0] + gap + backward_sub.shape[0]
        if total_h > H:
            return None
        top_r0 = random.randint(0, H - total_h)
        bottom_r0 = top_r0 + forward_sub.shape[0] + gap

        placements = [
            {
                "pair": 0,
                "side": "forward",
                "r0": top_r0,
                "c0": pair0_c,
                "sub": forward_sub,
                "ids": forward_ids,
                "direction": forward_dir,
                "active": forward_active,
            },
            {
                "pair": 0,
                "side": "backward",
                "r0": bottom_r0,
                "c0": pair0_c,
                "sub": backward_sub,
                "ids": backward_ids,
                "direction": backward_dir,
                "active": backward_active,
            },
            {
                "pair": 1,
                "side": "forward",
                "r0": top_r0,
                "c0": pair1_c,
                "sub": forward_sub,
                "ids": forward_ids,
                "direction": forward_dir,
                "active": forward_active,
            },
            {
                "pair": 1,
                "side": "backward",
                "r0": bottom_r0,
                "c0": pair1_c,
                "sub": backward_sub,
                "ids": backward_ids,
                "direction": backward_dir,
                "active": backward_active,
            },
        ]

    grid = np.full((H, W), _BG, dtype=int)
    for item in placements:
        sub = item["sub"]
        h, w = sub.shape
        r0 = int(item["r0"])
        c0 = int(item["c0"])
        if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
            return None
        if np.any(grid[r0:r0 + h, c0:c0 + w] != _BG):
            return None
        grid[r0:r0 + h, c0:c0 + w] = sub

    forward_by_pair = {item["pair"]: item for item in placements if item["side"] == "forward"}
    backward_by_pair = {item["pair"]: item for item in placements if item["side"] == "backward"}
    ref_forward = forward_by_pair[0]
    active_template_ids = sorted(int(ref_forward["ids"][lr, lc]) for lr, lc in ref_forward["active"])

    marker_lanes = _choose_marker_lanes(len(active_template_ids))
    if marker_lanes is None:
        return None
    lane_indices, pair_assignments = marker_lanes
    chosen_template_ids = [active_template_ids[idx] for idx in lane_indices]
    available = [c for c in _NON_BG_COLORS if c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return None
    marker_colors = random.sample(available, _NUM_MARKERS)

    for color, tid, pair_idx in zip(marker_colors, chosen_template_ids, pair_assignments):
        forward_sprite = forward_by_pair[pair_idx]
        backward_sprite = backward_by_pair[pair_idx]
        locs = np.argwhere(forward_sprite["ids"] == tid)
        if locs.shape[0] != 1:
            return None
        lr, lc = map(int, locs[0])
        if (lr, lc) not in forward_sprite["active"]:
            return None

        if axis == "horizontal":
            marker_r = int(forward_sprite["r0"]) + lr
            port_c = int(forward_sprite["c0"]) + lc
            marker_min = port_c + 2
            marker_max = int(backward_sprite["c0"]) - 2
            if marker_min > marker_max:
                return None
            marker_c = random.randint(marker_min, marker_max)
        else:
            marker_c = int(forward_sprite["c0"]) + lc
            port_r = int(forward_sprite["r0"]) + lr
            marker_min = port_r + 2
            marker_max = int(backward_sprite["r0"]) - 2
            if marker_min > marker_max:
                return None
            marker_r = random.randint(marker_min, marker_max)

        if grid[marker_r, marker_c] != _BG:
            return None
        for rr in range(marker_r - 1, marker_r + 2):
            for cc in range(marker_c - 1, marker_c + 2):
                if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] != _BG:
                    return None
        grid[marker_r, marker_c] = color

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
