# Combo variant of 64efde09 (index 14)
# combo_spec: D2/0+D3/1+D5/0
# Name: 对角循环穿透射线

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
# - The original task fires cardinal rays straight out from labeled active-edge ports and stops at
#   the first non-background pixel. Here the same template-port labeling mechanism is retained, but
#   every outward ray is rotated onto a 45-degree diagonal, its colors advance through the 3 marker
#   colors cyclically, and sprite pixels become transparent to the propagation.
# - Small trace: suppose marker colors A/B/C label three top-to-bottom ports on a left-facing sprite.
#   The top port now steps SW instead of W. Its visited cells follow A,B,C,A,... even when one of the
#   steps lands on another sprite: that occupied cell stays unchanged, but the phase still advances and
#   the next background cell keeps the sequence. A straight non-penetrating solver would both use the
#   wrong geometry and stop at the blocker.
#
# b) Complexity calibration:
# - The generator keeps the original repeated-template structure, exactly 3 labeled ports, and 3-4
#   sprite copies. All examples reuse the same 3-color cycle on every copy, so the color rule is
#   repeated enough to infer, but the rays still span long diagonals instead of becoming trivial.
# - To keep the output readable, generated sprites are vertical only. That means every emitted ray
#   lives on the same anti-diagonal family (r+c = const), so visual clutter from perpendicular
#   crossings is removed without changing the solver rule itself.
#
# c) Broken old assumptions:
# - Marker search can no longer stop at the first non-background cell; it must ignore intervening
#   sprite pixels and continue along the diagonal until it either reaches a marker or exits.
# - Output drawing can no longer break on object contact; it must preserve non-background pixels and
#   keep stepping to the boundary.
# - A single marker color is no longer enough to paint the whole ray. The solver now has to recover
#   an ordered 3-color cycle from the labeled template ports and advance phase along each diagonal.
#
# d) Placement legality:
# - Sprites remain dense non-square rectangles, are separated by at least one cell under 8-connectivity,
#   and stay fully in bounds.
# - Marker cells are boundary singletons with no 8-neighbor contact to sprites or to other markers.
# - The structured placement guarantees that at least one labeled diagonal from the marker-bearing
#   sprite passes through another sprite before it reaches the boundary.
#
# e) Visual observability:
# - The change is visible in two independent ways: outputs form diagonal rather than cardinal bands,
#   and those bands show an A-B-C repeating stripe rather than a single solid color.
# - Because the copied rays reappear on mirrored sprite copies, the user can still infer that markers
#   label template ports rather than absolute positions.
#
# f) Visual cleanliness:
# - All generated rays are parallel anti-diagonals, so different colors do not form dense X-shaped
#   clutter. Distinct labeled ports are chosen with spacing, which keeps boundary markers separated.
#
# g) Color inference:
# - Only 3 marker colors participate in the cycle, and their cycle order is tied to the top-to-bottom
#   order of the 3 labeled active-edge ports on the shared template. This gives a visible structural
#   cue instead of relying on numeric color ids.
#
# Post-change solver assumptions enforced by the generator:
# - Exactly 3 isolated marker colors exist.
# - Every labeled template id lies on the active edge of every generated sprite copy.
# - The 3 emitted anti-diagonal lines for different sprite copies use distinct r+c constants, so any
#   overlaps with non-background cells come from tunneling through sprites, not ray-ray conflicts.


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


def _sorted_coords(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda rc: (int(rc[0]), int(rc[1])))


def _diagonal_direction(direction: tuple[int, int]) -> tuple[int, int]:
    mapping = {
        (0, -1): (1, -1),
        (0, 1): (-1, 1),
        (-1, 0): (-1, -1),
        (1, 0): (1, 1),
    }
    assert direction in mapping, f"unsupported active-side direction {direction}"
    return mapping[direction]


def _first_marker_on_diagonal(
    markers: dict[tuple[int, int], int],
    r: int,
    c: int,
    dr: int,
    dc: int,
    H: int,
    W: int,
) -> int | None:
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        marker_color = markers.get((r, c))
        if marker_color is not None:
            return marker_color
        r += dr
        c += dc
    return None


def _boundary_endpoint(r: int, c: int, dr: int, dc: int, H: int, W: int) -> tuple[int, int]:
    while 0 <= r + dr < H and 0 <= c + dc < W:
        r += dr
        c += dc
    return int(r), int(c)


def _template_cycle_order(sprites: list[dict], template_id_to_color: dict[int, int]) -> list[tuple[int, int]]:
    ref = sprites[0]
    active_order: list[int] = [int(ref["ids"][r, c]) for r, c in _sorted_coords(ref["active"])]
    rank = {tid: idx for idx, tid in enumerate(active_order)}
    assert all(tid in rank for tid in template_id_to_color), "labeled ids must lie on the reference active edge"
    return sorted(template_id_to_color.items(), key=lambda item: rank[item[0]])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer boundary-labeled template ports across matching sprites, then emit diagonal cyclic rays that tunnel through sprites.

    Input structure:
    - The background is the majority color.
    - Several dense 2-by-N or N-by-2 rectangular sprites are dihedral copies of one shared template.
    - Exactly three isolated boundary markers with unique colors lie on diagonal outward lines from active-edge ports, and those sight-lines may pass through other sprites.

    Rule:
    - For each sprite, detect the active long edge as the side with the stronger repeated-color majority, then map its outward normal to a fixed 45-degree diagonal direction.
    - Scanning along that diagonal, ignore intervening non-background pixels until a boundary marker is found; that marker color labels the corresponding shared template port.
    - Order the three labeled template ports along the active edge to recover a 3-color cycle.
    - From every matching port on every sprite, propagate to the boundary along the same diagonal, preserving existing non-background pixels but coloring each background step with the cyclic sequence starting from that port's own marker color.

    Key insight:
    - The markers do not encode absolute destinations. They identify three template ports whose colors define a repeating cycle, and that cycle is replayed from the corresponding diagonal ports on every transformed sprite even across intervening objects.
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
    assert len(markers) == 3, "expected exactly three isolated control markers"
    assert len(set(markers.values())) == 3, "control marker colors must be unique"

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
        sprites.append(
            {
                **sprite,
                "ids": ids,
                "direction": direction,
                "active": active,
                "diag": _diagonal_direction(direction),
            }
        )

    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    H, W = g.shape
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["diag"]
        for lr, lc in sprite["active"]:
            marker_color = _first_marker_on_diagonal(markers, r0 + lr, c0 + lc, dr, dc, H, W)
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

    cycle_pairs = _template_cycle_order(sprites, template_id_to_color)
    cycle_template_ids = [tid for tid, _color in cycle_pairs]
    cycle_colors = [color for _tid, color in cycle_pairs]
    cycle_index = {tid: idx for idx, tid in enumerate(cycle_template_ids)}
    assert len(cycle_colors) == 3, "this combo uses a 3-color cycle"

    out = g.copy()
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["diag"]
        for template_id in cycle_template_ids:
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            assert (lr, lc) in sprite["active"], "mapped port must land on the sprite active edge"

            phase0 = cycle_index[template_id]
            r = r0 + lr + dr
            c = c0 + lc + dc
            step = 0
            while 0 <= r < H and 0 <= c < W:
                expected = int(cycle_colors[(phase0 + step) % len(cycle_colors)])
                if int(g[r, c]) == bg:
                    assert int(out[r, c]) in (bg, expected), "diagonal cycle rays must not conflict on background cells"
                    out[r, c] = expected
                else:
                    assert int(out[r, c]) == int(g[r, c]), "rays may pass through objects but may not rewrite them"
                r += dr
                c += dc
                step += 1

    return out


# --------------- generator ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (24, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 1
_PLACEMENT_RETRIES = 80
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
    if len(set(template_colors)) < 2:
        return None
    right_col = _build_column(N, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None
    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _vertical_variants(template: np.ndarray, ids_base: np.ndarray) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]]:
    result: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]] = {}
    seen: list[np.ndarray] = []
    for (_, sub, _), (_, _, ids_arr) in zip(_dihedral_variants(template), _dihedral_variants(ids_base)):
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            direction, active = _active_side(sub)
        except AssertionError:
            continue
        if direction not in {(0, -1), (0, 1)}:
            continue
        seen.append(sub.copy())
        result[direction] = (sub, ids_arr, _sorted_coords(active))
    assert (0, -1) in result and (0, 1) in result
    return result


def _make_sprite(r0: int, c0: int, sub: np.ndarray, ids_arr: np.ndarray, active_coords: list[tuple[int, int]]) -> dict:
    h, w = sub.shape
    direction, active = _active_side(sub)
    return {
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        "sub": sub,
        "ids": ids_arr,
        "direction": direction,
        "diag": _diagonal_direction(direction),
        "active": set(active_coords),
    }


def _bbox_clear(candidate: tuple[int, int, int, int], others: list[dict], pad: int = _PLACEMENT_PAD) -> bool:
    r0, c0, r1, c1 = candidate
    for sprite in others:
        sr0, sc0, sr1, sc1 = sprite["bbox"]
        if not (r1 + pad < sr0 or sr1 + pad < r0 or c1 + pad < sc0 or sc1 + pad < c0):
            return False
    return True


def _choose_label_rows(N: int) -> list[int]:
    for _ in range(100):
        rows = sorted(random.sample(range(N), _NUM_MARKERS))
        if min(rows[i + 1] - rows[i] for i in range(len(rows) - 1)) >= 2:
            return rows
    return [0, N // 2, N - 1]


def _find_port(sprite: dict, template_id: int) -> tuple[int, int]:
    locs = np.argwhere(sprite["ids"] == int(template_id))
    assert locs.shape[0] == 1
    lr, lc = map(int, locs[0])
    assert (lr, lc) in sprite["active"]
    return int(lr), int(lc)


def _selected_constants(sprite: dict, labeled_ids: list[int]) -> set[int]:
    r0, c0, _r1, _c1 = sprite["bbox"]
    values: set[int] = set()
    for tid in labeled_ids:
        lr, lc = _find_port(sprite, tid)
        values.add(int(r0 + lr + c0 + lc))
    return values


def _marker_endpoint(sprite: dict, template_id: int, H: int, W: int) -> tuple[int, int]:
    r0, c0, _r1, _c1 = sprite["bbox"]
    lr, lc = _find_port(sprite, template_id)
    dr, dc = sprite["diag"]
    return _boundary_endpoint(r0 + lr, c0 + lc, dr, dc, H, W)


def _marker_is_isolated(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != _BG:
        return False
    H, W = grid.shape
    for rr in range(max(0, r - 1), min(H, r + 2)):
        for cc in range(max(0, c - 1), min(W, c + 2)):
            if (rr, cc) != (r, c) and int(grid[rr, cc]) != _BG:
                return False
    return True


def _ray_intersects_sprite_body(sprite: dict, template_id: int, blocker: dict, H: int, W: int) -> bool:
    r0, c0, _r1, _c1 = sprite["bbox"]
    lr, lc = _find_port(sprite, template_id)
    dr, dc = sprite["diag"]
    br0, bc0, br1, bc1 = blocker["bbox"]
    r = r0 + lr + dr
    c = c0 + lc + dc
    while 0 <= r < H and 0 <= c < W:
        if br0 <= r <= br1 and bc0 <= c <= bc1:
            return True
        r += dr
        c += dc
    return False


def _place_extra_sprite(
    H: int,
    W: int,
    variant: tuple[np.ndarray, np.ndarray, list[tuple[int, int]]],
    placements: list[dict],
    labeled_ids: list[int],
    used_constants: set[int],
    row_range: tuple[int, int],
    col_range: tuple[int, int],
) -> dict | None:
    sub, ids_arr, active_coords = variant
    h, w = sub.shape
    r_lo, r_hi = row_range
    c_lo, c_hi = col_range
    r_hi = min(r_hi, H - h)
    c_hi = min(c_hi, W - w)
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(_PLACEMENT_RETRIES):
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        candidate_bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if not _bbox_clear(candidate_bbox, placements):
            continue
        sprite = _make_sprite(r0, c0, sub, ids_arr, active_coords)
        constants = _selected_constants(sprite, labeled_ids)
        if constants & used_constants:
            continue
        return sprite
    return None


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)
    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors = design
    variants = _vertical_variants(template, ids_base)
    left_variant = variants[(0, -1)]
    right_variant = variants[(0, 1)]

    label_rows = _choose_label_rows(N)
    labeled_ids = [int(ids_base[r, 0]) for r in label_rows]

    right_r0 = random.randint(max(N + 2, H - N - 6), H - N - 2)
    right_c0 = random.randint(1, 3)
    right_sprite = _make_sprite(right_r0, right_c0, *right_variant)

    low_r0 = random.randint(1, 3)
    blocker_min = right_r0 + right_c0
    blocker_max = right_r0 + right_c0 + N + 1
    left_c_lo = max(3, blocker_min - low_r0 - max(label_rows))
    left_c_hi = min(W - 4, blocker_max - low_r0 - min(label_rows))
    if left_c_lo > left_c_hi:
        return None
    preferred_lo = max(left_c_lo, W // 2)
    if preferred_lo <= left_c_hi:
        left_c0 = random.randint(preferred_lo, left_c_hi)
    else:
        left_c0 = random.randint(left_c_lo, left_c_hi)
    left_sprite = _make_sprite(low_r0, left_c0, *left_variant)
    if not _bbox_clear(left_sprite["bbox"], [right_sprite]):
        return None

    if not all(_ray_intersects_sprite_body(left_sprite, tid, right_sprite, H, W) for tid in labeled_ids):
        return None

    placements: list[dict] = [left_sprite, right_sprite]
    used_constants = _selected_constants(left_sprite, labeled_ids) | _selected_constants(right_sprite, labeled_ids)

    extra_left = _place_extra_sprite(
        H,
        W,
        left_variant,
        placements,
        labeled_ids,
        used_constants,
        row_range=(4, max(4, H // 2 - 2)),
        col_range=(max(5, left_c0 - 4), W - 4),
    )
    if extra_left is None:
        return None
    placements.append(extra_left)
    used_constants |= _selected_constants(extra_left, labeled_ids)

    if num_sprites == 4:
        extra_right = _place_extra_sprite(
            H,
            W,
            right_variant,
            placements,
            labeled_ids,
            used_constants,
            row_range=(max(H // 2, right_r0 - 5), H - N - 1),
            col_range=(1, min(max(1, left_c0 - 5), W // 2)),
        )
        if extra_right is None:
            return None
        placements.append(extra_right)
        used_constants |= _selected_constants(extra_right, labeled_ids)

    grid = np.full((H, W), _BG, dtype=int)
    for sprite in placements:
        r0, c0, r1, c1 = sprite["bbox"]
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != _BG):
            return None
        grid[r0 : r1 + 1, c0 : c1 + 1] = sprite["sub"]

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return None
    marker_colors = random.sample(available, _NUM_MARKERS)

    marker_cells: list[tuple[int, int]] = []
    for tid in labeled_ids:
        mr, mc = _marker_endpoint(left_sprite, tid, H, W)
        marker_cells.append((mr, mc))

    if len(set(marker_cells)) != _NUM_MARKERS:
        return None
    if any(not _marker_is_isolated(grid, r, c) for r, c in marker_cells):
        return None
    if any(max(abs(r1 - r2), abs(c1 - c2)) <= 1 for i, (r1, c1) in enumerate(marker_cells) for r2, c2 in marker_cells[i + 1 :]):
        return None

    for color, (r, c) in zip(marker_colors, marker_cells):
        grid[r, c] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None

    # Enforce visible combo behavior: at least one output ray must resume after crossing a sprite.
    resumed_segments = 0
    for tid in labeled_ids:
        lr, lc = _find_port(left_sprite, tid)
        r0, c0, _r1, _c1 = left_sprite["bbox"]
        dr, dc = left_sprite["diag"]
        hit_blocker = False
        r = r0 + lr + dr
        c = c0 + lc + dc
        while 0 <= r < H and 0 <= c < W:
            on_blocker = right_sprite["bbox"][0] <= r <= right_sprite["bbox"][2] and right_sprite["bbox"][1] <= c <= right_sprite["bbox"][3]
            if on_blocker:
                hit_blocker = True
            elif hit_blocker and int(grid[r, c]) == _BG and int(out[r, c]) != _BG:
                resumed_segments += 1
                break
            r += dr
            c += dc
    if resumed_segments == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
