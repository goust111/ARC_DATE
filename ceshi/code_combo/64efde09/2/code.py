# Combo variant of 64efde09 (index 2)
# combo_spec: D2/0+D3/0
# Name: 对角虚线射线

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the combo variant (D2/0 + D3/0)
# a) Concrete behavior and trace:
#    - The original task labels active-edge ports of a shared thin rectangle template, then emits
#      solid axis-aligned rays from the corresponding ports on every transformed sprite.
#    - Here the labeling and emission geometry both become diagonal. Example: take a left-facing
#      vertical sprite whose active edge is its left column. A port in the upper half now looks
#      along (-1, -1); a port in the lower half looks along (+1, -1). If a marker is found at the
#      first non-bg cell on that diagonal, its color labels that template port.
#    - In the output, every matching port on every sprite emits along the same local diagonal, but
#      only odd steps are painted: step 1 colored, step 2 blank, step 3 colored, ...
#    - Compared with the original, both the path geometry and the paint cadence change: the ray no
#      longer follows the edge normal as a solid strip, but fans toward an outer corner as a dash.
# b) Complexity calibration:
#    - A single diagonal direction per whole sprite would be too brittle and too close to an
#      arbitrary orientation tag. Instead, the direction depends on which half of the active edge
#      contains the port. That creates repeated structure within one sample while staying purely
#      geometric and rotation/reflection-equivariant.
#    - The generator always labels ports from both halves of the canonical active edge, so at least
#      two diagonal branches are visible across the sample set.
#    - Every accepted sample enforces a minimum visible corridor length before the first blocker, so
#      the dashed behavior does not collapse into a one-pixel line that looks effectively solid.
# c) Hidden assumptions audited:
#    - The original solver assumed one outward direction per sprite. That is no longer valid: the
#      ray direction must now be recomputed for each active-edge port from its local coordinate.
#    - `_first_marker_on_ray` itself remains valid because it only needs a step vector; the changed
#      assumption is how that vector is chosen.
#    - The original draw loop overwrote every traversed bg cell. That assumption breaks under the
#      intermittent rule, so conflict checks must happen only on painted odd steps.
#    - The original placement code only guaranteed clearance toward one cardinal boundary. Diagonal
#      rays depend on two axes at once, so the generator must instead keep four-sided margins and
#      explicitly verify all labeled diagonal corridors.
# d) Placement legality:
#    - Sprites remain dense non-square rectangles and may not overlap, even with padding.
#    - Boundary markers must stay isolated as single-pixel components under 8-connectivity, so the
#      generator rejects marker endpoints that touch any other non-bg cell diagonally or orthogonally.
#    - For every labeled port on every sprite, the diagonal corridor must stay inside bg for a
#      minimum number of steps before hitting a marker, another sprite, or the boundary.
# e) Visual observability:
#    - A human can see three repeated cues: marker colors copy across transformed sprites, the
#      copied trajectories are diagonal rather than horizontal/vertical, and every diagonal trail
#      visibly alternates colored / blank / colored.
#    - The rule does not depend on numeric color ordering. Colors only serve as reusable labels for
#      shared template ports, exactly as in the original task.
# f) Visual cleanliness:
#    - The generator validates with the solver, so odd-step painted cells from different colors do
#      not conflict.
#    - Requiring marker isolation and sprite padding prevents the input from turning into one large
#      8-connected non-bg blob, which would destroy object parsing.
# g) Change inferability:
#    - Marker colors still map one-to-one to template ids, so the only new inference burden is the
#      local geometry of the ray. Because upper/lower or left/right half ports repeat across several
#      transformed sprites in the same grid, that geometric rule is recoverable from 2-3 examples.


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


def _port_diag_direction(
    active_direction: tuple[int, int],
    lr: int,
    lc: int,
    shape: tuple[int, int],
) -> tuple[int, int]:
    h, w = shape
    if active_direction == (0, -1):
        return (-1, -1) if 2 * lr < h else (1, -1)
    if active_direction == (0, 1):
        return (-1, 1) if 2 * lr < h else (1, 1)
    if active_direction == (-1, 0):
        return (-1, -1) if 2 * lc < w else (-1, 1)
    assert active_direction == (1, 0)
    return (1, -1) if 2 * lc < w else (1, 1)


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


def _ray_bg_steps(
    g: np.ndarray,
    bg: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
):
    H, W = g.shape
    step = 1
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if int(g[r, c]) != bg:
            break
        yield r, c, step
        r += dr
        c += dc
        step += 1


def _ray_length_before_blocker(
    g: np.ndarray,
    bg: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> int:
    length = 0
    for _r, _c, _step in _ray_bg_steps(g, bg, r, c, dr, dc):
        length += 1
    return length


def _boundary_endpoint(r: int, c: int, dr: int, dc: int, H: int, W: int) -> tuple[int, int, int]:
    steps = 0
    r += dr
    c += dc
    last: tuple[int, int] | None = None
    while 0 <= r < H and 0 <= c < W:
        last = (int(r), int(c))
        steps += 1
        r += dr
        c += dc
    assert last is not None, "ray must leave the sprite before exiting the grid"
    return last[0], last[1], steps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer marker-labeled template ports across all sprite transforms, then emit dashed diagonal rays.

    Input structure:
    - The background is the most common color.
    - Several dense 2-by-N or N-by-2 rectangles are dihedral transforms of one thin template.
    - Single-pixel markers sit on boundary endpoints reached by diagonal sightlines from some cells of each sprite's uniquely dominant long edge.

    Rule:
    - For each sprite, find the active long edge as the side with the strongest same-color majority.
    - Each active-edge cell chooses one outward 45-degree direction from its position on that edge: upper/left half ports aim toward one outer corner, lower/right half ports toward the other.
    - If the first non-background cell on that diagonal is a marker, that marker color labels the corresponding template port.
    - For every sprite and every labeled template port, emit that color along the matching diagonal, painting only steps 1, 3, 5, ... until blocked or out of bounds.

    Key insight:
    - The markers are not instructions for absolute positions. They identify shared template edge cells, and both the chosen diagonal branch and the dashed paint rhythm must be reproduced after rotating or reflecting the template onto every sprite.
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
        shape = tuple(sprite["sub"].shape)
        for lr, lc in sprite["active"]:
            ray_dr, ray_dc = _port_diag_direction(sprite["direction"], lr, lc, shape)
            marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, ray_dr, ray_dc)
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
        shape = tuple(sprite["sub"].shape)
        for color, template_id in color_to_template_id.items():
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            assert (lr, lc) in sprite["active"], "mapped port must land on the sprite active edge"

            ray_dr, ray_dc = _port_diag_direction(sprite["direction"], lr, lc, shape)
            for r, c, step in _ray_bg_steps(g, bg, r0 + lr, c0 + lc, ray_dr, ray_dc):
                if step % 2 == 0:
                    continue
                assert int(out[r, c]) in (bg, color), "dashed diagonal rays must not conflict"
                out[r, c] = color

    return out


# --------------- generator (combo variant) ---------------

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
_BOUNDARY_MARGIN = 3
_MIN_RAY_BG_STEPS = 3
_PLACEMENT_RETRIES = 150
_GEN_RETRIES = 500


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
    if len(template_colors) == 1:
        return None
    right_col = _build_column(N, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _deduplicated_variants(
    template: np.ndarray,
    ids_base: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], set[tuple[int, int]]]]:
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
        result.append((sub, ids_arr, direction, active_set))
    return result


def _sample_variants(unique_variants, num_sprites: int):
    tall = [idx for idx, variant in enumerate(unique_variants) if variant[0].shape[0] > variant[0].shape[1]]
    wide = [idx for idx, variant in enumerate(unique_variants) if variant[0].shape[1] > variant[0].shape[0]]
    if not tall or not wide:
        return None

    chosen_ids = {random.choice(tall), random.choice(wide)}
    remaining_ids = [idx for idx in range(len(unique_variants)) if idx not in chosen_ids]
    if len(remaining_ids) < num_sprites - len(chosen_ids):
        return None
    chosen_ids.update(random.sample(remaining_ids, num_sprites - len(chosen_ids)))
    chosen = [unique_variants[idx] for idx in chosen_ids]
    random.shuffle(chosen)
    return chosen


def _occupied_set(placements: list[tuple]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _random_sprite_position(H: int, W: int, h: int, w: int) -> tuple[int, int] | None:
    row_low = _BOUNDARY_MARGIN
    row_high = H - h - _BOUNDARY_MARGIN
    col_low = _BOUNDARY_MARGIN
    col_high = W - w - _BOUNDARY_MARGIN
    if row_low > row_high or col_low > col_high:
        return None
    return random.randint(row_low, row_high), random.randint(col_low, col_high)


def _place_sprites(H: int, W: int, variants) -> list[tuple] | None:
    placements: list[tuple] = []
    pad = _PLACEMENT_PAD
    for sub, ids_arr, direction, active_coords in variants:
        h, w = sub.shape
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            pos = _random_sprite_position(H, W, h, w)
            if pos is None:
                return None
            r0, c0 = pos
            occ = _occupied_set(placements)
            conflict = False
            for rr in range(r0 - pad, r0 + h + pad):
                for cc in range(c0 - pad, c0 + w + pad):
                    if (rr, cc) in occ:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                continue
            placements.append((r0, c0, sub, ids_arr, direction, active_coords))
            placed = True
            break
        if not placed:
            return None
    return placements


def _choose_labeled_ids(ids_base: np.ndarray) -> list[int]:
    N = ids_base.shape[0]
    upper = [int(ids_base[r, 0]) for r in range(N) if 2 * r < N]
    lower = [int(ids_base[r, 0]) for r in range(N) if 2 * r >= N]
    assert upper and lower

    chosen = [random.choice(upper), random.choice(lower)]
    remaining = [int(ids_base[r, 0]) for r in range(N) if int(ids_base[r, 0]) not in chosen]
    chosen.extend(random.sample(remaining, _NUM_MARKERS - len(chosen)))
    random.shuffle(chosen)
    return chosen


def _all_sprite_cells(placements: list[tuple]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _is_isolated_marker(cell: tuple[int, int], occupied_nonbg: set[tuple[int, int]]) -> bool:
    r0, c0 = cell
    for r, c in occupied_nonbg:
        if max(abs(r - r0), abs(c - c0)) <= 1:
            return False
    return True


def _candidate_marker_hosts(
    grid: np.ndarray,
    placements: list[tuple],
    tid: int,
    occupied_nonbg: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    H, W = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for idx, (r0, c0, sub, ids_arr, direction, active_coords) in enumerate(placements):
        locs = np.argwhere(ids_arr == tid)
        if locs.shape[0] != 1:
            continue
        lr, lc = map(int, locs[0])
        if (lr, lc) not in active_coords:
            continue
        ray_dr, ray_dc = _port_diag_direction(direction, lr, lc, sub.shape)
        end_r, end_c, total_steps = _boundary_endpoint(r0 + lr, c0 + lc, ray_dr, ray_dc, H, W)
        bg_steps = total_steps - 1
        if bg_steps < _MIN_RAY_BG_STEPS:
            continue

        clear = True
        rr = r0 + lr + ray_dr
        cc = c0 + lc + ray_dc
        while (rr, cc) != (end_r, end_c):
            if int(grid[rr, cc]) != _BG:
                clear = False
                break
            rr += ray_dr
            cc += ray_dc
        if not clear:
            continue
        if int(grid[end_r, end_c]) != _BG:
            continue
        if not _is_isolated_marker((end_r, end_c), occupied_nonbg):
            continue
        candidates.append((idx, end_r, end_c))
    random.shuffle(candidates)
    return candidates


def _place_markers(
    grid: np.ndarray,
    placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> tuple[list[int], dict[int, int]] | None:
    labeled_ids = _choose_labeled_ids(ids_base)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return None
    marker_colors = random.sample(available, _NUM_MARKERS)

    occupied_nonbg = _all_sprite_cells(placements)
    color_to_tid: dict[int, int] = {}
    for color, tid in zip(marker_colors, labeled_ids):
        candidates = _candidate_marker_hosts(grid, placements, tid, occupied_nonbg)
        if not candidates:
            return None
        _host_idx, mr, mc = random.choice(candidates)
        grid[mr, mc] = color
        occupied_nonbg.add((mr, mc))
        color_to_tid[int(color)] = int(tid)

    return labeled_ids, color_to_tid


def _all_labeled_rays_visible(
    grid: np.ndarray,
    placements: list[tuple],
    labeled_ids: list[int],
) -> bool:
    for tid in labeled_ids:
        for r0, c0, sub, ids_arr, direction, active_coords in placements:
            locs = np.argwhere(ids_arr == tid)
            if locs.shape[0] != 1:
                return False
            lr, lc = map(int, locs[0])
            if (lr, lc) not in active_coords:
                return False
            ray_dr, ray_dc = _port_diag_direction(direction, lr, lc, sub.shape)
            length = _ray_length_before_blocker(grid, _BG, r0 + lr, c0 + lc, ray_dr, ray_dc)
            if length < _MIN_RAY_BG_STEPS:
                return False
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

    chosen = _sample_variants(unique_variants, num_sprites)
    if chosen is None:
        return None
    placements = _place_sprites(H, W, chosen)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        grid[r0:r0 + h, c0:c0 + w] = sub

    marker_info = _place_markers(grid, placements, ids_base, template_colors)
    if marker_info is None:
        return None
    labeled_ids, _color_to_tid = marker_info

    if not _all_labeled_rays_visible(grid, placements, labeled_ids):
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
