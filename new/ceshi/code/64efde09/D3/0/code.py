# Variant of 64efde09
# Dimension: D3 — 轨迹样式：射线的涂色模式 (trajectory)
# Change: keep_source — 射线以标记像素的颜色连续涂色，从端口到停止点全部填充... → 间歇涂色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the D3 variant
# a) Concrete behavior and trace:
#    - Input objects stay the same: several dense 2xN / Nx2 rectangular sprites, plus three
#      boundary markers that label template ports on the active long edge.
#    - The changed behavior is only in the emitted output rays. Originally, once a template port
#      is labeled, every background cell from that port to the first blocker/boundary is filled
#      with the marker color. In this variant, only every odd step away from the port is colored:
#      step 1 colored, step 2 blank, step 3 colored, step 4 blank, ...
#    - Small trace for one left-facing port: if the sprite edge is at column 5 and the marker is
#      at column 0, the continuous rule colors columns 4,3,2,1. The intermittent rule colors
#      only 4 and 2, leaving 3 and 1 as background; the boundary marker at 0 stays colored from
#      the input, so the visible path becomes colored/blank/colored/blank/marker.
# b) Complexity calibration:
#    - Only changing the solver would be too weak if some rays had length 1, because a one-cell
#      ray looks identical under both rules. The generator therefore enforces a minimum ray gap of
#      3 cells and odd sprite-to-boundary distance so the dashed pattern is always visible and the
#      marker lands on the same phase as the painted cells.
#    - All original orientation/template logic is preserved; the puzzle still requires recovering
#      template correspondences, not merely reading absolute positions.
# c) Old hidden assumptions audited:
#    - _first_marker_on_ray only inspects the input grid, so it does not depend on continuous
#      output painting and remains valid.
#    - The original draw loop assumed every traversed background cell should be overwritten.
#      That assumption is false now; we must track distance from the port and only write on odd
#      steps. The conflict assertion must also move inside that odd-step branch.
#    - Blocking still depends on the input grid, not partially drawn output, otherwise one ray
#      could incorrectly stop another ray after coloring an odd-step cell.
# d) Placement legality:
#    - Sprites still cannot overlap and still need clear boundary-facing corridors so markers are
#      the first non-background cell on their labeling rays.
#    - The generator now restricts placements to boundary gaps with odd length >= 3. That keeps
#      the dashed phase consistent while preserving the original spatial semantics.
# e) Visual observability:
#    - The dashed rule is visible in every training sample because each emitted corridor contains
#      at least one colored gap and one preserved blank gap.
#    - No color-id ordering or hidden metadata is used; the only cue is the repeated spatial
#      pattern of colored and blank cells along otherwise identical rays.
# f) Visual cleanliness:
#    - We retain the original no-overlap/no-conflict ray layout. Skipped cells remain background,
#      so the output is actually cleaner than the solid-ray version rather than more cluttered.
# g) Color inference:
#    - Marker colors still map one-to-one to template ports and are reused across all sprite
#      transforms. The only changed inference is trajectory style, which is supported by repeated
#      same-color dashed corridors in multiple places per example.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate marker-labeled ports across all transformed sprites with dashed outward rays.

    Input structure:
    - The background is the most frequent color.
    - Several dense rectangular 2-by-N or N-by-2 sprites are dihedral transforms of one template.
    - Single-pixel boundary markers lie on straight outward sightlines from some cells of each sprite's uniquely dominant long edge.

    Rule:
    - For each sprite, identify the active long edge as the side whose cells contain the strongest same-color majority.
    - If the first non-background cell seen from an active-edge cell is a boundary marker, that marker color labels the corresponding template port.
    - For every sprite and every labeled template port, emit the marker color outward through background only on alternating cells: paint the first step, skip the next, and continue until blocked or out of bounds.

    Key insight:
    - The markers do not encode absolute coordinates. They tag shared template edge cells, and those tags must be transferred through the sprites' rotations/reflections before drawing the same dashed trajectory from each corresponding port.
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

            for r, c, step in _ray_bg_steps(g, bg, r0 + lr, c0 + lc, dr, dc):
                if step % 2 == 0:
                    continue
                assert int(out[r, c]) in (bg, color), "dashed control rays must not conflict"
                out[r, c] = color

    return out


# --------------- generator (att1, D3 intermittent variant) ---------------

# New generator assumptions introduced by the solver:
# - Every labeled corridor must keep the marker as the first non-bg cell.
# - Every sprite-to-boundary gap along the active side must be odd and at least 3, otherwise the
#   dashed pattern can collapse into a solid-looking one-cell ray or end out of phase with marker.
# - Potential ray crossings are still rejected by validating with main(); skipped cells are allowed
#   to remain shared background, but colored odd-step cells may not conflict.

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
_PLACEMENT_CLEARANCE = 3
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


def _axis_positions(low: int, high: int) -> list[int]:
    if low > high:
        return []
    return list(range(low, high + 1))


def _parity_gap_positions(limit_low: int, limit_high: int, gap_of) -> list[int]:
    return [pos for pos in range(limit_low, limit_high + 1) if gap_of(pos) >= _PLACEMENT_CLEARANCE and gap_of(pos) % 2 == 1]


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int] | None:
    if dr == 0 and dc == -1:
        row_choices = _axis_positions(0, H - h)
        col_choices = _parity_gap_positions(_PLACEMENT_CLEARANCE, W - w, lambda c0: c0)
    elif dr == 0 and dc == 1:
        row_choices = _axis_positions(0, H - h)
        col_choices = _parity_gap_positions(0, W - w - _PLACEMENT_CLEARANCE, lambda c0: W - w - c0)
    elif dr == -1 and dc == 0:
        row_choices = _parity_gap_positions(_PLACEMENT_CLEARANCE, H - h, lambda r0: r0)
        col_choices = _axis_positions(0, W - w)
    else:
        row_choices = _parity_gap_positions(0, H - h - _PLACEMENT_CLEARANCE, lambda r0: H - h - r0)
        col_choices = _axis_positions(0, W - w)

    if not row_choices or not col_choices:
        return None
    return random.choice(row_choices), random.choice(col_choices)


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
            pos = _random_sprite_position(H, W, h, w, direction[0], direction[1])
            if pos is None:
                return None
            r0, c0 = pos
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


def _gap_to_boundary(r0: int, c0: int, sub: np.ndarray, direction: tuple[int, int], H: int, W: int) -> int:
    h, w = sub.shape
    dr, dc = direction
    if dr == 0 and dc == -1:
        return c0
    if dr == 0 and dc == 1:
        return W - c0 - w
    if dr == -1 and dc == 0:
        return r0
    return H - r0 - h


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    N = ids_base.shape[0]
    num_sprites = len(sprite_placements)

    active_template_ids = [int(ids_base[r, 0]) for r in range(N)]
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    for color, tid in zip(marker_colors, labeled_ids):
        si = random.randint(0, num_sprites - 1)
        r0, c0, sub, ids_arr, direction, active_coords = sprite_placements[si]
        dr, dc = direction

        if _gap_to_boundary(r0, c0, sub, direction, H, W) < _PLACEMENT_CLEARANCE:
            return False
        if _gap_to_boundary(r0, c0, sub, direction, H, W) % 2 == 0:
            return False

        locs = np.argwhere(ids_arr == tid)
        if locs.shape[0] != 1:
            return False
        lr, lc = int(locs[0, 0]), int(locs[0, 1])
        if (lr, lc) not in {(r, c) for r, c in active_coords}:
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
        grid[r0:r0 + h, c0:c0 + w] = sub

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
