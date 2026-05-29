# Combo variant of 64efde09 (index 18)
# combo_spec: D2/2+D19/2
# Name: 双向四射辐射

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the fused rule:
# a) Original behavior: a marker labels one active-edge port by sitting on the outward
#    normal ray, then each matching sprite emits a single outward ray from that port.
#    Fused behavior: the label can be read from either direction of the active normal
#    axis, including through the sprite interior to the opposite side, and each copied
#    port radiates in all four cardinal directions. Every direction skips over the
#    sprite body first and only paints outside it, so the output is a clear starburst
#    around the rectangle instead of a single half-line.
# b) Complexity calibration: if we kept three labeled ports from the base task, the
#    new tangential rays would force unavoidable same-cell conflicts inside a sprite
#    (e.g. several active-edge ports on one column would all emit up/down on that same
#    column). To keep the fusion visible and legal, the generator uses exactly one
#    labeled template port and shows the rule through repetition across 3-4 sprites.
# c) Hidden assumptions that changed:
#    - Marker lookup can no longer assume "outward only"; it must inspect both normal
#      directions while ignoring the current sprite's own pixels.
#    - Drawing can no longer start from the adjacent cell, because some directions pass
#      through the sprite interior before they can emerge into background.
#    - Marker placement must avoid accidental visibility from a different template id;
#      the generator validates this by solving the candidate grid.
# d) Placement legality: sprites stay dense non-square rectangles, do not overlap, keep
#    boundary clearance on all sides, and markers remain single boundary pixels with a
#    clear line to the chosen port along the active-axis direction.
# e) Visual observability: a single marker color is enough because the same colored
#    four-way star repeats on every transformed sprite. The reverse-side marker option
#    makes the bidirectional readout visible in the input distribution.
# f) Cleanliness: only one port color is used, so star rays may overlap only with the
#    same color; different-color collisions are designed out rather than hidden.
# g) Inferability: no color ordering or numeric property is used. The only signal is
#    geometric alignment between one boundary marker and a shared template port.


CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MARKER_SIDE_CYCLE = ("forward", "reverse")
_marker_side_index = 0


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


def _inside_bbox(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _first_external_cell(
    bbox: tuple[int, int, int, int],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> tuple[int, int]:
    r += dr
    c += dc
    while _inside_bbox(bbox, r, c):
        r += dr
        c += dc
    return r, c


def _first_marker_on_axis(
    g: np.ndarray,
    bg: int,
    markers: dict[tuple[int, int], int],
    bbox: tuple[int, int, int, int],
    r: int,
    c: int,
    direction: tuple[int, int],
) -> int | None:
    H, W = g.shape
    found: list[int] = []
    for dr, dc in (direction, (-direction[0], -direction[1])):
        rr, cc = _first_external_cell(bbox, r, c, dr, dc)
        while 0 <= rr < H and 0 <= cc < W:
            if int(g[rr, cc]) != bg:
                marker = markers.get((rr, cc))
                if marker is not None:
                    found.append(marker)
                break
            rr += dr
            cc += dc

    if not found:
        return None
    assert len(set(found)) == 1, "a port cannot see multiple marker colors"
    return found[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer one boundary-labeled active-edge port across all matching thin sprites, then radiate that port as a four-way star.

    Input structure:
    - The background is the most frequent color.
    - Several dense 2-by-N or N-by-2 rectangles are dihedral transforms of one shared template.
    - A single isolated marker pixel lies on a boundary cell aligned with one active-edge template port, either on the active-facing side or on the opposite side of that same axis.

    Rule:
    - For each sprite, detect its active long edge as the side with the stronger same-color majority.
    - Read the marker by scanning from each active-edge cell in both directions along that active normal axis, skipping through the sprite body itself.
    - Copy the labeled template port to every transformed sprite and emit rays from that corresponding cell in all four cardinal directions; each ray first exits the current sprite and then paints background cells until another non-background cell or the grid boundary stops it.

    Key insight:
    - The marker names a template port, not a location in the grid. Once that port is identified, every transformed instance of the template produces the same four-direction star from its own corresponding port.
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
    assert len(markers) == 1, "this fused variant expects exactly one control marker"

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
        sprites.append({**sprite, "ids": ids, "direction": direction, "active": active})

    color_to_template_id: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        for lr, lc in sprite["active"]:
            marker_color = _first_marker_on_axis(
                g,
                bg,
                markers,
                sprite["bbox"],
                r0 + lr,
                c0 + lc,
                sprite["direction"],
            )
            if marker_color is None:
                continue
            template_id = int(sprite["ids"][lr, lc])
            if marker_color in color_to_template_id:
                assert color_to_template_id[marker_color] == template_id
            color_to_template_id[marker_color] = template_id

    assert set(color_to_template_id) == set(markers.values()), "the boundary marker must identify one active-edge cell"

    out = g.copy()
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        for color, template_id in color_to_template_id.items():
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            assert (lr, lc) in sprite["active"], "the copied port must stay on the active edge"
            r = r0 + lr
            c = c0 + lc
            for dr, dc in CARDINAL_DIRS:
                rr, cc = _first_external_cell(sprite["bbox"], r, c, dr, dc)
                H, W = g.shape
                while 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) != bg:
                        break
                    assert int(out[rr, cc]) in (bg, color), "star rays must not conflict"
                    out[rr, cc] = color
                    rr += dr
                    cc += dc

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
_NUM_MARKERS = 1
_PLACEMENT_PAD = 3
_BOUNDARY_CLEARANCE = 2
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
        result.append((sub, ids_arr, direction, sorted(active_set)))
    return result


def _random_sprite_position(H: int, W: int, h: int, w: int) -> tuple[int, int]:
    return (
        random.randint(_BOUNDARY_CLEARANCE, H - h - _BOUNDARY_CLEARANCE),
        random.randint(_BOUNDARY_CLEARANCE, W - w - _BOUNDARY_CLEARANCE),
    )


def _expanded_overlap(
    placements: list[tuple[int, int, np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]],
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> bool:
    for pr0, pc0, psub, *_ in placements:
        ph, pw = psub.shape
        if not (
            r0 + h + _PLACEMENT_PAD - 1 < pr0
            or pr0 + ph + _PLACEMENT_PAD - 1 < r0
            or c0 + w + _PLACEMENT_PAD - 1 < pc0
            or pc0 + pw + _PLACEMENT_PAD - 1 < c0
        ):
            return True
    return False


def _place_sprites(H: int, W: int, variants) -> list[tuple] | None:
    placements: list[tuple] = []
    for sub, ids_arr, direction, active_coords in variants:
        h, w = sub.shape
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w)
            if _expanded_overlap(placements, r0, c0, h, w):
                continue
            placements.append((r0, c0, sub, ids_arr, direction, active_coords))
            placed = True
            break
        if not placed:
            return None
    return placements


def _boundary_position(r: int, c: int, dr: int, dc: int, H: int, W: int) -> tuple[int, int]:
    if dr == -1:
        return 0, c
    if dr == 1:
        return H - 1, c
    if dc == -1:
        return r, 0
    return r, W - 1


def _path_clear_to_boundary(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> bool:
    H, W = grid.shape
    rr, cc = _first_external_cell(bbox, r, c, dr, dc)
    while 0 <= rr < H and 0 <= cc < W:
        if int(grid[rr, cc]) != _BG:
            return False
        rr += dr
        cc += dc
    return True


def _place_marker(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
    preferred_side: str | None = None,
) -> bool:
    H, W = grid.shape
    active_template_ids = [int(ids_base[r, 0]) for r in range(ids_base.shape[0])]
    labeled_id = random.choice(active_template_ids)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_color = random.choice(available)

    candidates: list[tuple[int, int, int, str]] = []
    for r0, c0, sub, ids_arr, direction, active_coords in sprite_placements:
        locs = np.argwhere(ids_arr == labeled_id)
        if locs.shape[0] != 1:
            continue
        lr, lc = map(int, locs[0])
        if (lr, lc) not in active_coords:
            continue
        port_r = r0 + lr
        port_c = c0 + lc
        for side_name, (dr, dc) in zip(MARKER_SIDE_CYCLE, (direction, (-direction[0], -direction[1]))):
            if _path_clear_to_boundary(grid, (r0, c0, r0 + sub.shape[0] - 1, c0 + sub.shape[1] - 1), port_r, port_c, dr, dc):
                marker_r, marker_c = _boundary_position(port_r, port_c, dr, dc, H, W)
                if int(grid[marker_r, marker_c]) == _BG:
                    candidates.append((marker_r, marker_c, marker_color, side_name))

    if not candidates:
        return False

    if preferred_side is not None:
        preferred_candidates = [item for item in candidates if item[3] == preferred_side]
        if not preferred_candidates:
            return False
        candidates = preferred_candidates

    marker_r, marker_c, marker_color, _side_name = random.choice(candidates)
    grid[marker_r, marker_c] = marker_color
    return True


def _gen_attempt(preferred_side: str | None = None) -> np.ndarray | None:
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

    if not _place_marker(grid, placements, ids_base, template_colors, preferred_side=preferred_side):
        return None

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _marker_side_index
    for _ in range(_GEN_RETRIES):
        preferred_side = MARKER_SIDE_CYCLE[_marker_side_index % len(MARKER_SIDE_CYCLE)]
        result = _gen_attempt(preferred_side=preferred_side)
        if result is not None:
            _marker_side_index += 1
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
