# Combo variant of 64efde09 (index 1)
# combo_spec: D2/0+D5/0
# Name: 对角穿透射线

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the D2/0 + D5/0 combo
# ------------------------------------------
# a) Concrete behavior:
# - The original task labels active-edge template ports by shooting straight cardinal sight-lines
#   from long-edge cells to boundary markers, then copies those labels as straight rays that stop
#   at the first non-background obstruction.
# - This combo changes both parts of the path model. Each active-edge port now uses a fixed 45-degree
#   outward diagonal instead of the edge normal, and both the marker lookup and the output ray ignore
#   intervening sprites. Non-background cells stay visually unchanged, but they no longer terminate
#   the path.
# - Small trace: place a left-facing 9x2 emitter at top-left (6, 21) and a right-facing blocker at
#   (2, 16). The emitter's active port at local row 3 starts from absolute (9, 21) and follows
#   (-1, -1): (8,20),(7,19),(6,18),(5,17),(4,16),... . The blocker occupies rows 2..10 and cols
#   16..17, so the diagonal crosses it and still reaches a boundary marker at row 0. Under the old
#   task, the marker could not be seen through the blocker and the output ray would stop at the first
#   blocker pixel. In this combo, the marker is still found and the output continues past the blocker
#   to the boundary.
#
# b) Complexity calibration:
# - The template remains a dense N x 2 rectangle with a uniquely dominant long edge, and three marker
#   colors still label three shared template ids. That preserves the original template-matching core.
# - The generator uses three vertically oriented sprites: one upper right-facing blocker, one upper
#   left-facing emitter behind it, and one lower right-facing copy. This is constrained enough to keep
#   paths legible, but it still shows repeated evidence across multiple sprite instances and multiple
#   marker colors.
# - Using only two diagonal directions in generated samples is deliberate. It keeps the diagonal
#   geometry human-readable while still forcing the solver to perform diagonal stepping and
#   transparency-aware propagation.
#
# c) Old assumptions that break:
# - `_first_marker_on_ray(...)` cannot stop at the first non-background cell anymore; it must keep
#   stepping diagonally until it either finds a marker or leaves the grid.
# - The solver can no longer derive the path step directly from the active-edge normal. Each cardinal
#   active side now maps to a specific diagonal step.
# - The output drawing loop must not `break` on sprites or markers. It may color only background
#   cells, but it must continue stepping until it exits the grid.
# - The original generator's boundary-aligned marker placement and clear line-of-sight check are both
#   invalid. Markers must instead be placed on the diagonal boundary endpoint, and blocking sprites are
#   now required on at least one labeled path so the D5 change is observable.
#
# d) Placement legality:
# - Sprites are still dense non-square rectangles, fully in bounds, and never overlap or touch.
# - The upper blocker/emitter pair is constructed with a fixed gap `d` so every emitter active-edge
#   cell crosses the blocker on the same diagonal family. Marker cells are placed on the top boundary
#   with at least one background cell between them, preventing 8-connected merges.
# - The lower copy is positioned so its diagonal family has a disjoint `r-c` range from the upper
#   pair, preventing different-colored diagonal overlaps on background cells.
#
# e) Visual observability:
# - The D2 change is visible because all copied rays are diagonal rather than horizontal/vertical.
# - The D5 change is visible twice: the input markers sit behind another sprite along the diagonal, and
#   the output contains colored segments on both sides of that blocking sprite.
# - Repeating this over three marker colors gives enough evidence for a human to infer that the colors
#   attach to template ports, not to absolute positions.
#
# f) Visual cleanliness:
# - The generator keeps all markers on the top boundary and spaces them apart, so they remain isolated
#   single-pixel controls instead of merging into a boundary cluster.
# - Only one upper blocker/emitter pair uses transparency-critical geometry; the extra lower copy adds
#   repetition without introducing dense diagonal crossings throughout the grid.
#
# g) Color inference:
# - Marker colors remain unique and are sampled from colors not used by the template, so they are easy
#   to separate visually from sprite pixels.
# - No rule depends on numeric color ordering. Colors are only symbolic labels for shared template ids.
#
# Post-change solver assumptions checked against the generator:
# - Every sprite still matches exactly one dihedral transform of one template.
# - Every marker color labels exactly one active-edge template id.
# - At least one labeled diagonal path crosses a sprite before reaching its marker.
# - At least one output background segment exists beyond that crossed sprite, so a blocking-diagonal
#   baseline would produce a different output.


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


def _diagonal_step(direction: tuple[int, int]) -> tuple[int, int]:
    mapping = {
        (0, -1): (-1, -1),
        (0, 1): (1, 1),
        (-1, 0): (-1, 1),
        (1, 0): (1, -1),
    }
    assert direction in mapping, f"unexpected active-edge direction: {direction}"
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
    while True:
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            return int(r), int(c)
        r, c = nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy boundary-labeled template ports across matching sprites, using diagonal rays that tunnel through sprites.

    Input structure:
    - The background is the grid's majority color.
    - Several dense non-square rectangular sprites are dihedral transforms of one shared template.
    - Isolated single-pixel markers lie on the boundary at the endpoints of outward 45-degree diagonals from some active-edge ports, and those diagonals may cross other sprites before reaching the marker.

    Rule:
    - For each sprite, find the active long edge by the stronger repeated-color majority.
    - Convert that active-edge facing direction into a fixed outward diagonal direction; along that diagonal, ignore intervening non-background cells until a boundary marker is found, and use the marker color to label the corresponding template port.
    - From every matching port on every sprite, propagate that color along the same outward diagonal all the way to the grid edge, painting only background cells and preserving any sprite or marker pixels that the ray passes through.

    Key insight:
    - The markers still label shared template ids rather than absolute positions, but now both recognition and drawing depend on diagonal stepping with transparent sprites.
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
        ray_step = _diagonal_step(direction)
        sprites.append({**sprite, "ids": ids, "direction": direction, "active": active, "ray_step": ray_step})

    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    H, W = g.shape
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["ray_step"]
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

    out = g.copy()
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["ray_step"]
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
_NUM_MARKERS = 3
_GEN_RETRIES = 300
_PAIR_GAP = 4
_TOP_MARGIN = 2
_RIGHT_MARGIN = 6
_LOWER_LEFT_COL = 2
_LOWER_BOTTOM_MARGIN = 3


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    fill = others if others else colors
    for _ in range(n - majority):
        col.append(random.choice(fill))
    random.shuffle(col)
    return col


def _design_template(N: int) -> tuple[np.ndarray, list[int]] | None:
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
    if len(set(template.flat)) < num_colors:
        return None
    return template, template_colors


def _directional_variants(
    template: np.ndarray,
) -> dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]]]:
    raw = _dihedral_variants(template)

    seen: list[np.ndarray] = []
    by_direction: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]]] = {}
    for _name, sub, ids_arr in raw:
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        direction, active_set = _active_side(sub)
        seen.append(sub.copy())
        by_direction.setdefault(direction, []).append((sub, ids_arr, sorted(active_set)))
    return by_direction


def _active_id_order(ids_arr: np.ndarray, active_coords: list[tuple[int, int]]) -> list[int]:
    return [int(ids_arr[r, c]) for r, c in active_coords]


def _make_sprite(
    r0: int,
    c0: int,
    sub: np.ndarray,
    ids_arr: np.ndarray,
    active_coords: list[tuple[int, int]],
) -> dict:
    h, w = sub.shape
    direction, active = _active_side(sub)
    return {
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        "sub": sub,
        "ids": ids_arr,
        "direction": direction,
        "active": set(active_coords),
        "ray_step": _diagonal_step(direction),
    }


def _compatible_lr_pair(
    by_direction: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]]]
) -> tuple[tuple[np.ndarray, np.ndarray, list[tuple[int, int]]], tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]] | None:
    left_choices = by_direction.get((0, -1), [])
    right_choices = by_direction.get((0, 1), [])
    candidates = []
    for left_variant in left_choices:
        left_order = _active_id_order(left_variant[1], left_variant[2])
        for right_variant in right_choices:
            if _active_id_order(right_variant[1], right_variant[2]) == left_order:
                candidates.append((left_variant, right_variant))
    if not candidates:
        return None
    return random.choice(candidates)


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


def _choose_spaced_rows(N: int) -> list[int] | None:
    candidates = list(range(N))
    for _ in range(60):
        rows = sorted(random.sample(candidates, _NUM_MARKERS))
        if all(rows[i + 1] - rows[i] >= 2 for i in range(len(rows) - 1)):
            return rows
    return None


def _place_markers(grid: np.ndarray, emitter: dict, labeled_rows: list[int], template_colors: list[int]) -> bool:
    H, W = grid.shape
    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    r0, c0, _r1, _c1 = emitter["bbox"]
    dr, dc = emitter["ray_step"]
    active_col = min(c for _r, c in emitter["active"])
    used_cells: set[tuple[int, int]] = set()

    for color, lr in zip(marker_colors, labeled_rows):
        lc = active_col
        marker_r, marker_c = _boundary_endpoint(r0 + lr, c0 + lc, dr, dc, H, W)
        if marker_r != 0:
            return False
        if (marker_r, marker_c) in used_cells:
            return False
        if int(grid[marker_r, marker_c]) != _BG:
            return False
        grid[marker_r, marker_c] = color
        used_cells.add((marker_r, marker_c))
    return True


def _build_layout(H: int, W: int, N: int, left_variant, right_variant) -> list[dict] | None:
    left_sub, left_ids, left_active = left_variant
    right_sub, right_ids, right_active = right_variant

    top_blocker_r = _TOP_MARGIN
    top_blocker_c = W - _RIGHT_MARGIN - _PAIR_GAP - 1
    top_emitter_r = top_blocker_r + _PAIR_GAP
    top_emitter_c = top_blocker_c + _PAIR_GAP + 1

    copy_r = H - N - _LOWER_BOTTOM_MARGIN
    copy_c = _LOWER_LEFT_COL

    if top_blocker_c < 0 or top_emitter_c + 1 >= W:
        return None
    if top_emitter_r + N > H:
        return None
    if copy_r < 0 or copy_r + N > H:
        return None

    placements = [
        _make_sprite(top_blocker_r, top_blocker_c, right_sub, right_ids, right_active),
        _make_sprite(top_emitter_r, top_emitter_c, left_sub, left_ids, left_active),
        _make_sprite(copy_r, copy_c, right_sub, right_ids, right_active),
    ]
    return placements


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, template_colors = design

    by_direction = _directional_variants(template)
    pair = _compatible_lr_pair(by_direction)
    if pair is None:
        return None
    left_variant, right_variant = pair

    placements = _build_layout(H, W, N, left_variant, right_variant)
    if placements is None:
        return None

    grid = _paint_sprites(H, W, placements)
    if grid is None:
        return None

    labeled_rows = _choose_spaced_rows(N)
    if labeled_rows is None:
        return None
    emitter = placements[1]
    if not _place_markers(grid, emitter, labeled_rows, template_colors):
        return None

    # Rotate valid base layouts to cover all four active-side / diagonal branches
    # without changing the underlying "diagonal + transparent" rule.
    k = random.randint(0, 3)
    if k:
        grid = np.rot90(grid, k).copy()

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
