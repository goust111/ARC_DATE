# Combo variant of 64efde09 (index 21)
# combo_spec: D18/0+D20/0
# Name: 动态背景色+小网格

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) Concreteization and behavior trace
#    - Original task: the background is always azure 8, the grid is roomy, sprites are 7-9 by 2,
#      and isolated boundary markers label template ports whose colors then shoot outward on every
#      transformed copy.
#    - New combo: the same port-labeling rule remains, but the canvas shrinks to a compact 10-11
#      range and the background color is no longer fixed. A human must first infer which color is
#      the true background by frequency and emptiness, then apply the original ray logic.
#    - Small example: suppose a 10x10 grid uses color 3 as background, with four 5x2 sprites and
#      three boundary markers. Old generation would still use bg=8 and leave large empty margins;
#      new generation uses bg=3, keeps only two empty cells between each active edge and its
#      boundary marker, and packs the four orientations into the four corners. The solver must use
#      "most common color" instead of any fixed bg constant, but the mapping from markers to ports
#      and from ports to rays is unchanged.
# b) Complexity calibration
#    - The rule is not simplified into a trivial two-sprite case: every instance contains four
#      transformed sprites, one per active direction, so repeated evidence still exists.
#    - Freedom is limited in visible ways: thin 3-5 by 2 rectangles, exactly three markers, and a
#      four-corner layout. That keeps the learner focused on background inference plus port copying,
#      rather than on arbitrary packing accidents.
#    - To make the dynamic background observable instead of cosmetic, the generator favors dense
#      10-11 grids and templates with a strong repeated non-bg color, so the background remains the
#      most common color but not by the huge margin seen in the original large canvas.
# c) Implicit-assumption audit against main()
#    - `bg = get_bg(g)` remains valid, but only if generation guarantees a unique most-common color.
#      The generator checks that condition explicitly.
#    - Treating every area-1 component as a marker is now safe only if those singletons are boundary
#      cells; the new solver asserts boundary placement and the generator constructs markers that way.
#    - Matching each sprite against one unique dihedral orientation becomes riskier with N=3-5
#      because symmetric templates are more likely. The generator rejects templates whose chosen
#      directional variants are ambiguous.
#    - Shorter rays mean a sprite placed only one cell from its marker would produce no visible
#      output, because the boundary marker itself blocks drawing. The generator therefore enforces an
#      active-edge gap of 3 cells so every successful ray paints exactly two bg cells.
# d) Placement legality
#    - The generator uses four fixed corner slots: left-facing vertical in the upper-left, up-facing
#      horizontal in the upper-right, down-facing horizontal in the lower-left, and right-facing
#      vertical in the lower-right.
#    - Sprites may touch non-active boundaries, but each active edge stays three cells from the
#      boundary it faces. This keeps all objects in bounds, prevents overlap, and guarantees clear
#      marker sight lines.
#    - Markers are distributed across three distinct sides, so 8-connected component extraction
#      still sees three isolated single-pixel markers.
# e) Visual observability
#    - Across examples, the learner can see the same geometric rule on different background colors,
#      while the compact layout makes "which cells are empty background" less obvious than before.
#    - Four repeated sprite orientations still provide enough evidence that a marker labels a shared
#      template port rather than a location on one specific sprite.
# f) Clean visualization
#    - All added structure is still axis-aligned and sparse: dense 2-cell-thick rectangles, one-pixel
#      markers on boundaries, and straight outward rays in the output only. No region contains layered
#      decorations or overlapping ray colors.
# g) Inferrable color behavior
#    - No step depends on numeric color ordering. Background detection uses visible frequency; marker
#      colors remain unique and visually distinct from both the background and template colors.
# h) Generator guarantees for new main() assumptions
#    - There are exactly four dense rectangles and exactly three isolated boundary markers.
#    - The background color is unique-most-common and varies across generated samples.
#    - Grid dimensions stay in the small regime, and every active-direction ray has two drawable bg
#      cells between sprite edge and boundary marker.

_ALL_COLORS = list(range(10))
_GRID_SIDE_OPTIONS = [10, 10, 10, 11, 11]
_TEMPLATE_N_OPTIONS = [3, 4, 4, 5, 5]
_TEMPLATE_COLOR_COUNT = (3, 4)
_NUM_MARKERS = 3
_ACTIVE_GAP = 3  # marker boundary + 2 bg cells before the active edge
_BG_GAP_MAX = 60
_GEN_RETRIES = 500


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


def _is_boundary_cell(r: int, c: int, H: int, W: int) -> bool:
    return r in (0, H - 1) or c in (0, W - 1)


def _parse_scene(g: np.ndarray) -> tuple[int, list[dict], dict[tuple[int, int], int]]:
    bg = int(get_bg(g))
    H, W = g.shape
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
            assert _is_boundary_cell(r, c, H, W), "markers must be isolated boundary cells"
            markers[(r, c)] = int(g[r, c])
        else:
            assert area == h * w, "sprite components must be dense rectangles"
            sprites_raw.append({"bbox": (r0, c0, r1, c1), "sub": crop_bbox(g, (r0, c0, r1, c1))})

    assert len(sprites_raw) >= 2, "expected multiple transformed rectangular sprites"
    assert markers, "expected isolated control markers"
    assert len(set(markers.values())) == len(markers), "control marker colors must be unique"
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
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a non-fixed background, read boundary marker rays on compact rectangular sprites, and copy those labeled ports to every transformed sprite.

    Input structure:
    - The background is whichever color occupies the most cells, and it is not fixed across tasks.
    - Four dense 2-by-N or N-by-2 rectangles, with N in a small 3-5 range, are dihedral transforms of one shared template.
    - Three isolated single-pixel markers sit on grid boundaries, each aligned with an outward ray from one sprite's active long edge.

    Rule:
    - For each sprite, find the active long edge: the side whose cells have the stronger same-color majority than the opposite long side.
    - If the first non-background cell seen on an outward ray from one active-edge cell is a boundary marker, that marker color labels the shared template port for that cell.
    - On every sprite, draw that marker color outward from the corresponding transformed active-edge cell through background cells until another non-background cell or the grid edge blocks the ray.

    Key insight:
    - The extra difficulty is not a new geometric transformation; it is that the solver must first infer the background from a dense small grid, then treat boundary markers as labels on shared template ports that persist through rotation and reflection.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprites_raw, markers = _parse_scene(g)
    sprites = _identify_sprites(sprites_raw)

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

    return out


# --------------- generator ---------------


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    for _ in range(n - majority):
        col.append(random.choice(others if others else colors))
    random.shuffle(col)
    return col


def _design_template(bg: int, n: int) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    palette = [c for c in _ALL_COLORS if c != bg]
    if n == 3:
        # A dedicated 3x2 design keeps the compact branch solvable:
        # the left edge is unambiguously active, the right edge is asymmetric,
        # and the dominant non-bg color is frequent enough that bg is not trivially obvious.
        template_colors = random.sample(palette, 3)
        dominant = random.choice(template_colors)
        others = [c for c in template_colors if c != dominant]
        left_col = [dominant, dominant, dominant]
        right_col = [others[0], dominant, others[1]]
        if random.random() < 0.5:
            right_col.reverse()

        template = np.column_stack([left_col, right_col]).astype(int)
        ids_base = np.arange(n * 2, dtype=int).reshape(n, 2)
        return template, ids_base, template_colors

    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(palette, num_colors)

    dominant = random.choice(template_colors)
    active_majority = random.randint(max(2, n - 1), n)
    left_col = _build_column(n, template_colors, active_majority, dominant)

    max_right_majority = active_majority - 1
    if max_right_majority < 1:
        return None
    right_majority = max_right_majority
    if random.random() < 0.9:
        right_dominant = dominant
    else:
        right_dominant = random.choice(template_colors)
    right_col = _build_column(n, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None

    ids_base = np.arange(n * 2, dtype=int).reshape(n, 2)
    return template, ids_base, template_colors


def _directional_variants(
    template: np.ndarray, ids_base: np.ndarray
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]]:
    by_direction: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]]] = {}
    seen: list[np.ndarray] = []
    for (_, sub, _), (_, _, ids_arr) in zip(_dihedral_variants(template), _dihedral_variants(ids_base)):
        if any(prev.shape == sub.shape and np.array_equal(prev, sub) for prev in seen):
            continue
        seen.append(sub)
        try:
            direction, active = _active_side(sub)
        except AssertionError:
            continue
        by_direction.setdefault(direction, []).append((sub, ids_arr, sorted(active)))

    required = {(0, -1), (0, 1), (-1, 0), (1, 0)}
    if set(by_direction) != required:
        return {}

    choice: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]] = {}
    for direction in required:
        choice[direction] = random.choice(by_direction[direction])
    return choice


def _place_directional_sprites(
    H: int,
    W: int,
    variants: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]],
) -> list[tuple[int, int, np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
    left_sub, left_ids, left_active = variants[(0, -1)]
    right_sub, right_ids, right_active = variants[(0, 1)]
    up_sub, up_ids, up_active = variants[(-1, 0)]
    down_sub, down_ids, down_active = variants[(1, 0)]

    placements = [
        (0, _ACTIVE_GAP, left_sub, left_ids, (0, -1), left_active),
        (_ACTIVE_GAP, W - up_sub.shape[1], up_sub, up_ids, (-1, 0), up_active),
        (H - down_sub.shape[0] - _ACTIVE_GAP, 0, down_sub, down_ids, (1, 0), down_active),
        (H - right_sub.shape[0], W - right_sub.shape[1] - _ACTIVE_GAP, right_sub, right_ids, (0, 1), right_active),
    ]

    occupied: set[tuple[int, int]] = set()
    for r0, c0, sub, _ids, _direction, _active in placements:
        h, w = sub.shape
        if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
            return []
        sprite_cells = [(r0 + r, c0 + c) for r in range(h) for c in range(w)]
        for cell_r, cell_c in sprite_cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (cell_r + dr, cell_c + dc) in occupied:
                        return []
        occupied.update(sprite_cells)
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
    placements: list[tuple[int, int, np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]],
    ids_base: np.ndarray,
    bg: int,
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    active_template_ids = [int(ids_base[r, 0]) for r in range(ids_base.shape[0])]
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available_colors = [c for c in _ALL_COLORS if c != bg and c not in template_colors]
    if len(available_colors) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available_colors, _NUM_MARKERS)

    # Use three different sides to keep singleton markers visually separated in the compact grid.
    owner_indices = random.sample(range(len(placements)), _NUM_MARKERS)

    used_boundary_cells: set[tuple[int, int]] = set()
    for color, template_id, owner_idx in zip(marker_colors, labeled_ids, owner_indices):
        r0, c0, _sub, ids_arr, direction, active_coords = placements[owner_idx]
        dr, dc = direction

        locs = np.argwhere(ids_arr == template_id)
        if locs.shape[0] != 1:
            return False
        lr, lc = map(int, locs[0])
        if (lr, lc) not in active_coords:
            return False

        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)
        if (marker_r, marker_c) in used_boundary_cells:
            return False
        if any(max(abs(marker_r - rr), abs(marker_c - cc)) <= 1 for rr, cc in used_boundary_cells):
            return False

        cr, cc = r0 + lr + dr, c0 + lc + dc
        saw_bg = False
        while (cr, cc) != (marker_r, marker_c) and 0 <= cr < H and 0 <= cc < W:
            if int(grid[cr, cc]) != bg:
                return False
            saw_bg = True
            cr += dr
            cc += dc

        if not saw_bg or int(grid[marker_r, marker_c]) != bg:
            return False
        grid[marker_r, marker_c] = color
        used_boundary_cells.add((marker_r, marker_c))
    return True


def _bg_is_plausible(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.flat))
    most_common = counts.most_common(2)
    if not most_common or most_common[0][0] != bg:
        return False
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return False
    if len(most_common) > 1 and most_common[0][1] - most_common[1][1] > _BG_GAP_MAX:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_TEMPLATE_N_OPTIONS)
    if n == 5:
        H = 11
        W = 11
    else:
        H = random.choice(_GRID_SIDE_OPTIONS)
        W = random.choice(_GRID_SIDE_OPTIONS)
    bg = random.choice(_ALL_COLORS)

    design = _design_template(bg, n)
    if design is None:
        return None
    template, ids_base, template_colors = design

    variants = _directional_variants(template, ids_base)
    if not variants:
        return None

    placements = _place_directional_sprites(H, W, variants)
    if not placements or len(placements) != 4:
        return None

    grid = np.full((H, W), bg, dtype=int)
    for r0, c0, sub, *_rest in placements:
        h, w = sub.shape
        grid[r0 : r0 + h, c0 : c0 + w] = sub

    if not _place_markers(grid, placements, ids_base, bg, template_colors):
        return None

    if not _bg_is_plausible(grid, bg):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
