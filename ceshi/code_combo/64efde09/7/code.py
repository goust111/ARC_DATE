# Combo variant of 64efde09 (index 7)
# combo_spec: D8/0+D9/0
# Name: 宽矩形仅旋转变体

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Input still contains several dense rectangular sprites plus isolated one-cell markers on the boundary.
#    - The template is now an N x M rectangle with N in 7..9 and M in {3, 4}; this makes each sprite visibly wider
#      than the original 2-column seed task.
#    - Example trace:
#         * Suppose the canonical template is 8x3 and its left edge has color counts [2,2,2,2,2,4,4,2], so the
#           strongest repeated color on that edge appears 6 times.
#         * The other outer edges have at most 4 equal cells because width is only 3, and the right edge is
#           explicitly built with a smaller maximum count, so the left edge is the unique active edge.
#         * If that template is rotated 90 degrees, the same ports move onto the top edge of a 3x8 sprite; markers
#           still label those template ports, and rays are emitted outward from matching ports on every rotation.
#      Difference from the seed task:
#         * the solver can no longer assume a fixed 2-cell thickness or compare only one pair of long sides;
#         * the solver must also reject mirror-only matches, because the shared template family is rotational, not
#           fully dihedral.
# b) Complexity calibration
#    - Width 3/4 produces a clear visual change and more internal color structure, but the scoring rule is still local
#      and human-checkable: inspect each boundary and find the edge with the largest repeated-color multiplicity.
#    - The transform family is intentionally smaller than the original: only 4 rotations are legal. This keeps matching
#      manageable while still requiring the solver to understand orientation transfer.
#    - The generator enforces chirality and four distinct rotations so the "no reflection" restriction is observable
#      rather than vacuous.
# c) Hidden assumptions from the original main()
#    - The original `_active_side()` only inspected the two long sides. That breaks for wider rectangles, so the combo
#      version scores all four edges and requires a unique winner.
#    - The original orientation matcher accepted reflections. That is no longer valid; the combo version matches sprites
#      only against rotations of a shared template.
#    - The original generator could hard-code canonical ports as "left column of an Nx2". Here the active ports are
#      recorded from the actual base template because the wider rectangle may later be rotated to any side.
#    - The matching loop assumes every placed sprite has exactly one legal orientation. The generator guarantees this by
#      rejecting templates with repeated rotations or reflection symmetries.
# d) Placement legality
#    - Sprites remain dense, non-overlapping rectangles placed fully inside the grid.
#    - Each sprite is positioned so its active edge faces a boundary with clearance, letting a marker sit on that same
#      boundary and see a chosen port through background cells only.
#    - Final validation still runs `main()` to reject accidental ray conflicts.
# e) Visual observability
#    - The wider body is directly visible in the input/output pairs.
#    - The active edge can be inferred from repeated colors on one outer side; this uses visible multiplicity rather than
#      any hidden ordering on color ids.
#    - The lack of reflections is made visible by using chiral templates: mirrored copies would look different, but they
#      never appear among the sprites.
# f) Visual cleanliness
#    - Markers stay isolated on the border and sprite padding keeps the rays separated from unrelated bodies.
#    - Ray drawing only extends through background, so the output remains readable instead of stacking multiple object
#      types inside one region.
# g) Color inferability
#    - Marker colors are unique and disjoint from template colors, so each ray family has an obvious source.
#    - Template color count stays moderate even though the body is wider, preventing visually noisy color remapping.
# h) New solver assumptions after the change
#    - Every sprite is a rotation of one chiral, non-square template and matches exactly one rotation.
#    - The active edge is the unique highest-scoring edge among all four sides.
#    - Every marker labels exactly one active-edge template port, and generator construction enforces all three points.


def _rotation_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        variants.append((f"rot{k}", np.rot90(template, k), np.rot90(ids, k)))
    return variants


def _dihedral_variants(arr: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(arr, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _edge_specs(sub: np.ndarray) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"
    return [
        ((0, -1), [(r, 0) for r in range(h)]),
        ((0, 1), [(r, w - 1) for r in range(h)]),
        ((-1, 0), [(0, c) for c in range(w)]),
        ((1, 0), [(h - 1, c) for c in range(w)]),
    ]


def _active_side(sub: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    scored = []
    for direction, coords in _edge_specs(sub):
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((max(Counter(vals).values()), direction, set(coords)))

    scored.sort(key=lambda item: item[0], reverse=True)
    assert scored[0][0] > scored[1][0], "active side must be structurally unique across all edges"
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate boundary marker labels from one wide rectangular template to every rotationally matching sprite.

    Input structure:
    - The background is the most common color.
    - Several dense non-square N-by-M rectangles share one template with width M in {3, 4}, and each sprite is a pure
      rotation of that template rather than an arbitrary reflection.
    - Isolated one-cell markers sit on boundary cells along outward rays aligned with some active-edge ports.

    Rule:
    - Score all four edges of each sprite by the largest same-color multiplicity visible on that edge; the unique best
      edge is the active edge.
    - If the first non-background cell seen along the outward ray of an active-edge port is a marker, that marker color
      labels the corresponding template port.
    - For every sprite that matches the template by rotation, emit each learned marker color outward from the matching
      active-edge port through background cells until another object blocks the ray or the grid ends.

    Key insight:
    - Markers do not label absolute positions. They identify template ports on a shared wider rectangle, and those ports
      must be tracked across rotations only; mirrored placements are not part of the legal object family.
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

    variants = _rotation_variants(sprites_raw[0]["sub"])
    sprites = []
    for sprite in sprites_raw:
        matches = [
            (ids, name)
            for name, sub, ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "each sprite must match the template in exactly one rotational orientation"
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

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (24, 32)
_GRID_W_RANGE = (30, 35)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_WIDTH_CHOICES = (3, 3, 4)
_TEMPLATE_COLOR_COUNT = (4, 5)
_ACTIVE_EDGE_SCORE_RANGE = (4, 7)
_SHORT_ACTIVE_SCORE_MIN = 3
_NUM_ROTATION_SPRITES = 4
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_CLEARANCE = 2
_PLACEMENT_RETRIES = 150
_GEN_RETRIES = 500

_SIDE_NAME_TO_DIRECTION = {
    "left": (0, -1),
    "right": (0, 1),
    "top": (-1, 0),
    "bottom": (1, 0),
}


def _build_edge_with_max_count(
    length: int,
    colors: list[int],
    target: int,
    anchor: int | None = None,
) -> list[int] | None:
    if not (1 <= target <= length):
        return None
    if len(colors) < 2 and target < length:
        return None
    if length > len(colors) * target:
        return None

    dominant = random.choice(colors) if anchor is None else anchor
    others = [c for c in colors if c != dominant]
    if target < length and not others:
        return None

    counts = {dominant: target}
    remaining = length - target
    supply: list[int] = []
    for color in others:
        supply.extend([color] * target)
    random.shuffle(supply)
    if remaining > len(supply):
        return None
    for color in supply[:remaining]:
        counts[color] = counts.get(color, 0) + 1

    edge: list[int] = []
    for color, cnt in counts.items():
        edge.extend([color] * cnt)
    random.shuffle(edge)
    if max(Counter(edge).values()) != target:
        return None
    return edge


def _side_score(sub: np.ndarray, coords: list[tuple[int, int]]) -> int:
    vals = [int(sub[r, c]) for r, c in coords]
    return max(Counter(vals).values())


def _has_unique_rotations(template: np.ndarray) -> bool:
    seen: list[np.ndarray] = []
    for _name, sub, _ids in _rotation_variants(template):
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            return False
        seen.append(sub.copy())
    return True


def _is_chiral(template: np.ndarray) -> bool:
    rotations = [sub for _name, sub, _ids in _rotation_variants(template)]
    for reflected in _dihedral_variants(template)[1::2]:
        if any(rot.shape == reflected.shape and np.array_equal(rot, reflected) for rot in rotations):
            return False
    return True


def _design_template(
    n: int,
    m: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[int, int]]] | None:
    active_name = random.choice(["left", "right", "top", "bottom", "top", "bottom", "top", "bottom"])
    if active_name in ("top", "bottom"):
        num_colors = _TEMPLATE_COLOR_COUNT[1]
    else:
        num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)

    if active_name in ("left", "right"):
        min_feasible = (n + num_colors - 1) // num_colors
        max_active = min(n, _ACTIVE_EDGE_SCORE_RANGE[1])
        min_active = max(min_feasible, min(_ACTIVE_EDGE_SCORE_RANGE[0], max_active))
        if min_active > max_active:
            return None
        active_score = random.randint(min_active, max_active)
        active_color = random.choice(template_colors)

        columns: list[list[int]] = []
        active_col = _build_edge_with_max_count(n, template_colors, active_score, anchor=active_color)
        if active_col is None:
            return None
        columns.append(active_col)
        for _ in range(m - 1):
            max_other = max(min_feasible, active_score - 1)
            if max_other > n:
                return None
            target = random.randint(min_feasible, max_other)
            col = _build_edge_with_max_count(n, template_colors, target)
            if col is None:
                return None
            columns.append(col)
        template = np.column_stack(columns).astype(int)
        if active_name == "right":
            template = np.fliplr(template)
    else:
        min_long_edge = (n + num_colors - 1) // num_colors
        min_active = max(_SHORT_ACTIVE_SCORE_MIN, min_long_edge + 1)
        max_active = m
        if min_active > max_active:
            return None
        active_score = random.randint(min_active, max_active)
        active_color = random.choice(template_colors)

        rows: list[list[int]] = []
        active_row = _build_edge_with_max_count(m, template_colors, active_score, anchor=active_color)
        if active_row is None:
            return None
        rows.append(active_row)
        for _ in range(n - 1):
            max_other = max(1, active_score - 1)
            target = random.randint(1, max_other)
            row = _build_edge_with_max_count(m, template_colors, target)
            if row is None:
                return None
            rows.append(row)
        template = np.array(rows, dtype=int)
        if active_name == "bottom":
            template = np.flipud(template)

    if len(set(map(int, template.flat))) < num_colors:
        return None

    specs = _edge_specs(template)
    scores = [_side_score(template, coords) for _direction, coords in specs]
    if scores.count(max(scores)) != 1:
        return None
    expected_direction = _SIDE_NAME_TO_DIRECTION[active_name]
    direction, active_set = _active_side(template)
    if direction != expected_direction:
        return None
    if not _has_unique_rotations(template):
        return None
    if not _is_chiral(template):
        return None

    ids_base = np.arange(n * m, dtype=int).reshape(n, m)
    active_coords = sorted(active_set)
    return template, ids_base, template_colors, active_coords


def _deduplicated_rotation_variants(
    template: np.ndarray,
    ids_base: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
    result = []
    for _name, sub, ids_arr in _rotation_variants(template):
        direction, active_set = _active_side(sub)
        result.append((sub, ids_arr, direction, sorted(active_set)))
    return result


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int]:
    pad = _PLACEMENT_CLEARANCE
    if dr == 0 and dc == -1:
        return random.randint(0, H - h), random.randint(pad, W - w)
    if dr == 0 and dc == 1:
        return random.randint(0, H - h), random.randint(0, W - w - pad - 1)
    if dr == -1 and dc == 0:
        return random.randint(pad, H - h), random.randint(0, W - w)
    return random.randint(0, H - h - pad - 1), random.randint(0, W - w)


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
        dr, dc = direction
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w, dr, dc)
            occ = _occupied_set(placements)
            if any(
                (r0 + r, c0 + c) in occ
                for r in range(-pad, h + pad)
                for c in range(-pad, w + pad)
            ):
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


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    active_template_ids: list[int],
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    num_sprites = len(sprite_placements)

    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)
    sprite_indices = random.sample(range(num_sprites), _NUM_MARKERS)

    for color, tid, si in zip(marker_colors, labeled_ids, sprite_indices):
        r0, c0, _sub, ids_arr, direction, active_coords = sprite_placements[si]
        dr, dc = direction

        locs = np.argwhere(ids_arr == tid)
        if locs.shape[0] != 1:
            return False
        lr, lc = int(locs[0, 0]), int(locs[0, 1])
        if (lr, lc) not in active_coords:
            return False

        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)

        cr, cc = r0 + lr + dr, c0 + lc + dc
        while (cr, cc) != (marker_r, marker_c) and 0 <= cr < H and 0 <= cc < W:
            if int(grid[cr, cc]) != _BG:
                return False
            cr += dr
            cc += dc

        if int(grid[marker_r, marker_c]) != _BG:
            return False
        grid[marker_r, marker_c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_TEMPLATE_N_RANGE)
    m = random.choice(_TEMPLATE_WIDTH_CHOICES)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(n, m)
    if design is None:
        return None
    template, ids_base, template_colors, active_base_coords = design
    active_template_ids = [int(ids_base[r, c]) for r, c in active_base_coords]

    unique_variants = _deduplicated_rotation_variants(template, ids_base)
    if len(unique_variants) != _NUM_ROTATION_SPRITES:
        return None
    # Show the full rotation family each time so the "rotations only, no flips" rule is
    # visible from the training examples instead of being an unobserved generator promise.
    chosen = random.sample(unique_variants, len(unique_variants))

    placements = _place_sprites(H, W, chosen)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        grid[r0:r0 + h, c0:c0 + w] = sub

    if not _place_markers(grid, placements, active_template_ids, template_colors):
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
