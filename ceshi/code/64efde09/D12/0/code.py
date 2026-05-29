# Variant of 64efde09
# Dimension: D12 — 活跃边识别规则 (meta)
# Change: 活跃边是长边中最大同色计数（majority count）严格更高的那一边
#         → 活跃边由特定颜色存在性决定，这里取模板中唯一最少出现的颜色所在的长边

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for D12
# a) Concrete behavior and trace:
#    The sprites remain dense N x 2 or 2 x N rectangles, and markers remain isolated
#    boundary pixels whose colors label template ports. The changed step is only how the
#    active long edge is recognized. Instead of comparing same-color majorities between the
#    two long sides, each template now contains one unique rarest color ("signal color"),
#    and exactly one long side contains that color. Example on a canonical N x 2 sprite:
#    the left column contains one signal-color cell and otherwise mixed colors, while the
#    right column contains no signal-color cells but may have a strong majority. Original
#    logic would pick the right side because its majority is larger; the variant must pick
#    the left side because only that side contains the signal color. Marker discovery and
#    ray propagation then proceed exactly as before, but from the signal-bearing side.
# b) Complexity calibration:
#    To make the change visible rather than cosmetic, the generator intentionally makes the
#    majority-winning side the opposite side. This means the same input would often produce
#    a different output under the original solver. The template still uses only 3-4 colors
#    and only one signal-color category, so the recognition burden stays small.
# c) Hidden assumptions audited:
#    - The old _active_side depended on "strict majority gap". That no longer holds and is
#      replaced by two new guarantees: the template has a unique rarest color, and exactly
#      one long side contains it.
#    - Marker assignment loops previously assumed every active coordinate came from the
#      majority side; now they only assume the mapped template ids lie on the signal side.
#    - Variant matching still uses dihedral equivalence of the whole rectangle, which is
#      unchanged by D12.
# d) Placement legality:
#    Sprites still face a boundary with clearance, markers still sit on the corresponding
#    boundary ray, and placement rejects overlaps and blocked line-of-sight. Those spatial
#    constraints are unchanged and remain sufficient for the new solver.
# e) Visual observability:
#    The signal color is the unique globally rarest template color and is confined to the
#    active edge, making the active edge visually identifiable from a few examples. The
#    generator further forces the old majority heuristic to disagree, so D12 is observable.
# f) Visual cleanliness:
#    Only one extra cue is added: a rare color embedded in one side of the rectangle. Rays,
#    markers, and sprite spacing remain unchanged, so the grid stays readable.
# g) Color inferability:
#    No hidden color-id ordering is used. The chosen signal color is determined by visible
#    multiplicity in the sprite, not by numeric value. Humans can infer "the side containing
#    the uniquely rare color is the emitting side" from repeated examples.
# 9) New assumptions checked against the generator:
#    - main() assumes a unique rarest non-bg sprite color: generator enforces that.
#    - main() assumes exactly one long side contains that color: generator enforces that by
#      keeping the signal color entirely off the opposite side.
#    - main() assumes every marker labels an active-edge template id: generator samples only
#      from the canonical active column and transforms those ids with the selected sprite.


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


def _long_sides(sub: np.ndarray) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"
    if h > w:
        return [
            ((0, -1), [(r, 0) for r in range(h)]),
            ((0, 1), [(r, w - 1) for r in range(h)]),
        ]
    return [
        ((-1, 0), [(0, c) for c in range(w)]),
        ((1, 0), [(h - 1, c) for c in range(w)]),
    ]


def _signal_color(sub: np.ndarray) -> int:
    counts = Counter(map(int, sub.flat))
    rare_count = min(counts.values())
    rare_colors = [color for color, count in counts.items() if count == rare_count]
    assert len(rare_colors) == 1, "template must have a unique rarest color"
    return int(rare_colors[0])


def _active_side(sub: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    signal = _signal_color(sub)
    hits = []
    for direction, coords in _long_sides(sub):
        contains_signal = any(int(sub[r, c]) == signal for r, c in coords)
        hits.append((contains_signal, direction, set(coords)))
    active = [item for item in hits if item[0]]
    assert len(active) == 1, "exactly one long side must contain the signal color"
    _contains, direction, coords = active[0]
    return direction, coords


def _majority_side_direction(sub: np.ndarray) -> tuple[int, int] | None:
    scored = []
    for direction, coords in _long_sides(sub):
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((max(Counter(vals).values()), direction))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


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
    Transfer boundary marker labels from one signal-bearing edge of a rectangular sprite template to all transforms.

    Input structure:
    - The background is the most frequent color.
    - Several dense 2-by-N or N-by-2 rectangles are dihedral copies of one shared template.
    - Single-pixel markers lie on boundary-aligned rays extending outward from some sprites' active long edges.

    Rule:
    - For each sprite, find the unique rarest color appearing anywhere in that rectangle.
    - The active edge is the long side that contains that rarest color; a boundary marker first seen on a ray from one of its cells labels that template port.
    - Reproduce each labeled port on every transformed sprite by drawing the marker color outward from the corresponding active-edge cell until blocked or off-grid.

    Key insight:
    - The markers name template positions, not absolute coordinates, and the correct edge is identified by the presence of the template's rare signal color rather than by a majority comparison.
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


# --------------- generator (att1) ---------------

# --- CFG: tuneable generation parameters ---
_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_SIGNAL_COUNT_RANGE = (1, 2)
_ACTIVE_SIDE_MAX_REPEAT = 3
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_CLEARANCE = 2
_PLACEMENT_RETRIES = 100
_GEN_RETRIES = 400


def _build_balanced_column(length: int, colors: list[int], max_repeat: int) -> list[int] | None:
    if length < len(colors):
        return None
    col = list(colors)
    counts = Counter(col)
    while len(col) < length:
        allowed = [c for c in colors if counts[c] < max_repeat]
        if not allowed:
            return None
        min_count = min(counts[c] for c in allowed)
        choices = [c for c in allowed if counts[c] == min_count]
        choice = random.choice(choices)
        counts[choice] += 1
        col.append(choice)
    random.shuffle(col)
    return col


def _build_dominant_column(length: int, colors: list[int], dominant: int, majority: int) -> list[int] | None:
    others = [c for c in colors if c != dominant]
    if majority >= length or length - majority < len(others):
        return None
    col = [dominant] * majority + list(others)
    while len(col) < length:
        col.append(random.choice(others))
    random.shuffle(col)
    return col


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int], int] | None:
    """Create an Nx2 canonical template whose left edge is active by signal-color presence."""
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)
    signal_color = random.choice(template_colors)
    support_colors = [c for c in template_colors if c != signal_color]

    signal_count = random.randint(*_SIGNAL_COUNT_RANGE)
    left_fill = _build_balanced_column(N - signal_count, support_colors, _ACTIVE_SIDE_MAX_REPEAT)
    if left_fill is None:
        return None
    left_col = left_fill + [signal_color] * signal_count
    random.shuffle(left_col)
    left_max = max(Counter(left_col).values())

    dominant = random.choice(support_colors)
    max_majority = N - (len(support_colors) - 1)
    min_majority = max(left_max + 1, 4)
    if min_majority > max_majority:
        return None
    right_majority = random.randint(min_majority, max_majority)
    right_col = _build_dominant_column(N, support_colors, dominant, right_majority)
    if right_col is None:
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    counts = Counter(map(int, template.flat))
    rare_count = counts[signal_color]
    if sum(1 for count in counts.values() if count == rare_count) != 1:
        return None
    if any(int(template[r, 1]) == signal_color for r in range(N)):
        return None
    if not any(int(template[r, 0]) == signal_color for r in range(N)):
        return None

    new_direction, _active = _active_side(template)
    old_direction = _majority_side_direction(template)
    if new_direction != (0, -1):
        return None
    if old_direction != (0, 1):
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors, signal_color


def _deduplicated_variants(
    template: np.ndarray, ids_base: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
    """Return unique dihedral variants together with their signal-based active side."""
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


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int]:
    """Pick a top-left corner so the active edge faces its boundary with clearance."""
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
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    N = ids_base.shape[0]
    num_sprites = len(sprite_placements)

    active_template_ids = [ids_base[r, 0] for r in range(N)]
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    for color, tid in zip(marker_colors, labeled_ids):
        si = random.randint(0, num_sprites - 1)
        r0, c0, _, ids_arr, direction, active_coords = sprite_placements[si]
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
    template, ids_base, template_colors, _signal_color_value = design

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
        grid[r0 : r0 + h, c0 : c0 + w] = sub

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
