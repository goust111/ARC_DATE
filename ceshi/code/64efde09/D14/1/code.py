# Variant of 64efde09
# Dimension: D14 — 端口到模板的映射方式 (meta)
# Change: 通过二面体变换的ID追踪实现端口映射——每个模板像素有唯一ID，变换后ID跟随，标记标注的模板ID在所有精灵上找到对应位... → 按距离活跃边角落的偏移映射

from __future__ import annotations

from collections import Counter

import numpy as np
import random

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the D14 variant
# -----------------------------------
# source_task:
#   skipped: the variant description has no external source_task.
# concretization:
#   objects:
#     - Each sprite is still a dense 2xN or Nx2 rectangle obtained from one template.
#     - Each sprite still has one active long edge determined by the stronger same-color majority.
#     - Boundary markers still sit on rays projected outward from active-edge cells.
#   new port identity:
#     - Old rule: a marker labels an exact template cell via transformed per-pixel IDs.
#     - New rule: a marker labels offset k along the active edge, counted from the same
#       visual corner of that edge on every sprite.
#     - Reference corner is the active-edge endpoint nearest the sprite bbox upper-left:
#       vertical active edges count from top to bottom; horizontal active edges count
#       from left to right.
#   behavior trace:
#     - Suppose the canonical sprite is vertical with active edge on the left, and a
#       marker labels the second cell from the top.
#     - Under the old ID rule, a vertically flipped copy would emit from the second cell
#       from the bottom because that exact template pixel moved there.
#     - Under the new offset rule, the flipped copy still emits from the second cell from
#       the top, because offset is measured from the same visible corner, not by identity.
# complexity calibration:
#   - The variant only changes how corresponding ports are paired; ray drawing, active-edge
#     detection, dense-rectangle parsing, and marker placement remain unchanged.
#   - To make the difference visible, the generator requires both order-preserving and
#     order-reversing sprite orientations in one sample, so the new rule cannot collapse
#     back to the original ID-following behavior.
# implicit assumption review:
#   old_main_dependencies:
#     - _dihedral_variants(...) returned transformed ID grids; this was necessary only
#       because the old solver needed exact template-pixel identity.
#     - len(matches) == 1 assumed each sprite orientation had to be uniquely identified
#       before transferring IDs. In the new rule, exact transform identity is irrelevant:
#       any dihedral match is enough, because only edge order matters.
#     - np.argwhere(sprite["ids"] == template_id) assumed each color mapped to a unique
#       pixel under every transform. This is replaced by direct indexing into the ordered
#       active-edge list using offset k.
#   new_main_dependencies:
#     - Every sprite must still be a dihedral transform of the same template.
#     - Every sprite must have the same active-edge length.
#     - Active-edge cells, when sorted top-to-bottom or left-to-right, represent the
#       intended offset order. The generator enforces this by using thin rectangles and
#       keeping the active side structurally unique.
# placement legality:
#   - Sprites are still padded apart and placed so the active side faces a boundary.
#   - Markers remain on the corresponding boundary cell with clear line-of-sight.
#   - Solver validation rejects ray conflicts, so the generator keeps retrying until a
#     legal arrangement is found.
# visual observability:
#   - The template is forced to be asymmetric enough that flips are visible.
#   - At least one chosen sprite reverses active-edge order relative to the canonical
#     template, and at least one chosen sprite preserves it.
#   - Marker offsets avoid the exact middle cell when N is odd, preventing accidental
#     old/new agreement on reversed sprites.
# cleanliness and color inference:
#   - Marker colors remain unique and disjoint from template colors.
#   - No ordering depends on numeric color IDs; only geometric offset along the active
#     edge controls color propagation.


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        variants.append((f"rot{k}", rotated))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated)))
    return variants


def _dihedral_variants_with_ids(
    template: np.ndarray, ids: np.ndarray
) -> list[tuple[str, np.ndarray, np.ndarray]]:
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


def _ordered_active_edge(
    active: set[tuple[int, int]], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    if direction[0] == 0:
        return sorted(active, key=lambda rc: (rc[0], rc[1]))
    return sorted(active, key=lambda rc: (rc[1], rc[0]))


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
    Replicate colored control rays across transformed sprites by matching edge offsets instead of template-pixel IDs.

    Input structure:
    - The background is the most frequent color.
    - Several dense rectangular sprites share one 2xN or Nx2 template up to dihedral transforms.
    - Single-cell markers sit on boundary-directed rays coming out of active-edge cells.

    Rule:
    - For each sprite, find the active long edge: the long side with the stronger same-color majority.
    - If a marker is the first non-background cell seen on the outward ray from an active-edge cell, that marker color labels the cell's offset along the active edge.
    - On every sprite, emit that color outward from the active-edge cell with the same top-to-bottom or left-to-right offset until blocked or out of bounds.

    Key insight:
    - Correspondence is no longer "the same transformed template pixel"; it is "the kth port from the reference corner of the active edge", so flipped sprites keep the same visible offset rather than the same original identity.
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
    active_len = None
    for sprite in sprites_raw:
        matches = [
            name
            for name, sub in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert matches, "each sprite must be a dihedral transform of the template"
        direction, active = _active_side(sprite["sub"])
        ordered_active = _ordered_active_edge(active, direction)
        if active_len is None:
            active_len = len(ordered_active)
        assert len(ordered_active) == active_len, "active-edge lengths must agree"
        sprites.append(
            {
                **sprite,
                "direction": direction,
                "active": active,
                "ordered_active": ordered_active,
            }
        )

    color_to_offset: dict[int, int] = {}
    offset_to_color: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for offset, (lr, lc) in enumerate(sprite["ordered_active"]):
            marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if marker_color is None:
                continue
            if marker_color in color_to_offset:
                assert color_to_offset[marker_color] == offset
            if offset in offset_to_color:
                assert offset_to_color[offset] == marker_color
            color_to_offset[marker_color] = offset
            offset_to_color[offset] = marker_color

    assert set(color_to_offset) == set(markers.values()), "every control marker must label one active-edge offset"

    out = g.copy()
    H, W = g.shape
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for color, offset in color_to_offset.items():
            assert 0 <= offset < len(sprite["ordered_active"]), "offset must fit every active edge"
            lr, lc = sprite["ordered_active"][offset]

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
_ACTIVE_MAJORITY_RANGE = (5, 9)
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_CLEARANCE = 2
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
    if len(set(map(int, template.flat))) < num_colors:
        return None

    row_pairs = [tuple(map(int, row)) for row in template.tolist()]
    if row_pairs == list(reversed(row_pairs)):
        return None
    if row_pairs[0] == row_pairs[-1]:
        return None
    if len(set(row_pairs)) < min(5, N):
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _deduplicated_variants(template: np.ndarray, ids_base: np.ndarray) -> list[dict]:
    raw = _dihedral_variants_with_ids(template, ids_base)
    id_to_offset = {int(ids_base[r, 0]): r for r in range(ids_base.shape[0])}
    forward_seq = list(range(ids_base.shape[0]))
    reverse_seq = list(reversed(forward_seq))

    seen: list[np.ndarray] = []
    result: list[dict] = []
    for name, sub, ids_arr in raw:
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            direction, active_set = _active_side(sub)
        except AssertionError:
            continue

        ordered_active = _ordered_active_edge(active_set, direction)
        try:
            offset_sequence = [id_to_offset[int(ids_arr[r, c])] for r, c in ordered_active]
        except KeyError:
            continue

        if offset_sequence == forward_seq:
            reversed_order = False
        elif offset_sequence == reverse_seq:
            reversed_order = True
        else:
            continue

        seen.append(sub)
        result.append(
            {
                "name": name,
                "sub": sub,
                "direction": direction,
                "active_coords": list(active_set),
                "ordered_active": ordered_active,
                "reversed_order": reversed_order,
            }
        )
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


def _occupied_set(placements: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for sprite in placements:
        r0 = sprite["r0"]
        c0 = sprite["c0"]
        sub = sprite["sub"]
        h, w = sub.shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _place_sprites(H: int, W: int, variants: list[dict]) -> list[dict] | None:
    placements: list[dict] = []
    pad = _PLACEMENT_PAD
    for variant in variants:
        sub = variant["sub"]
        h, w = sub.shape
        dr, dc = variant["direction"]
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w, dr, dc)
            occ = _occupied_set(placements)
            if any((r0 + r, c0 + c) in occ for r in range(-pad, h + pad) for c in range(-pad, w + pad)):
                continue
            placements.append({**variant, "r0": r0, "c0": c0})
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


def _choose_variants(unique_variants: list[dict], num_sprites: int) -> list[dict] | None:
    forward = [v for v in unique_variants if not v["reversed_order"]]
    reversed_variants = [v for v in unique_variants if v["reversed_order"]]
    if not forward or not reversed_variants:
        return None

    indices = list(range(len(unique_variants)))
    for _ in range(80):
        chosen_idx = random.sample(indices, num_sprites)
        chosen = [unique_variants[i] for i in chosen_idx]
        if any(v["reversed_order"] for v in chosen) and any(not v["reversed_order"] for v in chosen):
            return chosen
    return None


def _candidate_offsets(N: int) -> list[int]:
    return [k for k in range(N) if 2 * k != N - 1]


def _place_markers(grid: np.ndarray, sprite_placements: list[dict], N: int, template_colors: list[int]) -> bool:
    H, W = grid.shape
    candidates = _candidate_offsets(N)
    if len(candidates) < _NUM_MARKERS:
        return False
    labeled_offsets = random.sample(candidates, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    forward_idxs = [i for i, sprite in enumerate(sprite_placements) if not sprite["reversed_order"]]
    reversed_idxs = [i for i, sprite in enumerate(sprite_placements) if sprite["reversed_order"]]
    if not forward_idxs or not reversed_idxs:
        return False

    host_indices = [random.choice(reversed_idxs), random.choice(forward_idxs)]
    while len(host_indices) < _NUM_MARKERS:
        host_indices.append(random.randrange(len(sprite_placements)))
    random.shuffle(host_indices)

    for color, offset, host_idx in zip(marker_colors, labeled_offsets, host_indices):
        sprite = sprite_placements[host_idx]
        r0, c0 = sprite["r0"], sprite["c0"]
        dr, dc = sprite["direction"]
        lr, lc = sprite["ordered_active"][offset]
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
    chosen = _choose_variants(unique_variants, num_sprites)
    if chosen is None:
        return None

    placements = _place_sprites(H, W, chosen)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for sprite in placements:
        r0, c0 = sprite["r0"], sprite["c0"]
        sub = sprite["sub"]
        h, w = sub.shape
        grid[r0 : r0 + h, c0 : c0 + w] = sub

    if not _place_markers(grid, placements, N, template_colors):
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
