# Combo variant of 64efde09 (index 11)
# combo_spec: D4/2+D14/0
# Name: 同色标记+颜色序列位置映射

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   combo_core:
#     - Marker pixels remain isolated single boundary cells aligned with outward rays from
#       active-edge ports, but their colors are borrowed from the template palette itself.
#     - Port identity is no longer tracked by transformed per-pixel IDs. Instead, a labeled
#       port is identified by the visible signature `(edge_color, kth_occurrence_of_that_color)`
#       along the active edge, read in top-to-bottom or left-to-right order.
#     - Because the marker color equals the active-edge color it refers to, color now serves
#       two roles at once: it helps detect the color class to count inside the active-edge
#       sequence, yet it is insufficient on its own because multiple markers may reuse it.
#   concreteization_and_trace:
#     - Original behavior: suppose the canonical active edge is `[2, 5, 2, 3, 2, 5, 2]`
#       and a unique outside marker color 7 lands on the third cell. The old solver stores
#       that exact template pixel ID and finds its transformed copy everywhere else.
#     - Combo behavior: if a boundary singleton with color 2 aligns with that same third cell,
#       it means "the second 2 on the active edge", not "template pixel ID #k". Another
#       boundary singleton can also be color 2 and mean "the fourth 2".
#     - On a flipped sprite whose visible active-edge sequence becomes `[2, 5, 2, 3, 2, 5, 2]`
#       reversed, the old ID rule would follow the transformed original pixel, while the new
#       rule still chooses "the second 2 from the visible start of the edge". That is the
#       intended divergence.
#   complexity_calibration:
#     - The generator always creates at least two markers with the same color and always labels
#       non-identical occurrences of that color, so the task cannot collapse to `marker_color ->
#       unique port`.
#     - Samples also include both order-preserving and order-reversing sprite orientations, so
#       sequence-position mapping produces visibly different launch ports from ID tracking.
#     - Freedom is kept small: only three markers, a 3-4 color template, and thin 2xN / Nx2
#       rectangles. The only extra reasoning burden is counting repeated colors along one edge.
#   implicit_assumption_audit:
#     - `len(set(markers.values())) == len(markers)` is invalid now because repeated marker
#       colors are required.
#     - `color -> template_id` is invalid because one color may label several different ports.
#     - Treating every area-1 component as a marker remains safe only if it is a boundary
#       singleton; the solver asserts that and the generator enforces 8-neighbor isolation.
#     - Exact dihedral-ID tracking is intentionally removed from `main()`. The new solver only
#       assumes that all sprites share the same ordered active-edge color sequence up to reversal.
#   placement_legality:
#     - Sprites stay at least one cell away from all boundaries and have extra clearance in the
#       facing direction, keeping boundary markers visually separate from sprite cells.
#     - Marker placement checks boundary isolation and clear line-of-sight to the chosen active
#       port. Validation by `main()` filters any accidental ray conflicts.
#   visual_observability:
#     - Every marker color already appears on the sprites, so novelty of color no longer marks
#       labels; their role is visible only through boundary isolation and ray alignment.
#     - At least one repeated marker color appears on two distinct ports, and at least one chosen
#       labeled occurrence is non-central under reversal, so the sequence-position rule changes
#       actual outputs on reversed sprites.
#   cleanliness_and_inference:
#     - No ordering depends on numeric color IDs. The only order used is visible edge order:
#       vertical edges count top-to-bottom, horizontal edges count left-to-right.
#     - The output remains tidy because only straight rays are added, and different colors are
#       never allowed to overlap on the same background cell.
#   generator_guarantees_for_new_main:
#     - All sprites are dense rectangles and all area-1 components are boundary markers.
#     - The canonical active edge contains at least two labeled ports sharing one color.
#     - Chosen sprite orientations include both preserved and reversed edge-order cases.
#     - The repeated-color labels are chosen so sequence mapping differs from ID-following on
#       at least one reversed sprite.


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


def _is_boundary_cell(r: int, c: int, H: int, W: int) -> bool:
    return r in (0, H - 1) or c in (0, W - 1)


def _ordered_active_edge(
    active: set[tuple[int, int]], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    if direction[0] == 0:
        return sorted(active, key=lambda rc: (rc[0], rc[1]))
    return sorted(active, key=lambda rc: (rc[1], rc[0]))


def _signatures_from_ordered_edge(
    sub: np.ndarray, ordered_active: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    seen: Counter[int] = Counter()
    signatures: list[tuple[int, int]] = []
    coord_to_signature: dict[tuple[int, int], tuple[int, int]] = {}
    for lr, lc in ordered_active:
        color = int(sub[lr, lc])
        seen[color] += 1
        signature = (color, int(seen[color]))
        signatures.append(signature)
        coord_to_signature[(lr, lc)] = signature
    return signatures, coord_to_signature


def _analyze_sprite(sub: np.ndarray) -> dict:
    direction, active = _active_side(sub)
    ordered_active = _ordered_active_edge(active, direction)
    signatures, coord_to_signature = _signatures_from_ordered_edge(sub, ordered_active)
    signature_to_coord = {
        signature: coord for coord, signature in coord_to_signature.items()
    }
    assert len(signature_to_coord) == len(ordered_active), "active-edge signatures must be unique"
    return {
        "sub": sub,
        "direction": direction,
        "active": active,
        "ordered_active": ordered_active,
        "signatures": signatures,
        "coord_to_signature": coord_to_signature,
        "signature_to_coord": signature_to_coord,
    }


def _first_marker_on_ray(
    g: np.ndarray,
    bg: int,
    markers: dict[tuple[int, int], int],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> tuple[int, int] | None:
    H, W = g.shape
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if int(g[r, c]) != bg:
            return (r, c) if (r, c) in markers else None
        r += dr
        c += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast boundary marker colors from repeated-color edge positions to every transformed strip sprite.

    Input structure:
    - The background is the dominant color.
    - Several dense 2-by-N or N-by-2 rectangles are dihedral transforms of one shared template.
    - Isolated one-pixel boundary markers lie on rays extending outward from active-edge cells, but their colors are reused from the template palette and may repeat.

    Rule:
    - For each sprite, locate the active long edge: the side whose cells have the stronger same-color majority.
    - If the first non-background cell on an outward ray is a boundary marker, that marker labels the active-edge signature `(cell_color, kth occurrence of that color along the edge)` for that sprite.
    - On every sprite, find the active-edge cell with the same signature and draw the marker color outward through background cells until blocked or out of bounds.

    Key insight:
    - A repeated marker color does not mean "all ports of that color"; it means "the kth visible port of that color on the active edge", so correspondence follows edge-sequence position instead of transformed pixel identity.
    """
    g = np.asarray(grid, dtype=int)
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
            assert _is_boundary_cell(r, c, H, W), "markers must be isolated boundary pixels"
            markers[(r, c)] = int(g[r, c])
        else:
            assert area == h * w, "sprite components must be dense rectangles"
            sprites_raw.append({"bbox": (r0, c0, r1, c1), "sub": crop_bbox(g, (r0, c0, r1, c1))})

    assert len(sprites_raw) >= 2, "expected multiple transformed rectangular sprites"
    assert markers, "expected isolated control markers"

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
        info = _analyze_sprite(sprite["sub"])
        if active_len is None:
            active_len = len(info["ordered_active"])
        assert len(info["ordered_active"]) == active_len, "all active edges must have the same length"
        sprites.append({**sprite, **info})

    marker_to_signature: dict[tuple[int, int], tuple[int, int]] = {}
    signature_to_color: dict[tuple[int, int], int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["ordered_active"]:
            marker_pos = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if marker_pos is None:
                continue
            signature = sprite["coord_to_signature"][(lr, lc)]
            marker_color = int(markers[marker_pos])
            assert marker_color == signature[0], "marker color must match the labeled edge color"
            if marker_pos in marker_to_signature:
                assert marker_to_signature[marker_pos] == signature
            if signature in signature_to_color:
                assert signature_to_color[signature] == marker_color
            marker_to_signature[marker_pos] = signature
            signature_to_color[signature] = marker_color

    assert set(marker_to_signature) == set(markers), "every boundary marker must label one active-edge cell"
    assert len(set(marker_to_signature.values())) == len(marker_to_signature), "markers must label distinct edge signatures"

    out = g.copy()
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for signature, color in signature_to_color.items():
            coord = sprite["signature_to_coord"].get(signature)
            assert coord is not None, "every labeled signature must exist on every sprite"
            lr, lc = coord
            assert (lr, lc) in sprite["active"], "mapped port must lie on the active edge"

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


# --------------- generator (combo D4/2 + D14/0) ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 8)
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_CLEARANCE = 2
_BOUNDARY_MARGIN = 1
_PLACEMENT_RETRIES = 120
_GEN_RETRIES = 500


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    assert others, "template must have at least one non-dominant color"
    while len(col) < n:
        col.append(random.choice(others))
    random.shuffle(col)
    return col


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[int, int]]] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)

    dominant = random.choice(template_colors)
    active_majority = random.randint(_ACTIVE_MAJORITY_RANGE[0], min(N - 1, _ACTIVE_MAJORITY_RANGE[1]))
    left_col = _build_column(N, template_colors, active_majority, dominant)
    if left_col == list(reversed(left_col)):
        return None

    left_counts = Counter(left_col)
    repeated_colors = [color for color, count in left_counts.items() if count >= 2]
    if not repeated_colors:
        return None

    max_right_majority = active_majority - 1
    min_right_majority = max(2, N // num_colors)
    if min_right_majority > max_right_majority:
        return None
    right_dominant = random.choice(template_colors)
    right_col = _build_column(N, template_colors, random.randint(min_right_majority, max_right_majority), right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(template.flat)) < num_colors:
        return None

    base_info = _analyze_sprite(template)
    repeated_signatures = [sig for sig in base_info["signatures"] if left_counts[sig[0]] >= 2]
    if len(repeated_signatures) < 2:
        return None

    repeated_color = max(repeated_colors, key=lambda color: (left_counts[color], -left_col.index(color)))
    repeated_occurrences = [sig for sig in base_info["signatures"] if sig[0] == repeated_color]
    if len(repeated_occurrences) < 2:
        return None

    chosen_signatures = [repeated_occurrences[0], repeated_occurrences[-1]]
    other_signatures = [sig for sig in base_info["signatures"] if sig not in chosen_signatures]
    preferred_other = [sig for sig in other_signatures if sig[0] != repeated_color]
    if preferred_other:
        chosen_signatures.append(random.choice(preferred_other))
    elif other_signatures:
        chosen_signatures.append(random.choice(other_signatures))
    else:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors, chosen_signatures


def _deduplicated_variants(template: np.ndarray, ids_base: np.ndarray) -> list[dict]:
    raw = _dihedral_variants_with_ids(template, ids_base)
    seen: list[np.ndarray] = []
    base_active_ids = [int(ids_base[r, 0]) for r in range(ids_base.shape[0])]
    result: list[dict] = []
    for name, sub, ids_arr in raw:
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            info = _analyze_sprite(sub)
        except AssertionError:
            continue
        ordered_ids = [int(ids_arr[r, c]) for r, c in info["ordered_active"]]
        if ordered_ids == base_active_ids:
            order_kind = "preserve"
        elif ordered_ids == list(reversed(base_active_ids)):
            order_kind = "reverse"
        else:
            continue
        seen.append(sub)
        result.append(
            {
                "name": name,
                "sub": sub,
                "ids_arr": ids_arr,
                "order_kind": order_kind,
                **info,
            }
        )
    return result


def _choose_variants(unique_variants: list[dict], num_sprites: int) -> list[dict] | None:
    preserve = [variant for variant in unique_variants if variant["order_kind"] == "preserve"]
    reverse = [variant for variant in unique_variants if variant["order_kind"] == "reverse"]
    if not preserve or not reverse:
        return None
    if len(unique_variants) < num_sprites:
        return None

    chosen = [random.choice(preserve), random.choice(reverse)]
    used_names = {variant["name"] for variant in chosen}
    remaining = [variant for variant in unique_variants if variant["name"] not in used_names]
    need = num_sprites - len(chosen)
    if need > len(remaining):
        return None
    chosen.extend(random.sample(remaining, need))
    random.shuffle(chosen)
    return chosen


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int] | None:
    pad = _PLACEMENT_CLEARANCE
    margin = _BOUNDARY_MARGIN

    if dr == 0 and dc == -1:
        row_lo, row_hi = margin, H - h - margin
        col_lo, col_hi = pad, W - w - margin
    elif dr == 0 and dc == 1:
        row_lo, row_hi = margin, H - h - margin
        col_lo, col_hi = margin, W - w - pad - 1
    elif dr == -1 and dc == 0:
        row_lo, row_hi = pad, H - h - margin
        col_lo, col_hi = margin, W - w - margin
    else:
        row_lo, row_hi = margin, H - h - pad - 1
        col_lo, col_hi = margin, W - w - margin

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _occupied_set(placements: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for placement in placements:
        r0, c0 = placement["r0"], placement["c0"]
        h, w = placement["sub"].shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _place_sprites(H: int, W: int, variants: list[dict]) -> list[dict] | None:
    placements: list[dict] = []
    pad = _PLACEMENT_PAD
    for variant in variants:
        h, w = variant["sub"].shape
        dr, dc = variant["direction"]
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            pos = _random_sprite_position(H, W, h, w, dr, dc)
            if pos is None:
                return None
            r0, c0 = pos
            occ = _occupied_set(placements)
            if any(
                (r0 + r, c0 + c) in occ
                for r in range(-pad, h + pad)
                for c in range(-pad, w + pad)
            ):
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


def _is_isolated_boundary_spot(grid: np.ndarray, r: int, c: int) -> bool:
    H, W = grid.shape
    if not _is_boundary_cell(r, c, H, W):
        return False
    if grid[r, c] != _BG:
        return False
    for rr in range(max(0, r - 1), min(H, r + 2)):
        for cc in range(max(0, c - 1), min(W, c + 2)):
            if (rr, cc) != (r, c) and grid[rr, cc] != _BG:
                return False
    return True


def _place_markers(grid: np.ndarray, placements: list[dict], signatures: list[tuple[int, int]]) -> bool:
    H, W = grid.shape
    sprite_order = list(range(len(placements)))
    for signature in signatures:
        color = signature[0]
        random.shuffle(sprite_order)
        placed = False
        for idx in sprite_order:
            sprite = placements[idx]
            coord = sprite["signature_to_coord"].get(signature)
            if coord is None:
                return False
            lr, lc = coord
            dr, dc = sprite["direction"]
            marker_r, marker_c = _boundary_position(sprite["r0"] + lr, sprite["c0"] + lc, dr, dc, H, W)

            cr, cc = sprite["r0"] + lr + dr, sprite["c0"] + lc + dc
            clear = True
            while (cr, cc) != (marker_r, marker_c) and 0 <= cr < H and 0 <= cc < W:
                if grid[cr, cc] != _BG:
                    clear = False
                    break
                cr += dr
                cc += dc
            if not clear:
                continue

            if not _is_isolated_boundary_spot(grid, marker_r, marker_c):
                continue
            grid[marker_r, marker_c] = color
            placed = True
            break
        if not placed:
            return False
    return True


def _scene_from_grid(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False)
    sprites: list[dict] = []
    markers: dict[tuple[int, int], int] = {}
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            if _is_boundary_cell(r, c, H, W):
                markers[(r, c)] = int(g[r, c])
        else:
            assert area == h * w
            sub = crop_bbox(g, (r0, c0, r1, c1))
            sprites.append({"bbox": (r0, c0, r1, c1), **_analyze_sprite(sub)})
    return sprites, markers


def fusion_diagnostics(grid: np.ndarray) -> dict[str, int]:
    sprites, markers = _scene_from_grid(grid)
    sequences = [
        [int(sprite["sub"][r, c]) for r, c in sprite["ordered_active"]]
        for sprite in sprites
    ]
    base_sequence = sequences[0]
    reversed_sequence = list(reversed(base_sequence))
    preserve_sprites = sum(seq == base_sequence for seq in sequences)
    reverse_sprites = sum(seq == reversed_sequence for seq in sequences)
    assert preserve_sprites + reverse_sprites == len(sprites), "active-edge sequences must match or reverse"

    marker_color_counts = Counter(markers.values())
    duplicate_marker_colors = sum(count - 1 for count in marker_color_counts.values() if count > 1)
    labeled_signatures: set[tuple[int, int]] = set()
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["ordered_active"]:
            marker_pos = _first_marker_on_ray(grid, int(get_bg(grid)), markers, r0 + lr, c0 + lc, dr, dc)
            if marker_pos is not None:
                labeled_signatures.add(sprite["coord_to_signature"][(lr, lc)])

    repeated_signature_labels = sum(1 for _color, occ in labeled_signatures if occ > 1)
    base_signature_to_index = {signature: idx for idx, signature in enumerate(sprites[0]["signatures"])}
    sequence_mapping_divergences = 0
    for sprite, seq in zip(sprites, sequences):
        if seq != reversed_sequence:
            continue
        reverse_signature_to_index = {
            signature: idx for idx, signature in enumerate(sprite["signatures"])
        }
        for signature in labeled_signatures:
            base_idx = base_signature_to_index[signature]
            old_id_following_idx = len(base_sequence) - 1 - base_idx
            new_sequence_idx = reverse_signature_to_index[signature]
            if old_id_following_idx != new_sequence_idx:
                sequence_mapping_divergences += 1

    return {
        "markers": len(markers),
        "duplicate_marker_colors": duplicate_marker_colors,
        "repeated_signature_labels": repeated_signature_labels,
        "sprites": len(sprites),
        "preserve_sprites_seen": preserve_sprites,
        "reverse_sprites_seen": reverse_sprites,
        "sequence_mapping_divergences": sequence_mapping_divergences,
    }


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, _template_colors, signatures = design

    unique_variants = _deduplicated_variants(template, ids_base)
    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    chosen = _choose_variants(unique_variants, num_sprites)
    if chosen is None:
        return None

    placements = _place_sprites(H, W, chosen)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for placement in placements:
        r0, c0 = placement["r0"], placement["c0"]
        h, w = placement["sub"].shape
        grid[r0 : r0 + h, c0 : c0 + w] = placement["sub"]

    if not _place_markers(grid, placements, signatures):
        return None

    try:
        main(grid)
    except Exception:
        return None

    info = fusion_diagnostics(grid)
    if info["duplicate_marker_colors"] < 1:
        return None
    if info["repeated_signature_labels"] < 2:
        return None
    if info["reverse_sprites_seen"] < 1:
        return None
    if info["sequence_mapping_divergences"] < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
