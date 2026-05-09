# Combo variant of 64efde09 (index 22)
# combo_spec: D2/3+D4/1
# Name: 质心射线+内部标记

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# variant_spec:
#   preserved_core:
#   - The scene still contains 3-4 dense rectangular sprites that are dihedral transforms
#     of one shared non-square template.
#   - The active side is still the long edge with the stronger same-color majority.
#   - Three unique marker colors still label three template-specific control identities, and
#     those identities are transferred to every transformed sprite.
#   fused_change:
#   - D4/1 moves the single-pixel markers off the boundary and into the interior.
#   - D2/3 breaks the original "edge port -> outward normal ray" chain. Instead of launching
#     from arbitrary active-edge cells, each sprite now launches from a discrete hub chosen next
#     to the geometric centroid (the true centroid falls between cells because the rectangles are
#     2-by-N or N-by-2). The three controllable template ports are the structurally distinguished
#     top/middle/bottom cells (or left/middle/right after rotation). Their colors radiate from
#     that centroid-adjacent hub along three cardinal directions.
#   concrete_behavior_trace:
#   - Original left-facing vertical sprite:
#       marker on boundary labels one chosen row of the left edge, then every copy emits leftward
#       from that exact edge cell.
#   - New left-facing vertical sprite:
#       the three legal control ports are left-edge top endpoint, left-edge midpoint, left-edge
#       bottom endpoint. Markers float on interior cells located respectively above, left of, and
#       below a central hub just inside the rectangle near the sprite centroid. Output rays also
#       start from that hub path: upward color corresponds to the top endpoint id, leftward color
#       to the midpoint id, downward color to the bottom endpoint id.
#   - Difference:
#       the label is no longer "which boundary-facing row got tagged". It is "which structural
#       control port of the shared template got tagged", and the visible emission starts from the
#       sprite center region rather than from the outer edge.
#   complexity_calibration:
#   - Restricting control ports to exactly three structural positions prevents centroid-launched
#     directions from collapsing into ambiguous same-direction aliases.
#   - Marker colors remain unique and disjoint from template colors, so the new difficulty comes
#     from changed geometry, not from hidden color remapping.
#   - The same three structural ports repeat on every transformed sprite, giving enough visual
#     support across 2-4 examples for a human to infer the rule.
#   hidden_assumptions_checked:
#   - The original solver assumed markers could be recovered by scanning outward from every
#     active-edge cell. This is false now because the launch origin moved to a centroid-adjacent
#     hub and the controlled ports are only three special cells, not all N edge cells.
#   - The original solver also assumed the first non-bg beyond an edge cell was the controlling
#     marker. Now the scan must begin at the hub, traverse through the sprite body if needed, exit
#     the rectangle, and only then look for an isolated marker in the interior.
#   - The original generator assumed boundary placement via a boundary-coordinate helper. That is
#     invalid here; candidate markers must instead be searched along interior hub rays with
#     explicit distance-to-sprite and distance-to-boundary constraints.
#   - Because the new rays can be tangent to the long edge, ray intersections are more likely than
#     in the base task. The generator therefore rejects placements where different template ids
#     would ever paint the same background cell.
#   placement_legality:
#   - Sprites are placed with padding from each other and with at least four cells of margin to
#     every boundary, ensuring that all three control rays have room for an interior marker.
#   - Marker cells must be background, strictly interior, isolated from sprites/markers in their
#     8-neighborhood, and uniquely owned by one hub ray.
#   visual_observability:
#   - The three rays on a sprite now form a center-anchored fan/T shape instead of three parallel
#     edge-normal beams. That makes the D2 change visible immediately.
#   - Markers are visibly floating inside the canvas, so the D4 change is also obvious.
#   cleanliness:
#   - Only three labels exist, rays are cardinal, and conflicting cross-color ray overlaps are
#     rejected. This keeps the canvas readable instead of turning into dense spaghetti.
#   inferability:
#   - No rule depends on raw color-number ordering.
#   - The three marker colors are tied to repeated structural ports, and every sample shows the
#     same "center-hub launch" geometry under several rotations/reflections.


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


def _is_interior_cell(r: int, c: int, H: int, W: int) -> bool:
    return 0 < r < H - 1 and 0 < c < W - 1


def _hub_local(shape: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    dr, dc = direction
    if dr == 0:
        assert h % 2 == 1
        return h // 2, w - 1 if dc == -1 else 0
    assert w % 2 == 1
    return h - 1 if dr == -1 else 0, w // 2


def _control_specs(ids_arr: np.ndarray, direction: tuple[int, int]) -> list[dict]:
    h, w = ids_arr.shape
    if direction == (0, -1):
        mid = h // 2
        return [
            {"template_id": int(ids_arr[0, 0]), "dir": (-1, 0), "port": (0, 0)},
            {"template_id": int(ids_arr[mid, 0]), "dir": (0, -1), "port": (mid, 0)},
            {"template_id": int(ids_arr[h - 1, 0]), "dir": (1, 0), "port": (h - 1, 0)},
        ]
    if direction == (0, 1):
        mid = h // 2
        return [
            {"template_id": int(ids_arr[0, w - 1]), "dir": (-1, 0), "port": (0, w - 1)},
            {"template_id": int(ids_arr[mid, w - 1]), "dir": (0, 1), "port": (mid, w - 1)},
            {"template_id": int(ids_arr[h - 1, w - 1]), "dir": (1, 0), "port": (h - 1, w - 1)},
        ]
    if direction == (-1, 0):
        mid = w // 2
        return [
            {"template_id": int(ids_arr[0, 0]), "dir": (0, -1), "port": (0, 0)},
            {"template_id": int(ids_arr[0, mid]), "dir": (-1, 0), "port": (0, mid)},
            {"template_id": int(ids_arr[0, w - 1]), "dir": (0, 1), "port": (0, w - 1)},
        ]
    assert direction == (1, 0)
    mid = w // 2
    return [
        {"template_id": int(ids_arr[h - 1, 0]), "dir": (0, -1), "port": (h - 1, 0)},
        {"template_id": int(ids_arr[h - 1, mid]), "dir": (1, 0), "port": (h - 1, mid)},
        {"template_id": int(ids_arr[h - 1, w - 1]), "dir": (0, 1), "port": (h - 1, w - 1)},
    ]


def _inside_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _outside_ray_cells(
    bbox: tuple[int, int, int, int],
    hub_abs: tuple[int, int],
    step: tuple[int, int],
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    r, c = hub_abs
    dr, dc = step
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W and _inside_bbox((r, c), bbox):
        r += dr
        c += dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < H and 0 <= c < W:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _first_marker_on_control_ray(
    g: np.ndarray,
    bg: int,
    markers: dict[tuple[int, int], int],
    bbox: tuple[int, int, int, int],
    hub_abs: tuple[int, int],
    step: tuple[int, int],
) -> int | None:
    H, W = g.shape
    for cell in _outside_ray_cells(bbox, hub_abs, step, H, W):
        if int(g[cell]) != bg:
            return markers.get(cell)
    return None


def _paint_control_ray(
    out: np.ndarray,
    g: np.ndarray,
    bg: int,
    bbox: tuple[int, int, int, int],
    hub_abs: tuple[int, int],
    step: tuple[int, int],
    color: int,
) -> None:
    H, W = g.shape
    for r, c in _outside_ray_cells(bbox, hub_abs, step, H, W):
        if int(g[r, c]) != bg:
            break
        assert int(out[r, c]) in (bg, color), "control rays must not conflict"
        out[r, c] = color


def _extract_scene(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict], dict[tuple[int, int], int]]:
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
            assert _is_interior_cell(r, c, H, W), "control markers must be interior pixels"
            markers[(r, c)] = int(g[r, c])
        else:
            assert area == h * w, "sprite components must be dense rectangles"
            sprites_raw.append({"bbox": (r0, c0, r1, c1), "sub": crop_bbox(g, (r0, c0, r1, c1))})

    assert len(sprites_raw) >= 2, "expected multiple transformed rectangular sprites"
    assert markers, "expected isolated control markers"
    assert len(set(markers.values())) == len(markers), "control marker colors must be unique"

    variants = _dihedral_variants(sprites_raw[0]["sub"])
    sprites: list[dict] = []
    for sprite in sprites_raw:
        matches = [
            ids
            for _name, sub, ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "each sprite must match the template in exactly one orientation"
        ids_arr = matches[0]
        direction, _active = _active_side(sprite["sub"])
        hub = _hub_local(sprite["sub"].shape, direction)
        controls = _control_specs(ids_arr, direction)
        r0, c0, _r1, _c1 = sprite["bbox"]
        sprites.append(
            {
                **sprite,
                "ids": ids_arr,
                "direction": direction,
                "hub_local": hub,
                "hub_abs": (r0 + hub[0], c0 + hub[1]),
                "controls": controls,
            }
        )

    legal_tids = {control["template_id"] for control in sprites[0]["controls"]}
    assert len(legal_tids) == 3, "expected exactly three structural control ports"
    for sprite in sprites[1:]:
        assert {control["template_id"] for control in sprite["controls"]} == legal_tids

    return g, bg, sprites, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read internal marker colors as labels for three structural ports of a shared sprite template, then emit those colors from each sprite's centroid-adjacent hub.

    Input structure:
    - bg is the most common color.
    - Several dense 2-by-N or N-by-2 rectangular sprites are dihedral transforms of one template.
    - Three isolated single-pixel markers sit strictly inside the canvas on clear cardinal rays extending away from a sprite's centroid-side hub.

    Rule:
    - For each sprite, the active side is the long edge whose cells have the strongest same-color majority.
    - Only three template ports are controllable: the two endpoints of the active edge and its midpoint. Their rays are anchored at a discrete hub chosen adjacent to the geometric centroid and head in the three cardinal directions that reach those structural ports.
    - If the first non-bg cell beyond a hub ray is an isolated marker, that marker color labels the corresponding template port. Then every transformed sprite emits that color outward from its own matching centroid-side hub ray through bg cells until blocked or out of bounds.

    Key insight:
    - The floating markers no longer tag arbitrary boundary-facing edge rows. They identify three repeated structural ports of the shared template, and those labels follow the sprite under rotation/reflection while the visible rays launch from the center region rather than from the outer edge.
    """
    g, bg, sprites, markers = _extract_scene(grid)

    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    for sprite in sprites:
        for control in sprite["controls"]:
            marker_color = _first_marker_on_control_ray(
                g,
                bg,
                markers,
                sprite["bbox"],
                sprite["hub_abs"],
                control["dir"],
            )
            if marker_color is None:
                continue
            template_id = int(control["template_id"])
            if marker_color in color_to_template_id:
                assert color_to_template_id[marker_color] == template_id
            if template_id in template_id_to_color:
                assert template_id_to_color[template_id] == marker_color
            color_to_template_id[marker_color] = template_id
            template_id_to_color[template_id] = marker_color

    assert set(color_to_template_id) == set(markers.values()), "every control marker must label one structural port"

    out = g.copy()
    for sprite in sprites:
        for control in sprite["controls"]:
            color = template_id_to_color.get(int(control["template_id"]))
            if color is None:
                continue
            _paint_control_ray(out, g, bg, sprite["bbox"], sprite["hub_abs"], control["dir"], color)

    return out


# --------------- generator (combo variant) ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (23, 29)
_GRID_W_RANGE = (28, 29)
_TEMPLATE_N_CHOICES = [7, 9]
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_MARGIN = 4
_MARKER_MIN_STEPS_FROM_SPRITE = 2
_MARKER_MIN_STEPS_TO_BOUNDARY = 2
_PLACEMENT_RETRIES = 120
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
    if not [c for c in template_colors if c != right_dominant]:
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
    template: np.ndarray, ids_base: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    raw = _dihedral_variants(template)
    raw_ids = _dihedral_variants(ids_base)

    seen: list[np.ndarray] = []
    result = []
    for (_, sub, _), (_, _, ids_arr) in zip(raw, raw_ids):
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            direction, _active = _active_side(sub)
        except AssertionError:
            continue
        seen.append(sub)
        result.append((sub, ids_arr, direction))
    return result


def _random_sprite_position(H: int, W: int, h: int, w: int) -> tuple[int, int]:
    return (
        random.randint(_PLACEMENT_MARGIN, H - h - _PLACEMENT_MARGIN),
        random.randint(_PLACEMENT_MARGIN, W - w - _PLACEMENT_MARGIN),
    )


def _occupied_set(placements: list[tuple]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _control_records(sprite_placements: list[tuple], H: int, W: int) -> list[dict]:
    records: list[dict] = []
    for si, placement in enumerate(sprite_placements):
        r0, c0, sub, ids_arr, direction = placement
        bbox = (r0, c0, r0 + sub.shape[0] - 1, c0 + sub.shape[1] - 1)
        hub_local = _hub_local(sub.shape, direction)
        hub_abs = (r0 + hub_local[0], c0 + hub_local[1])
        for control in _control_specs(ids_arr, direction):
            records.append(
                {
                    "sprite_index": si,
                    "template_id": int(control["template_id"]),
                    "dir": control["dir"],
                    "bbox": bbox,
                    "hub_abs": hub_abs,
                    "outside_ray": _outside_ray_cells(bbox, hub_abs, control["dir"], H, W),
                }
            )
    return records


def _ray_layout_ok(sprite_placements: list[tuple], H: int, W: int) -> bool:
    occupied = _occupied_set(sprite_placements)
    seen_by_cell: dict[tuple[int, int], int] = {}
    for record in _control_records(sprite_placements, H, W):
        tid = record["template_id"]
        for cell in record["outside_ray"]:
            if cell in occupied:
                return False
            other = seen_by_cell.get(cell)
            if other is not None and other != tid:
                return False
            seen_by_cell[cell] = tid
    return True


def _place_sprites(H: int, W: int, variants: list[tuple]) -> list[tuple] | None:
    placements: list[tuple] = []
    pad = _PLACEMENT_PAD
    for sub, ids_arr, direction in variants:
        h, w = sub.shape
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w)
            occ = _occupied_set(placements)
            blocked = False
            for r in range(-pad, h + pad):
                for c in range(-pad, w + pad):
                    if (r0 + r, c0 + c) in occ:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue
            candidate = placements + [(r0, c0, sub, ids_arr, direction)]
            if not _ray_layout_ok(candidate, H, W):
                continue
            placements.append((r0, c0, sub, ids_arr, direction))
            placed = True
            break
        if not placed:
            return None
    return placements


def _marker_is_isolated(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    markers: set[tuple[int, int]],
    H: int,
    W: int,
) -> bool:
    r, c = cell
    if not _is_interior_cell(r, c, H, W):
        return False
    if cell in occupied or cell in markers:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) in occupied or (rr, cc) in markers:
                return False
    return True


def _control_template_ids(ids_base: np.ndarray) -> list[int]:
    return [int(spec["template_id"]) for spec in _control_specs(ids_base, (0, -1))]


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    control_template_ids: list[int],
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    occupied = _occupied_set(sprite_placements)
    records = _control_records(sprite_placements, H, W)
    ray_owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for record in records:
        owner = (record["sprite_index"], record["template_id"])
        for cell in record["outside_ray"]:
            ray_owners.setdefault(cell, []).append(owner)

    placed_markers: set[tuple[int, int]] = set()
    for color, tid in zip(marker_colors, control_template_ids):
        candidates: list[tuple[int, int]] = []
        for record in records:
            if record["template_id"] != tid:
                continue
            owner = (record["sprite_index"], tid)
            ray = record["outside_ray"]
            for idx, cell in enumerate(ray, start=1):
                remaining = len(ray) - idx
                if idx < _MARKER_MIN_STEPS_FROM_SPRITE:
                    continue
                if remaining < _MARKER_MIN_STEPS_TO_BOUNDARY:
                    continue
                if ray_owners.get(cell) != [owner]:
                    continue
                if not _marker_is_isolated(cell, occupied, placed_markers, H, W):
                    continue
                candidates.append(cell)
        if not candidates:
            return False
        marker_r, marker_c = random.choice(candidates)
        grid[marker_r, marker_c] = color
        placed_markers.add((marker_r, marker_c))
    return True


def _choose_variants(unique_variants: list[tuple]) -> list[tuple] | None:
    if len(unique_variants) < 3:
        return None
    target = 4 if len(unique_variants) >= 4 and random.random() < 0.4 else 3
    if len(unique_variants) < target:
        target = 3

    by_axis = {
        "vertical": [v for v in unique_variants if v[2][0] == 0],
        "horizontal": [v for v in unique_variants if v[2][1] == 0],
    }
    if not by_axis["vertical"] or not by_axis["horizontal"]:
        return random.sample(unique_variants, target)

    random.shuffle(by_axis["vertical"])
    random.shuffle(by_axis["horizontal"])
    chosen: list[tuple] = [by_axis["vertical"].pop(), by_axis["horizontal"].pop()]

    remaining = [
        v
        for v in unique_variants
        if not any(np.array_equal(v[0], picked[0]) and v[2] == picked[2] for picked in chosen)
    ]
    direction_buckets: dict[tuple[int, int], list[tuple]] = {}
    for variant in remaining:
        direction_buckets.setdefault(variant[2], []).append(variant)
    for bucket in direction_buckets.values():
        random.shuffle(bucket)

    preferred = [
        bucket.pop()
        for direction, bucket in direction_buckets.items()
        if bucket and direction not in {picked[2] for picked in chosen}
    ]
    random.shuffle(preferred)
    for variant in preferred:
        if len(chosen) == target:
            break
        chosen.append(variant)

    if len(chosen) < target:
        leftovers = [
            v
            for v in unique_variants
            if not any(np.array_equal(v[0], picked[0]) and v[2] == picked[2] for picked in chosen)
        ]
        random.shuffle(leftovers)
        chosen.extend(leftovers[: target - len(chosen)])

    return chosen if len(chosen) == target else None


def _gen_attempt() -> np.ndarray | None:
    N = random.choice(_TEMPLATE_N_CHOICES)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors = design

    unique_variants = _deduplicated_variants(template, ids_base)
    chosen = _choose_variants(unique_variants)
    if chosen is None:
        return None

    placements = _place_sprites(H, W, chosen)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        grid[r0 : r0 + h, c0 : c0 + w] = sub

    control_template_ids = _control_template_ids(ids_base)
    if len(control_template_ids) != _NUM_MARKERS:
        return None
    if not _place_markers(grid, placements, control_template_ids, template_colors):
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
