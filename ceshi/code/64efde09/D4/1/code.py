# Variant of 64efde09
# Dimension: D4 — 源对象：标记像素的类型与位置 (src)
# Change: 单像素标记，放置在网格边界上，颜色与模板颜色不同且互不重复，通过射线对齐到精灵活跃边端口... → 标记放置在网格内部而非边界

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the D4 variant
# ----------------------------------
# variant_spec:
#   preserved_core:
#   - The scene still contains 3-4 dense rectangular sprites that are dihedral transforms
#     of one shared N-by-2 template.
#   - The active side of each sprite is still the long edge with the stronger same-color majority.
#   - Three unique single-pixel markers still label three active-edge template ports, and the
#     corresponding colors are then emitted from every transformed sprite's matching port.
#   changed_source_object:
#   - In the original task, each marker sat on the grid boundary. Here each marker must be an
#     interior isolated pixel: not on row 0 / row H-1 / col 0 / col W-1.
#   - The marker remains collinear with one active-edge cell and lies on that cell's outward ray,
#     but there is at least one background cell between the sprite and the marker and at least
#     one background cell between the marker and the boundary. This makes the "floating" marker
#     visibly different from the old boundary-anchor cue.
#   concrete_behavior_trace:
#   - Original local pattern on a left-facing sprite row:
#       boundary marker M . . sprite_edge
#     Solver scans left from sprite_edge and the first non-bg cell is immediately the boundary marker.
#   - New local pattern:
#       boundary . M . sprite_edge
#     Solver must first detect that M is an isolated interior pixel, then determine which active-edge
#     port it labels by tracing the outward ray back toward the sprite. The output still paints rays
#     from every homologous port; on the sprite that owns M, the ray stops when it reaches the marker.
#   complexity_calibration:
#   - Marker colors are still unique and disjoint from template colors, so the new difficulty comes
#     from spatial search rather than from hidden color remapping.
#   - Only the source-object position changes; we keep the number of markers fixed at 3 and preserve
#     the same sprite family so the task remains legible from 2-4 examples.
#   hidden_assumptions_checked:
#   - `_first_marker_on_ray` in the original solver did not truly depend on boundary placement; it
#     only depended on "first non-bg along the outward ray is the marker". Therefore the core solver
#     can stay nearly unchanged.
#   - The original generator *did* depend on boundary placement via `_boundary_position`. That logic
#     must be replaced by explicit interior candidate search.
#   - Because the solver scans every active-edge cell of every sprite, a floating marker must belong
#     to exactly one active-edge ray. The generator enforces unique ray ownership to avoid accidental
#     remapping of a marker color to the wrong template id.
#   placement_legality:
#   - Sprites are placed with extra clearance in the direction their active side faces so every
#     labeled port has room for an interior marker plus visible background before the boundary.
#   - Marker candidates must be background cells, 8-disconnected from all sprites and other markers,
#     and visible from their owning active-edge cell through a clear background corridor.
#   visual_observability:
#   - Every generated marker is strictly interior, and the boundary side beyond it remains visible.
#     This makes the D4 change obvious by inspection: the cue is no longer "color on the frame"
#     but "floating colored dot near a sprite".
#   cleanliness:
#   - We forbid markers from lying on multiple active-edge rays and keep sprite padding, so the
#     canvas does not devolve into overlapping cues or ambiguous occlusions.


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


def _is_interior_cell(r: int, c: int, H: int, W: int) -> bool:
    return 0 < r < H - 1 and 0 < c < W - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate control-ray colors from internally placed marker pixels to every matching sprite port.

    Input structure:
    - bg is the most common color.
    - Several dense 2-by-N or N-by-2 rectangular sprites are dihedral transforms of one template.
    - Isolated single-pixel control markers sit inside the canvas on outward rays from some active-edge cells, not on the boundary.

    Rule:
    - For each sprite, the active edge is the long side whose cells have the strongest same-color majority.
    - The first non-bg cell seen while scanning outward from an active-edge cell labels that template port if the cell is one of the isolated markers.
    - On every transformed sprite, draw that assigned color outward from the corresponding active-edge cell through bg cells until blocked or out of bounds.

    Key insight:
    - The floating marker dots are not absolute targets; they encode template-port identities via ray alignment with the dominant edge, and those identities transfer across every rotation/reflection of the shared sprite.
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
            assert _is_interior_cell(r, c, H, W), "control markers must be interior pixels in this variant"
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


# --------------- generator (att1, D4 interior-marker variant) ---------------

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
_PLACEMENT_CLEARANCE = 4
_MARKER_MIN_STEPS_FROM_EDGE = 2
_MARKER_MIN_STEPS_TO_BOUNDARY = 2
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


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int]:
    pad = _PLACEMENT_CLEARANCE
    if dr == 0 and dc == -1:
        return random.randint(0, H - h), random.randint(pad, W - w)
    if dr == 0 and dc == 1:
        return random.randint(0, H - h), random.randint(0, W - w - pad)
    if dr == -1 and dc == 0:
        return random.randint(pad, H - h), random.randint(0, W - w)
    return random.randint(0, H - h - pad), random.randint(0, W - w)


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
            r0, c0 = _random_sprite_position(H, W, h, w, *direction)
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


def _ray_cells(r: int, c: int, dr: int, dc: int, H: int, W: int) -> list[tuple[int, int]]:
    cells = []
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _clear_segment_to_marker(
    grid: np.ndarray, r: int, c: int, dr: int, dc: int, marker: tuple[int, int]
) -> bool:
    mr, mc = marker
    r += dr
    c += dc
    while (r, c) != (mr, mc):
        if int(grid[r, c]) != _BG:
            return False
        r += dr
        c += dc
    return int(grid[mr, mc]) == _BG


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


def _port_records(sprite_placements: list[tuple], H: int, W: int) -> list[dict]:
    ports: list[dict] = []
    for si, placement in enumerate(sprite_placements):
        r0, c0, _sub, ids_arr, direction, active_coords = placement
        dr, dc = direction
        active_set = set(active_coords)
        for lr, lc in active_set:
            rr = r0 + lr
            cc = c0 + lc
            ports.append(
                {
                    "sprite_index": si,
                    "tid": int(ids_arr[lr, lc]),
                    "origin": (rr, cc),
                    "direction": direction,
                    "ray": _ray_cells(rr, cc, dr, dc, H, W),
                }
            )
    return ports


def _place_markers(
    grid: np.ndarray, sprite_placements: list[tuple], ids_base: np.ndarray, template_colors: list[int]
) -> bool:
    H, W = grid.shape
    N = ids_base.shape[0]

    active_template_ids = [int(ids_base[r, 0]) for r in range(N)]
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    occupied = _occupied_set(sprite_placements)
    ports = _port_records(sprite_placements, H, W)
    ray_owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for port in ports:
        owner = (port["sprite_index"], port["tid"])
        for cell in port["ray"]:
            ray_owners.setdefault(cell, []).append(owner)

    placed_markers: set[tuple[int, int]] = set()
    for color, tid in zip(marker_colors, labeled_ids):
        candidates: list[tuple[int, int]] = []
        for port in ports:
            if port["tid"] != tid:
                continue
            si = port["sprite_index"]
            rr, cc = port["origin"]
            dr, dc = port["direction"]
            ray = port["ray"]
            for step, cell in enumerate(ray, start=1):
                remaining = len(ray) - step
                if step < _MARKER_MIN_STEPS_FROM_EDGE:
                    continue
                if remaining < _MARKER_MIN_STEPS_TO_BOUNDARY:
                    continue
                if ray_owners.get(cell) != [(si, tid)]:
                    continue
                if not _marker_is_isolated(cell, occupied, placed_markers, H, W):
                    continue
                if not _clear_segment_to_marker(grid, rr, cc, dr, dc, cell):
                    continue
                candidates.append(cell)
        if not candidates:
            return False

        marker_r, marker_c = random.choice(candidates)
        grid[marker_r, marker_c] = color
        placed_markers.add((marker_r, marker_c))
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
