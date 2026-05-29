# Combo variant of 64efde09 (index 15)
# combo_spec: D4/3+D12/3
# Name: 非活跃边标记+双活跃边射线

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg

"""
design_analysis:
  concretization:
    original:
      Markers sat on the same outward normal used for the emitted rays, and only
      one long edge of each thin rectangle was considered active.
    fused_variant:
      Every sprite is still a dense 2xN / Nx2 dihedral transform of one shared
      template, but both long edges now emit. Markers no longer sit on either
      emitting side; instead they appear 2-4 cells away on a short-edge ray.
      Because a thin rectangle has short edges of length 2, the only directly
      labelable cells from those rays are the four corner ports. Each labeled
      corner belongs to exactly one long edge, so the marker is discovered from a
      non-active side and then transferred to the corresponding long-edge family.
  behavior_trace:
    example:
      Consider a vertical Nx2 sprite. A marker above the top-right corner is
      found by scanning upward from the top short edge, so it labels template id
      ids[0,1]. In the output that color does not continue upward; it fires
      sideways from the right long edge of every transformed sprite. A second
      marker might sit below the bottom-left corner and therefore label ids[N-1,0],
      which later emits from the left long edge.
    delta_from_original:
      The cue direction and the emission direction are now orthogonal, and the
      output can contain rays leaving both long sides of the same sprite family.
  complexity_calibration:
    kept:
      Grid scale, thin-rectangle template family, 3 unique control colors, and
      dihedral-template matching remain unchanged so the task still reads like
      the same ARC family.
    controlled_change:
      Restricting labels to the 4 corner ports prevents the fused rule from
      becoming underdetermined after D12 makes both long edges active. The human
      still sees repeated evidence: markers consistently live on short-edge rays,
      while outputs consistently depart from the long side owning that corner.
  implicit_assumption_audit:
    original_assumptions_broken:
      - "scan the same side you later emit from" is false.
      - "there is exactly one active long edge" is false.
      - "a labeled port is any cell on one winning long edge" is false; only
        corner ports are discoverable from short-edge scans.
    new_main_assumptions:
      - every non-singleton component is a dense non-square rectangle
      - every sprite matches the shared template in exactly one dihedral variant
      - every marker is the first non-bg cell on exactly one intended short-edge ray
      - every labeled template id lands on exactly one long edge in each sprite
    generator_guarantees:
      - reject symmetric templates so orientation matching stays unique
      - reserve clearance on both long sides for visible bidirectional rays
      - reserve clearance on short sides for interior markers
      - keep markers off all future emission lanes for the chosen labeled ports
  placement_legality:
    sprites:
      Sprites are placed with padding between rectangles, enough room on both
      long sides for emitted rays, and enough room on both short sides for 2-4
      cell marker offsets.
    markers:
      Markers are isolated single pixels on bg-only short-edge corridors. A
      candidate is rejected if another short-edge port would see it first.
  observability:
    human_view:
      Across examples the visual pattern is stable: tiny colored pixels hover
      above/below or left/right of rectangle endpoints, but the resulting color
      streaks leave from the rectangle's long sides instead. Since generation
      forces labels from both canonical long-edge families, opposite-direction
      propagation is visible rather than merely possible.
  cleanliness:
    safeguards:
      - markers are never placed on emitted-ray lanes for labeled ports
      - 8-neighborhood isolation keeps control pixels visually distinct
      - solver validation remains part of rejection sampling
  non_degeneracy_check_plan:
    metrics:
      - each sample uses markers only on short-edge rays, never on long-edge scans
      - labeled ports cover both canonical long-edge families
      - the original / D12-only long-edge marker logic should fail or differ on
        generated samples, confirming the fusion is not collapsing back
"""


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


def _long_sides(sub: np.ndarray) -> list[dict]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"
    if h > w:
        raw = [
            ((0, -1), [(r, 0) for r in range(h)]),
            ((0, 1), [(r, w - 1) for r in range(h)]),
        ]
    else:
        raw = [
            ((-1, 0), [(0, c) for c in range(w)]),
            ((1, 0), [(h - 1, c) for c in range(w)]),
        ]
    return [{"direction": direction, "coords": coords, "coord_set": set(coords)} for direction, coords in raw]


def _short_sides(sub: np.ndarray) -> list[dict]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"
    if h > w:
        raw = [
            ((-1, 0), [(0, c) for c in range(w)]),
            ((1, 0), [(h - 1, c) for c in range(w)]),
        ]
    else:
        raw = [
            ((0, -1), [(r, 0) for r in range(h)]),
            ((0, 1), [(r, w - 1) for r in range(h)]),
        ]
    return [{"direction": direction, "coords": coords, "coord_set": set(coords)} for direction, coords in raw]


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
    Read corner labels from non-emitting short-edge rays, then project those labels from both long edges of every transformed sprite.

    Input structure:
    - bg is the most common color.
    - Several dense 2-by-N or N-by-2 rectangles are dihedral transforms of one shared template.
    - Isolated single-pixel markers of unique non-template colors sit 2-4 cells away on rays extending from short edges, so each marker aligns with one corner port rather than with an emitting side.

    Rule:
    - Treat both long edges of every sprite as active output sides.
    - Scan outward from each short-edge cell; if the first non-bg cell on that short-edge ray is a marker, that marker color labels the shared template corner at that exact transformed location.
    - For every labeled template corner, find the corresponding corner in every transformed sprite and emit that color outward along the long edge that owns that corner, continuing through bg cells until blocked or out of bounds.

    Key insight:
    - The marker does not sit where the output ray will go. It is discovered through a short-edge cue, but the labeled corner simultaneously belongs to one long-edge family, so the color must be transferred through the shared template orientation before the bidirectional long-edge rays can be drawn.
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
        long_sides = _long_sides(sprite["sub"])
        short_sides = _short_sides(sprite["sub"])
        long_dir_by_coord = {}
        for side in long_sides:
            for coord in side["coords"]:
                long_dir_by_coord[coord] = side["direction"]
        sprites.append(
            {
                **sprite,
                "ids": ids,
                "long_sides": long_sides,
                "short_sides": short_sides,
                "long_dir_by_coord": long_dir_by_coord,
            }
        )

    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        for side in sprite["short_sides"]:
            dr, dc = side["direction"]
            for lr, lc in side["coords"]:
                marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
                if marker_color is None:
                    continue
                assert (lr, lc) in sprite["long_dir_by_coord"], "short-edge marker must hit a corner port"
                template_id = int(sprite["ids"][lr, lc])
                if marker_color in color_to_template_id:
                    assert color_to_template_id[marker_color] == template_id
                if template_id in template_id_to_color:
                    assert template_id_to_color[template_id] == marker_color
                color_to_template_id[marker_color] = template_id
                template_id_to_color[template_id] = marker_color

    assert set(color_to_template_id) == set(markers.values()), "every control marker must label one short-edge corner"

    out = g.copy()
    H, W = g.shape
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        for color, template_id in color_to_template_id.items():
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            assert (lr, lc) in sprite["long_dir_by_coord"], "mapped corner must belong to one long edge"
            dr, dc = sprite["long_dir_by_coord"][(lr, lc)]

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
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_LONG_RAY_MARGIN = 2
_MARKER_GAP_RANGE = (2, 4)
_PLACEMENT_RETRIES = 120
_GEN_RETRIES = 700


def _random_column(n: int, colors: list[int]) -> list[int]:
    return [random.choice(colors) for _ in range(n)]


def _unique_variants_with_ports(template: np.ndarray, ids_base: np.ndarray) -> list[dict]:
    left_ids = {int(ids_base[r, 0]) for r in range(ids_base.shape[0])}
    right_ids = {int(ids_base[r, 1]) for r in range(ids_base.shape[0])}
    raw = _dihedral_variants(template)
    raw_ids = _dihedral_variants(ids_base)

    seen: list[np.ndarray] = []
    result: list[dict] = []
    for (_, sub, _), (_, _, ids_arr) in zip(raw, raw_ids):
        if any(prev.shape == sub.shape and np.array_equal(prev, sub) for prev in seen):
            continue

        long_sides = _long_sides(sub)
        short_sides = _short_sides(sub)
        long_dir_by_coord: dict[tuple[int, int], tuple[int, int]] = {}
        long_side_by_coord: dict[tuple[int, int], dict] = {}
        annotated_long_sides = []
        for side in long_sides:
            template_ids = [int(ids_arr[r, c]) for r, c in side["coords"]]
            tid_set = set(template_ids)
            if tid_set == left_ids:
                canonical_side = 0
            elif tid_set == right_ids:
                canonical_side = 1
            else:
                raise AssertionError("long edge must correspond to one canonical template column")
            side = {**side, "template_ids": template_ids, "canonical_side": canonical_side}
            annotated_long_sides.append(side)
            for coord in side["coords"]:
                long_dir_by_coord[coord] = side["direction"]
                long_side_by_coord[coord] = side

        ports: dict[int, dict] = {}
        for short_side in short_sides:
            for coord in short_side["coords"]:
                assert coord in long_side_by_coord, "short-edge cell must be a corner port"
                lr, lc = coord
                tid = int(ids_arr[lr, lc])
                ports[tid] = {
                    "coord": coord,
                    "short_direction": short_side["direction"],
                    "long_direction": long_side_by_coord[coord]["direction"],
                    "canonical_side": long_side_by_coord[coord]["canonical_side"],
                }

        if len(ports) != 4:
            raise AssertionError("expected exactly four corner ports")

        seen.append(sub.copy())
        result.append(
            {
                "sub": sub,
                "ids": ids_arr,
                "long_sides": annotated_long_sides,
                "short_sides": short_sides,
                "ports": ports,
            }
        )
    return result


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int], list[dict]] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)

    left_col = _random_column(N, template_colors)
    right_col = _random_column(N, template_colors)

    if len(set(left_col)) < 2 or len(set(right_col)) < 2:
        return None
    if left_col == right_col or left_col == list(reversed(right_col)):
        return None
    if left_col[0] == left_col[-1] and right_col[0] == right_col[-1]:
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None
    if sum(int(left_col[r] != right_col[r]) for r in range(N)) < 2:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    variants = _unique_variants_with_ports(template, ids_base)
    if len(variants) < 8:
        return None

    return template, ids_base, template_colors, variants


def _random_sprite_position(H: int, W: int, h: int, w: int) -> tuple[int, int]:
    short_margin = _MARKER_GAP_RANGE[1] + 1
    if h > w:
        row_lo = short_margin
        row_hi = H - h - short_margin
        col_lo = _LONG_RAY_MARGIN
        col_hi = W - w - _LONG_RAY_MARGIN
    else:
        row_lo = _LONG_RAY_MARGIN
        row_hi = H - h - _LONG_RAY_MARGIN
        col_lo = short_margin
        col_hi = W - w - short_margin
    assert row_lo <= row_hi and col_lo <= col_hi, "grid must allow required margins"
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
        sub = variant["sub"]
        h, w = sub.shape
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w)
            occ = _occupied_set(placements)
            if any((r0 + r, c0 + c) in occ for r in range(-pad, h + pad) for c in range(-pad, w + pad)):
                continue
            placements.append({**variant, "r0": r0, "c0": c0})
            placed = True
            break
        if not placed:
            return None
    return placements


def _trace_from_port(
    grid: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    H, W = grid.shape
    cells = []
    r = start_r + dr
    c = start_c + dc
    while 0 <= r < H and 0 <= c < W:
        if int(grid[r, c]) != _BG:
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _forbidden_long_ray_cells(grid: np.ndarray, placements: list[dict]) -> set[tuple[int, int]]:
    forbidden: set[tuple[int, int]] = set()
    for sprite in placements:
        r0, c0 = sprite["r0"], sprite["c0"]
        for side in _long_sides(sprite["sub"]):
            dr, dc = side["direction"]
            for lr, lc in side["coords"]:
                forbidden.update(_trace_from_port(grid, r0 + lr, c0 + lc, dr, dc))
    return forbidden


def _marker_candidates(
    base_r: int,
    base_c: int,
    dr: int,
    dc: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    cells = []
    for gap in range(_MARKER_GAP_RANGE[0], _MARKER_GAP_RANGE[1] + 1):
        r = base_r + dr * gap
        c = base_c + dc * gap
        if 0 < r < H - 1 and 0 < c < W - 1:
            cells.append((r, c))
    return cells


def _ray_clear(grid: np.ndarray, start_r: int, start_c: int, end_r: int, end_c: int, dr: int, dc: int) -> bool:
    r = start_r + dr
    c = start_c + dc
    while (r, c) != (end_r, end_c):
        if int(grid[r, c]) != _BG:
            return False
        r += dr
        c += dc
    return int(grid[end_r, end_c]) == _BG


def _isolated_marker_cell(grid: np.ndarray, r: int, c: int) -> bool:
    H, W = grid.shape
    if int(grid[r, c]) != _BG:
        return False
    for rr in range(max(0, r - 1), min(H, r + 2)):
        for cc in range(max(0, c - 1), min(W, c + 2)):
            if (rr, cc) != (r, c) and int(grid[rr, cc]) != _BG:
                return False
    return True


def _marker_seen_by_other_port(
    grid: np.ndarray,
    placements: list[dict],
    marker_r: int,
    marker_c: int,
    host_index: int,
    host_tid: int,
) -> bool:
    for si, sprite in enumerate(placements):
        r0, c0 = sprite["r0"], sprite["c0"]
        for tid, port in sprite["ports"].items():
            if si == host_index and tid == host_tid:
                continue
            pr, pc = port["coord"]
            dr, dc = port["short_direction"]
            cr = r0 + pr + dr
            cc = c0 + pc + dc
            while 0 <= cr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                if (cr, cc) == (marker_r, marker_c):
                    return True
                if int(grid[cr, cc]) != _BG:
                    break
                cr += dr
                cc += dc
    return False


def _sample_exhibits_fusion(grid: np.ndarray, out: np.ndarray) -> bool:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False)

    markers: dict[tuple[int, int], int] = {}
    sprites: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            markers[(r, c)] = int(grid[r, c])
        else:
            sprites.append({"bbox": (r0, c0, r1, c1), "sub": crop_bbox(grid, (r0, c0, r1, c1))})

    seen_on_short: set[int] = set()
    seen_on_long: set[int] = set()
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        for side in _short_sides(sprite["sub"]):
            dr, dc = side["direction"]
            for lr, lc in side["coords"]:
                marker_color = _first_marker_on_ray(grid, bg, markers, r0 + lr, c0 + lc, dr, dc)
                if marker_color is not None:
                    seen_on_short.add(marker_color)
        long_hits = 0
        for side in _long_sides(sprite["sub"]):
            dr, dc = side["direction"]
            for lr, lc in side["coords"]:
                marker_color = _first_marker_on_ray(grid, bg, markers, r0 + lr, c0 + lc, dr, dc)
                if marker_color is not None:
                    seen_on_long.add(marker_color)
                rr = r0 + lr + dr
                cc = c0 + lc + dc
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and int(out[rr, cc]) != int(grid[rr, cc]):
                    long_hits += 1
                    break
        if long_hits != 2:
            return False

    return seen_on_short == set(markers.values()) and not seen_on_long


def _choose_labeled_ids(ids_base: np.ndarray) -> list[int]:
    N = ids_base.shape[0]
    left_ids = [int(ids_base[0, 0]), int(ids_base[N - 1, 0])]
    right_ids = [int(ids_base[0, 1]), int(ids_base[N - 1, 1])]
    chosen = [random.choice(left_ids), random.choice(right_ids)]
    remaining = [tid for tid in left_ids + right_ids if tid not in chosen]
    chosen.append(random.choice(remaining))
    random.shuffle(chosen)
    return chosen


def _place_markers(
    grid: np.ndarray,
    placements: list[dict],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    labeled_ids = _choose_labeled_ids(ids_base)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)
    forbidden = _forbidden_long_ray_cells(grid, placements)

    jobs = list(zip(marker_colors, labeled_ids))
    random.shuffle(jobs)
    for color, tid in jobs:
        sprite_order = list(range(len(placements)))
        random.shuffle(sprite_order)
        placed = False
        for si in sprite_order:
            sprite = placements[si]
            port = sprite["ports"][tid]
            base_r = sprite["r0"] + port["coord"][0]
            base_c = sprite["c0"] + port["coord"][1]
            dr, dc = port["short_direction"]

            candidates = _marker_candidates(base_r, base_c, dr, dc, H, W)
            random.shuffle(candidates)
            for mr, mc in candidates:
                if (mr, mc) in forbidden:
                    continue
                if not _isolated_marker_cell(grid, mr, mc):
                    continue
                if not _ray_clear(grid, base_r, base_c, mr, mc, dr, dc):
                    continue
                if _marker_seen_by_other_port(grid, placements, mr, mc, si, tid):
                    continue
                grid[mr, mc] = color
                placed = True
                break
            if placed:
                break
        if not placed:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors, variants = design

    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    if len(variants) < num_sprites:
        return None
    chosen = random.sample(variants, num_sprites)

    placements = _place_sprites(H, W, chosen)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for sprite in placements:
        r0, c0 = sprite["r0"], sprite["c0"]
        sub = sprite["sub"]
        h, w = sub.shape
        grid[r0 : r0 + h, c0 : c0 + w] = sub

    if not _place_markers(grid, placements, ids_base, template_colors):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not _sample_exhibits_fusion(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
