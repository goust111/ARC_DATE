# Variant of 64efde09
# Dimension: D4 — 源对象：标记像素的类型与位置 (src)
# Change: 单像素标记，放置在网格边界上，颜色与模板颜色不同且互不重复，通过射线对齐到精灵活跃边端口... → 标记放在精灵的非活跃边一侧

from __future__ import annotations

"""
design_analysis:
  concretization:
    original:
      markers lived on the canvas boundary and were collinear with the outward
      rays emitted from the active long edge.
    variant:
      markers stay as isolated single pixels with unique non-template colors,
      but they are moved to the outward side of the opposite long edge of a
      sprite. For an Nx2 canonical template whose active side is the left
      column, a marker now sits somewhere to the right of the right column, on
      the same row as the labeled port. The output still emits from the active
      edge toward its boundary-facing direction.
    why_long_opposite_edge:
      The prompt allows "short edge or non-dominant edge". Using the opposite
      long edge keeps the D4 change visually obvious while preserving the
      original label count of 3 ports. Short edges only expose 2 cells on a
      thin Nx2 rectangle, which would undercut the original structure.
  behavior_trace:
    example:
      old: active-left vertical sprite, marker on the left boundary, same row as
        active cell (r,0) -> solver scans left from that active cell -> marker
        color labels template id ids[r,0].
      new: same sprite, marker is instead to the right of inactive cell (r,1) ->
        solver must first identify the active side, infer that the opposite long
        side is the relevant non-active side, scan right from (r,1), and then
        transfer the label back across the rectangle to ids[r,0].
      delta:
        The emitted ray direction is unchanged, but the observed source cue now
        sits on the opposite side of the sprite.
  complexity_calibration:
    kept:
      Template shape, color diversity, dihedral matching, and active-side
      majority logic stay unchanged so the task is still recognizably the same
      family.
    changed:
      Markers are never used as boundary beacons anymore. They appear in the
      interior, but always on one of only three labeled opposite-side rows/cols.
      This keeps the new freedom small enough to infer from 2-4 examples.
  implicit_assumption_audit:
    old_main_dependencies:
      - "_first_marker_on_ray(..., dr, dc) from active-edge cells assumed the
        marker ray and the output ray used the same direction."
      - "assert set(color_to_template_id) == set(markers.values()) assumed every
        marker was discoverable from active-side scans."
      - "generator placed sprites only with active-side boundary clearance,
        because marker placement reused the boundary reached by the active ray."
    variant_updates:
      - solver now scans from the opposite long edge using the opposite outward
        direction, then maps that inactive-edge location back to the paired
        active port before storing the template id.
      - generator must reserve interior clearance on the non-active side, keep
        markers isolated, and prevent markers from landing on any future active
        output ray so they do not become accidental blockers.
  placement_legality:
    sprites:
      Each sprite still faces its active side toward a boundary, so output rays
      remain easy to read and conflict checks remain local.
    markers:
      Markers are placed 2-4 cells away from the non-active long edge, never on
      the boundary, with a clear bg-only corridor back to that edge and with
      8-neighborhood isolation so they remain single-pixel components.
  observability:
    human_view:
      The key visual shift is that the colored pixels are now behind the sprites
      relative to the eventual colored rays. Across examples, the same colors
      repeatedly appear opposite the emitting side, making the inversion
      inferable without hidden metadata.
  cleanliness:
    safeguards:
      - markers are kept off all active-ray lanes
      - sprites retain padding from one another
      - solver validation is part of generation rejection
  color_inference:
    unchanged:
      Marker colors are still unique labels, not transformed by numeric order or
      any palette arithmetic. Their only role is to bind one shared template
      port to one output ray color.
"""

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


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


def _inactive_long_side(
    sub: np.ndarray,
    active_direction: tuple[int, int],
) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    h, w = sub.shape
    dr, dc = active_direction
    if (dr, dc) == (0, -1):
        return (0, 1), {(r, w - 1) for r in range(h)}
    if (dr, dc) == (0, 1):
        return (0, -1), {(r, 0) for r in range(h)}
    if (dr, dc) == (-1, 0):
        return (1, 0), {(h - 1, c) for c in range(w)}
    assert (dr, dc) == (1, 0)
    return (-1, 0), {(0, c) for c in range(w)}


def _paired_active_cell(
    shape: tuple[int, int],
    active_direction: tuple[int, int],
    lr: int,
    lc: int,
) -> tuple[int, int]:
    h, w = shape
    dr, dc = active_direction
    if (dr, dc) == (0, -1):
        return lr, 0
    if (dr, dc) == (0, 1):
        return lr, w - 1
    if (dr, dc) == (-1, 0):
        return 0, lc
    assert (dr, dc) == (1, 0)
    return h - 1, lc


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
    Copy port labels from opposite-side marker rays to every transformed sprite, then emit from the dominant edge.

    Input structure:
    - bg is the most common color.
    - Several dense 2-by-N or N-by-2 rectangles are dihedral transforms of one shared template.
    - Isolated single-pixel markers of unique non-template colors lie on rays extending outward from the non-active long edge of some sprites, not on the emitting side.

    Rule:
    - For each sprite, identify the active long edge as the side with the stronger same-color majority.
    - Scan outward from the opposite long edge; a first-seen marker labels the active-edge port paired to that opposite-side cell in the same transformed sprite.
    - On every sprite, draw the labeled color outward from the corresponding active-edge cell through bg until blocked or out of bounds.

    Key insight:
    - The marker is no longer colocated with the emission direction. You must pair each opposite-side hit back across the rectangle to the shared active-side template port, and that pairing must follow the sprite's rotation/reflection.
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
        inactive_direction, inactive = _inactive_long_side(sprite["sub"], direction)
        sprites.append(
            {
                **sprite,
                "ids": ids,
                "direction": direction,
                "active": active,
                "inactive_direction": inactive_direction,
                "inactive": inactive,
            }
        )

    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        scan_dr, scan_dc = sprite["inactive_direction"]
        for lr, lc in sprite["inactive"]:
            marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, scan_dr, scan_dc)
            if marker_color is None:
                continue
            ar, ac = _paired_active_cell(sprite["sub"].shape, sprite["direction"], lr, lc)
            template_id = int(sprite["ids"][ar, ac])
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
_ACTIVE_MAJORITY_RANGE = (5, 9)
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_CLEARANCE = 2
_MARKER_GAP_RANGE = (2, 4)
_PLACEMENT_RETRIES = 100
_GEN_RETRIES = 400


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
    if len([c for c in template_colors if c != right_dominant]) == 0:
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
    template: np.ndarray,
    ids_base: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]], tuple[int, int], list[tuple[int, int]]]]:
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
        inactive_direction, inactive_set = _inactive_long_side(sub, direction)
        seen.append(sub)
        result.append(
            (
                sub,
                ids_arr,
                direction,
                sorted(active_set),
                inactive_direction,
                sorted(inactive_set),
            )
        )
    return result


def _random_sprite_position(
    H: int,
    W: int,
    h: int,
    w: int,
    dr: int,
    dc: int,
) -> tuple[int, int]:
    active_pad = _PLACEMENT_CLEARANCE
    marker_pad = _MARKER_GAP_RANGE[1] + 2
    if (dr, dc) == (0, -1):
        return random.randint(0, H - h), random.randint(active_pad, W - w - marker_pad)
    if (dr, dc) == (0, 1):
        return random.randint(0, H - h), random.randint(marker_pad, W - w - active_pad)
    if (dr, dc) == (-1, 0):
        return random.randint(active_pad, H - h - marker_pad), random.randint(0, W - w)
    return random.randint(marker_pad, H - h - active_pad), random.randint(0, W - w)


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
    for sub, ids_arr, direction, active_coords, inactive_direction, inactive_coords in variants:
        h, w = sub.shape
        dr, dc = direction
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w, dr, dc)
            occ = _occupied_set(placements)
            if any((r0 + r, c0 + c) in occ for r in range(-pad, h + pad) for c in range(-pad, w + pad)):
                continue
            placements.append((r0, c0, sub, ids_arr, direction, active_coords, inactive_direction, inactive_coords))
            placed = True
            break
        if not placed:
            return None
    return placements


def _trace_to_boundary(
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    cells = []
    r = start_r + dr
    c = start_c + dc
    while 0 <= r < H and 0 <= c < W:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _active_ray_cells(grid: np.ndarray, sprite_placements: list[tuple]) -> set[tuple[int, int]]:
    H, W = grid.shape
    cells: set[tuple[int, int]] = set()
    for r0, c0, _sub, _ids_arr, direction, active_coords, *_rest in sprite_placements:
        dr, dc = direction
        for lr, lc in active_coords:
            for cell in _trace_to_boundary(r0 + lr, c0 + lc, dr, dc, H, W):
                if int(grid[cell]) != _BG:
                    break
                cells.add(cell)
    return cells


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
    return True


def _isolated_marker_cell(grid: np.ndarray, r: int, c: int) -> bool:
    H, W = grid.shape
    if int(grid[r, c]) != _BG:
        return False
    for rr in range(max(0, r - 1), min(H, r + 2)):
        for cc in range(max(0, c - 1), min(W, c + 2)):
            if (rr, cc) != (r, c) and int(grid[rr, cc]) != _BG:
                return False
    return True


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    N = ids_base.shape[0]
    num_sprites = len(sprite_placements)

    active_template_ids = [int(ids_base[r, 0]) for r in range(N)]
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    forbidden = _active_ray_cells(grid, sprite_placements)
    placements = list(zip(marker_colors, labeled_ids))
    random.shuffle(placements)

    for color, tid in placements:
        order = list(range(num_sprites))
        random.shuffle(order)
        placed = False
        for si in order:
            r0, c0, sub, ids_arr, direction, _active_coords, inactive_direction, inactive_coords = sprite_placements[si]
            locs = np.argwhere(ids_arr == tid)
            if locs.shape[0] != 1:
                return False
            ar, ac = map(int, locs[0])
            if (ar, ac) not in _active_coords:
                continue

            ir, ic = None, None
            for cand_r, cand_c in inactive_coords:
                pair = _paired_active_cell(sub.shape, direction, cand_r, cand_c)
                if pair == (ar, ac):
                    ir, ic = cand_r, cand_c
                    break
            if ir is None or ic is None:
                return False

            scan_dr, scan_dc = inactive_direction
            candidate_cells = _marker_candidates(r0 + ir, c0 + ic, scan_dr, scan_dc, H, W)
            random.shuffle(candidate_cells)
            for mr, mc in candidate_cells:
                if (mr, mc) in forbidden:
                    continue
                if not _isolated_marker_cell(grid, mr, mc):
                    continue
                if not _ray_clear(grid, r0 + ir, c0 + ic, mr, mc, scan_dr, scan_dc):
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
