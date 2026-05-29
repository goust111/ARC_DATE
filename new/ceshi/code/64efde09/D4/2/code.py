# Variant of 64efde09
# Dimension: D4 — 源对象：标记像素的类型与位置 (src)
# Change: 单像素标记，放置在网格边界上，颜色与模板颜色不同且互不重复，通过射线对齐到精灵活跃边端口... → 标记颜色与模板颜色重叠

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


# Design analysis for the D4 variant
# ----------------------------------
# spec:
#   variant_core:
#     - Marker pixels stay as isolated single boundary cells aligned with active-edge rays.
#     - Their colors now come from the sprite template palette instead of a disjoint label palette.
#     - To make the change operational rather than cosmetic, the generator always reuses one
#       marker color twice among the three markers, so color can no longer serve as a unique key.
#   concreteization_and_trace:
#     - Original behavior: if three markers have colors [6, 7, 9], the solver can store
#       `color -> template_port` because each color appears once and nowhere on the sprites.
#     - Variant behavior: markers may have colors [2, 2, 4] while the template itself already
#       uses colors {2, 3, 4}. Two different boundary singletons can therefore share color 2.
#     - Consequence: the solver must recognize markers by position and boundary isolation, then
#       store `template_port -> marker_color`. Using `color -> template_port` would merge the two
#       color-2 markers and lose one labeled port.
#   complexity_calibration:
#     - Only one new freedom is introduced: marker colors overlap and may repeat.
#     - The repeated-color pattern is controlled to exactly two distinct colors across three
#       markers, which is enough to invalidate the old shortcut without making examples noisy.
#   implicit_assumption_audit:
#     - `assert len(set(markers.values())) == len(markers)` is no longer valid.
#     - `color_to_template_id` / `template_id_to_color` as a bijection is no longer valid because
#       several markers can legitimately share the same color.
#     - Treating all area-1 components as markers remains safe only if generation guarantees they
#       are boundary singletons; the new solver asserts that explicitly because boundary position is
#       now the visible cue that distinguishes a marker from sprite structure.
#   placement_legality:
#     - Sprites are kept at least one cell away from every boundary, except for a larger clearance
#       on the boundary their active edge faces. This makes boundary singletons visually clean.
#     - Marker placement checks clear line-of-sight, occupancy, and 8-neighbor isolation so each
#       marker remains its own component and cannot merge diagonally with another boundary marker.
#   visual_observability:
#     - Every marker color is guaranteed to already appear on the sprites.
#     - Because one marker color is reused, the input visibly departs from the original "three new
#       colors" pattern, and the output shows that two different ports can emit the same color.
#   neatness_and_color_inference:
#     - The task still relies on only three markers and a 3-4 color template palette.
#     - Rays are still straight and boundary-facing; the only new inference burden is recognizing
#       that isolated boundary pixels, not novel colors, are the labels.


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


def _is_boundary_cell(r: int, c: int, H: int, W: int) -> bool:
    return r in (0, H - 1) or c in (0, W - 1)


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
    Broadcast boundary marker rays from labeled ports of one rectangle template to every transformed copy.

    Input structure:
    - The background is the dominant color.
    - Several dense 2-by-N or N-by-2 rectangular sprites are dihedral transforms of one shared template.
    - Some boundary cells contain isolated one-pixel markers; their colors are borrowed from the same palette as the sprites and may repeat.

    Rule:
    - For each sprite, find the active long edge: the side whose cells have the stronger same-color majority.
    - If an outward ray from an active-edge cell first meets an isolated boundary marker, that marker labels the corresponding template port and contributes its own color.
    - For every sprite, extend that color outward from the matching transformed port through background cells until the ray leaves the grid or hits a non-background cell.

    Key insight:
    - The markers are identified by spatial role rather than novel colors: they are lone boundary pixels on clear outward rays, and the template-port correspondence still follows the shared rotation/reflection structure.
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
    for sprite in sprites_raw:
        matches = [
            ids
            for _name, sub, ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "each sprite must match the template in exactly one orientation"
        direction, active = _active_side(sprite["sub"])
        sprites.append({**sprite, "ids": matches[0], "direction": direction, "active": active})

    template_id_to_color: dict[int, int] = {}
    marker_to_template_id: dict[tuple[int, int], int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["active"]:
            marker_pos = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if marker_pos is None:
                continue
            template_id = int(sprite["ids"][lr, lc])
            marker_color = int(markers[marker_pos])

            if marker_pos in marker_to_template_id:
                assert marker_to_template_id[marker_pos] == template_id
            if template_id in template_id_to_color:
                assert template_id_to_color[template_id] == marker_color

            marker_to_template_id[marker_pos] = template_id
            template_id_to_color[template_id] = marker_color

    assert set(marker_to_template_id) == set(markers), "every boundary marker must label one active-edge cell"
    assert len(set(marker_to_template_id.values())) == len(marker_to_template_id), "markers must label distinct template ports"

    out = g.copy()
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for template_id, color in template_id_to_color.items():
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


# --------------- generator (att1 D4) ---------------

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
_BOUNDARY_MARGIN = 1
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
    if all(c == right_dominant for c in template_colors):
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
            pos = _random_sprite_position(H, W, h, w, *direction)
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


def _choose_marker_colors(template_colors: list[int]) -> list[int] | None:
    if len(template_colors) < 2:
        return None
    repeated = random.choice(template_colors)
    other = random.choice([c for c in template_colors if c != repeated])
    marker_colors = [repeated, repeated, other]
    random.shuffle(marker_colors)
    return marker_colors


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

    marker_colors = _choose_marker_colors(template_colors)
    if marker_colors is None:
        return False

    for color, tid in zip(marker_colors, labeled_ids):
        si = random.randint(0, num_sprites - 1)
        r0, c0, _sub, ids_arr, direction, active_coords = sprite_placements[si]
        dr, dc = direction

        locs = np.argwhere(ids_arr == tid)
        if locs.shape[0] != 1:
            return False
        lr, lc = map(int, locs[0])
        if (lr, lc) not in set(active_coords):
            return False

        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)

        cr, cc = r0 + lr + dr, c0 + lc + dc
        while (cr, cc) != (marker_r, marker_c) and 0 <= cr < H and 0 <= cc < W:
            if grid[cr, cc] != _BG:
                return False
            cr += dr
            cc += dc

        if not _is_isolated_boundary_spot(grid, marker_r, marker_c):
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
