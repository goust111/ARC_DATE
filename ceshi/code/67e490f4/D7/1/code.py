# Variant of 67e490f4
# Dimension: D7 — 目标结构：洞的形态与连通性 (dst)
# Change: 洞为 4-连通背景色区域，形状 bbox 1x1 到 5x6，面积 1-8 像素，3-6 种不同形状，每种 1-5 个实例 → 大型矩形洞（cavity风格）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)

# Design analysis for the D7 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     original_dst: frame interior contains several small irregular 4-connected holes; external sprites
#       are matched by exact mask shape under rotation/flip.
#     new_dst: frame interior contains only large rectangular cavities with bbox sizes in 3x3..5x5.
#       External sprites are large connected chunks whose *bounding boxes* match those cavity sizes,
#       but their masks may have small bites/notches. The solver must therefore ignore exact silhouette
#       and key only on cavity height/width.
#   concrete_behavior_trace:
#     example:
#       old_logic: a 4x5 hole would require an external 4x5 full rectangle (or exact same mask after
#         transforms). A 4x5 chunk with one missing corner would not match.
#       new_logic: the same 4x5 cavity is filled by the plurality color among all outside chunks whose
#         bbox is 4x5, regardless of whether each chunk is full, corner-bitten, or has a tiny internal bite.
#       visible_difference: humans can see that multiple slightly different 4x5 outside chunks all point
#         to the same 4x5 empty rectangle inside the frame, so the rule is "same size", not "same mask".
#   complexity_calibration:
#     - Keep one large frame and plurality-based color selection from the original task.
#     - Use 3-4 distinct cavity sizes so the task does not collapse to a single trivial lookup.
#     - Keep repeated cavity instances via symmetric placement, so each size appears multiple times.
#     - Avoid hidden tie-break rules by generating a unique plurality color for every size group.
#   main_assumption_audit:
#     - Largest monochrome non-bg component is still the frame: generator keeps frame area much larger
#       than every sprite component.
#     - Every hole component inside the frame bbox must now be a full rectangle: solver asserts that the
#       hole mask fills its bbox, and generator carves only axis-aligned rectangles.
#     - External sprite matching must not depend on exact mask bytes anymore: solver groups by bbox size
#       only, and generator ensures many sprites are non-rectangular so the original exact-mask rule would fail.
#   placement_legality:
#     - Cavities stay at least one frame-colored cell away from the frame border and from each other.
#     - External sprites use a solid blocker bbox during placement, so notch-shaped chunks cannot overlap
#       through their transparent bites and remain visually separated from the frame and from each other.
#   visual_observability:
#     - Large 3x3..5x5 cavities are conspicuous and visually unlike the original tiny polyomino holes.
#     - Outside chunks are also large; their minor bites are visible but still read as size-coded pieces.
#   anti_degeneracy:
#     - No size group relies on a full rectangle sprite. This guarantees that the original solver's exact
#       shape matching would not solve the same inputs, while the new bbox-only solver does.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill large rectangular cavities inside a frame using size-matched outside chunks.

    Input structure:
    - bg = most common color. One large monochrome frame encloses several big rectangular cavities.
    - Outside the frame sit large monochrome chunks; they may have small bites, but each chunk still spans a rectangular bbox.

    Rule:
    - Crop to the frame's bounding box and treat every non-frame connected region inside it as a cavity.
    - Match each cavity to outside chunks by bbox size only, not by exact occupied mask.
    - For each cavity size, fill all matching cavities with the unique plurality color among same-sized outside chunks.

    Key insight:
    - The outside pieces are size keys rather than stencil keys: only their height and width matter, so slightly notched chunks still map to full rectangular cavities.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    fb = frame["bbox"]
    r0, c0, r1, c1 = map(int, fb)

    sub = crop_bbox(g, fb)
    th, tw = map(int, sub.shape)

    hole_mask = sub != int(frame_col)
    holes: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        assert bool(m.all())
        holes.append(((hh, ww), (int(hr0), int(hc0))))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_size: dict[tuple[int, int], dict[int, int]] = {}
    for comp in spr_comps:
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)
        vals = [int(v) for v in g[comp["coords"][:, 0], comp["coords"][:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])
        bucket = sprites_by_size.setdefault((hh, ww), {})
        bucket[scol] = int(bucket.get(scol, 0)) + 1

    out = np.full((th, tw), int(frame_col), dtype=int)
    for size, (hr0, hc0) in holes:
        cmap = sprites_by_size.get(size)
        assert cmap is not None
        best = max(int(v) for v in cmap.values())
        winners = [int(col) for col, cnt in cmap.items() if int(cnt) == best]
        assert len(winners) == 1
        fill_col = int(winners[0])
        hh, ww = map(int, size)
        out[hr0 : hr0 + hh, hc0 : hc0 + ww] = int(fill_col)

    return out


NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (17, 22)
FRAME_W_RANGE = (17, 22)

RECT_SIDE_RANGE = (3, 5)
SIZE_COUNT_RANGE = (3, 4)
MIN_SIZE_COUNT = 3
MIN_TOTAL_HOLES = 5
FRAME_DENSITY_RANGE = (0.45, 0.78)
PLACEMENT_RETRIES = 120

WINNING_SPRITE_RANGE = (2, 3)
LOSING_COLOR_MAX = 1
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 400


def _build_rect_library() -> list[np.ndarray]:
    rects = []
    for h in range(RECT_SIDE_RANGE[0], RECT_SIDE_RANGE[1] + 1):
        for w in range(RECT_SIDE_RANGE[0], RECT_SIDE_RANGE[1] + 1):
            rects.append(np.ones((h, w), dtype=bool))
    return rects


RECT_LIBRARY = _build_rect_library()


def _mirror_pos(r: int, c: int, sh: int, sw: int, th: int, tw: int) -> list[tuple[int, int]]:
    return list(
        {
            (int(r), int(c)),
            (int(r), int(tw - sw - c)),
            (int(th - sh - r), int(c)),
            (int(th - sh - r), int(tw - sw - c)),
        }
    )


def _validate_positions(
    occ: np.ndarray, shape: np.ndarray, positions: list[tuple[int, int]], th: int, tw: int
) -> list[tuple[int, int]] | None:
    sh, sw = map(int, shape.shape)
    new_cells: set[tuple[int, int]] = set()
    for pr, pc in positions:
        if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
            return None
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    new_cells.add((int(pr + dr), int(pc + dc)))

    for rr, cc in new_cells:
        if not bool(occ[rr, cc]):
            return None
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < th and 0 <= nc < tw and (not bool(occ[nr, nc])) and (nr, nc) not in new_cells:
                return None
    return positions


def _try_place_quadrant(occ: np.ndarray, shape: np.ndarray, th: int, tw: int) -> list[tuple[int, int]] | None:
    sh, sw = map(int, shape.shape)
    r_max = th // 2 - sh
    c_max = tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r = random.randint(1, r_max)
    c = random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ: np.ndarray, shape: np.ndarray, th: int, tw: int) -> list[tuple[int, int]] | None:
    if th % 2 == 0:
        return None
    sh, sw = map(int, shape.shape)
    r = th // 2 - sh // 2
    if r < 1 or r + sh > th - 1:
        return None
    c_max = tw // 2 - sw
    if c_max < 1:
        return None
    c = random.randint(1, c_max)
    c_m = tw - sw - c
    return _validate_positions(occ, shape, list({(int(r), int(c)), (int(r), int(c_m))}), th, tw)


def _try_place_vcenter(occ: np.ndarray, shape: np.ndarray, th: int, tw: int) -> list[tuple[int, int]] | None:
    if tw % 2 == 0:
        return None
    sh, sw = map(int, shape.shape)
    c = tw // 2 - sw // 2
    if c < 1 or c + sw > tw - 1:
        return None
    r_max = th // 2 - sh
    if r_max < 1:
        return None
    r = random.randint(1, r_max)
    r_m = th - sh - r
    return _validate_positions(occ, shape, list({(int(r), int(c)), (int(r_m), int(c))}), th, tw)


def _try_place_center(occ: np.ndarray, shape: np.ndarray, th: int, tw: int) -> list[tuple[int, int]] | None:
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = map(int, shape.shape)
    r = th // 2 - sh // 2
    c = tw // 2 - sw // 2
    return _validate_positions(occ, shape, [(int(r), int(c))], th, tw)


def _try_place_symmetric(occ: np.ndarray, shape: np.ndarray, th: int, tw: int) -> list[tuple[int, int]] | None:
    pool = [_try_place_quadrant, _try_place_quadrant]
    if th % 2 == 1:
        pool.extend([_try_place_hcenter] * 3)
    if tw % 2 == 1:
        pool.extend([_try_place_vcenter] * 3)
    if th % 2 == 1 and tw % 2 == 1:
        pool.extend([_try_place_center] * 4)
    random.shuffle(pool)
    for placer in pool:
        positions = placer(occ, shape, th, tw)
        if positions is not None:
            return positions
    return None


def _carve_hole(occ: np.ndarray, shape: np.ndarray, positions: list[tuple[int, int]]) -> None:
    sh, sw = map(int, shape.shape)
    for pr, pc in positions:
        occ[pr : pr + sh, pc : pc + sw] = False


def _build_hole_layout(th: int, tw: int) -> tuple[np.ndarray, list[tuple[np.ndarray, list[tuple[int, int]]]]] | None:
    max_sh = max(1, (th - 2) // 2)
    max_sw = max(1, (tw - 2) // 2)
    available = [s for s in RECT_LIBRARY if int(s.shape[0]) <= max_sh and int(s.shape[1]) <= max_sw]
    if len(available) < MIN_SIZE_COUNT:
        return None

    random.shuffle(available)
    k_target = random.randint(SIZE_COUNT_RANGE[0], min(SIZE_COUNT_RANGE[1], len(available)))
    occ = np.ones((th, tw), dtype=bool)
    placements: list[tuple[np.ndarray, list[tuple[int, int]]]] = []

    available.sort(key=lambda s: (-(int(s.shape[0]) * int(s.shape[1])), -int(s.shape[0]), -int(s.shape[1])))
    for shape in available:
        if len(placements) >= k_target:
            break
        for _ in range(PLACEMENT_RETRIES):
            pos = _try_place_symmetric(occ, shape, th, tw)
            if pos is not None:
                _carve_hole(occ, shape, pos)
                placements.append((shape, pos))
                break

    if len(placements) < MIN_SIZE_COUNT:
        return None

    total_holes = sum(len(pos) for _, pos in placements)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    return occ, placements


def _build_output(
    occ: np.ndarray, hole_placements: list[tuple[np.ndarray, list[tuple[int, int]]]], frame_col: int, bg: int
) -> tuple[np.ndarray, list[tuple[tuple[int, int], int, list[tuple[int, int]]]]] | None:
    th, tw = map(int, occ.shape)
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None

    output = np.full((th, tw), int(frame_col), dtype=int)
    shape_colors = random.sample(avail_colors, len(hole_placements))
    shape_info: list[tuple[tuple[int, int], int, list[tuple[int, int]]]] = []
    for (shape, positions), fill_col in zip(hole_placements, shape_colors):
        h, w = map(int, shape.shape)
        for pr, pc in positions:
            output[pr : pr + h, pc : pc + w] = int(fill_col)
        shape_info.append(((int(h), int(w)), int(fill_col), [(int(pr), int(pc)) for pr, pc in positions]))
    return output, shape_info


def _sprite_variants(h: int, w: int) -> list[np.ndarray]:
    full = np.ones((h, w), dtype=bool)
    variants: list[np.ndarray] = []
    remove_sets = [
        [(0, 0)],
        [(0, w - 1)],
        [(h - 1, 0)],
        [(h - 1, w - 1)],
    ]
    if w >= 4:
        remove_sets.extend(
            [
                [(0, 0), (0, 1)],
                [(h - 1, w - 2), (h - 1, w - 1)],
            ]
        )
    if h >= 4:
        remove_sets.extend(
            [
                [(0, 0), (1, 0)],
                [(h - 2, w - 1), (h - 1, w - 1)],
            ]
        )
    if h >= 3 and w >= 3:
        remove_sets.append([(h // 2, w // 2)])

    seen: set[bytes] = set()
    for cells in remove_sets:
        m = full.copy()
        for rr, cc in cells:
            m[rr, cc] = False
        key = m.tobytes()
        if key not in seen and bool(m.any()) and not bool(m.all()):
            seen.add(key)
            variants.append(m)
    assert variants
    return variants


def _place_sprite(inp: np.ndarray, placement: np.ndarray, mask: np.ndarray, color: int, bg: int) -> bool:
    h, w = map(int, mask.shape)
    blocker = np.full((h, w), int(color), dtype=int)
    patch = np.full((h, w), int(bg), dtype=int)
    patch[mask] = int(color)
    try:
        r, c = random_free_location(placement, blocker, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, blocker, int(r), int(c))
    paste(inp, patch, int(r), int(c), transparent=bg)
    return True


def _scatter_sprites(
    inp: np.ndarray, shape_info: list[tuple[tuple[int, int], int, list[tuple[int, int]]]], bg: int, frame_col: int
) -> bool:
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    for (h, w), fill_color, _ in shape_info:
        variants = _sprite_variants(h, w)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != fill_color]
        if not losers:
            return False
        n_other_colors = random.randint(1, min(LOSING_COLOR_MAX, len(losers)))
        loser_cols = random.sample(losers, n_other_colors)

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(variants), fill_color, bg):
                return False

        for oc in loser_cols:
            n_losing = random.randint(1, max(1, n_winning - 1))
            for _ in range(n_losing):
                if not _place_sprite(inp, placement, random.choice(variants), oc, bg):
                    break
    return True


def _choose_frame_origin(th: int, tw: int) -> tuple[int, int]:
    corners = [
        (0, 0),
        (0, GRID_SIZE - tw),
        (GRID_SIZE - th, 0),
        (GRID_SIZE - th, GRID_SIZE - tw),
    ]
    if random.random() < 0.8:
        return random.choice(corners)
    return random.randint(0, GRID_SIZE - th), random.randint(0, GRID_SIZE - tw)


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_placements = layout

    built = _build_output(occ, hole_placements, frame_col, bg)
    if built is None:
        return None
    output, shape_info = built

    fr, fc = _choose_frame_origin(th, tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    frame_patch = np.where(occ, int(frame_col), int(bg)).astype(int)
    paste(inp, frame_patch, int(fr), int(fc))

    if not _scatter_sprites(inp, shape_info, bg, frame_col):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
