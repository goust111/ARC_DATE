from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg


BG = 0
PALETTE = tuple(range(1, 10))
GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
EXTRA_OBJECT_RANGE = (2, 4)
ANCHOR_AREA_RANGE = (4, 8)
EXTRA_AREA_RANGE = (3, 8)
ANCHOR_BBOX_H_RANGE = (2, 4)
ANCHOR_BBOX_W_RANGE = (2, 5)
EXTRA_BBOX_H_RANGE = (2, 4)
EXTRA_BBOX_W_RANGE = (2, 5)
PLACEMENT_PADDING = 1
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _shape_key_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_key_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0 = coords.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _half_turn_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _shape_key_from_mask(np.rot90(mask.astype(bool), 2))


def _matches_half_turn(comp_a: dict, comp_b: dict) -> tuple[int, int] | None:
    if int(comp_a["color"]) != int(comp_b["color"]):
        return None
    if int(comp_a["area"]) != int(comp_b["area"]):
        return None

    r0a, c0a, r1a, c1a = map(int, comp_a["bbox"])
    r0b, c0b, r1b, c1b = map(int, comp_b["bbox"])
    sum_r_1 = r0a + r1b
    sum_r_2 = r1a + r0b
    sum_c_1 = c0a + c1b
    sum_c_2 = c1a + c0b
    if sum_r_1 != sum_r_2 or sum_c_1 != sum_c_2:
        return None

    sum_r = int(sum_r_1)
    sum_c = int(sum_c_1)
    coords_b = _coord_set(comp_b["coords"])
    mapped = {(sum_r - int(r), sum_c - int(c)) for r, c in comp_a["coords"]}
    if mapped != coords_b:
        return None
    return sum_r, sum_c


def _find_anchor_pair(grid: np.ndarray) -> tuple[int, list[dict], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    matches: list[tuple[int, int, int, int]] = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            center = _matches_half_turn(components[i], components[j])
            if center is not None:
                matches.append((i, j, center[0], center[1]))
    assert len(matches) == 1, "expected exactly one anchor pair"
    return bg, components, matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden 180-degree rotation center from the unique anchor pair, then complete every other object by point symmetry.

    Input structure:
    - The grid has a dominant background and several scattered monochromatic 4-connected objects.
    - Exactly one same-color object pair is already related by a 180-degree rotation around one hidden center, and that center is not marked.
    - Every other object's rotated copy would stay in-bounds and land only on background cells.

    Rule:
    - Find the unique same-color component pair that is an exact 180-degree rotation of one another.
    - Use the shared midpoint of corresponding pixels in that pair as the global rotation center.
    - Keep the input unchanged, and for each remaining object paint its 180-degree rotated copy in the same color.
    - Do not add anything for the anchor pair itself because it is already complete.

    Key insight:
    - The center is not given explicitly; it must be reconstructed from the only pre-existing rotation pair before any completion can be done.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, components, (anchor_a, anchor_b, sum_r, sum_c) = _find_anchor_pair(g)
    out = g.copy()
    anchor_ids = {int(anchor_a), int(anchor_b)}
    h, w = g.shape

    for idx, comp in enumerate(components):
        if idx in anchor_ids:
            continue
        color = int(comp["color"])
        mirrored = [(int(sum_r - r), int(sum_c - c)) for r, c in comp["coords"]]
        for rr, cc in mirrored:
            assert 0 <= rr < h and 0 <= cc < w, "rotated copy must stay in bounds"
            assert int(g[rr, cc]) == bg, "rotated copy must land on background"
            out[rr, cc] = color
    return out


def _generate_polyomino(
    area_range: tuple[int, int],
    h_range: tuple[int, int],
    w_range: tuple[int, int],
    *,
    forbid_half_turn: bool = False,
    area_override: int | None = None,
) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(*h_range)
        w = random.randint(*w_range)
        min_area = max(area_range[0], 3)
        max_area = min(area_range[1], h * w)
        if min_area > max_area:
            continue
        area = int(area_override) if area_override is not None else random.randint(min_area, max_area)
        if area < 3 or area > h * w:
            continue

        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            nbrs = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                continue
            cells.add(random.choice(nbrs))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _normalize_mask(mask)
        if int(mask.sum()) < 3:
            continue
        if mask.shape[0] == 1 or mask.shape[1] == 1:
            continue
        if forbid_half_turn and _shape_key_from_mask(mask) == _half_turn_key(mask):
            continue
        return mask
    raise RuntimeError("failed to generate shape")


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in np.argwhere(mask)}


def _mirror_coords(coords: set[tuple[int, int]], sum_r: int, sum_c: int) -> set[tuple[int, int]]:
    return {(int(sum_r - r), int(sum_c - c)) for r, c in coords}


def _expanded(coords: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((int(nr), int(nc)))
    return out


def _choose_center_sum(length: int) -> int:
    margin = 5
    low = margin
    high = 2 * length - 2 - margin
    parity = random.randint(0, 1)
    candidates = [v for v in range(low, high + 1) if v % 2 == parity]
    if not candidates:
        candidates = list(range(low, high + 1))
    return random.choice(candidates)


def _candidate_top_lefts(h: int, w: int, mask: np.ndarray, sum_r: int, sum_c: int) -> list[tuple[int, int]]:
    ph, pw = mask.shape
    out = []
    for r0 in range(h - ph + 1):
        mr0 = sum_r - (r0 + ph - 1)
        if not (0 <= mr0 <= h - ph):
            continue
        for c0 in range(w - pw + 1):
            mc0 = sum_c - (c0 + pw - 1)
            if 0 <= mc0 <= w - pw:
                out.append((int(r0), int(c0)))
    random.shuffle(out)
    return out


def _place_pair_or_source(
    grid: np.ndarray,
    mask: np.ndarray,
    color: int,
    sum_r: int,
    sum_c: int,
    occupied_input: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
    blocked_output: set[tuple[int, int]],
    *,
    paint_mirror_in_input: bool,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)

    for r0, c0 in _candidate_top_lefts(h, w, mask, sum_r, sum_c):
        src = _coords_from_mask(mask, r0, c0)
        mirror = _mirror_coords(src, sum_r, sum_c)
        if not src.isdisjoint(mirror):
            continue
        if _expanded(src, h, w, PLACEMENT_PADDING) & mirror:
            continue
        union = src | mirror
        if union & blocked_output:
            continue
        if src & reserved_output:
            continue
        if mirror & occupied_input:
            continue

        if paint_mirror_in_input:
            if mirror & reserved_output:
                continue
            for rr, cc in union:
                grid[rr, cc] = int(color)
            return src, mirror

        if mirror & reserved_output:
            continue
        for rr, cc in src:
            grid[rr, cc] = int(color)
        return src, mirror
    return None


def _build_expected_output(
    grid: np.ndarray,
    components: list[dict],
    anchor_info: tuple[int, int, int, int],
) -> np.ndarray:
    anchor_a, anchor_b, sum_r, sum_c = anchor_info
    out = np.asarray(grid, dtype=int).copy()
    anchor_ids = {int(anchor_a), int(anchor_b)}
    for idx, comp in enumerate(components):
        if idx in anchor_ids:
            continue
        color = int(comp["color"])
        for r, c in comp["coords"]:
            out[int(sum_r - r), int(sum_c - c)] = color
    return out


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        bg, components, anchor = _find_anchor_pair(grid)
    except AssertionError:
        return False

    anchor_a, anchor_b, sum_r, sum_c = anchor
    anchor_ids = {int(anchor_a), int(anchor_b)}
    if len(components) < 4:
        return False

    anchor_color = int(components[anchor_a]["color"])
    if sum(1 for comp in components if int(comp["color"]) == anchor_color) < 3:
        return False

    h, w = grid.shape
    expected = _build_expected_output(grid, components, anchor)
    if np.array_equal(expected, grid):
        return False

    painted_any = False
    for idx, comp in enumerate(components):
        if idx in anchor_ids:
            continue
        src = _coord_set(comp["coords"])
        dst = _mirror_coords(src, sum_r, sum_c)
        if src & dst:
            return False
        if _expanded(src, h, w, PLACEMENT_PADDING) & dst:
            return False
        for rr, cc in dst:
            if not (0 <= rr < h and 0 <= cc < w):
                return False
            if int(grid[rr, cc]) != bg:
                return False
        painted_any = painted_any or bool(dst)

    if not painted_any:
        return False

    actual = main(grid)
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    sum_r = _choose_center_sum(h)
    sum_c = _choose_center_sum(w)
    grid = np.full((h, w), BG, dtype=int)

    occupied_input: set[tuple[int, int]] = set()
    reserved_output: set[tuple[int, int]] = set()
    blocked_output: set[tuple[int, int]] = set()

    anchor_color = random.choice(PALETTE)
    anchor_mask = _generate_polyomino(
        ANCHOR_AREA_RANGE,
        ANCHOR_BBOX_H_RANGE,
        ANCHOR_BBOX_W_RANGE,
        forbid_half_turn=True,
    )
    placed = _place_pair_or_source(
        grid,
        anchor_mask,
        anchor_color,
        sum_r,
        sum_c,
        occupied_input,
        reserved_output,
        blocked_output,
        paint_mirror_in_input=True,
    )
    if placed is None:
        return None
    anchor_src, anchor_mirror = placed
    occupied_input |= anchor_src | anchor_mirror
    reserved_output |= anchor_src | anchor_mirror
    blocked_output = _expanded(reserved_output, h, w, PLACEMENT_PADDING)

    extra_count = random.randint(*EXTRA_OBJECT_RANGE)
    target_specs: list[tuple[np.ndarray, int]] = []

    same_color_area = int(anchor_mask.sum()) if random.random() < 0.5 else None
    for _ in range(SHAPE_ATTEMPTS):
        distractor_mask = _generate_polyomino(
            EXTRA_AREA_RANGE,
            EXTRA_BBOX_H_RANGE,
            EXTRA_BBOX_W_RANGE,
            area_override=same_color_area,
        )
        if _shape_key_from_mask(distractor_mask) != _shape_key_from_mask(anchor_mask):
            target_specs.append((distractor_mask, anchor_color))
            break
    if not target_specs:
        return None

    colors = list(PALETTE)
    random.shuffle(colors)
    for _ in range(extra_count - 1):
        color = random.choice(colors)
        mask = _generate_polyomino(EXTRA_AREA_RANGE, EXTRA_BBOX_H_RANGE, EXTRA_BBOX_W_RANGE)
        target_specs.append((mask, color))

    random.shuffle(target_specs)
    for mask, color in target_specs:
        placed = _place_pair_or_source(
            grid,
            mask,
            color,
            sum_r,
            sum_c,
            occupied_input,
            reserved_output,
            blocked_output,
            paint_mirror_in_input=False,
        )
        if placed is None:
            return None
        src, mirror = placed
        occupied_input |= src
        reserved_output |= src | mirror
        blocked_output = _expanded(reserved_output, h, w, PLACEMENT_PADDING)

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 11")
