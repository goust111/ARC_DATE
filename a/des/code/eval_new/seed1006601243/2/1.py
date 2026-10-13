from __future__ import annotations

import random

import numpy as np


BG = 0
STRUCTURE_COLORS = (5, 6, 8)
GRID_RANGE = (16, 30)
OBJECT_COUNT_RANGE = (1, 4)
FRAME_SIDE_RANGE = (4, 9)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
RESERVE_MARGIN = 1
SEED_TIE_BIAS = 0.45
QUADRANTS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_rectangular_frame(frame_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    sub = frame_mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    if h < 3 or w < 3:
        return False
    return bool(np.array_equal(sub, _frame_mask((h, w))))


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _quadrant_sign(seed_r: int, seed_c: int, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    dr = -1 if 2 * seed_r <= r0 + r1 else 1
    dc = -1 if 2 * seed_c <= c0 + c1 else 1
    return dr, dc


def _copy_bbox(
    bbox: tuple[int, int, int, int],
    dr: int,
    dc: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    src_h = r1 - r0 + 1
    src_w = c1 - c0 + 1
    copy_h = src_h // 2
    copy_w = src_w // 2
    assert copy_h >= 1 and copy_w >= 1

    if dr < 0:
        nr1 = r0 - 1
        nr0 = nr1 - copy_h + 1
    else:
        nr0 = r1 + 1
        nr1 = nr0 + copy_h - 1

    if dc < 0:
        nc1 = c0 - 1
        nc0 = nc1 - copy_w + 1
    else:
        nc0 = c1 + 1
        nc1 = nc0 + copy_w - 1

    return nr0, nc0, nr1, nc1


def _reserve_rect(mask: np.ndarray, bbox: tuple[int, int, int, int], margin: int) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _valid_seed_positions(
    bbox: tuple[int, int, int, int],
    dr: int,
    dc: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    positions: list[tuple[int, int]] = []
    tie_positions: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            qdr, qdc = _quadrant_sign(r, c, bbox)
            if (qdr, qdc) != (dr, dc):
                continue
            positions.append((r, c))
            if 2 * r == r0 + r1 or 2 * c == c0 + c1:
                tie_positions.append((r, c))
    return positions, tie_positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a half-size colored rectangular frame outside each fence in the seed-indicated diagonal direction.

    Input structure:
    - The background is a uniform color.
    - One structural color forms several disjoint one-pixel-thick closed axis-aligned rectangular fences.
    - Each fence contains exactly one interior seed pixel of a non-structural color.

    Rule:
    - For each fence, compare the seed row and column to the fence center; upper/lower and left/right determine the growth diagonal, with centerline ties choosing upper and left.
    - Build a new rectangular frame whose outer height and width are the original fence dimensions halved with floor rounding.
    - Place that new frame diagonally outside the matching corner so the two rectangles only meet at the geometric corner, leaving the original fence and seed unchanged.

    Key insight:
    - The seed position encodes a quadrant direction, while the fence geometry itself supplies the size of the externally attached half-scale copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _most_common_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one non-background color"

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    frame_color = max(colors, key=lambda color: (counts[color], -color))
    frame_components = _components(g == frame_color)
    assert frame_components, "expected at least one fence"

    out = g.copy()
    for comp in frame_components:
        bbox = _bbox(comp)
        assert _is_rectangular_frame(g == frame_color, bbox), "structure color must form rectangular fences"

        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere((interior != bg) & (interior != frame_color))
        assert seed_coords.shape[0] == 1, "each fence must contain exactly one seed"
        sr, sc = map(int, seed_coords[0])
        sr += r0 + 1
        sc += c0 + 1
        seed_color = int(g[sr, sc])

        dr, dc = _quadrant_sign(sr, sc, bbox)
        copy_box = _copy_bbox(bbox, dr, dc)
        nr0, nc0, nr1, nc1 = copy_box
        assert 0 <= nr0 <= nr1 < h and 0 <= nc0 <= nc1 < w, "copy must stay inside the canvas"
        assert np.all(out[nr0 : nr1 + 1, nc0 : nc1 + 1] == bg), "copy region must be empty"
        _draw_frame(out, copy_box, seed_color)

    return out


def _choose_scene_size(rng: random.Random, n_objects: int) -> tuple[int, int]:
    min_side = max(GRID_RANGE[0], 12 + 3 * (n_objects - 1))
    h = rng.randint(min_side, GRID_RANGE[1])
    w = rng.randint(min_side, GRID_RANGE[1])
    return h, w


def _sample_frame_side(rng: random.Random, n_objects: int) -> int:
    lo, hi = FRAME_SIDE_RANGE
    if n_objects >= 4:
        hi = min(hi, 7)
    elif n_objects == 3:
        hi = min(hi, 8)
    return rng.randint(lo, hi)


def _place_object(
    rng: random.Random,
    reserved: np.ndarray,
    fence_color: int,
    seed_color: int,
    dr: int,
    dc: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int]] | None:
    h, w = reserved.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        box_h = _sample_frame_side(rng, 1)
        box_w = _sample_frame_side(rng, 1)
        if rng.random() < 0.4:
            box_h = _sample_frame_side(rng, 3)
        if rng.random() < 0.4:
            box_w = _sample_frame_side(rng, 3)

        copy_h = box_h // 2
        copy_w = box_w // 2
        if copy_h < 2 or copy_w < 2:
            continue

        r_min = RESERVE_MARGIN + (copy_h if dr < 0 else 0)
        r_max = h - RESERVE_MARGIN - box_h - (copy_h if dr > 0 else 0)
        c_min = RESERVE_MARGIN + (copy_w if dc < 0 else 0)
        c_max = w - RESERVE_MARGIN - box_w - (copy_w if dc > 0 else 0)
        if r_min > r_max or c_min > c_max:
            continue

        r0 = rng.randint(r_min, r_max)
        c0 = rng.randint(c_min, c_max)
        bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        copy_box = _copy_bbox(bbox, dr, dc)

        candidate = np.zeros_like(reserved)
        _reserve_rect(candidate, bbox, RESERVE_MARGIN)
        _reserve_rect(candidate, copy_box, RESERVE_MARGIN)
        if np.any(candidate & reserved):
            continue

        seed_positions, tie_positions = _valid_seed_positions(bbox, dr, dc)
        if not seed_positions:
            continue
        if tie_positions and rng.random() < SEED_TIE_BIAS:
            seed = rng.choice(tie_positions)
        else:
            seed = rng.choice(seed_positions)

        reserved |= candidate
        return bbox, copy_box, seed
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(*OBJECT_COUNT_RANGE)
    h, w = _choose_scene_size(rng, n_objects)
    fence_color = rng.choice(STRUCTURE_COLORS)
    palette = [color for color in range(10) if color not in (BG, fence_color)]
    if n_objects > len(palette):
        return None
    seed_colors = rng.sample(palette, n_objects)
    directions = rng.sample(QUADRANTS, n_objects)

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    placed: list[tuple[tuple[int, int, int, int], tuple[int, int], int]] = []
    for seed_color, (dr, dc) in zip(seed_colors, directions):
        result = _place_object(rng, reserved, fence_color, seed_color, dr, dc)
        if result is None:
            return None
        bbox, _copy_box, seed = result
        placed.append((bbox, seed, seed_color))

    for bbox, seed, seed_color in placed:
        _draw_frame(grid, bbox, fence_color)
        sr, sc = seed
        grid[sr, sc] = seed_color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
