from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

QUAD_SIZE = 6
INPUT_SIZE = QUAD_SIZE * 2 + 1
MAX_GENERATION_ATTEMPTS = 500
_AREA_BAG: list[tuple[int, int, int, int]] = []

_BASE_SHAPES: dict[int, tuple[np.ndarray, ...]] = {
    2: (
        np.array([[1, 1]], dtype=bool),
    ),
    3: (
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
    ),
    4: (
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    ),
    5: (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    ),
    6: (
        np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    ),
    7: (
        np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    ),
    8: (
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    ),
    9: (
        np.ones((3, 3), dtype=bool),
    ),
}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_bool(mask.astype(bool))
    return cropped.shape, cropped.astype(np.uint8).tobytes()


def _transform_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(mask, copy=True)
    for _ in range(int(rng.integers(0, 4))):
        out = np.rot90(out)
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    return _crop_bool(out)


def _quadrant_views(grid: np.ndarray, cross_r: int, cross_c: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        grid[:cross_r, :cross_c],
        grid[:cross_r, cross_c + 1 :],
        grid[cross_r + 1 :, :cross_c],
        grid[cross_r + 1 :, cross_c + 1 :],
    )


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    assert h % 2 == 1 and w % 2 == 1, "grid must have odd dimensions"
    grey_rows = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    grey_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(grey_rows) == 1 and len(grey_cols) == 1, "expected one full grey row and one full grey column"
    cross_r = int(grey_rows[0])
    cross_c = int(grey_cols[0])
    assert cross_r == h // 2 and cross_c == w // 2, "grey cross must be centered"
    return cross_r, cross_c


def _extract_pattern(quadrant: np.ndarray) -> np.ndarray:
    assert quadrant.shape == (QUAD_SIZE, QUAD_SIZE), "quadrants must be 6x6"
    allowed = {BLACK, RED}
    assert set(map(int, np.unique(quadrant))) <= allowed, "quadrant may only contain black and red"
    red_mask = quadrant == RED
    components = _components_mask(red_mask)
    assert len(components) == 1, "each quadrant must contain exactly one red component"
    coords = components[0]
    area = int(coords.shape[0])
    assert 2 <= area <= 9, "component area must be in [2, 9]"
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 + 1 <= 3 and c1 - c0 + 1 <= 3, "cropped bbox must be at most 3x3"
    return quadrant[r0 : r1 + 1, c0 : c1 + 1].copy()


def _recolor_pattern(cropped: np.ndarray) -> tuple[np.ndarray, int]:
    area = int(np.count_nonzero(cropped == RED))
    color = GREEN if area % 2 == 1 else BLUE
    recolored = np.where(cropped == RED, color, BLACK).astype(int)
    return recolored, color


def _tile_pattern(unit: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    h, w = unit.shape
    out_h, out_w = out_shape
    out = np.zeros((out_h, out_w), dtype=int)
    for r0 in range(0, out_h, h):
        for c0 in range(0, out_w, w):
            rr1 = min(out_h, r0 + h)
            cc1 = min(out_w, c0 + w)
            patch = unit[: rr1 - r0, : cc1 - c0]
            mask = patch != BLACK
            out[r0:rr1, c0:cc1][mask] = patch[mask]
    return out


def _solve_quadrant(quadrant: np.ndarray) -> tuple[np.ndarray, int, int]:
    cropped = _extract_pattern(quadrant)
    recolored, color = _recolor_pattern(cropped)
    tiled = _tile_pattern(recolored, quadrant.shape)
    area = int(np.count_nonzero(cropped == RED))
    return tiled, area, color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop each corner-quadrant pattern, recolor it by area parity, then tile it across its quadrant before removing the separator cross.

    Input structure:
    - The grid is a black canvas split by one full grey row and one full grey column crossing at the center.
    - The grey cross creates four 6x6 quadrants.
    - Each quadrant contains exactly one small 4-connected red pattern whose cropped bounding box is at most 3x3.

    Rule:
    - For each quadrant, crop the red pattern to its minimal bounding box.
    - Count its red foreground pixels; odd area becomes green, even area becomes blue.
    - Use the cropped recolored pattern as a tile unit and repeat it from the quadrant's top-left corner, clipping partial tiles at the quadrant boundary and drawing only non-black cells.
    - Remove the grey separator cross and concatenate the four tiled quadrants into the output.

    Key insight:
    - The output color is determined by the parity of the cropped pattern's pixel area, while the cropped bounding box determines the tiling period.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    cross_r, cross_c = _find_cross(g)
    tl, tr, bl, br = _quadrant_views(g, cross_r, cross_c)
    assert tl.shape == tr.shape == bl.shape == br.shape == (QUAD_SIZE, QUAD_SIZE)

    quadrants = []
    for quadrant in (tl, tr, bl, br):
        tiled, _area, _color = _solve_quadrant(quadrant)
        quadrants.append(tiled)

    top = np.hstack([quadrants[0], quadrants[1]])
    bottom = np.hstack([quadrants[2], quadrants[3]])
    return np.vstack([top, bottom]).astype(int)


def _shape_pool_by_area() -> dict[int, list[np.ndarray]]:
    pool: dict[int, list[np.ndarray]] = {area: [] for area in _BASE_SHAPES}
    for area, shapes in _BASE_SHAPES.items():
        seen: set[tuple[tuple[int, int], bytes]] = set()
        for shape in shapes:
            for k in range(4):
                rotated = np.rot90(shape, k)
                for flipped in (rotated, np.fliplr(rotated)):
                    cropped = _crop_bool(flipped)
                    sig = _canonical_signature(cropped)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    pool[area].append(cropped)
    return pool


_SHAPE_POOL = _shape_pool_by_area()


def _sample_areas(rng: np.random.Generator) -> list[int]:
    if not _AREA_BAG:
        combos_by_total: dict[int, list[tuple[int, int, int, int]]] = {}
        for a0 in range(2, 10):
            for a1 in range(2, 10):
                for a2 in range(2, 10):
                    for a3 in range(2, 10):
                        areas = (a0, a1, a2, a3)
                        if not any(area % 2 == 1 for area in areas):
                            continue
                        if not any(area % 2 == 0 for area in areas):
                            continue
                        if len(set(areas)) < 2:
                            continue
                        combos_by_total.setdefault(sum(areas), []).append(areas)
        totals = sorted(combos_by_total)
        rng.shuffle(totals)
        for total in totals:
            choices = combos_by_total[total]
            _AREA_BAG.append(choices[int(rng.integers(len(choices)))])
    areas = list(_AREA_BAG.pop())
    rng.shuffle(areas)
    return areas


def _sample_shape_for_area(area: int, rng: np.random.Generator) -> np.ndarray:
    base = _SHAPE_POOL[area][int(rng.integers(len(_SHAPE_POOL[area])))]
    return _transform_mask(base, rng)


def _place_shape_in_quadrant(shape: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    quadrant = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=int)
    h, w = shape.shape
    r0 = int(rng.integers(0, QUAD_SIZE - h + 1))
    c0 = int(rng.integers(0, QUAD_SIZE - w + 1))
    quadrant[r0 : r0 + h, c0 : c0 + w][shape] = RED
    return quadrant


def _build_input(quadrants: list[np.ndarray]) -> np.ndarray:
    inp = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=int)
    inp[QUAD_SIZE, :] = GREY
    inp[:, QUAD_SIZE] = GREY
    inp[:QUAD_SIZE, :QUAD_SIZE] = quadrants[0]
    inp[:QUAD_SIZE, QUAD_SIZE + 1 :] = quadrants[1]
    inp[QUAD_SIZE + 1 :, :QUAD_SIZE] = quadrants[2]
    inp[QUAD_SIZE + 1 :, QUAD_SIZE + 1 :] = quadrants[3]
    return inp


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != (INPUT_SIZE, INPUT_SIZE) or out.shape != (QUAD_SIZE * 2, QUAD_SIZE * 2):
        return False
    try:
        cross_r, cross_c = _find_cross(inp)
    except AssertionError:
        return False

    if set(map(int, np.unique(out))) - {BLACK, BLUE, GREEN}:
        return False
    if GREY in out or RED in out:
        return False

    quadrant_inputs = _quadrant_views(inp, cross_r, cross_c)
    quadrant_outputs = (
        out[:QUAD_SIZE, :QUAD_SIZE],
        out[:QUAD_SIZE, QUAD_SIZE:],
        out[QUAD_SIZE:, :QUAD_SIZE],
        out[QUAD_SIZE:, QUAD_SIZE:],
    )

    seen_output_colors: set[int] = set()
    for q_in, q_out in zip(quadrant_inputs, quadrant_outputs, strict=True):
        try:
            cropped = _extract_pattern(q_in)
        except AssertionError:
            return False
        area = int(np.count_nonzero(cropped == RED))
        expected_color = GREEN if area % 2 == 1 else BLUE
        non_black = set(map(int, np.unique(q_out))) - {BLACK}
        if len(non_black) != 1 or next(iter(non_black)) != expected_color:
            return False
        expected = _tile_pattern(np.where(cropped == RED, expected_color, BLACK), (QUAD_SIZE, QUAD_SIZE))
        if not np.array_equal(q_out, expected):
            return False
        seen_output_colors.add(expected_color)

    return seen_output_colors == {BLUE, GREEN}


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    areas = _sample_areas(rng)

    shapes = [_sample_shape_for_area(area, rng) for area in areas]
    if len({_canonical_signature(shape) for shape in shapes}) < 3:
        return None

    quadrants = [_place_shape_in_quadrant(shape, rng) for shape in shapes]
    inp = _build_input(quadrants)

    # Encourage visible cropping by rejecting cases where every pattern is already corner-anchored.
    pattern_offsets = []
    for quadrant in quadrants:
        coords = np.argwhere(quadrant == RED)
        r0, c0, _r1, _c1 = _bbox(coords)
        pattern_offsets.append((r0, c0))
    if all(offset == (0, 0) for offset in pattern_offsets):
        return None

    try:
        out = main(inp)
    except AssertionError:
        return None

    if not _semantic_check(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-tiling ARC input")
