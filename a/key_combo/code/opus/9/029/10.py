from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np


_BG = 0
_BLOCK = 5
_RAY = 8
_SEED_COLORS = (1, 2, 3, 4)
_ARMS = ("up", "down", "left", "right")
_GRID_MIN = 5
_GRID_MAX = 8
_MAX_ATTEMPTS = 400

Region = Tuple[int, int, int, int]


def _scale2(grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


def _find_source(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == _RAY)
    assert coords.shape == (1, 2), "expected exactly one teal source pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _nearest_boundaries(grid: np.ndarray, sr: int, sc: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    top = max((r for r in range(0, sr) if int(grid[r, sc]) == _BLOCK), default=-1)
    bottom = min((r for r in range(sr + 1, h) if int(grid[r, sc]) == _BLOCK), default=h)
    left = max((c for c in range(0, sc) if int(grid[sr, c]) == _BLOCK), default=-1)
    right = min((c for c in range(sc + 1, w) if int(grid[sr, c]) == _BLOCK), default=w)
    return top, bottom, left, right


def _region_boxes(sr: int, sc: int, top: int, bottom: int, left: int, right: int) -> Dict[str, Region]:
    boxes: Dict[str, Region] = {}
    spans_v = {"top": (top + 1, sr - 1), "bottom": (sr + 1, bottom - 1)}
    spans_h = {"left": (left + 1, sc - 1), "right": (sc + 1, right - 1)}
    for v_name, (r0, r1) in spans_v.items():
        if r0 > r1:
            continue
        for h_name, (c0, c1) in spans_h.items():
            if c0 > c1:
                continue
            boxes[f"{v_name}_{h_name}"] = (r0, c0, r1, c1)
    return boxes


def _paint_rays(grid: np.ndarray, sr: int, sc: int, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    out = grid.copy()
    out[sr, sc] = _RAY
    if top + 1 <= sr - 1:
        out[top + 1 : sr, sc] = _RAY
    if sr + 1 <= bottom - 1:
        out[sr + 1 : bottom, sc] = _RAY
    if left + 1 <= sc - 1:
        out[sr, left + 1 : sc] = _RAY
    if sc + 1 <= right - 1:
        out[sr, sc + 1 : right] = _RAY
    return out


def _seed_color(region: np.ndarray) -> int | None:
    colors = sorted(set(map(int, np.unique(region))) - {_BG, _BLOCK, _RAY})
    assert len(colors) <= 1, "each region must contain at most one seed color"
    return colors[0] if colors else None


def _solve_coarse(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sr, sc = _find_source(g)
    top, bottom, left, right = _nearest_boundaries(g, sr, sc)
    out = _paint_rays(g, sr, sc, top, bottom, left, right)

    for r0, c0, r1, c1 in _region_boxes(sr, sc, top, bottom, left, right).values():
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        color = _seed_color(patch)
        if color is None:
            continue
        fill_mask = patch == _BG
        patch[fill_mask] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the grid, project a teal cross until grey blockers or edges, then fill each bounded rectangle from its local seed.

    Input structure:
    - The grid has black background, exactly one teal source pixel, several grey blocker pixels, and 2-4 colored seed pixels.
    - Grey blockers that matter lie on the source row or source column and stop the cardinal projections before their cells.
    - Colored seeds sit inside the rectangles determined by the nearest blocker-or-edge above, below, left, and right of the source.

    Rule:
    - First interpret the small grid as a 2x-upscaled pixel-art canvas, so every input cell becomes a solid 2x2 block in the output.
    - From the teal source block, cast a width-2 teal ray upward, downward, leftward, and rightward, stopping immediately before the nearest grey blocker block on that arm, or at the grid edge if there is no blocker.
    - These four stopping positions define up to four axis-aligned interior rectangles around the source; for each rectangle, if it contains a non-black, non-teal, non-grey seed block, fill every black cell in that rectangle with the seed color and keep the seed itself unchanged.

    Key insight:
    - The enlargement is structural rather than cosmetic: it turns the source, blockers, rays, and seeds into aligned 2x2 blocks, while the blocker positions on the source row and column determine the four fillable rectangles and each rectangle gets its color only from the seed already inside it.
    """
    return _scale2(_solve_coarse(grid))


def _pick_source(h: int, w: int) -> tuple[int, int]:
    return random.randint(2, h - 3), random.randint(2, w - 3)


def _place_arm_blockers(grid: np.ndarray, sr: int, sc: int, arms: tuple[str, ...]) -> None:
    h, w = grid.shape
    for arm in arms:
        if arm == "up":
            grid[random.randint(0, sr - 2), sc] = _BLOCK
        elif arm == "down":
            grid[random.randint(sr + 2, h - 1), sc] = _BLOCK
        elif arm == "left":
            grid[sr, random.randint(0, sc - 2)] = _BLOCK
        else:
            grid[sr, random.randint(sc + 2, w - 1)] = _BLOCK


def _random_seed_colors(n: int) -> list[int]:
    if n == 1:
        return [random.choice(_SEED_COLORS)]
    if random.random() < 0.3:
        colors = [random.choice(_SEED_COLORS) for _ in range(n)]
        if len(set(colors)) >= 2:
            return colors
    return random.sample(_SEED_COLORS, n)


def _random_cell(box: Region) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return random.randint(r0, r1), random.randint(c0, c1)


def _is_uniform_2x2(output: np.ndarray) -> bool:
    h, w = output.shape
    if h % 2 or w % 2:
        return False
    for r in range(0, h, 2):
        for c in range(0, w, 2):
            block = output[r : r + 2, c : c + 2]
            if not np.all(block == int(block[0, 0])):
                return False
    return True


def _validate_sample(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2
    assert _GRID_MIN <= h <= _GRID_MAX and _GRID_MIN <= w <= _GRID_MAX

    sr, sc = _find_source(g)
    assert 1 <= sr < h - 1 and 1 <= sc < w - 1

    blockers = np.argwhere(g == _BLOCK)
    seeds = np.argwhere(np.isin(g, _SEED_COLORS))
    assert 2 <= len(blockers) <= 4
    assert 2 <= len(seeds) <= 4

    for r, c in blockers:
        rr, cc = int(r), int(c)
        assert rr == sr or cc == sc, "generator blockers must lie on a source arm"
        assert abs(rr - sr) + abs(cc - sc) >= 2, "blockers must not touch the source"

    top, bottom, left, right = _nearest_boundaries(g, sr, sc)
    boxes = _region_boxes(sr, sc, top, bottom, left, right)
    assert len(boxes) == 4, "generator must create four visible candidate rectangles"

    seed_regions = 0
    for box in boxes.values():
        r0, c0, r1, c1 = box
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted(set(map(int, np.unique(patch))) - {_BG, _BLOCK, _RAY})
        assert len(colors) <= 1
        if colors:
            seed_regions += 1
    assert 2 <= seed_regions <= 4

    out = main(g)
    assert out.shape == (2 * h, 2 * w)
    assert _is_uniform_2x2(out), "output must be a pure 2x block upscale of some coarse state"

    coarse_out = out[::2, ::2]
    assert np.array_equal(out, _scale2(coarse_out))
    assert coarse_out[sr, sc] == _RAY
    assert np.array_equal(coarse_out[g == _BLOCK], np.full(len(blockers), _BLOCK))

    expected_rays = _paint_rays(g, sr, sc, top, bottom, left, right)
    ray_mask = expected_rays == _RAY
    assert np.array_equal(coarse_out[ray_mask], np.full(int(ray_mask.sum()), _RAY))

    for box in boxes.values():
        r0, c0, r1, c1 = box
        inp_patch = g[r0 : r1 + 1, c0 : c1 + 1]
        out_patch = coarse_out[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted(set(map(int, np.unique(inp_patch))) - {_BG, _BLOCK, _RAY})
        if colors:
            color = colors[0]
            assert np.all(np.isin(out_patch, (_BLOCK, color)) | (out_patch == color))
            assert not np.any(out_patch == _BG), "seeded rectangles must be fully filled"
        else:
            assert np.all(out_patch == _BG), "unseeded rectangles must stay black"


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(_GRID_MIN, _GRID_MAX)
    w = random.randint(_GRID_MIN, _GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    sr, sc = _pick_source(h, w)
    grid[sr, sc] = _RAY

    arm_count = random.randint(2, 4)
    arms = tuple(random.sample(_ARMS, arm_count))
    _place_arm_blockers(grid, sr, sc, arms)

    top, bottom, left, right = _nearest_boundaries(grid, sr, sc)
    boxes = _region_boxes(sr, sc, top, bottom, left, right)
    if len(boxes) != 4:
        return None

    seeded_names = random.sample(sorted(boxes), random.randint(2, 4))
    for color, name in zip(_random_seed_colors(len(seeded_names)), seeded_names):
        r, c = _random_cell(boxes[name])
        if grid[r, c] != _BG:
            return None
        grid[r, c] = color

    try:
        _validate_sample(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
