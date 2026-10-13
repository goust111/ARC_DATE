import random
from typing import Optional

import numpy as np


def _bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every colored sprite by applying the union of all aligned partial sprite masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) >= 2, "expected at least two foreground sprites"

    sprites: list[tuple[int, int, int, np.ndarray]] = []
    for color in colors:
        mask = g == color
        box = _bbox(mask)
        assert box is not None, "every foreground color must appear"
        r0, c0, r1, c1 = box
        local = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert local.any(), "local sprite mask must be non-empty"
        sprites.append((color, r0, c0, local))

    template_h = max(local.shape[0] for _, _, _, local in sprites)
    template_w = max(local.shape[1] for _, _, _, local in sprites)
    assert template_h > 0 and template_w > 0

    template = np.zeros((template_h, template_w), dtype=bool)
    for _, _, _, local in sprites:
        h, w = local.shape
        template[:h, :w] |= local
    assert template.any(), "recovered template must be non-empty"

    out = g.copy()
    grid_h, grid_w = g.shape
    for color, r0, c0, _ in sprites:
        r1 = r0 + template_h
        c1 = c0 + template_w
        assert r1 <= grid_h and c1 <= grid_w, "full sprite window must fit inside the grid"
        region = out[r0:r1, c0:c1]
        assert np.all((region == bg) | (region == color)), "sprite windows must be separated"
        region[template] = color
    return out


_GRID_SIDE_MIN = 14
_GRID_SIDE_MAX = 26
_TEMPLATE_SIDE_MIN = 5
_TEMPLATE_SIDE_MAX = 8
_SPRITE_COUNT_MIN = 3
_SPRITE_COUNT_MAX = 5
_TARGET_FILL_FRACTION = (0.38, 0.68)
_PLACEMENT_GAP = 1
_MAX_ATTEMPTS = 400


def _sample_prefixes(height: int, width: int, count: int) -> list[tuple[int, int]]:
    heights = sorted(random.sample(range(1, height), count - 1)) + [height]
    inner_widths = sorted(random.sample(range(1, width), count - 1), reverse=True)
    widths = [width] + inner_widths
    prefixes = list(zip(heights, widths))
    assert len(prefixes) == count
    assert prefixes[0][1] == width and prefixes[-1][0] == height
    assert all(h < height or w < width for h, w in prefixes)
    return prefixes


def _union_mask(height: int, width: int, prefixes: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for h, w in prefixes:
        mask[:h, :w] = True
    return mask


def _grow_template(allowed: np.ndarray) -> np.ndarray:
    height, width = allowed.shape
    template = np.zeros_like(allowed, dtype=bool)

    template[0, :] = True
    template[:, 0] = True

    target = random.randint(
        max(int(template.sum()), int(allowed.sum() * _TARGET_FILL_FRACTION[0])),
        max(int(template.sum()), int(allowed.sum() * _TARGET_FILL_FRACTION[1])),
    )

    def neighbors(r: int, c: int):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width:
                yield rr, cc

    frontier = set()
    for r, c in np.argwhere(template):
        for rr, cc in neighbors(int(r), int(c)):
            if allowed[rr, cc] and not template[rr, cc]:
                frontier.add((rr, cc))

    while template.sum() < target and frontier:
        rr, cc = random.choice(tuple(frontier))
        frontier.remove((rr, cc))
        template[rr, cc] = True
        for nr, nc in neighbors(rr, cc):
            if allowed[nr, nc] and not template[nr, nc]:
                frontier.add((nr, nc))

    template &= allowed
    assert template[0, 0]
    assert template[0, width - 1]
    assert template[height - 1, 0]
    return template


def _local_mask(template: np.ndarray, h: int, w: int) -> np.ndarray:
    local = template[:h, :w]
    box = _bbox(local)
    assert box is not None
    r0, c0, r1, c1 = box
    cropped = local[r0 : r1 + 1, c0 : c1 + 1]
    assert r0 == 0 and c0 == 0
    return cropped


def _place_windows(grid_h: int, grid_w: int, height: int, width: int, count: int) -> Optional[list[tuple[int, int]]]:
    placements: list[tuple[int, int]] = []
    for _ in range(200):
        placements.clear()
        occupied: list[tuple[int, int, int, int]] = []
        order = [(r, c) for r in range(grid_h - height + 1) for c in range(grid_w - width + 1)]
        random.shuffle(order)
        for r0, c0 in order:
            r1, c1 = r0 + height - 1, c0 + width - 1
            ok = True
            for ar0, ac0, ar1, ac1 in occupied:
                if not (r1 + _PLACEMENT_GAP < ar0 or ar1 + _PLACEMENT_GAP < r0 or c1 + _PLACEMENT_GAP < ac0 or ac1 + _PLACEMENT_GAP < c0):
                    ok = False
                    break
            if ok:
                placements.append((r0, c0))
                occupied.append((r0, c0, r1, c1))
                if len(placements) == count:
                    return placements
        if len(placements) == count:
            return placements
    return None


def _expected_output(
    grid: np.ndarray,
    template: np.ndarray,
    placements: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    h, w = template.shape
    for (r0, c0), color in zip(placements, colors):
        out[r0 : r0 + h, c0 : c0 + w][template] = color
    return out


def _gen_attempt() -> Optional[np.ndarray]:
    sprite_count = random.randint(_SPRITE_COUNT_MIN, _SPRITE_COUNT_MAX)
    template_h = random.randint(max(_TEMPLATE_SIDE_MIN, sprite_count), _TEMPLATE_SIDE_MAX)
    template_w = random.randint(max(_TEMPLATE_SIDE_MIN, sprite_count), _TEMPLATE_SIDE_MAX)

    prefixes = _sample_prefixes(template_h, template_w, sprite_count)
    allowed = _union_mask(template_h, template_w, prefixes)
    template = _grow_template(allowed)

    local_masks = [_local_mask(template, h, w) for h, w in prefixes]
    recovered_h = max(mask.shape[0] for mask in local_masks)
    recovered_w = max(mask.shape[1] for mask in local_masks)
    if recovered_h != template_h or recovered_w != template_w:
        return None
    if any(mask.shape == template.shape and np.array_equal(mask, template) for mask in local_masks):
        return None

    union = np.zeros_like(template)
    for mask in local_masks:
        h, w = mask.shape
        union[:h, :w] |= mask
    if not np.array_equal(union, template):
        return None

    min_h = max(_GRID_SIDE_MIN, template_h + 2, sprite_count * (template_h + _PLACEMENT_GAP) - _PLACEMENT_GAP)
    min_w = max(_GRID_SIDE_MIN, template_w + 2)
    grid_h = random.randint(min_h, max(min_h, _GRID_SIDE_MAX))
    grid_w = random.randint(min_w, max(min_w, _GRID_SIDE_MAX))

    placements = _place_windows(grid_h, grid_w, template_h, template_w, sprite_count)
    if placements is None:
        return None

    bg = random.randint(0, 9)
    colors = random.sample([c for c in range(10) if c != bg], sprite_count)
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    for (r0, c0), color, mask in zip(placements, colors, local_masks):
        h, w = mask.shape
        grid[r0 : r0 + h, c0 : c0 + w][mask] = color

    expected = _expected_output(grid, template, placements, colors)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(grid, produced):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
