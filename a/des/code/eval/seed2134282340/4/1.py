from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7

STRIP = 3
DIRS = ("up", "right", "down", "left")

# The four 2x2 L-corners are interpreted clockwise as up/right/down/left.
L_TO_DIR = {
    frozenset({(0, 0), (0, 1), (1, 0)}): "up",
    frozenset({(0, 0), (0, 1), (1, 1)}): "right",
    frozenset({(0, 1), (1, 0), (1, 1)}): "down",
    frozenset({(0, 0), (1, 0), (1, 1)}): "left",
}
DIR_TO_L = {
    direction: tuple(sorted(shape))
    for shape, direction in L_TO_DIR.items()
}


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
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _horizontal_tile(strip: np.ndarray) -> np.ndarray:
    _, w = strip.shape
    for period in range(1, w + 1):
        if all(np.array_equal(strip[:, c], strip[:, c % period]) for c in range(w)):
            return strip[:, :period].copy()
    return strip.copy()


def _vertical_tile(strip: np.ndarray) -> np.ndarray:
    h, _ = strip.shape
    for period in range(1, h + 1):
        if all(np.array_equal(strip[r, :], strip[r % period, :]) for r in range(h)):
            return strip[:period, :].copy()
    return strip.copy()


def _direction_for_region(core: np.ndarray, comp: list[tuple[int, int]]) -> str:
    red_cells = [(r, c) for r, c in comp if int(core[r, c]) == RED]
    assert len(red_cells) == 3, "each enclosed region must contain exactly one 3-pixel red L"
    r0 = min(r for r, _ in red_cells)
    r1 = max(r for r, _ in red_cells)
    c0 = min(c for _, c in red_cells)
    c1 = max(c for _, c in red_cells)
    assert r1 - r0 == 1 and c1 - c0 == 1, "red marker must fit in a 2x2 box"
    local = frozenset((r - r0, c - c0) for r, c in red_cells)
    assert local in L_TO_DIR, "red marker must be an L triomino"
    return L_TO_DIR[local]


def _tile_into_region(
    out: np.ndarray,
    comp: list[tuple[int, int]],
    tile: np.ndarray,
) -> None:
    r0 = min(r for r, _ in comp)
    c0 = min(c for _, c in comp)
    th, tw = tile.shape
    for r, c in comp:
        out[r, c] = int(tile[(r - r0) % th, (c - c0) % tw])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the edge texture selected by its internal red L marker.

    Input structure
    - Black background.
    - The outer frame contains four texture strips: top blue, bottom green,
      left yellow, right orange, each 3 cells thick.
    - The core contains one-cell-thick grey closed boundaries.
    - Each enclosed region contains exactly one red 3-pixel L marker.

    Rule
    - Remove the outer texture strips.
    - For each enclosed region, read the red L orientation and choose the
      corresponding edge strip.
    - Tile that strip's repeating pattern into the region, aligned to the
      region's top-left interior cell.
    - Preserve the grey boundaries.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h > 2 * STRIP and w > 2 * STRIP, "grid must contain a 3-cell strip on every side"

    top_strip = g[:STRIP, STRIP : w - STRIP]
    bottom_strip = g[h - STRIP :, STRIP : w - STRIP]
    left_strip = g[STRIP : h - STRIP, :STRIP]
    right_strip = g[STRIP : h - STRIP, w - STRIP :]

    textures = {
        "up": _horizontal_tile(top_strip),
        "down": _horizontal_tile(bottom_strip),
        "left": _vertical_tile(left_strip),
        "right": _vertical_tile(right_strip),
    }

    core = g[STRIP : h - STRIP, STRIP : w - STRIP].copy()
    out = core.copy()
    ch, cw = core.shape

    for comp in _components(core != GRAY):
        if _touches_border(comp, ch, cw):
            continue
        direction = _direction_for_region(core, comp)
        _tile_into_region(out, comp, textures[direction])

    return out


CORE_H_RANGE = (14, 24)
CORE_W_RANGE = (14, 24)
REGION_COUNT_CHOICES = [1, 2, 3, 4]
REGION_COUNT_WEIGHTS = [1, 3, 3, 2]
INNER_SIZE_RANGE = (4, 8)
REGION_MARGIN = 1
REGION_PLACE_RETRIES = 120
TEXTURE_PERIOD_RANGE = (2, 5)
GEN_ATTEMPTS = 400


def _repeat_h(tile: np.ndarray, width: int) -> np.ndarray:
    reps = (width + tile.shape[1] - 1) // tile.shape[1]
    return np.tile(tile, (1, reps))[:, :width]


def _repeat_v(tile: np.ndarray, height: int) -> np.ndarray:
    reps = (height + tile.shape[0] - 1) // tile.shape[0]
    return np.tile(tile, (reps, 1))[:height, :]


def _random_texture_tile(height: int, width: int, color: int) -> np.ndarray | None:
    for _ in range(80):
        density = random.uniform(0.35, 0.7)
        tile = np.full((height, width), BLACK, dtype=int)
        for r in range(height):
            for c in range(width):
                if random.random() < density:
                    tile[r, c] = color
        non_black = int(np.count_nonzero(tile))
        if non_black == 0 or non_black == height * width:
            continue
        if len({tuple(row) for row in tile.tolist()}) == 1 and len({tuple(tile[:, c]) for c in range(width)}) == 1:
            continue
        return tile
    return None


def _sample_textures(core_h: int, core_w: int) -> dict[str, np.ndarray] | None:
    top_w = random.randint(TEXTURE_PERIOD_RANGE[0], min(TEXTURE_PERIOD_RANGE[1], core_w))
    bot_w = random.randint(TEXTURE_PERIOD_RANGE[0], min(TEXTURE_PERIOD_RANGE[1], core_w))
    left_h = random.randint(TEXTURE_PERIOD_RANGE[0], min(TEXTURE_PERIOD_RANGE[1], core_h))
    right_h = random.randint(TEXTURE_PERIOD_RANGE[0], min(TEXTURE_PERIOD_RANGE[1], core_h))

    top_tile = _random_texture_tile(STRIP, top_w, BLUE)
    bottom_tile = _random_texture_tile(STRIP, bot_w, GREEN)
    left_tile = _random_texture_tile(left_h, STRIP, YELLOW)
    right_tile = _random_texture_tile(right_h, STRIP, ORANGE)
    if any(tile is None for tile in (top_tile, bottom_tile, left_tile, right_tile)):
        return None

    return {
        "up": top_tile,
        "down": bottom_tile,
        "left": left_tile,
        "right": right_tile,
    }


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _place_regions(core_h: int, core_w: int, n_regions: int) -> list[dict] | None:
    regions: list[dict] = []
    directions = random.sample(DIRS, k=min(n_regions, len(DIRS)))
    while len(directions) < n_regions:
        directions.append(random.choice(DIRS))
    random.shuffle(directions)

    blocked: list[tuple[int, int, int, int]] = []
    for direction in directions:
        placed = False
        for _ in range(REGION_PLACE_RETRIES):
            inner_h = random.randint(*INNER_SIZE_RANGE)
            inner_w = random.randint(*INNER_SIZE_RANGE)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            max_r0 = core_h - outer_h - REGION_MARGIN
            max_c0 = core_w - outer_w - REGION_MARGIN
            if max_r0 < REGION_MARGIN or max_c0 < REGION_MARGIN:
                return None
            r0 = random.randint(REGION_MARGIN, max_r0)
            c0 = random.randint(REGION_MARGIN, max_c0)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            padded = (bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1)
            if any(_boxes_overlap(padded, other) for other in blocked):
                continue

            lr0 = r0 + 2
            lr1 = r0 + inner_h - 2
            lc0 = c0 + 2
            lc1 = c0 + inner_w - 2
            if lr0 > lr1 or lc0 > lc1:
                continue
            l_r = random.randint(lr0, lr1)
            l_c = random.randint(lc0, lc1)

            regions.append(
                {
                    "outer": bbox,
                    "inner": (r0 + 1, c0 + 1, r0 + inner_h, c0 + inner_w),
                    "direction": direction,
                    "marker_origin": (l_r, l_c),
                }
            )
            blocked.append(padded)
            placed = True
            break
        if not placed:
            return None
    return regions


def _render_core(core_h: int, core_w: int, regions: list[dict]) -> np.ndarray:
    core = np.full((core_h, core_w), BLACK, dtype=int)
    for region in regions:
        r0, c0, r1, c1 = region["outer"]
        core[r0 : r1 + 1, c0] = GRAY
        core[r0 : r1 + 1, c1] = GRAY
        core[r0, c0 : c1 + 1] = GRAY
        core[r1, c0 : c1 + 1] = GRAY

        mr, mc = region["marker_origin"]
        for dr, dc in DIR_TO_L[region["direction"]]:
            core[mr + dr, mc + dc] = RED
    return core


def _compose_full_grid(core: np.ndarray, textures: dict[str, np.ndarray]) -> np.ndarray:
    core_h, core_w = core.shape
    full = np.full((core_h + 2 * STRIP, core_w + 2 * STRIP), BLACK, dtype=int)
    full[STRIP:-STRIP, STRIP:-STRIP] = core
    full[:STRIP, STRIP:-STRIP] = _repeat_h(textures["up"], core_w)
    full[-STRIP:, STRIP:-STRIP] = _repeat_h(textures["down"], core_w)
    full[STRIP:-STRIP, :STRIP] = _repeat_v(textures["left"], core_h)
    full[STRIP:-STRIP, -STRIP:] = _repeat_v(textures["right"], core_h)
    return full


def _expected_core(core: np.ndarray, regions: list[dict], textures: dict[str, np.ndarray]) -> np.ndarray:
    out = core.copy()
    for region in regions:
        r0, c0, r1, c1 = region["inner"]
        tile = textures[region["direction"]]
        th, tw = tile.shape
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r, c] = int(tile[(r - r0) % th, (c - c0) % tw])
    return out


def _gen_attempt() -> np.ndarray | None:
    core_h = random.randint(*CORE_H_RANGE)
    core_w = random.randint(*CORE_W_RANGE)
    n_regions = random.choices(REGION_COUNT_CHOICES, weights=REGION_COUNT_WEIGHTS)[0]

    textures = _sample_textures(core_h, core_w)
    if textures is None:
        return None
    regions = _place_regions(core_h, core_w, n_regions)
    if regions is None:
        return None

    core = _render_core(core_h, core_w, regions)
    closed = [comp for comp in _components(core != GRAY) if not _touches_border(comp, core_h, core_w)]
    if len(closed) != n_regions:
        return None

    full = _compose_full_grid(core, textures)
    expected = _expected_core(core, regions, textures)

    try:
        actual = main(full)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, core):
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to build a valid puzzle instance")
