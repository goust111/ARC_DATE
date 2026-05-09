from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import label_mask


# Variant of 7491f3cf
# Dimension: D1 - region split rule (match_cond)
# Change: line boundary -> connected-component split. Tile 0 contains several
# dispersed blocker cells plus one distinct indicator cell. The 4-connected
# background component containing the indicator takes tile 1; every other
# background component takes tile 2. Blocker cells are boundary cells and use
# the same local t1/t2 fallback as a visible separator.

TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10
T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300

_ORBITS = [
    [(0, 0), (4, 4)],
    [(0, 4), (4, 0)],
    [(2, 2)],
    [(1, 1), (3, 3)],
    [(1, 3), (3, 1)],
    [(0, 1), (1, 0), (3, 4), (4, 3)],
    [(0, 2), (2, 0), (2, 4), (4, 2)],
    [(0, 3), (3, 0), (1, 4), (4, 1)],
    [(1, 2), (2, 1), (2, 3), (3, 2)],
]

_VALID_INDICATORS = {(0, 0), (0, 4), (4, 0), (4, 4), (0, 2), (2, 0), (2, 4), (4, 2)}

_BASE_BLOCKERS = [
    frozenset({(0, 0), (0, 2), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)}),
    frozenset({(0, 2), (1, 0), (1, 3), (2, 3), (3, 1), (4, 1), (4, 4)}),
]


def _transform_cell(cell: tuple[int, int], kind: int) -> tuple[int, int]:
    r, c = cell
    if kind == 0:
        return r, c
    if kind == 1:
        return c, 4 - r
    if kind == 2:
        return 4 - r, 4 - c
    if kind == 3:
        return 4 - c, r
    if kind == 4:
        return r, 4 - c
    if kind == 5:
        return 4 - r, c
    if kind == 6:
        return c, r
    return 4 - c, 4 - r


def _background_labels(blockers: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[np.ndarray, int]:
    passable = np.ones((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in blockers:
        passable[r, c] = False
    return label_mask(passable, connectivity=4)


def _valid_indicator_for(blockers: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> bool:
    if indicator in blockers:
        return False
    ir, ic = indicator
    if any(abs(ir - r) <= 1 and abs(ic - c) <= 1 for r, c in blockers):
        return False
    labels, n_comp = _background_labels(blockers)
    if n_comp < 2:
        return False
    inside_label = int(labels[indicator])
    inside_size = int(np.count_nonzero(labels == inside_label))
    other_sizes = [int(np.count_nonzero(labels == k)) for k in range(1, n_comp + 1) if k != inside_label]
    return 3 <= inside_size <= 16 and any(size >= 2 for size in other_sizes)


def _blocker_8_component_count(blockers: frozenset[tuple[int, int]]) -> int:
    mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in blockers:
        mask[r, c] = True
    _lab, n_comp = label_mask(mask, connectivity=8)
    return int(n_comp)


def _build_configs() -> list[tuple[tuple[int, int], frozenset[tuple[int, int]]]]:
    configs: list[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = []
    seen = set()
    for base in _BASE_BLOCKERS:
        for kind in range(8):
            blockers = frozenset(_transform_cell(cell, kind) for cell in base)
            if _blocker_8_component_count(blockers) < 2:
                continue
            for indicator in sorted(_VALID_INDICATORS):
                if not _valid_indicator_for(blockers, indicator):
                    continue
                key = (indicator, tuple(sorted(blockers)))
                if key not in seen:
                    seen.add(key)
                    configs.append((indicator, blockers))
    assert configs, "no D1 connected-component blocker configs"
    return configs


_D1_CONFIGS = _build_configs()


def _sample_symmetric_pattern(
    bg: int,
    fg: int,
    rng: random.Random,
    *,
    min_px: int,
    max_px: int,
) -> np.ndarray | None:
    for _ in range(PATTERN_RETRIES):
        order = list(range(len(_ORBITS)))
        rng.shuffle(order)
        p = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
        count = 0
        for i in order:
            orbit = _ORBITS[i]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    p[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return p
    return None


def _parse_cue_tile(t0: np.ndarray) -> tuple[int, tuple[int, int], set[tuple[int, int]]]:
    t0_bg = int(Counter(map(int, t0.flatten())).most_common(1)[0][0])
    color_counts = Counter(int(v) for v in t0.flatten() if int(v) != t0_bg)
    assert len(color_counts) >= 2, "tile0 needs blocker color and indicator color"

    singleton_colors = [color for color, count in color_counts.items() if count == 1]
    assert len(singleton_colors) == 1, "indicator must be the only singleton cue color"
    indicator_color = int(singleton_colors[0])
    coords = np.argwhere(t0 == indicator_color)
    indicator = (int(coords[0, 0]), int(coords[0, 1]))
    assert indicator in _VALID_INDICATORS, f"indicator {indicator} not at corner/edge-midpoint"

    blockers = {
        (int(r), int(c))
        for r in range(TILE_SZ)
        for c in range(TILE_SZ)
        if int(t0[r, c]) != t0_bg and int(t0[r, c]) != indicator_color
    }
    assert len(blockers) >= 5, "not enough blocker cells"
    assert _blocker_8_component_count(frozenset(blockers)) >= 2, "blockers must be dispersed"
    assert _valid_indicator_for(frozenset(blockers), indicator), "invalid component split or indicator"
    return t0_bg, indicator, blockers


def _composite(
    t0: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    tile_bg: int,
) -> np.ndarray:
    _t0_bg, indicator, blockers = _parse_cue_tile(t0)
    labels, n_comp = _background_labels(blockers)
    assert n_comp >= 2, "blockers must split the background"
    inside_label = int(labels[indicator])
    assert inside_label > 0, "indicator must sit in a background component"

    out3 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in blockers:
                if v1 != tile_bg:
                    out3[r, c] = v1
                elif v2 != tile_bg:
                    out3[r, c] = v2
            elif int(labels[r, c]) == inside_label:
                if v1 != tile_bg:
                    out3[r, c] = v1
            else:
                if v2 != tile_bg:
                    out3[r, c] = v2
    return out3


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    tile_bg = int(Counter(map(int, t1.flatten())).most_common(1)[0][0])
    border = int(g[0, 0])
    assert border != tile_bg, "border color must differ from tile background"

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = _composite(t0, t1, t2, tile_bg)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    blocker_color, indicator_color, t1_color, t2_color = rng.sample(other, 4)
    border = rng.choice([c for c in other if c != blocker_color])

    indicator, blockers = rng.choice(_D1_CONFIGS)

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in blockers:
        t0[r, c] = blocker_color
    t0[indicator] = indicator_color

    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([t0, t1, t2, t3_in]):
        g[1:6, 1 + 6 * j : 1 + 6 * j + 5] = tile

    expected = g.copy()
    expected[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = _composite(t0, t1, t2, tile_bg)
    got = main(g)
    if not np.array_equal(got, expected):
        return None

    # Degeneracy guard: if every non-indicator component would yield the same
    # tile choice as the indicator component, the split is visually irrelevant.
    labels, _n = _background_labels(blockers)
    inside = int(labels[indicator])
    other_has_t2 = any(int(labels[r, c]) != inside and int(t2[r, c]) != tile_bg for r in range(5) for c in range(5))
    inside_has_t1 = any(int(labels[r, c]) == inside and int(t1[r, c]) != tile_bg for r in range(5) for c in range(5))
    if not (inside_has_t1 and other_has_t2):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
