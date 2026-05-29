# Combo variant of 7491f3cf (index 6)
# combo_spec: D1/3+D12/1
# Name: 连通分量分割+无显式指示像素

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


_THIS_FILE = Path(__file__).resolve()
ROOT = None
for _parent in _THIS_FILE.parents:
    if (_parent / "solvers" / "arc_utils.py").exists():
        ROOT = _parent
        break
assert ROOT is not None, "could not locate repo root containing solvers/arc_utils.py"
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import label_mask


# Design analysis for the fused variant:
# a) Concrete behavior / trace
#    - Original 7491f3cf: tile 0 had a continuous 5-pixel wall plus one detached
#      indicator pixel. Flood-fill from that explicit dot picked the tile-1 side.
#    - D1/3 removes the line-wall assumption: tile 0 now contains several small
#      blocker cells. The relevant structure is no longer a path but the way these
#      blockers split the background into 4-connected components.
#    - D12/1 removes the explicit indicator: no singleton cue pixel remains. The
#      tile-1 region must therefore be recovered from a visible property of the
#      background components themselves.
#    - Fused rule used here: compute all 4-connected background components of
#      tile 0 after treating the cue cells as blockers, then take the uniquely
#      smallest component as "inside". Example: a 7-blocker layout may split the
#      5x5 background into components of sizes 5 and 13; the size-5 pocket uses
#      tile 1, the size-13 region uses tile 2, and blocker cells themselves are
#      overlaid using tile 1 first then tile 2 fallback.
# b) Complexity calibration
#    - The generator uses only 8 transformed copies of one blocker template.
#      This keeps the latent rule small enough to repeat across 2-4 examples,
#      but still visually different from the original straight/bent walls.
#    - Every emitted cue splits the background into exactly two components with
#      sizes 5 and 13. The size contrast is large and stable, so "small pocket"
#      is inferable by eye rather than by hidden arithmetic.
# c) Original-code dependency audit
#    - "There is one 8-isolated indicator pixel": invalid now; removed.
#    - "Indicator must be on a corner/edge midpoint": invalid now; removed.
#    - "Cue pixels form one symmetric wall through the center": invalid now;
#      replaced by dispersed blockers with at least two 8-components.
#    - "Flood-fill starts at the indicator": invalid now; replaced by selecting
#      the smallest background component by area.
#    - "Wall 4-connectivity controls boundary fallback": specific to line walls
#      and dropped. Blocker cells are simply separator cells using t1 then t2.
# d) Placement legality
#    - Tile size and global 7x25 frame are unchanged.
#    - The precomputed blocker templates are all in-bounds 5x5 coordinate sets.
#      They never overlap each other beyond a single cell because they are just a
#      set of occupied cue positions written once into tile 0.
# e) Visual observability
#    - The absence of a detached dot is directly visible.
#    - The cue is visibly not a line: it appears as several scattered islands.
#    - The smallest enclosed/pocket-like background region repeats across all
#      samples, so a human can infer "that little region gets tile 1" from a few
#      input/output pairs without relying on color ids.
# f) Visual neatness
#    - Tile 0 uses one cue color only, avoiding clutter from extra symbol colors.
#    - The blocker template is sparse enough that the two background regions are
#      still easy to trace visually.
# g) Color reasoning
#    - Colors remain purely role-based: border, shared tile background, cue,
#      tile-1 pattern color, tile-2 pattern color. No numeric color ordering is
#      used anywhere in the solver.
# h) Generator guarantees
#    - It emits only blocker layouts with exactly two background components and a
#      unique smallest one.
#    - It rejects samples where tile 1 contributes nothing to the smallest
#      component or tile 2 contributes nothing to the larger component, so the
#      area-based split is behaviorally visible in tile 3.
#    - It also rejects samples where flipping the choice to the largest component
#      would produce the same tile-3 output; this guards against degeneration.
# i) New solver assumptions
#    - Tile 0 contains one cue color forming several dispersed blocker islands.
#    - Background components must have a unique smallest area.
#    - The generator constructs exactly those conditions, so main() is not making
#      unsupported guesses on the produced distribution.


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

_BASE_BLOCKERS = frozenset({(0, 0), (0, 2), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)})


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


def _background_labels(
    blockers: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> tuple[np.ndarray, int]:
    passable = np.ones((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in blockers:
        passable[r, c] = False
    return label_mask(passable, connectivity=4)


def _blocker_8_component_count(blockers: frozenset[tuple[int, int]]) -> int:
    mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in blockers:
        mask[r, c] = True
    _labels, n_comp = label_mask(mask, connectivity=8)
    return int(n_comp)


def _smallest_component_label(labels: np.ndarray, n_comp: int) -> int:
    assert n_comp >= 2, "blockers must split background into at least two components"
    sizes = [(k, int(np.count_nonzero(labels == k))) for k in range(1, n_comp + 1)]
    min_size = min(size for _k, size in sizes)
    winners = [k for k, size in sizes if size == min_size]
    assert len(winners) == 1, f"smallest component must be unique, got sizes {sizes}"
    return int(winners[0])


def _build_configs() -> list[frozenset[tuple[int, int]]]:
    configs: list[frozenset[tuple[int, int]]] = []
    seen = set()
    for kind in range(8):
        blockers = frozenset(_transform_cell(cell, kind) for cell in _BASE_BLOCKERS)
        if blockers in seen:
            continue
        seen.add(blockers)
        labels, n_comp = _background_labels(blockers)
        if n_comp != 2:
            continue
        sizes = sorted(int(np.count_nonzero(labels == k)) for k in range(1, n_comp + 1))
        if sizes != [5, 13]:
            continue
        if _blocker_8_component_count(blockers) < 2:
            continue
        _smallest_component_label(labels, n_comp)
        configs.append(blockers)
    assert len(configs) == 8, f"expected 8 fused blocker configs, got {len(configs)}"
    return configs


_FUSED_CONFIGS = _build_configs()


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


def _parse_cue_tile(t0: np.ndarray) -> tuple[int, frozenset[tuple[int, int]]]:
    t0_bg = int(Counter(map(int, t0.flatten())).most_common(1)[0][0])
    cue_colors = {int(v) for v in t0.flatten() if int(v) != t0_bg}
    assert cue_colors, "no cue pixels found"
    assert len(cue_colors) == 1, f"expected one cue color, got {sorted(cue_colors)}"

    blockers = frozenset(
        (int(r), int(c))
        for r in range(TILE_SZ)
        for c in range(TILE_SZ)
        if int(t0[r, c]) != t0_bg
    )
    assert len(blockers) >= 5, "not enough blocker cells"
    assert _blocker_8_component_count(blockers) >= 2, "cue must be dispersed, not one connected wall"

    labels, n_comp = _background_labels(blockers)
    _smallest_component_label(labels, n_comp)
    return t0_bg, blockers


def _inside_and_labels(
    blockers: frozenset[tuple[int, int]],
) -> tuple[set[tuple[int, int]], np.ndarray, int, int]:
    labels, n_comp = _background_labels(blockers)
    inside_label = _smallest_component_label(labels, n_comp)
    inside = {
        (int(r), int(c))
        for r in range(TILE_SZ)
        for c in range(TILE_SZ)
        if int(labels[r, c]) == inside_label
    }
    assert inside, "inside component is empty"
    return inside, labels, n_comp, inside_label


def _composite_output_tile(
    t0: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
) -> np.ndarray:
    _t0_bg, blockers = _parse_cue_tile(t0)
    inside, labels, n_comp, inside_label = _inside_and_labels(blockers)
    assert n_comp == 2, "generator-supported distribution expects exactly two background components"

    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in blockers:
                if v1 != bg:
                    out3[r, c] = v1
                elif v2 != bg:
                    out3[r, c] = v2
            elif int(labels[r, c]) == inside_label:
                if v1 != bg:
                    out3[r, c] = v1
            else:
                if v2 != bg:
                    out3[r, c] = v2
    return out3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two pattern tiles into the fourth panel using dispersed blocker
    cells and an implicit connected-component cue.

    Input structure:
    - Four 5x5 panels in one row, separated by a 1-cell border, inside a fixed
      7x25 grid.
    - Panel 0 contains one cue color forming several dispersed blocker cells.
      There is no explicit indicator pixel.
    - Panels 1-2 contain source patterns; panel 3 is blank in the input.

    Rule:
    - Treat the cue cells in panel 0 as blockers and label the remaining
      background cells by 4-connected components.
    - The uniquely smallest background component is the implicit inside region
      and takes non-background pixels from panel 1.
    - Every other background component takes non-background pixels from panel 2.
      Blocker cells themselves show panel 1 first, then panel 2 as fallback.

    Key insight:
    - The cue is no longer a wall-plus-dot instruction. The solver must notice
      that the scattered cue cells partition the background, then infer inside
      from component size rather than from any explicit marker.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    bg = int(Counter(map(int, t1.flatten())).most_common(1)[0][0])
    assert border != bg, "border color must differ from tile background"

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = _composite_output_tile(t0, t1, t2, bg)
    return out


def _largest_component_output(
    blockers: frozenset[tuple[int, int]],
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
) -> np.ndarray:
    labels, n_comp = _background_labels(blockers)
    assert n_comp >= 2
    sizes = [(k, int(np.count_nonzero(labels == k))) for k in range(1, n_comp + 1)]
    max_size = max(size for _k, size in sizes)
    largest = [k for k, size in sizes if size == max_size]
    assert len(largest) == 1
    inside_label = largest[0]

    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in blockers:
                if v1 != bg:
                    out3[r, c] = v1
                elif v2 != bg:
                    out3[r, c] = v2
            elif int(labels[r, c]) == inside_label:
                if v1 != bg:
                    out3[r, c] = v1
            else:
                if v2 != bg:
                    out3[r, c] = v2
    return out3


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice([c for c in other if c != cue_color])

    blockers = rng.choice(_FUSED_CONFIGS)
    inside, labels, n_comp, inside_label = _inside_and_labels(blockers)
    assert n_comp == 2
    outside = {
        (r, c)
        for r in range(TILE_SZ)
        for c in range(TILE_SZ)
        if int(labels[r, c]) > 0 and int(labels[r, c]) != inside_label
    }
    assert len(inside) == 5 and len(outside) == 13, "unexpected fused component sizes"

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    if not any(int(t1[r, c]) != tile_bg for r, c in inside):
        return None
    if not any(int(t2[r, c]) != tile_bg for r, c in outside):
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in blockers:
        t0[r, c] = cue_color

    out3 = _composite_output_tile(t0, t1, t2, tile_bg)
    counterfactual = _largest_component_output(blockers, t1, t2, tile_bg)
    if np.array_equal(out3, counterfactual):
        return None

    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([t0, t1, t2, t3_in]):
        g[1:6, 1 + 6 * j : 1 + 6 * j + 5] = tile

    expected = g.copy()
    expected[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    got = main(g)
    if not np.array_equal(got, expected):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
